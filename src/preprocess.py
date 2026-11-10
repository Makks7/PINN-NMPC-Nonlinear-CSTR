"""preprocess.py - loads the CSV, scales everything to [0,1] and splits into train/val/test."""

import os
import numpy as np
import pandas as pd
import torch
import config as C


class MinMaxScaler:
    """I rolled my own so the min/max are trivial to export to MATLAB later."""
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, Xs):
        return Xs * (self.max_ - self.min_) + self.min_


def load_one(path):
    """I load a single CSV and build (x,u) -> xdot pairs from it alone."""
    df = pd.read_csv(path)
    state = df[C.COL_STATES].values.astype(np.float32)
    inp   = df[C.COL_INPUTS].values.astype(np.float32)

    # I use derivative columns if the file has them, else finite-difference.
    if "dy1_dt" in df.columns and "dy2_dt" in df.columns:
        xdot = df[["dy1_dt", "dy2_dt"]].values.astype(np.float32)
    else:
        xdot  = np.diff(state, axis=0) / C.DT
        state = state[:-1]
        inp   = inp[:-1]
        # If the file holds many short trajectories, differencing straight
        # through the file invents a huge fake derivative at every join.
        # I drop those boundary rows, they are pure numerical garbage.
        if "traj_id" in df.columns:
            tid  = df["traj_id"].values[:-1]
            keep = np.diff(df["traj_id"].values) == 0
            state, inp, xdot = state[keep], inp[keep], xdot[keep]
            n_drop = (~keep).sum()
            if n_drop:
                print(f"    dropped {n_drop} trajectory-boundary rows")

    X = np.hstack([state, inp]).astype(np.float32)   # [CA, T, CA0, Q]
    Y = xdot.astype(np.float32)                      # [dCA/dt, dT/dt]
    return X, Y


def load_data():
    """I concatenate every file in C.DATA_FILES, differencing each separately."""
    Xs, Ys = [], []
    for path in C.DATA_FILES:
        X, Y = load_one(path)
        Xs.append(X); Ys.append(Y)
        print(f"  loaded {os.path.basename(path)}: {len(X)} samples")
    X_raw = np.vstack(Xs); Y_raw = np.vstack(Ys)
    return X_raw, Y_raw, len(X_raw)


def make_datasets():
    rng = np.random.default_rng(C.SEED)
    X_raw, Y_raw, N = load_data()

    idx    = rng.permutation(N)
    n_test = int(C.TEST_FRAC * N)
    n_val  = int(C.VAL_FRAC * N)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test+n_val], idx[n_test+n_val:]

    # I fit the scalers on train only. The net reads the STATE only, u enters via g_nn(x)@u.
    xscaler = MinMaxScaler().fit(X_raw[train_idx][:, 0:2])
    yscaler = MinMaxScaler().fit(Y_raw[train_idx])

    def pack(ix):
        return {
            "X_raw":    torch.tensor(X_raw[ix]),
            "Y_raw":    torch.tensor(Y_raw[ix]),
            "X_scaled": torch.tensor(xscaler.transform(X_raw[ix][:, 0:2])),
            "Y_scaled": torch.tensor(yscaler.transform(Y_raw[ix])),
        }

    return {
        "train": pack(train_idx),
        "val":   pack(val_idx),
        "test":  pack(test_idx),
        "xscaler": xscaler,
        "yscaler": yscaler,
        "n": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    }


if __name__ == "__main__":
    d = make_datasets()
    print("Sample counts:", d["n"])
    print("X_scaled range:", d["train"]["X_scaled"].min().item(), "to",
          d["train"]["X_scaled"].max().item())