"""sample_size_study.py - repeats the Fig. 2-3 study from the paper, i.e. how
the model quality changes as the training set gets bigger.

For every sample size a new PINN is trained, then I record the following on
a held-out test set and on a fixed 113x113 grid:
    - test R2 for both derivatives
    - error in the recovered g (%)
    - how many grid points have V_dot >= 0 on the true plant
Each size is repeated N_REPEATS times to get the spread (error bars).

The test set is the same for every run. If it changed, the small-sample
models could get an easier test and the comparison would not be fair.

Run:  python sample_size_study.py
Results are appended to outputs/sample_size_results.csv after every run, so
it can be stopped and restarted without losing anything.
"""

import os
import csv
import time
import numpy as np
import torch

import config as C
import physics as phys
from model import CSTR_PINN
from preprocess import load_data, MinMaxScaler
from losses import total_loss

# ---------------------------------------------------------------- settings
SAMPLE_SIZES = [500, 1000, 2500, 5000, 10000, 20000, 31920]
N_REPEATS    = 3
EPOCHS       = 400          # I cap epochs so the sweep finishes in reasonable time
TEST_FRAC    = 0.15
VAL_FRAC     = 0.15
RESULTS_CSV  = os.path.join(C.OUT_DIR, "sample_size_results.csv")


def build_pools():
    """I split once, then draw training subsets from the same pool so every run
    shares an identical held-out test set."""
    rng = np.random.default_rng(C.SEED)
    X, Y, N = load_data()
    idx = rng.permutation(N)
    n_test = int(TEST_FRAC * N)
    n_val  = int(VAL_FRAC * N)
    test_i  = idx[:n_test]
    val_i   = idx[n_test:n_test + n_val]
    train_i = idx[n_test + n_val:]
    return X, Y, train_i, val_i, test_i


def grid_states(n=113):
    """Same grid the paper uses (~12,600 states over the operating region)."""
    ca = np.linspace(-2.0, 2.0, n) + C.CAS
    t  = np.linspace(-100.0, 100.0, n) + C.TS
    CA, T = np.meshgrid(ca, t)
    return torch.tensor(np.stack([CA.ravel(), T.ravel()], 1).astype(np.float32))


def r2(true, pred):
    var = ((true - true.mean()) ** 2).sum()
    return float("nan") if var < 1e-12 else 1 - ((true - pred) ** 2).sum() / var


def train_one(Xtr, Ytr, seed, device):
    """I train a fresh model on the given subset and return it plus its scalers."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    xs = MinMaxScaler().fit(Xtr[:, 0:2])
    ys = MinMaxScaler().fit(Ytr)
    batch = {
        "X_raw":    torch.tensor(Xtr),
        "Y_raw":    torch.tensor(Ytr),
        "X_scaled": torch.tensor(xs.transform(Xtr[:, 0:2])),
        "Y_scaled": torch.tensor(ys.transform(Ytr)),
    }

    model = CSTR_PINN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.StepLR(opt, C.LR_DECAY_STEP, C.LR_DECAY_GAMMA)

    n = len(Xtr)
    bs = min(C.BATCH_SIZE, max(16, n // 8))
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            b = {k: v[sel].to(device) for k, v in batch.items()}
            loss, _ = total_loss(model, b, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
        sched.step()
    return model, xs


def evaluate(model, xs, Xte, Yte, grid, device):
    """I score prediction, g recovery, and Lyapunov violations on the TRUE plant."""
    model.eval()
    with torch.no_grad():
        # --- prediction on the fixed held-out test set ---
        Xs = torch.tensor(xs.transform(Xte[:, 0:2])).to(device)
        U  = torch.tensor(Xte[:, 2:4]).to(device)
        yp = model(Xs, U).cpu().numpy()
        r2_ca = r2(Yte[:, 0], yp[:, 0])
        r2_t  = r2(Yte[:, 1], yp[:, 1])

        # --- g recovery over the region (true g is constant) ---
        G  = grid.to(device)
        Gs = torch.tensor(xs.transform(grid.numpy())).to(device)
        f_nn, g_nn = model.f_g(Gs)
        g = g_nn.cpu().numpy()
        g11_true = C.F / C.V
        g22_true = 1.0 / (C.RHOL * C.CP * C.V)
        e11 = float(np.sqrt(((g[:, 0, 0] - g11_true) ** 2).mean()) / g11_true * 100)
        e22 = float(np.sqrt(((g[:, 1, 1] - g22_true) ** 2).mean()) / g22_true * 100)

        # --- Lyapunov: V_dot under this model's Sontag law, on the TRUE plant ---
        us = torch.tensor([C.CA0S, C.QS], device=device)
        u  = us + phys.sontag_controller(f_nn, g_nn, G)
        Vdot = phys.v_dot(G, phys.cstr_ode(G, u)).cpu().numpy()
        V    = phys.lyapunov_V(G).cpu().numpy()
        outside = V > C.RHO_S
        n_pos = int((Vdot[outside] >= 0).sum())
        n_out = int(outside.sum())

    return {"r2_dCA": r2_ca, "r2_dT": r2_t, "g11_err": e11, "g22_err": e22,
            "n_states": n_out, "n_Vdot_pos": n_pos,
            "pct_neg": 100.0 * (n_out - n_pos) / n_out}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    X, Y, train_i, val_i, test_i = build_pools()
    Xte, Yte = X[test_i], Y[test_i]
    grid = grid_states()
    print(f"Pool: {len(train_i)} train / {len(test_i)} test (fixed)")
    print(f"Grid: {len(grid)} states\n")

    # I append as I go so a crash or Ctrl-C doesn't lose completed runs.
    new_file = not os.path.exists(RESULTS_CSV)
    fh = open(RESULTS_CSV, "a", newline="")
    fields = ["n_samples", "repeat", "r2_dCA", "r2_dT", "g11_err", "g22_err",
              "n_states", "n_Vdot_pos", "pct_neg", "seconds"]
    w = csv.DictWriter(fh, fieldnames=fields)
    if new_file:
        w.writeheader()

    rng = np.random.default_rng(0)
    for m in SAMPLE_SIZES:
        m = min(m, len(train_i))
        for rep in range(N_REPEATS):
            t0 = time.time()
            sub = rng.choice(train_i, size=m, replace=False)
            model, xs = train_one(X[sub], Y[sub], seed=C.SEED + rep, device=device)
            res = evaluate(model, xs, Xte, Yte, grid, device)
            res.update({"n_samples": m, "repeat": rep,
                        "seconds": round(time.time() - t0, 1)})
            w.writerow(res); fh.flush()
            print(f"  m={m:>6} rep{rep}:  R2 {res['r2_dCA']:.4f}/{res['r2_dT']:.4f}"
                  f"   g11 {res['g11_err']:5.2f}%   Vdot>=0 {res['n_Vdot_pos']:>4}"
                  f"   ({res['seconds']:.0f}s)")
    fh.close()
    print(f"\nSaved -> {RESULTS_CSV}")
    print("Now run:  python plot_sample_size.py")


if __name__ == "__main__":
    main()