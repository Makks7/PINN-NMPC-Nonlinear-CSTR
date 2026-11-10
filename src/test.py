"""test.py - scores the trained model on train/val/test and makes the parity plots."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import physics as phys
from model import CSTR_PINN
from preprocess import make_datasets


def metrics(y_true, y_pred):
    """I compute R2, RMSE and MAE per output channel."""
    out = {}
    for i, name in enumerate(["dCA/dt", "dT/dt"]):
        t, p = y_true[:, i], y_pred[:, i]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        out[name] = {
            "R2":   1 - ss_res / ss_tot,
            "RMSE": np.sqrt(((t - p) ** 2).mean()),
            "MAE":  np.abs(t - p).mean(),
        }
    return out


def predict(model, split, device):
    """I run the net forward on a whole split and return physical xdot."""
    model.eval()
    with torch.no_grad():
        xs = split["X_scaled"].to(device)
        up = split["X_raw"][:, 2:4].to(device)
        return model(xs, up).cpu().numpy(), split["Y_raw"].numpy()


def lyap_report(model, split, device):
    """I check what fraction of states have V_dot < 0 under the model's own Sontag law."""
    model.eval()
    with torch.no_grad():
        xs = split["X_scaled"].to(device)
        x_phys = split["X_raw"][:, 0:2].to(device)
        f, g = model.f_g(xs)
        us = torch.tensor([C.CA0S, C.QS], dtype=f.dtype, device=f.device)
        u = us + phys.sontag_controller(f, g, x_phys)
        gu = torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)
        Vdot = phys.v_dot(x_phys, f + gu)
        V = phys.lyapunov_V(x_phys)

    outside = V > C.RHO_S
    n_out = outside.sum().item()
    frac_neg = (Vdot[outside] < 0).float().mean().item() * 100
    n_pos = (Vdot[outside] >= 0).sum().item()
    return n_out, frac_neg, n_pos, Vdot[outside].min().item(), Vdot[outside].max().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(C.OUT_DIR, "pinn_best.pt"),
                    map_location=device, weights_only=False)
    model = CSTR_PINN().to(device)
    model.load_state_dict(ck["model"])

    data = make_datasets()

    print("=" * 66)
    print("PREDICTION METRICS  (physical units)")
    print("=" * 66)
    preds = {}
    for name in ["train", "val", "test"]:
        yp, yt = predict(model, data[name], device)
        preds[name] = (yt, yp)
        m = metrics(yt, yp)
        print(f"\n--- {name.upper()}  (n={len(yt)}) ---")
        print(f"{'Variable':<10} {'R2':>10} {'RMSE':>12} {'MAE':>12}")
        for k, v in m.items():
            print(f"{k:<10} {v['R2']:>10.4f} {v['RMSE']:>12.4g} {v['MAE']:>12.4g}")

    print("\n" + "=" * 66)
    print("LYAPUNOV STABILITY  (V_dot under the model's own Sontag controller)")
    print("=" * 66)
    print(f"{'Split':<8} {'states>rho_s':>13} {'V_dot<0':>10} {'V_dot>=0':>10} {'V_dot range':>28}")
    for name in ["train", "val", "test"]:
        n_out, frac, n_pos, vmin, vmax = lyap_report(model, data[name], device)
        print(f"{name:<8} {n_out:>13} {frac:>9.1f}% {n_pos:>10} "
              f"{'[%.1f, %.1f]' % (vmin, vmax):>28}")
    print("\n(paper reports 178 positive out of 12,603 -> ~98.6% negative)")

    # I plot parity for the test split, one panel per output.
    yt, yp = preds["test"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for i, name in enumerate(["dCA/dt", "dT/dt"]):
        lo = min(yt[:, i].min(), yp[:, i].min())
        hi = max(yt[:, i].max(), yp[:, i].max())
        ax[i].scatter(yt[:, i], yp[:, i], s=5, alpha=0.3, c="#2471a3")
        ax[i].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
        r2 = metrics(yt, yp)[name]["R2"]
        ax[i].set_xlabel(f"true {name}")
        ax[i].set_ylabel(f"predicted {name}")
        ax[i].set_title(f"{name}   R² = {r2:.4f}")
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT_DIR, "parity_test.png"), dpi=120)
    print("\nSaved outputs/parity_test.png")


if __name__ == "__main__":
    main()