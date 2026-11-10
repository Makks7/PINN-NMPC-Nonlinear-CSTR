"""make_figures.py - builds the figures for the thesis/paper and saves all the metrics.

I kept the plots simple: data, labelled axes with units, and a small stats box
only where the number matters. Any explanation goes in the caption, not on the
figure itself, and I only compare against results I generated.
"""

import os
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import physics as phys
from preprocess import make_datasets
from model import CSTR_PINN
from validate import load_model, scale_x, true_f_g, rollout, grid

# I set one restrained style. Thin spines, muted grid, generous whitespace
# the conventions of a journal figure instead of a dashboard.
plt.rcParams.update({
    "font.size": 10.5,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11.5,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "figure.dpi": 110,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "legend.fontsize": 9.5,
})

# I use a single muted palette everywhere so the figure set reads as one piece.
INK    = "#16324f"   # deep navy, primary line
ACCENT = "#c1666b"   # muted brick, secondary line
STEEL  = "#4a7c9b"
SAGE   = "#5a7d5a"
SAND   = "#b8934a"
TRAJ   = [INK, ACCENT, SAGE, SAND]   # the four initial conditions


def r2(t, p):
    var = ((t - t.mean()) ** 2).sum()
    return float("nan") if var < 1e-12 else 1 - ((t - p) ** 2).sum() / var


def rmse(t, p): return float(np.sqrt(((t - p) ** 2).mean()))
def mae(t, p):  return float(np.abs(t - p).mean())


def _stat_box(ax, text, loc="upper left"):
    """I use one consistent annotation style for the stats, nothing else goes
    inside the axes."""
    pos = {"upper left":  (0.025, 0.965, "top", "left"),
           "lower left":  (0.025, 0.04,  "bottom", "left"),
           "lower right": (0.975, 0.04,  "bottom", "right")}[loc]
    ax.text(pos[0], pos[1], text, transform=ax.transAxes,
            va=pos[2], ha=pos[3], fontsize=9.5, color=INK, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#d5d5d5",
                      lw=0.8, alpha=0.94))


# ----------------------------------------------------------------------------
def collect_metrics(model, device, model_scalers):
    """I compute every metric once and return one dict, figures and files share it."""
    M = {}
    data = make_datasets()

    # --- prediction metrics on each split ---
    M["prediction"] = {}
    preds = {}
    for split in ["train", "val", "test"]:
        d = data[split]
        with torch.no_grad():
            yp = model(d["X_scaled"].to(device),
                       d["X_raw"][:, 2:4].to(device)).cpu().numpy()
        yt = d["Y_raw"].numpy()
        preds[split] = (yt, yp)
        M["prediction"][split] = {
            "n": int(len(yt)),
            **{nm: {"R2": float(r2(yt[:, i], yp[:, i])),
                    "RMSE": rmse(yt[:, i], yp[:, i]),
                    "MAE": mae(yt[:, i], yp[:, i])}
               for i, nm in enumerate(["dCA_dt", "dT_dt"])}}

    # --- f/g recovery over the region (paper Lemma 3.1) ---
    CA, T, x_abs = grid()
    x_abs = x_abs.to(device)
    xmin, xmax = model_scalers
    with torch.no_grad():
        f_nn, g_nn = model.f_g(scale_x(x_abs, xmin, xmax))
        f_t, g_t = true_f_g(x_abs)
    fn, ft = f_nn.cpu().numpy(), f_t.cpu().numpy()
    gn, gt = g_nn.cpu().numpy(), g_t.cpu().numpy()

    M["fg_recovery"] = {
        "f1": {"R2": float(r2(ft[:, 0], fn[:, 0])), "RMSE": rmse(ft[:, 0], fn[:, 0])},
        "f2": {"R2": float(r2(ft[:, 1], fn[:, 1])), "RMSE": rmse(ft[:, 1], fn[:, 1])},
    }
    for (i, j), nm, tv in [((0, 0), "g11", C.F / C.V),
                           ((1, 1), "g22", 1.0 / (C.RHOL * C.CP * C.V))]:
        e = rmse(gt[:, i, j], gn[:, i, j])
        M["fg_recovery"][nm] = {"RMSE": e, "true": float(tv),
                                "pct_error": float(e / tv * 100)}

    # --- Lyapunov over the region ---
    with torch.no_grad():
        us = torch.tensor([C.CA0S, C.QS], device=device)
        u = us + phys.sontag_controller(f_nn, g_nn, x_abs)     # phi_nn(x)
        # I evaluate V_dot on the TRUE PLANT under my controller. Using the
        # model's own xdot would only tell me what the model BELIEVES, a wrong
        # model can score 100% while the real closed loop diverges.
        Vdot = phys.v_dot(x_abs, phys.cstr_ode(x_abs, u)).cpu().numpy()
        Vdot_model = phys.v_dot(
            x_abs, f_nn + torch.bmm(g_nn, u.unsqueeze(-1)).squeeze(-1)).cpu().numpy()
        V = phys.lyapunov_V(x_abs).cpu().numpy()

    rho = 368.0
    ins = V > C.RHO_S
    n_in, n_pos = int(ins.sum()), int((Vdot[ins] >= 0).sum())
    ell = (V <= rho) & (V > C.RHO_S)
    M["lyapunov_region"] = {
        "rho": rho, "n_states": n_in, "n_Vdot_positive": n_pos,
        "pct_negative": float((n_in - n_pos) / n_in * 100),
        "Vdot_min": float(Vdot[ins].min()), "Vdot_max": float(Vdot[ins].max()),
        "evaluated_on": "true plant under phi_nn",
        "n_states_ellipse": int(ell.sum()),
        "n_positive_ellipse": int((Vdot[ell] >= 0).sum()),
        "pct_negative_model_own_xdot": float(
            (ins.sum() - (Vdot_model[ins] >= 0).sum()) / ins.sum() * 100),
    }

    # --- closed-loop rollouts on the TRUE plant ---
    x0s = [[C.CAS - 1.0, C.TS + 60.0], [C.CAS + 1.0, C.TS - 60.0],
           [C.CAS - 1.4, C.TS + 30.0], [C.CAS + 1.4, C.TS - 30.0]]
    M["closed_loop"] = []
    trajs = []
    for x0 in x0s:
        tr, uu = rollout(model, xmin, xmax, x0, 3000, C.ROLLOUT_DT, device)
        trajs.append((np.array(x0), tr, uu))
        xf = torch.tensor(tr[-1:], dtype=torch.float32)
        Vf = float(phys.lyapunov_V(xf).item())
        d0 = np.array(x0) - np.array([C.CAS, C.TS])
        M["closed_loop"].append({
            "x0_dev": [float(d0[0]), float(d0[1])],
            "V_final": Vf,
            "err_final": float(np.linalg.norm(tr[-1] - np.array([C.CAS, C.TS]))),
            "converged": bool(Vf < C.RHO_S)})

    # --- OPEN-LOOP: the true plant with NO control (u = u_s) ---
    ol_true = []
    for x0 in x0s:
        x = torch.tensor(x0, dtype=torch.float32, device=device).view(1, 2)
        us_t = torch.tensor([[C.CA0S, C.QS]], device=device)
        tr = [x.cpu().numpy().ravel()]
        for _ in range(3000):
            k1 = phys.cstr_ode(x, us_t); k2 = phys.cstr_ode(x + .5*C.ROLLOUT_DT*k1, us_t)
            k3 = phys.cstr_ode(x + .5*C.ROLLOUT_DT*k2, us_t)
            k4 = phys.cstr_ode(x + C.ROLLOUT_DT*k3, us_t)
            x = x + C.ROLLOUT_DT/6*(k1 + 2*k2 + 2*k3 + k4)
            if not torch.isfinite(x).all(): break
            tr.append(x.cpu().numpy().ravel())
        ol_true.append(np.array(tr))

    # --- OPEN-LOOP model fidelity under a VARYING input ---
    # I drive the plant with a random-step input that switches every HOLD steps,
    # matching how the identification data is excited. A constant input would
    # relax smoothly and would not test the model against fast changes.
    rng = np.random.default_rng(7)
    N_OL, HOLD = 250, 12
    x0_f = [C.CAS - 0.3, C.TS + 15.0]
    xt = torch.tensor(x0_f, dtype=torch.float32, device=device).view(1, 2)
    xp = xt.clone()
    tt, pp, one_step, u_log = [xt.cpu().numpy().ravel()], [xp.cpu().numpy().ravel()], [], []
    u_cur = None
    for k in range(N_OL):
        if k % HOLD == 0:
            du1 = rng.uniform(-2.5, 2.5); du2 = rng.uniform(-3.0e5, 3.0e5)
            u_cur = torch.tensor([[C.CA0S + du1, C.QS + du2]], device=device)
        u_log.append(u_cur.cpu().numpy().ravel())
        k1 = phys.cstr_ode(xt, u_cur); k2 = phys.cstr_ode(xt + .5*C.ROLLOUT_DT*k1, u_cur)
        k3 = phys.cstr_ode(xt + .5*C.ROLLOUT_DT*k2, u_cur)
        k4 = phys.cstr_ode(xt + C.ROLLOUT_DT*k3, u_cur)
        xt_next = xt + C.ROLLOUT_DT/6*(k1 + 2*k2 + 2*k3 + k4)
        with torch.no_grad():
            xd1 = model(scale_x(xt, xmin, xmax), u_cur)
        one_step.append((xt + C.ROLLOUT_DT*xd1).cpu().numpy().ravel())
        with torch.no_grad():
            xdp = model(scale_x(xp, xmin, xmax), u_cur)
        xp = xp + C.ROLLOUT_DT*xdp
        xt = xt_next
        tt.append(xt.cpu().numpy().ravel()); pp.append(xp.cpu().numpy().ravel())
    tt, pp, one_step, u_log = np.array(tt), np.array(pp), np.array(one_step), np.array(u_log)

    M["open_loop"] = {
        "unstable_from_all_ICs": bool(all(
            phys.lyapunov_V(torch.tensor(t[-1:], dtype=torch.float32)).item() > 368.0
            or len(t) < 3000 for t in ol_true)),
        "free_running": {nm: {"R2": float(r2(tt[:, i], pp[:, i])),
                              "RMSE": rmse(tt[:, i], pp[:, i])}
                         for i, nm in enumerate(["CA", "T"])},
        "one_step": {nm: {"R2": float(r2(tt[1:, i], one_step[:, i])),
                          "RMSE": rmse(tt[1:, i], one_step[:, i])}
                     for i, nm in enumerate(["CA", "T"])},
    }
    M["_ol_u"] = u_log
    M["_ol_onestep"] = one_step

    # --- MULTI-STEP-AHEAD prediction ---
    def h_step_pred(x_start, u_seq, h):
        x = x_start.clone()
        for j in range(h):
            with torch.no_grad():
                xd = model(scale_x(x, xmin, xmax), u_seq[j])
            x = x + C.ROLLOUT_DT * xd
        return x

    steps_list = [1, 2]
    ms = {"steps": steps_list, "curves": {}, "r2": {}}
    for hh in steps_list:
        pred = []
        for k in range(len(tt) - hh):
            xs = torch.tensor(tt[k:k+1], dtype=torch.float32, device=device)
            useq = [torch.tensor(u_log[min(k+j, len(u_log)-1)][None, :],
                                 dtype=torch.float32, device=device) for j in range(hh)]
            pred.append(h_step_pred(xs, useq, hh).cpu().numpy().ravel())
        pred = np.array(pred)
        true_h = tt[hh:hh+len(pred)]
        tgrid = np.arange(hh, hh+len(pred)) * C.ROLLOUT_DT
        ms["curves"][hh] = {"true": true_h, "pred": pred, "t": tgrid}
        ms["r2"][hh] = {"CA": float(r2(true_h[:, 0], pred[:, 0])),
                        "T":  float(r2(true_h[:, 1], pred[:, 1]))}
    M["_multistep"] = ms
    M["multistep_r2"] = ms["r2"]

    return M, preds, (CA, T, V, Vdot, rho), trajs, (fn, ft, gn, gt), \
           ol_true, (tt, pp, one_step)


# ----------------------------------------------------------------------------
def fig_parity(preds, M):
    """Predicted vs true derivatives on the test split."""
    yt, yp = preds["test"]
    names = [(r"$\dot{C}_A$  (kmol m$^{-3}$ hr$^{-1}$)", "dCA_dt"),
             (r"$\dot{T}$  (K hr$^{-1}$)", "dT_dt")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    for i, (ax, (lbl, key)) in enumerate(zip(axes, names)):
        m = M["prediction"]["test"][key]
        lo = min(yt[:, i].min(), yp[:, i].min())
        hi = max(yt[:, i].max(), yp[:, i].max())
        pad = 0.04 * (hi - lo)
        # I draw the identity line first, in grey, so the data sits on top of it.
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-",
                c="#b0b0b0", lw=1.2, zorder=2)
        ax.scatter(yt[:, i], yp[:, i], s=8, alpha=0.30, c=STEEL,
                   edgecolors="none", rasterized=True, zorder=3)
        ax.set_xlabel(f"true {lbl}")
        ax.set_ylabel(f"predicted {lbl}")
        _stat_box(ax, f"$R^2$ = {m['R2']:.4f}\nRMSE = {m['RMSE']:.3g}")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_parity_test.png"))
    plt.close(fig)


def fig_vdot_region(reg, M):
    """V_dot over the operating region, evaluated on the true plant."""
    CA, T, V, Vdot, rho = reg
    n = CA.shape[0]
    Vg, Vd = V.reshape(n, n), Vdot.reshape(n, n)
    masked = np.where(Vg <= rho, Vd, np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    pc = ax.pcolormesh(CA, T, masked, cmap="viridis", shading="gouraud",
                       rasterized=True)
    cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"$\dot{V}(x,\phi_{nn}(x))$   (hr$^{-1}$)", fontsize=11)
    cb.outline.set_linewidth(0.8)

    # I outline both level sets so the region boundary is unambiguous.
    ax.contour(CA, T, Vg, levels=[rho], colors="k", linewidths=1.4)
    ax.contour(CA, T, Vg, levels=[C.RHO_S], colors="k", linewidths=1.1,
               linestyles="--")

    # I mark any violation, if one exists it must be visible, not buried.
    bad = (Vg <= rho) & (Vd >= 0)
    if bad.any():
        ax.scatter(CA[bad], T[bad], s=30, c=ACCENT, marker="x", lw=1.7,
                   label=r"$\dot{V}\geq 0$", zorder=5)
        ax.legend(loc="upper right")

    ax.set_xlabel(r"$C_A - C_{As}$   (kmol m$^{-3}$)")
    ax.set_ylabel(r"$T - T_s$   (K)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig5_vdot_region.png"))
    plt.close(fig)


def fig_closed_loop(trajs, M):
    """Closed-loop phase portrait from the four initial conditions."""
    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    P = np.array(C.P_MATRIX)
    th = np.linspace(0, 2 * np.pi, 600)
    Lc = np.linalg.cholesky(np.linalg.inv(P))
    for rho, ls, lbl in [(368.0, "-", r"$\Omega_\rho$"),
                         (C.RHO_S, "--", r"$\Omega_{\rho_s}$")]:
        e = Lc @ (np.sqrt(rho) * np.stack([np.cos(th), np.sin(th)]))
        ax.plot(e[0], e[1], ls, c="#8a8a8a", lw=1.3, label=lbl, zorder=2)

    for (x0, tr, _), col in zip(trajs, TRAJ):
        z = tr - np.array([C.CAS, C.TS])
        ax.plot(z[:, 0], z[:, 1], c=col, lw=1.5, alpha=0.95, zorder=3)
        ax.plot(z[0, 0], z[0, 1], "o", c=col, ms=7, mec="white", mew=1.1, zorder=4)
        ax.plot(z[-1, 0], z[-1, 1], "s", c=col, ms=5, mec="white", mew=1.0, zorder=4)
    ax.plot(0, 0, "+", c="k", ms=11, mew=1.8, zorder=5)

    ax.set_xlabel(r"$C_A - C_{As}$   (kmol m$^{-3}$)")
    ax.set_ylabel(r"$T - T_s$   (K)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig8_closed_loop.png"))
    plt.close(fig)


def fig_time_response(trajs):
    """States and control inputs against time, with the input limits marked."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharex=True)
    for (x0, tr, uu), col in zip(trajs, TRAJ):
        z = tr - np.array([C.CAS, C.TS])
        tt = np.arange(len(tr)) * C.ROLLOUT_DT
        tu = np.arange(len(uu)) * C.ROLLOUT_DT
        axes[0, 0].plot(tt, z[:, 0], c=col, lw=1.4)
        axes[0, 1].plot(tt, z[:, 1], c=col, lw=1.4)
        axes[1, 0].plot(tu, uu[:, 0] - C.CA0S, c=col, lw=1.1, alpha=0.9)
        axes[1, 1].plot(tu, uu[:, 1] - C.QS, c=col, lw=1.1, alpha=0.9)

    for a in axes.ravel():
        a.axhline(0, c="#c0c0c0", lw=0.9, ls=":")
    for a, lim in [(axes[1, 0], C.DU_MAX[0]), (axes[1, 1], C.DU_MAX[1])]:
        a.axhline(+lim, c="#8a8a8a", ls="--", lw=1.0)
        a.axhline(-lim, c="#8a8a8a", ls="--", lw=1.0)

    axes[0, 0].set_ylabel(r"$C_A - C_{As}$   (kmol m$^{-3}$)")
    axes[0, 1].set_ylabel(r"$T - T_s$   (K)")
    axes[1, 0].set_ylabel(r"$\Delta C_{A0}$   (kmol m$^{-3}$)")
    axes[1, 1].set_ylabel(r"$\Delta Q$   (kJ hr$^{-1}$)")
    for a in axes[1]:
        a.set_xlabel("time   (hr)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_time_response.png"))
    plt.close(fig)


def fig_open_loop(ol_true, trajs, M):
    """Uncontrolled plant beside the controlled one, on shared axes."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)

    P = np.array(C.P_MATRIX)
    th = np.linspace(0, 2*np.pi, 600)
    Lc = np.linalg.cholesky(np.linalg.inv(P))
    for ax in axes:
        for rho, ls in [(368.0, "-"), (C.RHO_S, "--")]:
            e = Lc @ (np.sqrt(rho) * np.stack([np.cos(th), np.sin(th)]))
            ax.plot(e[0], e[1], ls, c="#8a8a8a", lw=1.3, zorder=2)
        ax.plot(0, 0, "+", c="k", ms=10, mew=1.8, zorder=5)
        ax.set_xlabel(r"$C_A - C_{As}$   (kmol m$^{-3}$)")

    for tr, col in zip(ol_true, TRAJ):
        z = tr - np.array([C.CAS, C.TS])
        axes[0].plot(z[:, 0], z[:, 1], c=col, lw=1.5, zorder=3)
        axes[0].plot(z[0, 0], z[0, 1], "o", c=col, ms=7, mec="white", mew=1.1, zorder=4)
    axes[0].set_ylabel(r"$T - T_s$   (K)")
    axes[0].set_title(r"open loop   ($u = u_s$)", fontsize=11)

    for (x0, tr, _), col in zip(trajs, TRAJ):
        z = tr - np.array([C.CAS, C.TS])
        axes[1].plot(z[:, 0], z[:, 1], c=col, lw=1.5, zorder=3)
        axes[1].plot(z[0, 0], z[0, 1], "o", c=col, ms=7, mec="white", mew=1.1, zorder=4)
        axes[1].plot(z[-1, 0], z[-1, 1], "s", c=col, ms=5, mec="white", mew=1.0, zorder=4)
    axes[1].set_title(r"closed loop   ($u = \phi_{nn}(x)$)", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_open_vs_closed_loop.png"))
    plt.close(fig)


def fig_open_loop_prediction(ol_pred, M):
    """Each state stacked over the input driving it."""
    tt, pp, one_step = ol_pred
    u = M["_ol_u"]
    t = np.arange(len(tt)) * C.ROLLOUT_DT
    tu = np.arange(len(u)) * C.ROLLOUT_DT
    ones = M["_ol_onestep"]

    specs = [("CA", 0, r"$C_A$   (kmol m$^{-3}$)", 0,
              r"$C_{A0}$   (kmol m$^{-3}$)", "kmol m$^{-3}$"),
             ("T",  1, r"$T$   (K)", 1,
              r"$Q$   (kJ hr$^{-1}$)", "K")]
    for key, i, lbl, ui, ulbl, unit in specs:
        fig, (ax, axu) = plt.subplots(2, 1, figsize=(9.0, 5.4), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(t, tt[:, i], c=INK, lw=1.9, label="true plant", zorder=3)
        ax.plot(t[1:], ones[:, i], "--", c=ACCENT, lw=1.5, label="PINN", zorder=4)
        o = M["open_loop"]["one_step"][key]
        _stat_box(ax, f"$R^2$ = {o['R2']:.5f}\nRMSE = {o['RMSE']:.3g} {unit}",
                  loc="lower left")
        ax.set_ylabel(lbl)
        ax.legend(loc="upper right")

        axu.step(tu, u[:, ui], where="post", c="#7a7a7a", lw=1.0)
        axu.set_ylabel(ulbl)
        axu.set_xlabel("time   (hr)")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIG_DIR, f"fig_open_loop_{key}.png"))
        plt.close(fig)


def fig_rollout_h(M, hh):
    """A single h-step-ahead rollout, both states stacked."""
    c = M["_multistep"]["curves"][hh]
    r = M["_multistep"]["r2"][hh]
    t = c["t"]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True)
    specs = [(0, r"$C_A$   (kmol m$^{-3}$)", "CA", "kmol m$^{-3}$"),
             (1, r"$T$   (K)", "T", "K")]
    for ax, (i, lbl, key, unit) in zip(axes, specs):
        ax.plot(t, c["true"][:, i], c=INK, lw=1.9, label="true plant", zorder=2)
        ax.plot(t, c["pred"][:, i], "--", c=ACCENT, lw=1.5, label="PINN", zorder=3)
        e = rmse(c["true"][:, i], c["pred"][:, i])
        ax.set_ylabel(lbl)
        ax.legend(loc="upper right")
        _stat_box(ax, f"$R^2$ = {r[key]:.5f}\nRMSE = {e:.3g} {unit}", loc="lower left")
    axes[1].set_xlabel("time   (hr)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, f"fig_rollout_{hh}step.png"))
    plt.close(fig)


def _smooth(y, k=15):
    """I take a centred rolling mean so the trend shows through the noise."""
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, ker, mode="same")[pad:-pad]


def fig_loss_total():
    """Total train/val loss. I fade the raw traces and draw the smoothed on top."""
    path = os.path.join(C.OUT_DIR, "history.csv")
    h = list(csv.DictReader(open(path)))
    ep = np.array([int(r["epoch"]) for r in h])
    tr = np.array([float(r["tr_total"]) for r in h])
    va = np.array([float(r["va_total"]) for r in h])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(ep, tr, c=STEEL,  lw=0.5, alpha=0.16, zorder=1)
    ax.plot(ep, va, c=ACCENT, lw=0.5, alpha=0.16, zorder=1)
    ax.plot(ep, _smooth(tr), c=STEEL,  lw=1.9, label="train", zorder=3)
    ax.plot(ep, _smooth(va), c=ACCENT, lw=1.9, label="validation", zorder=4)

    kb = int(np.argmin(va))
    ax.scatter([ep[kb]], [va[kb]], s=48, c="white", ec=ACCENT, lw=1.6, zorder=6)
    ax.annotate(f"best  (epoch {ep[kb]})", xy=(ep[kb], va[kb]),
                xytext=(12, 16), textcoords="offset points",
                fontsize=9.5, color=INK)

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_loss_total.png"))
    plt.close(fig)


def fig_loss_components():
    """The four loss terms on one log axis.

    Note on L_data and L_phys: both are RMS-normalised against the same
    derivative targets, so they track each other almost exactly (in my run
    both span ~0.027 to ~1.09). Drawn as two solid lines the second simply
    covers the first. I draw L_data thick underneath and L_phys dashed on
    top, so the overlap is visible AS overlap instead of looking like a
    missing series.

    I also do not clip values or set a hard y-floor. An earlier version did
    both and that quietly erased any curve below the floor while still
    drawing its legend entry.
    """
    path = os.path.join(C.OUT_DIR, "history.csv")
    h = list(csv.DictReader(open(path)))
    ep = np.array([int(r["epoch"]) for r in h])

    # (column, colour, label, linewidth, linestyle, z-order)
    comps = [("tr_data", INK,    r"$\mathcal{L}_{\mathrm{data}}$", 3.0, "-",  3),
             ("tr_phys", STEEL,  r"$\mathcal{L}_{\mathrm{phys}}$", 1.5, "--", 4),
             ("tr_g",    SAND,   r"$\mathcal{L}_{g}$",             1.8, "-",  5),
             ("tr_lyap", ACCENT, r"$\mathcal{L}_{\mathrm{lyap}}$", 1.8, "-",  5)]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = 0
    for key, col, lbl, lw, ls, z in comps:
        if key not in h[0]:
            print(f"  [loss plot] column '{key}' not in history.csv -- skipped")
            continue
        y = np.array([float(r[key]) for r in h])
        pos = y[y > 0]
        if pos.size == 0:
            print(f"  [loss plot] '{key}' is all <= 0 -- no legend entry drawn")
            continue
        # I mask non-positive points instead of clipping them to a floor, so
        # each curve keeps its true magnitude wherever it is defined.
        ys = _smooth(y)
        ys = np.where(ys > 0, ys, np.nan)
        ax.plot(ep, ys, c=col, lw=lw, ls=ls, label=lbl, zorder=z,
                alpha=0.95 if ls == "-" else 1.0)
        plotted += 1
        print(f"  [loss plot] {key}: min={pos.min():.3e}  max={pos.max():.3e}")

    ax.set_yscale("log")   # autoscaled: no manual floor
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if plotted:
        ax.legend(ncol=2, loc="upper right")
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_loss_components.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
def write_metrics(M):
    """I dump every metric to JSON and a flat CSV so I can quote them anywhere."""
    M = {k: v for k, v in M.items() if not k.startswith("_")}
    with open(os.path.join(C.OUT_DIR, "metrics.json"), "w") as f:
        json.dump(M, f, indent=2)

    rows = []
    for split, d in M["prediction"].items():
        for var in ["dCA_dt", "dT_dt"]:
            rows.append({"section": "prediction", "split": split, "quantity": var,
                         "R2": f"{d[var]['R2']:.6f}",
                         "RMSE": f"{d[var]['RMSE']:.6g}",
                         "MAE": f"{d[var]['MAE']:.6g}", "extra": f"n={d['n']}"})
    for k, v in M["fg_recovery"].items():
        rows.append({"section": "fg_recovery", "split": "region", "quantity": k,
                     "R2": f"{v['R2']:.6f}" if "R2" in v else "n/a (const)",
                     "RMSE": f"{v['RMSE']:.6g}", "MAE": "",
                     "extra": f"err={v['pct_error']:.2f}%" if "pct_error" in v else ""})
    L = M["lyapunov_region"]
    rows.append({"section": "lyapunov", "split": "region", "quantity": "Vdot<0",
                 "R2": "", "RMSE": "", "MAE": "",
                 "extra": f"{L['pct_negative']:.2f}% of {L['n_states']} "
                          f"({L['n_Vdot_positive']} positive)"})
    for i, c in enumerate(M["closed_loop"]):
        rows.append({"section": "closed_loop", "split": f"IC{i+1}",
                     "quantity": f"x0=({c['x0_dev'][0]:+.2f},{c['x0_dev'][1]:+.1f})",
                     "R2": "", "RMSE": "", "MAE": "",
                     "extra": f"V_final={c['V_final']:.2f}, "
                              f"converged={'YES' if c['converged'] else 'NO'}"})
    for mode in ["one_step", "free_running"]:
        for var, v in M["open_loop"][mode].items():
            rows.append({"section": "open_loop", "split": mode, "quantity": var,
                         "R2": f"{v['R2']:.6f}", "RMSE": f"{v['RMSE']:.6g}",
                         "MAE": "", "extra": ""})

    with open(os.path.join(C.OUT_DIR, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "split", "quantity",
                                          "R2", "RMSE", "MAE", "extra"])
        w.writeheader(); w.writerows(rows)


def print_summary(M):
    print("=" * 72)
    print("PINN-LSNN RESULTS SUMMARY")
    print("=" * 72)
    t = M["prediction"]["test"]
    print(f"Test  R2  dCA/dt = {t['dCA_dt']['R2']:.4f}   dT/dt = {t['dT_dt']['R2']:.4f}")
    print(f"      RMSE       = {t['dCA_dt']['RMSE']:.4g}       {t['dT_dt']['RMSE']:.4g}")
    fg = M["fg_recovery"]
    print(f"g recovery: g11 err = {fg['g11']['pct_error']:.2f}%   "
          f"g22 err = {fg['g22']['pct_error']:.2f}%")
    L = M["lyapunov_region"]
    print(f"Lyapunov (V_dot on the TRUE plant under phi_nn):")
    print(f"          V_dot<0 on {L['pct_negative']:.2f}% of {L['n_states']} states "
          f"({L['n_Vdot_positive']} positive)")
    print(f"          (scoring on the model's own xdot would read "
          f"{L['pct_negative_model_own_xdot']:.2f}%; I report the honest number)")
    n_ok = sum(c["converged"] for c in M["closed_loop"])
    print(f"Closed loop: {n_ok}/4 rollouts converge to Omega_rho_s on the true plant")
    for i, c in enumerate(M["closed_loop"]):
        print(f"             IC{i+1}  V_final = {c['V_final']:7.3f}   "
              f"{'converged' if c['converged'] else 'DID NOT CONVERGE'}")
    o = M["open_loop"]
    print(f"Open loop  : one-step R2  CA={o['one_step']['CA']['R2']:.4f}  "
          f"T={o['one_step']['T']['R2']:.4f}")
    print(f"             free-run R2  CA={o['free_running']['CA']['R2']:.4f}  "
          f"T={o['free_running']['T']['R2']:.4f}")
    print("\nWrote outputs/metrics.json and outputs/metrics.csv")
    print("Saved figures ->", C.FIG_DIR)


def clean_figures():
    """I clear old figures first so every run leaves only the current set."""
    if os.path.isdir(C.FIG_DIR):
        for fn in os.listdir(C.FIG_DIR):
            if fn.lower().endswith(".png"):
                os.remove(os.path.join(C.FIG_DIR, fn))


if __name__ == "__main__":
    clean_figures()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, xmin, xmax = load_model(device)
    model_scalers = (xmin, xmax)

    M, preds, reg, trajs, fg, ol_true, ol_pred = collect_metrics(
        model, device, model_scalers)

    fig_parity(preds, M)
    fig_vdot_region(reg, M)
    fig_closed_loop(trajs, M)
    fig_open_loop(ol_true, trajs, M)
    fig_open_loop_prediction(ol_pred, M)
    for hh in M["_multistep"]["steps"]:
        fig_rollout_h(M, hh)
    fig_time_response(trajs)
    fig_loss_total()
    fig_loss_components()

    write_metrics(M)
    print_summary(M)