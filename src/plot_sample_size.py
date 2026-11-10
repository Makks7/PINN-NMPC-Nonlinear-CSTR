"""plot_sample_size.py - plots sample_size_results.csv in the style of Fig. 2-3
of the paper (test error and Lyapunov violations against sample size).
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})
BLUE, ORANGE, GREEN = "#2471a3", "#e67e22", "#1e8449"


def load():
    rows = list(csv.DictReader(open(os.path.join(C.OUT_DIR, "sample_size_results.csv"))))
    d = {}
    for r in rows:
        m = int(r["n_samples"])
        d.setdefault(m, []).append(r)
    return dict(sorted(d.items()))


def band(ax, ms, series, color, label):
    """I plot the mean with a shaded min-max band, like the paper's error tube."""
    mean = np.array([np.mean(s) for s in series])
    lo   = np.array([np.min(s) for s in series])
    hi   = np.array([np.max(s) for s in series])
    ax.fill_between(ms, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(ms, mean, "o-", color=color, lw=2, ms=5, label=label)


def main():
    d = load()
    ms = list(d.keys())

    # ---- Figure A: test error vs sample size (paper Fig. 3) ----
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    band(ax, ms, [[1 - float(r["r2_dCA"]) for r in d[m]] for m in ms], BLUE,
         r"$1-R^2$  ($\dot{C}_A$)")
    band(ax, ms, [[1 - float(r["r2_dT"]) for r in d[m]] for m in ms], ORANGE,
         r"$1-R^2$  ($\dot{T}$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of training samples")
    ax.set_ylabel(r"test error  $1-R^2$")
    ax.set_title("Test error vs training sample size", weight="semibold")
    ax.legend()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_samplesize_testerror.png"))
    plt.close(fig)

    # ---- Figure B: Lyapunov violations vs sample size (paper Fig. 4) ----
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    band(ax, ms, [[int(r["n_Vdot_pos"]) for r in d[m]] for m in ms], GREEN,
         r"states with $\dot{V}\geq 0$")
    n_states = int(d[ms[0]][0]["n_states"])
    ax.set_xscale("log")
    ax.set_xlabel("number of training samples")
    ax.set_ylabel(r"number of states with $\dot{V}\geq 0$")
    ax.set_title(f"Lyapunov violations vs sample size  (of {n_states} grid states)",
                 weight="semibold")
    ax.legend()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_samplesize_violations.png"))
    plt.close(fig)

    # ---- Figure C: g recovery vs sample size (my addition, not in the paper) ----
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    band(ax, ms, [[float(r["g11_err"]) for r in d[m]] for m in ms], BLUE, r"$g_{11}$")
    band(ax, ms, [[float(r["g22_err"]) for r in d[m]] for m in ms], ORANGE, r"$g_{22}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of training samples")
    ax.set_ylabel("g recovery error (%)")
    ax.set_title("Control-matrix recovery vs sample size", weight="semibold")
    ax.legend()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_samplesize_grecovery.png"))
    plt.close(fig)

    # ---- summary table ----
    print(f"{'m':>7} {'R2 dCA':>9} {'R2 dT':>9} {'g11 %':>8} {'Vdot>=0':>9}")
    for m in ms:
        rs = d[m]
        print(f"{m:>7} {np.mean([float(r['r2_dCA']) for r in rs]):>9.4f}"
              f" {np.mean([float(r['r2_dT']) for r in rs]):>9.4f}"
              f" {np.mean([float(r['g11_err']) for r in rs]):>8.2f}"
              f" {np.mean([int(r['n_Vdot_pos']) for r in rs]):>9.1f}")
    print("\nSaved 3 figures ->", C.FIG_DIR)


if __name__ == "__main__":
    main()