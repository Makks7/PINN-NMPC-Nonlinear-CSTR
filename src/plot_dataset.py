"""
plot_dataset.py

Plots the whole training dataset. There are four panels on one x-axis: the
two inputs applied and the two states that came out of them.

The x-axis is sample index and not time. The dataset is 2500 short runs joined
end to end, so sample 5 and sample 6 can come from different runs with
different initial conditions. Plotting against time would make it look like
one continuous trajectory, which it is not.

There is also a second figure zoomed into the first few runs, since you
cannot see the run boundaries at full length.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- steady state and constraint box ---
CAS, TS = 1.9537, 401.8727
CA0S, QS = 4.0, 0.0
DCA0_MAX, DQ_MAX = 3.5, 5.0e5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

CSV = os.path.join(DATA, "cstr_open_clean.csv")
STEPS_PER_TRAJ = 5      # I use this only to draw the segment boundaries

INK    = "#16324f"
ACCENT = "#c1666b"
STEEL  = "#4a7c9b"
SAND   = "#b8934a"

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})


def load():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    return (df["u1"].to_numpy(), df["u2"].to_numpy(),
            df["y1"].to_numpy(), df["y2"].to_numpy())


def panels(u1, u2, y1, y2, idx, fname, seg_lines=False, title_note=""):
    """I draw the four-panel stack over whatever index range I'm given."""
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.2), sharex=True)

    specs = [
        (axes[0], u1, INK,    r"$C_{A0}$   (kmol m$^{-3}$)",
         [CA0S - DCA0_MAX, CA0S + DCA0_MAX], CA0S),
        (axes[1], u2, STEEL,  r"$Q$   (kJ hr$^{-1}$)",
         [QS - DQ_MAX, QS + DQ_MAX], QS),
        (axes[2], y1, SAND,   r"$C_A$   (kmol m$^{-3}$)", None, CAS),
        (axes[3], y2, ACCENT, r"$T$   (K)",               None, TS),
    ]

    for ax, y, col, lbl, box, ss in specs:
        ax.plot(idx, y, c=col, lw=0.45, alpha=0.85, zorder=3)
        # I mark the steady state, so deviation is readable at a glance.
        ax.axhline(ss, c="#8a8a8a", ls=":", lw=1.0, zorder=2)
        if box is not None:
            # I draw the admissible input limits, so coverage is visible.
            for b in box:
                ax.axhline(b, c="#8a8a8a", ls="--", lw=1.0, zorder=2)
            pad = 0.08 * (box[1] - box[0])
            ax.set_ylim(box[0] - pad, box[1] + pad)
        ax.set_ylabel(lbl)

    if seg_lines:
        # I show where one trajectory ends and the next begins.
        for b in range(idx[0], idx[-1] + 2, STEPS_PER_TRAJ):
            for ax in axes:
                ax.axvline(b - 0.5, c="#c8c8c8", lw=0.7, zorder=1)

    axes[3].set_xlabel("sample index" + title_note)
    axes[3].set_xlim(idx[0], idx[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname))
    plt.close(fig)


def main():
    u1, u2, y1, y2 = load()
    n = len(u1)
    idx = np.arange(n)

    print(f"{CSV}")
    print(f"  samples            : {n}")
    print(f"  trajectories (est) : {n / STEPS_PER_TRAJ:.0f}  "
          f"({STEPS_PER_TRAJ} steps each)")
    print(f"  CA0   range        : [{u1.min():.3f}, {u1.max():.3f}]   "
          f"box [{CA0S-DCA0_MAX:.1f}, {CA0S+DCA0_MAX:.1f}]")
    print(f"  Q     range        : [{u2.min():.3e}, {u2.max():.3e}]   "
          f"box [{-DQ_MAX:.0e}, {DQ_MAX:.0e}]")
    print(f"  CA    range        : [{y1.min():.3f}, {y1.max():.3f}]")
    print(f"  T     range        : [{y2.min():.2f}, {y2.max():.2f}]")

    # --- full length ---
    panels(u1, u2, y1, y2, idx, "fig_dataset_full.png",
           seg_lines=False, title_note=f"   (all {n} samples)")

    # --- zoom on the first 20 trajectories, so the structure is visible ---
    k = min(20 * STEPS_PER_TRAJ, n)
    panels(u1[:k], u2[:k], y1[:k], y2[:k], np.arange(k),
           "fig_dataset_zoom.png", seg_lines=True,
           title_note=f"   (first {k} samples, {k//STEPS_PER_TRAJ} trajectories; "
                      f"vertical lines mark trajectory boundaries)")

    print(f"\nSaved -> {os.path.join(FIGS, 'fig_dataset_full.png')}")
    print(f"Saved -> {os.path.join(FIGS, 'fig_dataset_zoom.png')}")


if __name__ == "__main__":
    main()