"""Figure 8(b): normed-field std trajectories at 3 S_width values across
4 grid sizes.

Axis: y = std(n / n_max) per generation t.  Shaded bands are +-1 sigma
across 100 independent runs per (S_width, grid_size).

Setup (mirrors the original Exp23.* CPU experiment):
    denom = 180
    S_low = 10/180,  B = [60/180, 160/180]
    S_width in {55, 60, 62} / 180
    Grid sizes: 50x37, 100x75, 150x112, 200x150
    rho_0 = 0.25, max_gen = 2000
    100 runs per (S_width, grid_size)

Aggregation: terminated runs contribute 0 to the mean/std at generations
past their stop_generation (zero-padding).  This matches the original
figure's convention.  The plotted curve is truncated at the maximum
trajectory length within each (S_width, grid) group.

Total sims: 3 widths * 4 grids * 100 = 1200.  Single H100 < 1 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent))

from sfca.simulate import simulate_batch  # noqa: E402


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 18


# ============================================================
DENOM = 180
S_LOW_UNIT  = 10
B_LOW_UNIT  = 60
B_HIGH_UNIT = 160
PANELS = [55, 60, 62]                    # S_width / 180 per panel
GRIDS  = [(50, 37), (100, 75), (150, 112), (200, 150)]
GRID_LABELS = [f"{w}x{h}" for (w, h) in GRIDS]

MAX_GEN = 2000
INIT_DENSITY = 0.25
N_RUNS = 100
CYCLE_BUFFER = MAX_GEN + 1

GRID_COLORS = {
    "50x37":   "#FA7F6F",
    "100x75":  "#FFBE7A",
    "150x112": "#8ECFC9",
    "200x150": "#82B0D2",
}


def simulate_one_panel(s_width: int, grid_wh: tuple) -> dict:
    """Run N_RUNS independent sims and return per-run (trajectory, stop_gen)."""
    w, h = grid_wh
    s_low  = np.full(N_RUNS, S_LOW_UNIT / DENOM,            dtype=np.float32)
    s_high = np.full(N_RUNS, (S_LOW_UNIT + s_width) / DENOM, dtype=np.float32)
    b_low  = np.full(N_RUNS, B_LOW_UNIT / DENOM,             dtype=np.float32)
    b_high = np.full(N_RUNS, B_HIGH_UNIT / DENOM,            dtype=np.float32)
    seeds = (np.int64(s_width) * 1_000_003
             + np.int64(w) * 9_991
             + np.int64(h) * 97
             + np.arange(N_RUNS, dtype=np.int64) + 1)

    res = simulate_batch(
        s_low=s_low, s_high=s_high, b_low=b_low, b_high=b_high,
        seeds=seeds,
        height=h, width=w,
        max_generations=MAX_GEN,
        initial_density=INIT_DENSITY,
        cycle_buffer=CYCLE_BUFFER,
        record_normed_std_first_n=MAX_GEN + 1,
    )
    return {
        "normed_std": res["normed_std_trajectory"],   # (N_RUNS, MAX_GEN+1)
        "stop_gen":   res["stop_generation"],         # (N_RUNS,)
        "outcome":    res["outcome"],
    }


def aggregate(panel_data: dict) -> dict:
    """Apply zero-padding past stop_gen, then take mean/std over runs.
    Returns truncated arrays (t, mean, std) per the longest run's stop_gen."""
    traj = panel_data["normed_std"].copy()             # (N, T)
    stop = panel_data["stop_gen"]
    outc = panel_data["outcome"]
    N, T = traj.shape

    # Zero out gens past stop_gen for terminated runs.  Long-transient runs
    # have stop_gen = MAX_GEN, so they retain their full trajectory.
    for i in range(N):
        if outc[i] != "long_transient":
            sg = int(stop[i])
            if sg + 1 < T:
                traj[i, sg + 1:] = 0.0

    mean = traj.mean(axis=0)
    std  = traj.std(axis=0)

    # Truncate plot at the largest stop_gen.
    max_len = int(stop.max()) + 1
    if max_len < T:
        return {"t":    np.arange(max_len),
                "mean": mean[:max_len],
                "std":  std[:max_len]}
    return {"t":    np.arange(T),
            "mean": mean,
            "std":  std}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)
    handles = None
    labels  = None

    for ax, sw in zip(axes, PANELS):
        for (w, h) in GRIDS:
            gl = f"{w}x{h}"
            print(f"  S_width={sw}/180, grid={gl}")
            t0 = time.time()
            data = simulate_one_panel(sw, (w, h))
            print(f"    sim done in {time.time()-t0:.1f}s")
            agg = aggregate(data)
            color = GRID_COLORS[gl]
            ax.fill_between(agg["t"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(agg["t"], agg["mean"], color=color, linewidth=1.5,
                    label=gl)
        ax.set_title(f"S width = {sw}/180")
        ax.set_xlabel("Generation")
        if ax is axes[0]:
            ax.set_ylabel(r"$\sigma(n / n_{\max})$")
        ax.set_xlim(0, MAX_GEN)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels,
               loc="upper center", ncol=len(GRIDS),
               bbox_to_anchor=(0.5, 1.02), fontsize=13, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p_png = THIS_DIR / "figure.png"
    p_pdf = THIS_DIR / "figure.pdf"
    fig.savefig(p_png, dpi=200, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p_png}\n       {p_pdf}")


if __name__ == "__main__":
    main()
