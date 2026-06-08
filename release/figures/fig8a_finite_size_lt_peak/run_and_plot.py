"""Figure 8(a): long-transient share vs S_width, finite-size scaling
across 4 grid sizes.

Setup:
    denom = 180,  S_low = 10/180,  B = [60/180, 160/180]
    S_width = 50..70 (21 values),  N = 100 runs per (grid, width)
    max_generations = 100_000,  initial_density = 0.25
    Grid sizes: 50x37, 100x75, 150x112, 200x150
    saturation_threshold = simulator default (1.01, never triggers)

For each grid we run the full S_width sweep on GPU and cache the per-run
outcomes to raw_<label>.csv.  Single H100 wall time per grid: ~10-15 min,
~50 min total for all 4 grids.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent))

from sfca.simulate import simulate_batch  # noqa: E402


# ============================================================
# Parameters (mirror Exp22 fine config)
# ============================================================
DENOMINATOR = 180
S_LOW_UNIT = 10
B_LOW_UNIT = 60
B_HIGH_UNIT = 160
S_WIDTHS = list(range(50, 71))         # 50..70 in units of 1/180
N_RUNS_PER_PARAM = 100
MAX_GENERATIONS = 100_000
INITIAL_DENSITY = 0.25
CYCLE_BUFFER = MAX_GENERATIONS + 1

# (label, (width, height))
GRIDS = [
    ("50x37",   (50,  37)),
    ("100x75",  (100, 75)),
    ("150x112", (150, 112)),
    ("200x150", (200, 150)),
]

# Four distinct hues from the project pastel palette, ordered warm -> cool
# from smallest to largest grid.  Each color is from a different hue family so
# curves stay readable side by side and survive a BW print test (lightness
# also varies monotonically: coral light, orange lighter, teal medium, blue
# darker).
GRID_COLORS = {
    "50x37":   "#FA7F6F",   # warm coral
    "100x75":  "#FFBE7A",   # orange
    "150x112": "#8ECFC9",   # teal / mint
    "200x150": "#82B0D2",   # cool steel blue
}
GRID_MARKERS = {
    "50x37":   "o",
    "100x75":  "s",
    "150x112": "^",
    "200x150": "D",
}

def cache_path(label):
    return THIS_DIR / f"raw_{label}.csv"


# Fonts (match cycle_lt_vs_swidth_S1_B6-16)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["legend.fontsize"] = 17


# ============================================================
# Per-grid simulation
# ============================================================
def simulate_grid(label, width, height):
    """Run the fine S_width sweep on a single grid size; cache to CSV."""
    cache = cache_path(label)
    if cache.exists():
        print(f"Loading cached sims for {label}: {cache}")
        df = pd.read_csv(cache)
        if set(df["s_width"].unique()) >= set(S_WIDTHS):
            return df
        print("  cache incomplete -- re-running")

    n_w = len(S_WIDTHS)
    total = n_w * N_RUNS_PER_PARAM
    s_width  = np.repeat(S_WIDTHS, N_RUNS_PER_PARAM).astype(np.int32)
    s_low_u  = np.full(total, S_LOW_UNIT, dtype=np.int32)
    s_high_u = s_low_u + s_width
    b_low_u  = np.full(total, B_LOW_UNIT, dtype=np.int32)
    b_high_u = np.full(total, B_HIGH_UNIT, dtype=np.int32)
    run_id   = np.tile(np.arange(N_RUNS_PER_PARAM), n_w).astype(np.int32)

    # Per-grid seed offset keeps the four sweeps independent.
    seeds = (s_width.astype(np.uint64) * np.uint64(7_777_001)
             + run_id.astype(np.uint64) * np.uint64(13)
             + np.uint64(width) * np.uint64(101)
             + np.uint64(height) * np.uint64(7)).astype(np.int64)

    s_low_f  = (s_low_u  / DENOMINATOR).astype(np.float32)
    s_high_f = (s_high_u / DENOMINATOR).astype(np.float32)
    b_low_f  = (b_low_u  / DENOMINATOR).astype(np.float32)
    b_high_f = (b_high_u / DENOMINATOR).astype(np.float32)

    print(f"Simulating {label} fine sweep: {total} sims "
          f"(max_gen={MAX_GENERATIONS})")
    t0 = time.time()
    res = simulate_batch(
        s_low=s_low_f, s_high=s_high_f, b_low=b_low_f, b_high=b_high_f,
        seeds=seeds, height=height, width=width,
        max_generations=MAX_GENERATIONS,
        initial_density=INITIAL_DENSITY,
        cycle_buffer=CYCLE_BUFFER,
    )
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s  ({total/dt:.1f} sims/s)")

    df = pd.DataFrame({
        "s_width":         s_width,
        "run_id":          run_id,
        "outcome":         res["outcome"],
        "stop_generation": res["stop_generation"],
        "cycle_period":    res["cycle_period"],
    })
    df.to_csv(cache, index=False)
    print(f"  cached to {cache}")
    return df


def lt_share_from_df(df):
    shares = []
    for w in S_WIDTHS:
        sub = df[df["s_width"] == w]
        shares.append(float((sub["outcome"] == "long_transient").mean()))
    return np.array(shares)


# ============================================================
# Plot
# ============================================================
def plot(shares):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.array(S_WIDTHS) / DENOMINATOR
    for label, _ in GRIDS:
        color = GRID_COLORS[label]
        ax.plot(x, shares[label], "--",
                color=color, linewidth=2.4,
                marker=GRID_MARKERS[label], markersize=8,
                markerfacecolor=color, markeredgecolor=color,
                label=f"grid {label}")
    ax.set_xlabel("S width (units of 1/180)")
    ax.set_ylabel("long transient share")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(min(x) - 0.5 / DENOMINATOR, max(x) + 0.5 / DENOMINATOR)

    major = [w / DENOMINATOR for w in S_WIDTHS if w % 5 == 0]
    ax.set_xticks(major)
    ax.set_xticklabels([f"{w}/180" for w in S_WIDTHS if w % 5 == 0])
    ax.set_xticks(x, minor=True)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = THIS_DIR / f"figure_lt_share_vs_swidth.{ext}"
        fig.savefig(p, dpi=200)
        print(f"  saved {p}")
    plt.close(fig)


def main():
    shares = {}
    for label, (w, h) in GRIDS:
        df = simulate_grid(label, w, h)
        shares[label] = lt_share_from_df(df)
        s = shares[label]
        argmax_w = S_WIDTHS[int(s.argmax())]
        print(f"{label:>8s}: LT share min={s.min():.3f}, max={s.max():.3f}, "
              f"argmax @ s_width={argmax_w}/180")

    with open(THIS_DIR / "data.json", "w") as f:
        json.dump({
            "parameters": {
                "denominator": DENOMINATOR,
                "S_low_unit":  S_LOW_UNIT,
                "B_low_unit":  B_LOW_UNIT,
                "B_high_unit": B_HIGH_UNIT,
                "S_widths":    S_WIDTHS,
                "n_runs_per_param":     N_RUNS_PER_PARAM,
                "max_generations":      MAX_GENERATIONS,
                "initial_density":      INITIAL_DENSITY,
                "saturation_threshold": "simulate.py default (1.01, never triggers)",
            },
            "grid_colors_hex": dict(GRID_COLORS),
            "lt_shares": {k: v.tolist() for k, v in shares.items()},
        }, f, indent=2)
    print("saved data.json")

    plot(shares)


if __name__ == "__main__":
    main()
