"""Grid snapshots for the 4 outcome classes at canonical SCA parameters.

Parameters (same as figures/outcome_distribution_S3-11_B7-9):
    S = [3/18, 11/18], B = [7/18, 9/18], rho_0 = 0.25
    grid 75 x 100, max_gen = 2000

For each of the 4 outcome classes we pick one representative seed from a
scan of N_SCAN seeds and save its grid state at gens 0, 400, 800, 1000,
1998, 1999, 2000.  The figure is a 4 x 7 panel; each row is one outcome,
each column is one snapshot generation.  Within a row, alive cells are
drawn with that outcome's color; dead cells are white.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent))

from sfca.simulate import simulate_batch  # noqa: E402


# ============================================================
# Parameters
# ============================================================
S_LOW  = 3 / 18
S_HIGH = 11 / 18
B_LOW  = 7 / 18
B_HIGH = 9 / 18

GRID_HEIGHT = 75
GRID_WIDTH = 100
MAX_GEN = 2_000
INIT_DENSITY = 0.25
SATURATION_THRESHOLD = 0.90
CYCLE_BUFFER = MAX_GEN + 1

SNAP_GENS = [0, 400, 800, 1000, 1998, 1999, 2000]

# Scan budget for finding representative seeds.  At 60% fp / 17% ext / 16% LT
# / 4% cycle, N_SCAN = 500 yields ~20 cycle sims (enough to pick one with a
# short period).
N_SCAN = 500

ROW_ORDER = ["extinction", "cycle", "long_transient", "fixed_point"]
ROW_COLORS = {
    "extinction":     "#FA7F6F",
    "cycle":          "#8ECFC9",
    "long_transient": "#FFBE7A",
    "fixed_point":    "#82B0D2",
}


def pick_index(outcomes_4cat, stop_gens, cycle_periods, target):
    """Pick one representative sim index for the target outcome.

    Heuristics aim for snapshots that look "interesting":
      extinction      : stop_gen between 300 and 1200 (lives a while then dies)
      cycle           : cycle_period == 2 (period-2 oscillation; 1998 == 2000 != 1999)
      long_transient  : any (all reach gen 2000 by definition)
      fixed_point     : stop_gen between 200 and 1500 (settled well before end)
    For the cycle case period-2 is *required* (no fallback to other periods).
    Other cases fall back to the first matching index if no candidate meets
    the heuristic.
    """
    base_mask = (outcomes_4cat == target)
    if target == "extinction":
        good = base_mask & (stop_gens >= 300) & (stop_gens <= 1200)
        idxs = np.where(good)[0]
        if len(idxs) == 0:
            idxs = np.where(base_mask)[0]
    elif target == "cycle":
        # Strict: must be period-2.  Don't fall back to other periods.
        good = base_mask & (cycle_periods == 2)
        idxs = np.where(good)[0]
    elif target == "long_transient":
        idxs = np.where(base_mask)[0]
    elif target == "fixed_point":
        good = base_mask & (stop_gens >= 200) & (stop_gens <= 1500)
        idxs = np.where(good)[0]
        if len(idxs) == 0:
            idxs = np.where(base_mask)[0]
    else:
        idxs = np.where(base_mask)[0]

    if len(idxs) == 0:
        return None
    return int(idxs[0])


def main():
    sl = np.full(N_SCAN, S_LOW,  dtype=np.float32)
    sh = np.full(N_SCAN, S_HIGH, dtype=np.float32)
    bl = np.full(N_SCAN, B_LOW,  dtype=np.float32)
    bh = np.full(N_SCAN, B_HIGH, dtype=np.float32)
    seeds = np.arange(N_SCAN, dtype=np.int64) + 1

    print(f"Scanning {N_SCAN} sims with snapshots at gens {SNAP_GENS}")
    t0 = time.time()
    res = simulate_batch(
        s_low=sl, s_high=sh, b_low=bl, b_high=bh, seeds=seeds,
        height=GRID_HEIGHT, width=GRID_WIDTH,
        max_generations=MAX_GEN,
        initial_density=INIT_DENSITY,
        saturation_threshold=SATURATION_THRESHOLD,
        cycle_buffer=CYCLE_BUFFER,
        snapshot_gens=SNAP_GENS,
    )
    elapsed = time.time() - t0
    print(f"Scan done in {elapsed:.1f}s ({N_SCAN/elapsed:.1f} sims/s)")

    outcomes = res["outcome"]
    outcomes_4cat = np.where(outcomes == "saturation", "fixed_point", outcomes)
    stop_gens = res["stop_generation"]
    cycle_periods = res["cycle_period"]
    snapshots = res["snapshots"]   # dict gen -> (B, H, W) uint8

    print("\nOutcome counts in scan:")
    for o in ROW_ORDER:
        n = int((outcomes_4cat == o).sum())
        print(f"  {o:16s}: {n}")

    picks = {}
    for o in ROW_ORDER:
        i = pick_index(outcomes_4cat, stop_gens, cycle_periods, o)
        if i is None:
            raise SystemExit(f"No sim found for outcome '{o}' in scan of {N_SCAN}")
        picks[o] = i
        print(f"  picked {o:16s}: idx={i}, seed={int(seeds[i])}, "
              f"stop_gen={int(stop_gens[i])}, cycle_period={int(cycle_periods[i])}")

    meta = {
        "parameters": {
            "S_low": S_LOW, "S_high": S_HIGH,
            "B_low": B_LOW, "B_high": B_HIGH,
            "initial_density": INIT_DENSITY,
            "max_generations": MAX_GEN,
            "grid": [GRID_HEIGHT, GRID_WIDTH],
            "N_scan": N_SCAN,
            "snapshot_gens": SNAP_GENS,
        },
        "picks": {o: {
            "scan_index": picks[o],
            "seed": int(seeds[picks[o]]),
            "stop_generation": int(stop_gens[picks[o]]),
            "cycle_period": int(cycle_periods[picks[o]]),
        } for o in ROW_ORDER},
    }
    with open(THIS_DIR / "picks.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ============================================================
    # Plot 4 rows x 7 cols
    # ============================================================
    n_rows = len(ROW_ORDER)
    n_cols = len(SNAP_GENS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.2, n_rows * 1.7),
        squeeze=False,
    )

    for r, outcome in enumerate(ROW_ORDER):
        cmap = ListedColormap(["white", ROW_COLORS[outcome]])
        idx = picks[outcome]
        for c, gen in enumerate(SNAP_GENS):
            ax = axes[r, c]
            grid_state = snapshots[gen][idx]
            ax.imshow(grid_state, cmap=cmap, vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("black")
            if r == 0:
                ax.set_title(f"gen {gen}", fontsize=18)
            if c == 0:
                ax.set_ylabel(outcome.replace("_", " "),
                              fontsize=18, rotation=90,
                              ha="center", va="center", labelpad=12)

    fig.tight_layout()
    p_png = THIS_DIR / "figure.png"
    p_pdf = THIS_DIR / "figure.pdf"
    fig.savefig(p_png, dpi=200)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"\nSaved: {p_png}\n       {p_pdf}")


if __name__ == "__main__":
    main()
