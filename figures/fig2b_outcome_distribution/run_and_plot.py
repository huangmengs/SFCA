"""Outcome distribution at canonical SCA parameters:
    S = [3/18, 11/18], B = [7/18, 9/18], rho_0 = 0.25
    grid 75 x 100, max_gen = 2000, N = 100000 independent runs.

Produces figure.png / figure.pdf (4-bar histogram of outcome shares) and
results.json (raw counts and percentages, plus wall time).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16


THIS_DIR = Path(__file__).resolve().parent
# this script lives at SCA-research/figures/<this>/, simulator lives at
# SCA-research/mengsha_experiment/common/.
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
N_RUNS = 100_000
INIT_DENSITY = 0.25
SATURATION_THRESHOLD = 0.90
CYCLE_BUFFER = MAX_GEN + 1   # full-history cycle detection


def main():
    sl = np.full(N_RUNS, S_LOW,  dtype=np.float32)
    sh = np.full(N_RUNS, S_HIGH, dtype=np.float32)
    bl = np.full(N_RUNS, B_LOW,  dtype=np.float32)
    bh = np.full(N_RUNS, B_HIGH, dtype=np.float32)
    seeds = np.arange(N_RUNS, dtype=np.int64) + 1

    print(f"Running {N_RUNS} sims at "
          f"S=[{S_LOW:.4f}, {S_HIGH:.4f}], B=[{B_LOW:.4f}, {B_HIGH:.4f}], "
          f"rho_0={INIT_DENSITY}, max_gen={MAX_GEN}")
    print(f"Grid {GRID_HEIGHT} x {GRID_WIDTH}, cycle_buffer={CYCLE_BUFFER}")
    t0 = time.time()
    res = simulate_batch(
        s_low=sl, s_high=sh, b_low=bl, b_high=bh, seeds=seeds,
        height=GRID_HEIGHT, width=GRID_WIDTH,
        max_generations=MAX_GEN,
        initial_density=INIT_DENSITY,
        saturation_threshold=SATURATION_THRESHOLD,
        cycle_buffer=CYCLE_BUFFER,
    )
    elapsed = time.time() - t0
    print(f"Sim done in {elapsed:.1f}s ({N_RUNS/elapsed:.1f} sims/s)")

    # Fold saturation into fixed_point for the 4-cat view (the convention used
    # throughout mengsha_experiment).
    outcomes = res["outcome"]
    outcomes_4cat = np.where(outcomes == "saturation", "fixed_point", outcomes)

    ORDER = ["extinction", "cycle", "long transient", "fixed point"]
    counts = {o: int((outcomes_4cat == o).sum()) for o in ORDER}
    n_sat_only = int((outcomes == "saturation").sum())
    total = sum(counts.values())
    pcts = {o: 100.0 * counts[o] / total for o in ORDER}

    print(f"\nOutcome counts (N={total}):")
    for o in ORDER:
        print(f"  {o:16s}: {counts[o]:7d}  ({pcts[o]:6.3f}%)")
    print(f"  (of which saturation -> fixed_point: {n_sat_only})")

    with open(THIS_DIR / "results.json", "w") as f:
        json.dump({
            "parameters": {
                "S_low": S_LOW, "S_high": S_HIGH,
                "B_low": B_LOW, "B_high": B_HIGH,
                "initial_density": INIT_DENSITY,
                "max_generations": MAX_GEN,
                "grid": [GRID_HEIGHT, GRID_WIDTH],
                "N_runs": N_RUNS,
                "saturation_threshold": SATURATION_THRESHOLD,
                "cycle_buffer": CYCLE_BUFFER,
            },
            "counts": counts,
            "percentages": pcts,
            "n_saturation_folded_into_fixed_point": n_sat_only,
            "wall_time_seconds": elapsed,
        }, f, indent=2)

    # ============================================================
    # Plot: 4-bar histogram of outcome shares.
    # Colors: extinction=C0, cycle=C1, long_transient=C2, fixed_point=C3.
    # ============================================================
    colors = ["#FA7F6F", "#8ECFC9", "#FFBE7A", "#82B0D2"]
    values = [pcts[o] for o in ORDER]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(ORDER, values, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(values) * 0.012,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=16)
    ax.set_ylabel("share of runs (%)")
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    p_png = THIS_DIR / "figure.png"
    p_pdf = THIS_DIR / "figure.pdf"
    fig.savefig(p_png, dpi=200)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"\nSaved: {p_png}\n       {p_pdf}")


if __name__ == "__main__":
    main()
