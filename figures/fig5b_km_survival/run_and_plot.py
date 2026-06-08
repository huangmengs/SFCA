"""Kaplan-Meier-style survival curves at several S_width values.

Survival = "sim has not yet terminated to a cycle / fixed_point / extinction".
LT sims (those that run all the way to max_gen without termination) are
treated as still surviving at every gen up to max_gen.

Base configuration (mirrors exp02 / exp03):
    denominator = 180
    S_low  = 10/180  (fixed)
    B      = [60/180, 160/180]  (fixed)
    grid 75 x 100, rho_0 = 0.25, max_gen = 2000

S_widths used: 50, 58, 60, 62, 70 (units of 1/180).  (Note: the user-supplied
list included 62/180 twice -- read as a typo and deduplicated to five
distinct widths.)

Colour gradient runs from "#FA7F6F" (red) at the lowest width through white
at the middle to "#82B0D2" (blue) at the highest, sampled at five evenly
spaced positions; the middle curve is white so each line is given a thin
light-grey stroke for visibility on white background.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent))

from sfca.simulate import simulate_batch  # noqa: E402


# ============================================================
DENOM = 180
S_LOW_UNIT = 10
B_LOW_UNIT = 60
B_HIGH_UNIT = 160

S_WIDTH_UNITS = [50, 53, 56, 58, 60, 62, 65, 70]

GRID_HEIGHT = 75
GRID_WIDTH = 100
MAX_GEN = 100_000
INIT_DENSITY = 0.25
CYCLE_BUFFER = MAX_GEN + 1
N_RUNS_PER_WIDTH = 1000

CMAP = LinearSegmentedColormap.from_list(
    "FA_white_82", ["#FA7F6F", "#FFFFFF", "#82B0D2"], N=256)


def main():
    n_widths = len(S_WIDTH_UNITS)
    total = n_widths * N_RUNS_PER_WIDTH
    print(f"Widths: {S_WIDTH_UNITS} (/{DENOM})")
    print(f"N per width: {N_RUNS_PER_WIDTH}, total sims: {total}, "
          f"max_gen={MAX_GEN}")

    sw = np.repeat(S_WIDTH_UNITS, N_RUNS_PER_WIDTH).astype(np.int32)
    run = np.tile(np.arange(N_RUNS_PER_WIDTH), n_widths).astype(np.int32)

    s_low  = np.full(total, S_LOW_UNIT / DENOM,                dtype=np.float32)
    s_high = (S_LOW_UNIT + sw).astype(np.float32) / DENOM
    b_low  = np.full(total, B_LOW_UNIT / DENOM,                dtype=np.float32)
    b_high = np.full(total, B_HIGH_UNIT / DENOM,               dtype=np.float32)
    seeds = (sw.astype(np.int64) * 1_000_003
             + run.astype(np.int64) + 1)

    t0 = time.time()
    res = simulate_batch(
        s_low=s_low, s_high=s_high, b_low=b_low, b_high=b_high,
        seeds=seeds,
        height=GRID_HEIGHT, width=GRID_WIDTH,
        max_generations=MAX_GEN,
        initial_density=INIT_DENSITY,
        cycle_buffer=CYCLE_BUFFER,
    )
    elapsed = time.time() - t0
    print(f"sim done in {elapsed:.1f}s ({total/elapsed:.1f} sims/s)")

    outcomes = res["outcome"]
    stop_gens = res["stop_generation"]

    # ----------------------------------------------------------------
    # Build the survival curve for each width.
    # An event = termination to extinction / cycle / fixed_point / saturation.
    # LT sims are treated as censored at max_gen (still surviving throughout).
    # S(t) = #{sims whose event time > t} / N, with LT events placed at
    #        max_gen + 1 so they always count as surviving for t <= max_gen.
    # ----------------------------------------------------------------
    # Log-spaced t_grid: 1000 points spanning [1, MAX_GEN] then deduped.
    # KM is a step function so any dense-enough log grid is visually identical.
    log_grid = np.unique(np.round(
        np.logspace(0, np.log10(MAX_GEN), 1200)).astype(np.int64))
    t_grid = np.concatenate([[0], log_grid])
    curves = {}
    counts_by_outcome = {}

    for w in S_WIDTH_UNITS:
        mask = (sw == w)
        ev_gen = stop_gens[mask].copy()
        outc = outcomes[mask]
        # LT sims: treat as surviving past max_gen.
        ev_gen[outc == "long_transient"] = MAX_GEN + 1
        # Sort and use search-style counting.
        ev_sorted = np.sort(ev_gen)
        N = len(ev_sorted)
        # number of events with ev_gen <= t
        n_event_le_t = np.searchsorted(ev_sorted, t_grid, side="right")
        surv = (N - n_event_le_t) / N
        curves[w] = surv
        counts_by_outcome[w] = {
            o: int((outc == o).sum())
            for o in ["extinction", "cycle", "fixed_point",
                      "long_transient", "saturation"]
        }

    # ----------------------------------------------------------------
    # Save data
    # ----------------------------------------------------------------
    with open(THIS_DIR / "km_data.json", "w") as f:
        json.dump({
            "parameters": {
                "denominator": DENOM,
                "S_low_unit": S_LOW_UNIT,
                "B_low_unit": B_LOW_UNIT,
                "B_high_unit": B_HIGH_UNIT,
                "S_width_units": S_WIDTH_UNITS,
                "grid": [GRID_HEIGHT, GRID_WIDTH],
                "max_generations": MAX_GEN,
                "initial_density": INIT_DENSITY,
                "N_runs_per_width": N_RUNS_PER_WIDTH,
                "saturation_threshold": "simulate.py default (1.01, never triggers)",
            },
            "outcome_counts": counts_by_outcome,
            "survival_curves_note": (
                "survival[t] = fraction of sims still running at generation t; "
                "LT runs are censored at max_gen (treated as surviving throughout)."
            ),
            "t_grid": t_grid.tolist(),
            "survival_curves": {
                str(w): curves[w].tolist() for w in S_WIDTH_UNITS
            },
        }, f, indent=2)
    print("saved km_data.json")

    plot_km(t_grid, curves)


def plot_km(t_grid, curves):
    fig, ax = plt.subplots(figsize=(9, 6))
    n = len(S_WIDTH_UNITS)
    handles = []
    for i, w in enumerate(S_WIDTH_UNITS):
        # evenly spaced colours across the FA->white->82 gradient
        color = CMAP(i / (n - 1))
        line, = ax.plot(t_grid, curves[w], color=color, linewidth=2.6,
                        label=f"S width = {w}/180")
        # Light-grey halo so the white-ish middle curves remain visible on
        # a white axes background; minimal impact on red/blue endpoints.
        line.set_path_effects([
            path_effects.Stroke(linewidth=4.0,
                                foreground="#9E9E9E", alpha=0.7),
            path_effects.Normal(),
        ])
        handles.append(line)

    ax.set_xscale("log")
    ax.set_xlim(1, MAX_GEN)
    ax.set_ylim(-0.02, 1.02)
    # Drop the implicit t=0 point so the log axis doesn't squash everything
    # against the left edge; the lines start cleanly at t=1.
    ax.set_xlabel("generation $t$ (log scale)")
    ax.set_ylabel("survival probability  $S(t)$")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(handles=handles, loc="lower left", fontsize=12, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    p_png = THIS_DIR / "figure.png"
    p_pdf = THIS_DIR / "figure.pdf"
    fig.savefig(p_png, dpi=200)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"Saved: {p_png}\n       {p_pdf}")


if __name__ == "__main__":
    main()
