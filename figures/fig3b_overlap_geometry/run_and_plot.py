"""Phase-diagram heatmap with constraint  b_low - s_low = 5/18.

For each cell (w_S, w_B):
  * enumerate valid (S_low, B_low) positions with the additional constraints
        B_low = S_low + 5/18   AND   S_low >= 1/18
  * with S_low >= 1, the number of valid positions is min(18 - w_S, 13 - w_B)
    if >= 1, else 0.  Cells with no valid position appear as gray "invalid"
    tiles.  This excludes the previously-valid w_B = 13 row (and the already-
    invalid w_B in {14, 15}).
  * distribute N_RUNS_PER_CELL = 1000 runs across those positions
    (uniform-as-possible).  Cell outcome share = mean indicator over the
    1000 runs in the cell.

Saturation is no longer triggered (simulate.py default threshold is 1.01).

Two boundary lines partition the (w_S, w_B) plane into 3 regions of S vs B
overlap:
  * No overlap        : w_S <  5
  * Partial overlap   : 5 <= w_S <= w_B + 4
  * B fully inside S  : w_S >= w_B + 5
  Line 1 (vertical at w_S = 4.5) separates "no overlap" from "partial".
  Line 2 (slope-1 diagonal w_S = w_B + 4.5) separates "partial" from "B subset S".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 18


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent))

from sfca.simulate import simulate_batch  # noqa: E402


# ============================================================
DENOM = 18
S_WIDTH_UNITS = list(range(1, 17))   # 1..16
B_WIDTH_UNITS = list(range(1, 16))   # 1..15

GRID_HEIGHT = 75
GRID_WIDTH = 100
MAX_GEN = 2000
INIT_DENSITY = 0.25
CYCLE_BUFFER = MAX_GEN + 1
N_RUNS_PER_CELL = 1000
CHUNK_SIZE = 60_000

GAP = 5   # b_low - s_low = GAP / DENOM

OUTCOME_ORDER = ["extinction", "cycle", "long_transient", "fixed_point"]
OUTCOME_COLORS = {
    "extinction":     "#FA7F6F",
    "cycle":          "#8ECFC9",
    "long_transient": "#FFBE7A",
    "fixed_point":    "#82B0D2",
}

LINE_NO_PARTIAL  = "#000000"   # no-overlap | partial-overlap boundary
LINE_PARTIAL_SUB = "#7B1FA2"   # partial-overlap | B-subset-S boundary


def build_params():
    rows = []
    valid_cells = set()
    for w_S in S_WIDTH_UNITS:
        for w_B in B_WIDTH_UNITS:
            # b_low = s_low + GAP. s_low in [1, min(18-w_S, 18-w_B-GAP)].
            s_low_max = min(DENOM - w_S, DENOM - w_B - GAP)
            if s_low_max < 1:
                continue
            positions = [(s, s + GAP) for s in range(1, s_low_max + 1)]
            n_pos = len(positions)
            valid_cells.add((w_S, w_B))
            base = N_RUNS_PER_CELL // n_pos
            extra = N_RUNS_PER_CELL - base * n_pos
            for i, (s_pos, b_pos) in enumerate(positions):
                c = base + 1 if i < extra else base
                for run in range(c):
                    rows.append((w_S, w_B, s_pos, b_pos, run))

    arr = np.array(rows, dtype=np.int32)
    w_S_arr = arr[:, 0]
    w_B_arr = arr[:, 1]
    s_pos   = arr[:, 2]
    b_pos   = arr[:, 3]
    run     = arr[:, 4]

    s_low  = s_pos.astype(np.float32) / DENOM
    s_high = (s_pos + w_S_arr).astype(np.float32) / DENOM
    b_low  = b_pos.astype(np.float32) / DENOM
    b_high = (b_pos + w_B_arr).astype(np.float32) / DENOM

    seeds = (w_S_arr.astype(np.int64) * 1_000_000_007
             + w_B_arr.astype(np.int64) * 10_000_019
             + s_pos.astype(np.int64) * 100_003
             + b_pos.astype(np.int64) * 1_009
             + run.astype(np.int64) + 1)

    return s_low, s_high, b_low, b_high, seeds, w_S_arr, w_B_arr, valid_cells


def main():
    (s_low, s_high, b_low, b_high, seeds,
     w_S_arr, w_B_arr, valid_cells) = build_params()
    N = len(seeds)
    expected = len(valid_cells) * N_RUNS_PER_CELL
    assert N == expected, f"sim count {N} != expected {expected}"
    print(f"Valid cells: {len(valid_cells)}/240; total sims: {N}")
    print(f"Running in chunks of {CHUNK_SIZE}")

    outcomes_all = np.empty(N, dtype=object)
    t0 = time.time()
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        tc = time.time()
        res = simulate_batch(
            s_low=s_low[start:end], s_high=s_high[start:end],
            b_low=b_low[start:end], b_high=b_high[start:end],
            seeds=seeds[start:end],
            height=GRID_HEIGHT, width=GRID_WIDTH,
            max_generations=MAX_GEN,
            initial_density=INIT_DENSITY,
            cycle_buffer=CYCLE_BUFFER,
        )
        outcomes_all[start:end] = res["outcome"]
        dt = time.time() - tc
        print(f"  chunk {start:>7d}..{end:>7d}: "
              f"{(end-start)/dt:.1f} sims/s, {dt:.1f}s, "
              f"elapsed {time.time()-t0:.1f}s")
    total_time = time.time() - t0
    print(f"\nSim done in {total_time:.1f}s")

    outcomes_4cat = np.where(outcomes_all == "saturation",
                             "fixed_point", outcomes_all)

    n_S = len(S_WIDTH_UNITS)
    n_B = len(B_WIDTH_UNITS)
    cell_idx = (w_B_arr - 1) * n_S + (w_S_arr - 1)

    valid_mask = np.zeros((n_B, n_S), dtype=bool)
    for (w_S, w_B) in valid_cells:
        valid_mask[w_B - 1, w_S - 1] = True

    pcts = np.full((n_B, n_S, 4), np.nan, dtype=np.float64)
    for k, o in enumerate(OUTCOME_ORDER):
        ind = (outcomes_4cat == o).astype(np.float64)
        cnt = np.bincount(cell_idx, weights=ind, minlength=n_B * n_S)
        share = (cnt / N_RUNS_PER_CELL).reshape(n_B, n_S)
        pcts[..., k] = np.where(valid_mask, share * 100, np.nan)

    with open(THIS_DIR / "heatmap_data.json", "w") as f:
        json.dump({
            "parameters": {
                "denominator": DENOM,
                "S_width_units": S_WIDTH_UNITS,
                "B_width_units": B_WIDTH_UNITS,
                "grid": [GRID_HEIGHT, GRID_WIDTH],
                "max_generations": MAX_GEN,
                "initial_density": INIT_DENSITY,
                "saturation_threshold": "simulate.py default (1.01, never triggers)",
                "N_runs_per_cell": N_RUNS_PER_CELL,
                "position_constraint": f"b_low - s_low = {GAP}/18 and s_low >= 1/18",
            },
            "outcome_order": OUTCOME_ORDER,
            "percentages_shape": "n_B x n_S x 4 (outcome share %); null where no valid position",
            "percentages": [
                [[None if np.isnan(x) else float(x) for x in row]
                 for row in pcts[..., k]]
                for k in range(4)
            ],
            "valid_mask": valid_mask.tolist(),
            "total_sims": int(N),
            "wall_time_seconds": total_time,
        }, f, indent=2)
    print("saved heatmap_data.json")

    plot_heatmap(pcts)


def plot_heatmap(pcts):
    n_S = len(S_WIDTH_UNITS)
    n_B = len(B_WIDTH_UNITS)

    # Trim off the all-invalid (all-NaN) tail rows so the figure shows only
    # cells that actually have a sampled outcome share.
    valid_row = ~np.all(np.isnan(pcts), axis=(1, 2))   # shape (n_B,)
    last_valid = int(np.where(valid_row)[0].max())     # 0-indexed
    n_B_show = last_valid + 1                          # number of rows to draw
    pcts_show = pcts[:n_B_show]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    line_handles = None
    for k, outcome in enumerate(OUTCOME_ORDER):
        ax = axes[k]
        cmap = LinearSegmentedColormap.from_list(
            f"cm_{outcome}", ["white", OUTCOME_COLORS[outcome]], N=256)
        cmap.set_bad("#D9D9D9")  # invalid cells: light gray

        data = np.ma.masked_invalid(pcts_show[..., k])
        vmax = max(float(data.max()), 1e-6)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                       origin="lower", aspect="auto",
                       extent=(0.5, n_S + 0.5, 0.5, n_B_show + 0.5))
        ax.set_xticks(S_WIDTH_UNITS)
        ax.set_yticks(list(range(1, n_B_show + 1)))
        ax.set_xlabel("S width (units of 1/18)")
        if k == 0:
            ax.set_ylabel("B width (units of 1/18)")
        ax.set_title(outcome.replace("_", " "))

        # Boundary 1: no overlap | partial overlap (vertical at w_S = 4.5)
        l1, = ax.plot([4.5, 4.5], [0.5, n_B_show + 0.5],
                      color=LINE_NO_PARTIAL, linewidth=2,
                      label="no overlap | partial overlap")
        # Boundary 2: partial overlap | B subset S (diagonal w_S = w_B + 4.5).
        # Clip the diagonal to the displayed plot region.
        x_left, y_left = 5.0, 0.5
        x_right_full, y_right_full = 16.5, 12.0
        if y_right_full > n_B_show + 0.5:
            # diagonal exits through the top edge
            y_top = n_B_show + 0.5
            x_top = y_top + 4.5
            x_right, y_right = x_top, y_top
        else:
            x_right, y_right = x_right_full, y_right_full
        l2, = ax.plot([x_left, x_right], [y_left, y_right],
                      color=LINE_PARTIAL_SUB, linewidth=2,
                      label="partial overlap | B $\\subseteq$ S")
        ax.set_xlim(0.5, n_S + 0.5)
        ax.set_ylim(0.5, n_B_show + 0.5)

        if line_handles is None:
            line_handles = (l1, l2)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("share (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    # Single shared legend above the row.
    fig.legend(line_handles,
               [h.get_label() for h in line_handles],
               loc="upper center", ncol=2,
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
