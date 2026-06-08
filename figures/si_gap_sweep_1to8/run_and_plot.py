"""SI: 8 phase-diagram heatmaps, one per Δ_low = b_low - s_low value in 1..8.

For each gap value G in {1, 2, ..., 8} and each (w_S, w_B) cell:
  * enumerate valid (s_low, b_low) positions satisfying b_low = s_low + G,
    s_low >= 1, s_high = s_low + w_S <= 18, b_high = b_low + w_B <= 18;
  * each valid position contributes N_RUNS_PER_POSITION = 100 runs;
  * cell outcome share = sum over (position, run) of indicators / total
    runs in the cell.

The two boundary lines generalize the gap-5 figure:
    line 1 (vertical):  w_S = G - 0.5
    line 2 (diagonal):  w_S = w_B + G - 0.5

Single H100 wall time per gap: ~6 min, ~50 min for all eight figures.
Each gap's run is cached to raw_gap<G>.csv next to this script.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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
DENOM = 18
S_WIDTH_UNITS = list(range(1, 17))   # 1..16
B_WIDTH_UNITS = list(range(1, 16))   # 1..15
N_RUNS_PER_POSITION = 100
GRID_HEIGHT = 75
GRID_WIDTH = 100
MAX_GEN = 2000
INIT_DENSITY = 0.25
CYCLE_BUFFER = MAX_GEN + 1
CHUNK_SIZE = 60_000

GAPS = list(range(1, 9))             # 1..8

OUTCOME_ORDER = ["extinction", "cycle", "long_transient", "fixed_point"]
OUTCOME_COLORS = {
    "extinction":     "#FA7F6F",
    "cycle":          "#8ECFC9",
    "long_transient": "#FFBE7A",
    "fixed_point":    "#82B0D2",
}
LINE_NO_PARTIAL  = "#000000"
LINE_PARTIAL_SUB = "#7B1FA2"


# ============================================================
# Simulation: one gap value at a time
# ============================================================
def build_positions(gap: int):
    """Per-(w_S, w_B) list of (s_low, b_low) positions with b_low - s_low == gap,
    s_low >= 1.  Returns a dict cell -> [(s, b), ...]."""
    positions = {}
    for w_S in S_WIDTH_UNITS:
        for w_B in B_WIDTH_UNITS:
            # s_low in [1, 18 - w_S]
            # b_low = s_low + gap in [0, 18 - w_B]  -> s_low in [-gap, 18 - w_B - gap]
            s_low_max = min(DENOM - w_S, DENOM - w_B - gap)
            if s_low_max < 1:
                continue
            cell_positions = [(s, s + gap) for s in range(1, s_low_max + 1)]
            positions[(w_S, w_B)] = cell_positions
    return positions


def simulate_gap(gap: int) -> pd.DataFrame:
    """Run all sims for a single gap and return a long-format DataFrame."""
    cache = THIS_DIR / f"raw_gap{gap}.csv"
    if cache.exists():
        print(f"[gap={gap}] loading cached sims: {cache}")
        return pd.read_csv(cache)

    positions = build_positions(gap)
    rows = []
    for (w_S, w_B), pos_list in positions.items():
        for (s_pos, b_pos) in pos_list:
            for run in range(N_RUNS_PER_POSITION):
                rows.append((w_S, w_B, s_pos, b_pos, run))
    n_total = len(rows)
    print(f"[gap={gap}] {len(positions)} cells, {n_total} sims")

    arr = np.array(rows, dtype=np.int32)
    w_S_arr = arr[:, 0]
    w_B_arr = arr[:, 1]
    s_pos   = arr[:, 2]
    b_pos   = arr[:, 3]
    run_arr = arr[:, 4]
    s_low_f  = s_pos.astype(np.float32) / DENOM
    s_high_f = (s_pos + w_S_arr).astype(np.float32) / DENOM
    b_low_f  = b_pos.astype(np.float32) / DENOM
    b_high_f = (b_pos + w_B_arr).astype(np.float32) / DENOM
    seeds = (np.int64(gap) * 1_000_000_007
             + w_S_arr.astype(np.int64) * 10_000_019
             + w_B_arr.astype(np.int64) * 100_003
             + s_pos.astype(np.int64) * 1_009
             + run_arr.astype(np.int64) + 1)

    outcomes = np.empty(n_total, dtype=object)
    t0 = time.time()
    for start in range(0, n_total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_total)
        res = simulate_batch(
            s_low=s_low_f[start:end], s_high=s_high_f[start:end],
            b_low=b_low_f[start:end], b_high=b_high_f[start:end],
            seeds=seeds[start:end],
            height=GRID_HEIGHT, width=GRID_WIDTH,
            max_generations=MAX_GEN,
            initial_density=INIT_DENSITY,
            cycle_buffer=CYCLE_BUFFER,
        )
        outcomes[start:end] = res["outcome"]
        print(f"  chunk {start:>7d}..{end:>7d}: elapsed {time.time() - t0:.1f}s")
    print(f"[gap={gap}] sim done in {time.time() - t0:.1f}s")

    df = pd.DataFrame({
        "w_S": w_S_arr, "w_B": w_B_arr,
        "s_pos": s_pos, "b_pos": b_pos,
        "run":   run_arr,
        "outcome": outcomes,
    })
    df["outcome_4cat"] = np.where(df["outcome"] == "saturation",
                                  "fixed_point", df["outcome"])
    df.to_csv(cache, index=False)
    print(f"[gap={gap}] cached to {cache}")
    return df


def compute_pcts(df: pd.DataFrame, gap: int):
    n_S = len(S_WIDTH_UNITS)
    n_B = len(B_WIDTH_UNITS)
    pcts = np.full((n_B, n_S, 4), np.nan, dtype=np.float64)
    valid_mask = np.zeros((n_B, n_S), dtype=bool)
    n_positions = np.zeros((n_B, n_S), dtype=np.int64)

    # how many positions per cell
    pos_cnt = (df.groupby(["w_S", "w_B"])[["s_pos", "b_pos"]]
                 .agg(lambda g: g.drop_duplicates().shape[0]).iloc[:, 0])
    for (w_S, w_B), n_pos in pos_cnt.items():
        if 1 <= w_S <= n_S and 1 <= w_B <= n_B and n_pos > 0:
            valid_mask[w_B - 1, w_S - 1] = True
            n_positions[w_B - 1, w_S - 1] = int(n_pos)

    grp = df.groupby(["w_S", "w_B", "outcome_4cat"]).size().reset_index(name="count")
    for k, o in enumerate(OUTCOME_ORDER):
        sub = grp[grp["outcome_4cat"] == o]
        for _, row in sub.iterrows():
            ws, wb, c = int(row["w_S"]), int(row["w_B"]), int(row["count"])
            if not valid_mask[wb - 1, ws - 1]:
                continue
            total = n_positions[wb - 1, ws - 1] * N_RUNS_PER_POSITION
            pcts[wb - 1, ws - 1, k] = c / total * 100

    # cells where the outcome simply didn't occur -> share 0 (not NaN)
    for k in range(4):
        mask_valid_zero = valid_mask & np.isnan(pcts[..., k])
        pcts[..., k] = np.where(mask_valid_zero, 0.0, pcts[..., k])

    return pcts, valid_mask, n_positions


def plot_one_gap(pcts, valid_mask, n_positions, gap, out_dir):
    n_S = len(S_WIDTH_UNITS)
    n_B = len(B_WIDTH_UNITS)

    valid_row = valid_mask.any(axis=1)
    if not valid_row.any():
        print(f"[gap={gap}] no valid cell; skip")
        return
    last_valid = int(np.where(valid_row)[0].max())
    n_B_show = last_valid + 1
    pcts_show = pcts[:n_B_show]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2))
    line_handles = None
    for k, outcome in enumerate(OUTCOME_ORDER):
        ax = axes[k]
        cmap = LinearSegmentedColormap.from_list(
            f"cm_{outcome}", ["white", OUTCOME_COLORS[outcome]], N=256)
        cmap.set_bad("#D9D9D9")
        data = np.ma.masked_invalid(pcts_show[..., k])
        vmax = max(float(data.max()) if data.count() > 0 else 1.0, 1e-6)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                       origin="lower", aspect="auto",
                       extent=(0.5, n_S + 0.5, 0.5, n_B_show + 0.5))
        ax.set_xticks(S_WIDTH_UNITS)
        ax.set_yticks(list(range(1, n_B_show + 1)))
        ax.set_xlabel("S width (units of 1/18)")
        if k == 0:
            ax.set_ylabel("B width (units of 1/18)")
        ax.set_title(outcome.replace("_", " "))

        # Boundary 1: vertical at w_S = gap - 0.5
        x_line1 = gap - 0.5
        l1 = None
        if 0.5 < x_line1 < n_S + 0.5:
            l1, = ax.plot([x_line1, x_line1], [0.5, n_B_show + 0.5],
                          color=LINE_NO_PARTIAL, linewidth=2,
                          label="no overlap | partial overlap")

        # Boundary 2: diagonal w_S = w_B + gap - 0.5
        x_left, y_left = float(gap), 0.5
        x_right_full = float(n_S) + 0.5
        y_right_full = x_right_full - (gap - 0.5)
        if y_right_full > n_B_show + 0.5:
            y_top = n_B_show + 0.5
            x_top = y_top + (gap - 0.5)
            x_right, y_right = x_top, y_top
        else:
            x_right, y_right = x_right_full, y_right_full
        l2 = None
        if x_left < n_S + 0.5 and y_left < n_B_show + 0.5:
            l2, = ax.plot([x_left, x_right], [y_left, y_right],
                          color=LINE_PARTIAL_SUB, linewidth=2,
                          label=r"partial overlap | B $\subseteq$ S")

        ax.set_xlim(0.5, n_S + 0.5)
        ax.set_ylim(0.5, n_B_show + 0.5)
        if line_handles is None:
            collected = [h for h in (l1, l2) if h is not None]
            if collected:
                line_handles = collected

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("share (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    fig.suptitle(rf"$b_{{\rm low}} - s_{{\rm low}} = {gap}/18$",
                 y=1.06, fontsize=18)
    if line_handles:
        fig.legend(line_handles, [h.get_label() for h in line_handles],
                   loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.00), fontsize=13, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p_png = out_dir / f"figure_gap{gap}.png"
    p_pdf = out_dir / f"figure_gap{gap}.pdf"
    fig.savefig(p_png, dpi=200, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.close(fig)
    total_cells = int(valid_mask.sum())
    total_pos = int(n_positions.sum())
    print(f"[gap={gap}] saved {p_png.name} + .pdf  "
          f"(n_B_show={n_B_show}, valid_cells={total_cells}, "
          f"total_positions={total_pos})")


def main():
    for gap in GAPS:
        df = simulate_gap(gap)
        pcts, valid_mask, n_positions = compute_pcts(df, gap)
        plot_one_gap(pcts, valid_mask, n_positions, gap, THIS_DIR)


if __name__ == "__main__":
    main()
