"""Phase-diagram heatmap with "per-position" run protocol.

For each cell (w_S, w_B):
  * enumerate all valid (S_low, B_low) positions subject to S_low < B_low.
  * run N_RUNS_PER_POSITION independent sims AT EACH position.
  * cell outcome share = mean indicator over all (position, run) pairs in
    the cell -- so positions are equally weighted within a cell, and cells
    with more positions contribute more total runs.

Total positions over the 16 x 15 grid: 13,060.
Total sims at N=200 per position: 2.61M.
Split across 4 H100 GPUs for ~20 min wall time.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
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
PROJECT_ROOT = THIS_DIR.parent.parent


# ============================================================
DENOM = 18
S_WIDTH_UNITS = list(range(1, 17))   # 1..16
B_WIDTH_UNITS = list(range(1, 16))   # 1..15

GRID_HEIGHT = 75
GRID_WIDTH = 100
MAX_GEN = 2000
INIT_DENSITY = 0.25
SATURATION_THRESHOLD = 0.90
CYCLE_BUFFER = MAX_GEN + 1
N_RUNS_PER_POSITION = 200

OUTCOME_ORDER = ["extinction", "cycle", "long_transient", "fixed_point"]
OUTCOME_COLORS = {
    "extinction":     "#FA7F6F",
    "cycle":          "#8ECFC9",
    "long_transient": "#FFBE7A",
    "fixed_point":    "#82B0D2",
}
LABEL_TO_INT = {
    "running": 0, "extinction": 1, "saturation": 2,
    "fixed_point": 3, "cycle": 4, "long_transient": 5,
}

NUM_GPUS = 4
CHUNK_SIZE = 60_000


def build_cells():
    """Per-cell metadata: w_S, w_B, list of valid (S_low, B_low) positions."""
    cells = []
    for w_S in S_WIDTH_UNITS:
        n_s = DENOM - w_S + 1
        for w_B in B_WIDTH_UNITS:
            n_b = DENOM - w_B + 1
            positions = [(s, b) for s in range(n_s)
                                  for b in range(n_b) if s < b]
            cells.append({
                "w_S": w_S, "w_B": w_B,
                "positions": positions, "n_pos": len(positions),
            })
    return cells


def shard_cells(cells, num_shards):
    """Greedy LPT (longest-processing-time) sharding to balance work."""
    shards = [[] for _ in range(num_shards)]
    weights = [0] * num_shards
    for c in sorted(cells, key=lambda x: -x["n_pos"]):
        i = min(range(num_shards), key=lambda j: weights[j])
        shards[i].append(c)
        weights[i] += c["n_pos"]
    return shards, weights


def worker(gpu_id: int, shard: list, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(PROJECT_ROOT))
    from sfca.simulate import simulate_batch  # noqa: E402

    rows = []
    for c in shard:
        w_S, w_B = c["w_S"], c["w_B"]
        for (s_pos, b_pos) in c["positions"]:
            for run in range(N_RUNS_PER_POSITION):
                rows.append((w_S, w_B, s_pos, b_pos, run))
    arr = np.array(rows, dtype=np.int32)
    w_S_arr = arr[:, 0]
    w_B_arr = arr[:, 1]
    s_pos   = arr[:, 2]
    b_pos   = arr[:, 3]
    run_arr = arr[:, 4]

    s_low  = s_pos.astype(np.float32) / DENOM
    s_high = (s_pos + w_S_arr).astype(np.float32) / DENOM
    b_low  = b_pos.astype(np.float32) / DENOM
    b_high = (b_pos + w_B_arr).astype(np.float32) / DENOM
    seeds = (w_S_arr.astype(np.int64) * 1_000_000_007
             + w_B_arr.astype(np.int64) * 10_000_019
             + s_pos.astype(np.int64) * 100_003
             + b_pos.astype(np.int64) * 1_009
             + run_arr.astype(np.int64) + 1)

    N = len(seeds)
    print(f"[GPU {gpu_id}] {len(shard)} cells, {N} sims", flush=True)

    outcomes_all = np.empty(N, dtype=object)
    t0 = time.time()
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        res = simulate_batch(
            s_low=s_low[start:end], s_high=s_high[start:end],
            b_low=b_low[start:end], b_high=b_high[start:end],
            seeds=seeds[start:end],
            height=GRID_HEIGHT, width=GRID_WIDTH,
            max_generations=MAX_GEN,
            initial_density=INIT_DENSITY,
            saturation_threshold=SATURATION_THRESHOLD,
            cycle_buffer=CYCLE_BUFFER,
        )
        outcomes_all[start:end] = res["outcome"]
        print(f"[GPU {gpu_id}] chunk {start:>7d}..{end:>7d} "
              f"elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"[GPU {gpu_id}] done in {time.time()-t0:.1f}s", flush=True)

    # Convert outcomes to int for compact transfer
    outcomes_int = np.array([LABEL_TO_INT[o] for o in outcomes_all],
                            dtype=np.uint8)
    result_queue.put({
        "gpu_id": gpu_id,
        "w_S": w_S_arr, "w_B": w_B_arr,
        "s_pos": s_pos, "b_pos": b_pos, "run": run_arr,
        "outcomes_int": outcomes_int,
    })


def main():
    cells = build_cells()
    total_pos = sum(c["n_pos"] for c in cells)
    total_sims = total_pos * N_RUNS_PER_POSITION
    print(f"Cells: {len(cells)}, total positions: {total_pos}, "
          f"N per position: {N_RUNS_PER_POSITION}, total sims: {total_sims}")

    shards, weights = shard_cells(cells, NUM_GPUS)
    print("Shard sizes (positions, sims):")
    for i, (sh, w) in enumerate(zip(shards, weights)):
        print(f"  GPU {i}: {len(sh):3d} cells, {w:5d} positions, "
              f"{w * N_RUNS_PER_POSITION:>7d} sims")

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    procs = []
    t0 = time.time()
    for gpu_id, sh in enumerate(shards):
        if not sh:
            continue
        p = mp.Process(target=worker, args=(gpu_id, sh, result_queue))
        p.start()
        procs.append(p)

    results = [result_queue.get() for _ in procs]
    for p in procs:
        p.join()
    print(f"\nAll workers done in {time.time()-t0:.1f}s")

    w_S_arr   = np.concatenate([r["w_S"]   for r in results])
    w_B_arr   = np.concatenate([r["w_B"]   for r in results])
    s_pos_arr = np.concatenate([r["s_pos"] for r in results])
    b_pos_arr = np.concatenate([r["b_pos"] for r in results])
    run_arr   = np.concatenate([r["run"]   for r in results])
    outcomes_int = np.concatenate([r["outcomes_int"] for r in results])

    # Map ints back to labels for aggregation logic
    INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
    outcomes = np.array([INT_TO_LABEL[int(i)] for i in outcomes_int],
                        dtype=object)
    outcomes_4cat = np.where(outcomes == "saturation",
                             "fixed_point", outcomes)

    n_S = len(S_WIDTH_UNITS)
    n_B = len(B_WIDTH_UNITS)
    cell_idx = (w_B_arr - 1) * n_S + (w_S_arr - 1)
    n_cells = n_B * n_S
    cell_run_count = np.bincount(cell_idx, minlength=n_cells)

    pcts = np.zeros((n_B, n_S, 4), dtype=np.float64)
    for k, o in enumerate(OUTCOME_ORDER):
        ind = (outcomes_4cat == o).astype(np.float64)
        cnt = np.bincount(cell_idx, weights=ind, minlength=n_cells)
        share = cnt / np.maximum(cell_run_count, 1)
        pcts[..., k] = (share * 100).reshape(n_B, n_S)

    np.save(THIS_DIR / "raw_outcomes.npy", outcomes_int)
    np.savez(THIS_DIR / "raw_indices.npz",
             w_S=w_S_arr, w_B=w_B_arr,
             s_pos=s_pos_arr, b_pos=b_pos_arr, run=run_arr)

    with open(THIS_DIR / "heatmap_data.json", "w") as f:
        json.dump({
            "parameters": {
                "denominator": DENOM,
                "S_width_units": S_WIDTH_UNITS,
                "B_width_units": B_WIDTH_UNITS,
                "grid": [GRID_HEIGHT, GRID_WIDTH],
                "max_generations": MAX_GEN,
                "initial_density": INIT_DENSITY,
                "saturation_threshold": SATURATION_THRESHOLD,
                "N_runs_per_position": N_RUNS_PER_POSITION,
                "position_constraint": "S_low < B_low (strict)",
            },
            "outcome_order": OUTCOME_ORDER,
            "percentages_shape": "n_B x n_S x 4 (outcome share %)",
            "percentages": pcts.tolist(),
            "cell_run_count": cell_run_count.reshape(n_B, n_S).tolist(),
            "cells_meta": [
                {"w_S": int(c["w_S"]), "w_B": int(c["w_B"]),
                 "n_pos": int(c["n_pos"]),
                 "total_runs": int(c["n_pos"] * N_RUNS_PER_POSITION)}
                for c in cells
            ],
            "total_sims": int(len(outcomes)),
        }, f, indent=2)

    plot_outcome_heatmap(pcts)


def plot_outcome_heatmap(pcts):
    """4-subplot percentage heatmap in a single row.  pcts shape (n_B, n_S, 4)."""
    n_S = len(S_WIDTH_UNITS)
    n_B = len(B_WIDTH_UNITS)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    for k, outcome in enumerate(OUTCOME_ORDER):
        ax = axes[k]
        cmap = LinearSegmentedColormap.from_list(
            f"cm_{outcome}", ["white", OUTCOME_COLORS[outcome]], N=256)
        data = pcts[..., k]
        vmax = max(float(data.max()), 1e-6)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                       origin="lower", aspect="auto",
                       extent=(0.5, n_S + 0.5, 0.5, n_B + 0.5))
        ax.set_xticks(S_WIDTH_UNITS)
        ax.set_yticks(B_WIDTH_UNITS)
        ax.set_xlabel("S width (units of 1/18)")
        if k == 0:
            ax.set_ylabel("B width (units of 1/18)")
        ax.set_title(outcome.replace("_", " "))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("share (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    p_png = THIS_DIR / "figure.png"
    p_pdf = THIS_DIR / "figure.pdf"
    fig.savefig(p_png, dpi=200)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"Saved: {p_png}\n       {p_pdf}")


if __name__ == "__main__":
    main()
