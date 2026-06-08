"""
Batched SCA simulator with outcome detection.

Outcomes (first hit wins, in this priority):
  extinction     -- grid sum reaches 0
  saturation     -- mean(grid) >= saturation_threshold
  fixed_point    -- new hash equals previous step's hash
  cycle          -- new hash equals an earlier hash in the rolling buffer
  long_transient -- none of the above by max_generations

All sims run concurrently as a single (B, H, W) batch on GPU.
Sims that have already terminated continue to run (cheap on GPU), but their
outcome / stop_generation are frozen at first detection.
"""
from __future__ import annotations

import numpy as np
import cupy as cp

from .rule import update_batch, make_hash_weights, hash_batch


OUTCOME_LABELS = {
    0: "running",
    1: "extinction",
    2: "saturation",
    3: "fixed_point",
    4: "cycle",
    5: "long_transient",
}


def simulate_batch(
    s_low: np.ndarray,                # (B,) float
    s_high: np.ndarray,                # (B,)
    b_low: np.ndarray,                 # (B,)
    b_high: np.ndarray,                # (B,)
    seeds: np.ndarray,                 # (B,) int64
    height: int,
    width: int,
    max_generations: int,
    initial_density=0.25,              # float OR (B,) array
    saturation_threshold: float = 1.01,
    cycle_buffer: int = 512,
    hash_seed: int = 12345,
    grid_init_seed: int = 7,
    record_change_rate_first_n: int = 0,
    record_density_first_n: int = 0,
    record_normed_std_first_n: int = 0,
    snapshot_gens=None,
) -> dict:
    """Run B independent SCA sims on a single GPU.

    Returns a dict of numpy arrays of length B:
      outcome (str), stop_generation (int), final_density (float),
      mean_density (float), final_n_max (float).
    """
    B = s_low.shape[0]
    assert s_high.shape == (B,) and b_low.shape == (B,) and b_high.shape == (B,)
    assert seeds.shape == (B,)

    s_low_g  = cp.asarray(s_low,  dtype=cp.float32)
    s_high_g = cp.asarray(s_high, dtype=cp.float32)
    b_low_g  = cp.asarray(b_low,  dtype=cp.float32)
    b_high_g = cp.asarray(b_high, dtype=cp.float32)

    # Per-sim initial grids: combine a master seed with each sim's seed so
    # different (param, run) combos get independent IID inits.
    rng = cp.random.default_rng(grid_init_seed)
    base = rng.random((B, height, width), dtype=cp.float32)
    # cheap per-sim "jitter" via seed-derived offsets to decorrelate batch elements
    jitter = cp.asarray((seeds.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFF), dtype=cp.uint32)
    jitter_f = jitter.astype(cp.float32) / cp.float32(2**32)
    # XOR-like decorrelation: shift each sim's noise pattern
    base = (base + jitter_f[:, None, None]) % cp.float32(1.0)
    init_dens_np = np.asarray(initial_density, dtype=np.float32)
    if init_dens_np.ndim == 0:
        grid = (base < cp.float32(float(init_dens_np))).astype(cp.uint8)
    else:
        assert init_dens_np.shape == (B,), (
            f"initial_density shape {init_dens_np.shape} does not match batch ({B},)"
        )
        density_thresh = cp.asarray(init_dens_np)[:, None, None]
        grid = (base < density_thresh).astype(cp.uint8)

    weights = make_hash_weights(height, width, seed=hash_seed)

    # Outcome bookkeeping on GPU
    outcome = cp.zeros(B, dtype=cp.uint8)          # 0 = running
    stop_gen = cp.full(B, -1, dtype=cp.int64)
    density_at_stop = cp.full(B, cp.nan, dtype=cp.float64)
    # cycle period (= cycle length, in generations). 0 = no cycle detected (or non-cycle outcome).
    cycle_period = cp.zeros(B, dtype=cp.int64)

    # Density accumulators for mean_density (only counted while sim is running)
    density_sum = cp.zeros(B, dtype=cp.float64)
    density_count = cp.zeros(B, dtype=cp.int64)

    # Change-rate accumulators (accumulated over ALL max_generations transitions,
    # regardless of outcome; sims that hit extinction/fixed_point produce 0 diff
    # for the rest of the run, sims in a cycle keep producing the cycle's diff).
    change_sum = cp.zeros(B, dtype=cp.float64)
    change_sumsq = cp.zeros(B, dtype=cp.float64)

    # Optional per-step change-rate trajectory, recorded for the first N transitions.
    record_n = int(record_change_rate_first_n)
    if record_n > 0:
        # shape (record_n, B) so each column is one sim's trace
        cr_trajectory = cp.zeros((record_n, B), dtype=cp.float32)
    else:
        cr_trajectory = None

    # Optional per-step density trajectory, recorded for the first M generations.
    record_dn = int(record_density_first_n)
    if record_dn > 0:
        density_trajectory = cp.zeros((record_dn, B), dtype=cp.float32)
    else:
        density_trajectory = None

    # Optional per-step std(n / n_max) trajectory.
    record_nstd = int(record_normed_std_first_n)
    if record_nstd > 0:
        nstd_trajectory = cp.zeros((record_nstd, B), dtype=cp.float32)
    else:
        nstd_trajectory = None

    # Rolling hash buffer (cycle_buffer, B), filled with sentinel 0
    buffer = cp.zeros((cycle_buffer, B), dtype=cp.uint64)
    buffer_filled = 0
    buffer_pos = 0  # next slot to write
    n_max_last = cp.zeros(B, dtype=cp.float32)

    # Optional grid snapshots at requested generations. snapshots[gen] is the
    # full (B, H, W) uint8 grid state at the START of iteration `gen` (i.e.,
    # after `gen` update steps from the initial state).
    if snapshot_gens is not None:
        snapshot_gens_set = {int(g) for g in snapshot_gens}
        snapshots = {}
    else:
        snapshot_gens_set = set()
        snapshots = None

    cell_count = height * width

    for gen in range(max_generations + 1):
        if gen in snapshot_gens_set:
            snapshots[gen] = cp.asnumpy(grid)

        cell_sum = grid.sum(axis=(1, 2), dtype=cp.int64)         # (B,)
        density = cell_sum.astype(cp.float64) / cell_count        # (B,)

        if density_trajectory is not None and gen < record_dn:
            density_trajectory[gen] = density.astype(cp.float32)

        if nstd_trajectory is not None and gen < record_nstd:
            # Recompute the rank-1 separable field for the current grid in
            # order to read off std(n / n_max).  This duplicates work that
            # update_batch will do shortly, but the recording path is
            # opt-in and only used when this trajectory is requested.
            g_f = grid.astype(cp.float32)
            cs = g_f.sum(axis=1)
            rs = g_f.sum(axis=2)
            px = cp.roll(cs, 1, axis=1) + cs + cp.roll(cs, -1, axis=1)
            py = cp.roll(rs, 1, axis=1) + rs + cp.roll(rs, -1, axis=1)
            n_field = py[:, :, None] * px[:, None, :]
            nm = n_field.reshape(B, -1).max(axis=1)
            safe_nm = cp.where(nm == 0, cp.float32(1.0), nm)
            n_normed = n_field / safe_nm[:, None, None]
            nstd_trajectory[gen] = n_normed.reshape(B, -1).std(axis=1).astype(cp.float32)

        running = outcome == 0
        density_sum = cp.where(running, density_sum + density, density_sum)
        density_count = cp.where(running, density_count + 1, density_count)

        # extinction
        ext = running & (cell_sum == 0)
        outcome = cp.where(ext, cp.uint8(1), outcome)
        stop_gen = cp.where(ext, gen, stop_gen)
        density_at_stop = cp.where(ext, density, density_at_stop)

        # saturation
        running = outcome == 0
        sat = running & (density >= saturation_threshold)
        outcome = cp.where(sat, cp.uint8(2), outcome)
        stop_gen = cp.where(sat, gen, stop_gen)
        density_at_stop = cp.where(sat, density, density_at_stop)

        # hash-based cycle detection
        h = hash_batch(grid, weights)
        running = outcome == 0

        if buffer_filled > 0:
            # fixed point: hash equals the most recently written entry (period = 1)
            last_idx = (buffer_pos - 1) % cycle_buffer
            fix = running & (h == buffer[last_idx])
            outcome = cp.where(fix, cp.uint8(3), outcome)
            stop_gen = cp.where(fix, gen, stop_gen)
            density_at_stop = cp.where(fix, density, density_at_stop)
            cycle_period = cp.where(fix, cp.int64(1), cycle_period)

            # cycle (period >= 2): hash matches any earlier entry but not the latest
            running = outcome == 0
            if buffer_filled > 1:
                slot_idx = cp.arange(cycle_buffer)
                if buffer_filled < cycle_buffer:
                    valid_slot = slot_idx < buffer_filled
                else:
                    valid_slot = cp.ones(cycle_buffer, dtype=cp.bool_)
                valid_slot = valid_slot & (slot_idx != last_idx)
                match_mask = (buffer == h[None, :]) & valid_slot[:, None]  # (T, B)
                match_any = match_mask.any(axis=0)
                # period: distance (in slot steps) back to the most-recent prior match,
                # using cyclic-buffer arithmetic so cycle_buffer < max_gen also works.
                slot_idx_col = slot_idx.astype(cp.int64)[:, None]
                masked_idx = cp.where(match_mask, slot_idx_col, cp.int64(-1))
                last_match_idx = masked_idx.max(axis=0)            # (B,) or -1
                # period = 1 + (buffer_pos - 1 - last_match_idx) mod cycle_buffer
                period = (cp.int64(buffer_pos - 1) - last_match_idx) % cp.int64(cycle_buffer) + cp.int64(1)

                cyc = running & match_any
                outcome = cp.where(cyc, cp.uint8(4), outcome)
                stop_gen = cp.where(cyc, gen, stop_gen)
                density_at_stop = cp.where(cyc, density, density_at_stop)
                cycle_period = cp.where(cyc, period, cycle_period)

        # write current hash into rolling buffer
        buffer[buffer_pos] = h
        buffer_pos = (buffer_pos + 1) % cycle_buffer
        buffer_filled = min(buffer_filled + 1, cycle_buffer)

        if gen == max_generations:
            break

        new_grid, n_max_last = update_batch(grid, s_low_g, s_high_g, b_low_g, b_high_g)
        # change rate at this transition = fraction of cells that flipped
        diff = (new_grid != grid).astype(cp.float32).mean(axis=(1, 2)).astype(cp.float64)
        change_sum = change_sum + diff
        change_sumsq = change_sumsq + diff * diff
        if cr_trajectory is not None and gen < record_n:
            cr_trajectory[gen] = diff.astype(cp.float32)
        grid = new_grid

    # Anything still running -> long_transient; freeze density at MAX_GEN
    final_density_max_gen = grid.sum(axis=(1, 2), dtype=cp.int64).astype(cp.float64) / cell_count
    long_t = outcome == 0
    outcome = cp.where(long_t, cp.uint8(5), outcome)
    stop_gen = cp.where(long_t, max_generations, stop_gen)
    density_at_stop = cp.where(long_t, final_density_max_gen, density_at_stop)

    final_density = density_at_stop
    mean_density = density_sum / cp.maximum(density_count, 1)

    # Per-sim change-rate mean and std over the full max_generations transitions
    mg = cp.float64(max_generations)
    change_rate_mean = change_sum / mg
    change_rate_var = cp.maximum(change_sumsq / mg - change_rate_mean * change_rate_mean,
                                 cp.float64(0))
    change_rate_std = cp.sqrt(change_rate_var)

    outcome_np = cp.asnumpy(outcome)
    outcome_str = np.array([OUTCOME_LABELS[int(o)] for o in outcome_np], dtype=object)

    result = {
        "outcome": outcome_str,
        "stop_generation": cp.asnumpy(stop_gen),
        "final_density": cp.asnumpy(final_density),
        "mean_density": cp.asnumpy(mean_density),
        "final_n_max": cp.asnumpy(n_max_last),
        "cycle_period": cp.asnumpy(cycle_period),
        "change_rate_mean": cp.asnumpy(change_rate_mean),
        "change_rate_std": cp.asnumpy(change_rate_std),
    }
    if cr_trajectory is not None:
        # transpose to (B, N) so each row is one sim's trajectory
        result["change_rate_trajectory"] = cp.asnumpy(cr_trajectory.T)
    if density_trajectory is not None:
        result["density_trajectory"] = cp.asnumpy(density_trajectory.T)
    if nstd_trajectory is not None:
        result["normed_std_trajectory"] = cp.asnumpy(nstd_trajectory.T)
    if snapshots is not None:
        result["snapshots"] = snapshots
    return result
