"""
Separable Cellular Automaton (SCA) rule.

The neighborhood count is the outer product of two 1-D blurred sums:

    col_sum[i] = sum_j grid[j, i]
    row_sum[j] = sum_i grid[j, i]
    product_x[i] = col_sum[i-1] + col_sum[i] + col_sum[i+1]   (along x)
    product_y[j] = row_sum[j-1] + row_sum[j] + row_sum[j+1]   (along y)
    n[j, i] = product_y[j] * product_x[i]
    n_max   = max(n)

A cell survives iff it is alive and S_low * n_max <= n <= S_high * n_max.
A cell is born iff it is dead  and B_low * n_max <= n <= B_high * n_max.

Thresholds are stored as fractions in {0/D .. D/D}; D = DENOMINATOR.

This module provides a single batched, GPU update via CuPy:
each batch element b can have its own (S_low, S_high, B_low, B_high) tuple,
which is what the rule-position scan needs.
"""
from __future__ import annotations

import cupy as cp


DENOMINATOR = 18


def make_grids(
    batch_size: int,
    height: int,
    width: int,
    initial_density: float,
    seed: int,
) -> cp.ndarray:
    rng = cp.random.default_rng(seed)
    return (rng.random((batch_size, height, width), dtype=cp.float32) < initial_density).astype(cp.uint8)


def update_batch(
    grid: cp.ndarray,        # (B, H, W) uint8
    s_low: cp.ndarray,       # (B,) float32   in units of 1
    s_high: cp.ndarray,      # (B,) float32
    b_low: cp.ndarray,       # (B,) float32
    b_high: cp.ndarray,      # (B,) float32
) -> tuple[cp.ndarray, cp.ndarray]:
    """One synchronous step of the SCA rule, batched over independent sims.

    Returns the new grid and the per-sim n_max (useful for diagnostics).
    Each batch index b uses its own (s_low[b], s_high[b], b_low[b], b_high[b]).
    """
    g = grid.astype(cp.float32)  # work in float for the outer product

    col_sum = g.sum(axis=1)      # (B, W)
    row_sum = g.sum(axis=2)      # (B, H)

    product_x = cp.roll(col_sum, 1, axis=1) + col_sum + cp.roll(col_sum, -1, axis=1)  # (B, W)
    product_y = cp.roll(row_sum, 1, axis=1) + row_sum + cp.roll(row_sum, -1, axis=1)  # (B, H)

    # n[b, j, i] = product_y[b, j] * product_x[b, i]
    n = product_y[:, :, None] * product_x[:, None, :]  # (B, H, W)

    n_max = n.reshape(g.shape[0], -1).max(axis=1)  # (B,)
    # Avoid divide-by-zero when n_max == 0 (extinct or near-extinct grid):
    # any positive thresholds will fail, and we want the cell to die.
    safe_nmax = cp.where(n_max == 0, cp.float32(1.0), n_max)  # (B,)

    s_lo = (s_low * safe_nmax)[:, None, None]
    s_hi = (s_high * safe_nmax)[:, None, None]
    b_lo = (b_low * safe_nmax)[:, None, None]
    b_hi = (b_high * safe_nmax)[:, None, None]

    alive = grid.astype(cp.bool_)
    survive_mask = alive & (n >= s_lo) & (n <= s_hi)
    born_mask = (~alive) & (n >= b_lo) & (n <= b_hi)

    # Any sim with n_max == 0 -> all zero (nothing alive could create new life).
    new_grid = (survive_mask | born_mask).astype(cp.uint8)
    new_grid = cp.where((n_max == 0)[:, None, None], cp.uint8(0), new_grid)

    return new_grid, n_max


def make_hash_weights(height: int, width: int, seed: int = 12345) -> cp.ndarray:
    """Pre-generate 64-bit random weights for a GPU-friendly grid hash.

    hash(grid) = sum_{j,i} (grid[j,i] * weights[j,i])   mod 2^64
    Collision probability ~ 2^-64; cheap one-shot reduction per step.
    """
    # cupy's rng.integers rejects high=2**32 (int32 bound). Generate uint64 on CPU.
    import numpy as np
    rng = np.random.default_rng(seed)
    weights_np = rng.integers(0, 2**64, size=(height * width,), dtype=np.uint64)
    return cp.asarray(weights_np)


def hash_batch(grid: cp.ndarray, weights: cp.ndarray) -> cp.ndarray:
    """Return one uint64 hash per sim. grid: (B,H,W) uint8; weights: (H*W,) uint64."""
    flat = grid.reshape(grid.shape[0], -1).astype(cp.uint64)
    return (flat * weights[None, :]).sum(axis=1, dtype=cp.uint64)
