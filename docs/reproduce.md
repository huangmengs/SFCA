# Reproducing the paper figures

Every figure in the paper is reproduced by a single `run_and_plot.py`
script under `figures/<fig_id>/`.  Each script is self-contained: it
imports `sfca.simulate.simulate_batch`, runs the necessary GPU
simulations, caches the per-run outcomes to a CSV next to the script,
and writes `figure.png` / `figure.pdf`.

All wall-time numbers below are measured on one NVIDIA H100 80 GB GPU
with CUDA 12.x and CuPy 12.  Memory peaks listed are for the
single-batch sim path (the chunked figures stay well below 10 GB).

## Quick-running figures (under 5 min on one H100)

| Figure | Folder | Sims | Wall time | Peak GPU mem |
|---|---|---|---|---|
| Fig 2(a) snapshots | `fig2a_snapshots` | 500 (selection scan) | < 10 s | < 2 GB |
| Fig 2(b) outcome bars | `fig2b_outcome_distribution` | 100,000 | ~3 min | ~13 GB |
| Fig 3(b) overlap geometry | `fig3b_overlap_geometry` | 208,000 | ~6 min | ~8 GB |
| Fig 4 initial density | `fig4_initial_density` | 7,600 | ~15 s | < 2 GB |
| Fig 8(b) normed-std traj. | `fig8b_normed_std_trajectories` | 1,200 | < 1 min | < 1 GB |

## Medium figures (20–60 min on one H100)

| Figure | Folder | Sims | Wall time | Peak GPU mem |
|---|---|---|---|---|
| Fig 3(a) width phase diagram | `fig3a_width_phase_diagram` | 2,612,000 | ~80 min single GPU; ~20 min on 4 H100s via multiprocessing | ~10 GB per GPU |
| Fig 5(A)/6 fine transition | `fig5a_6_fine_transition` | 3,100 at `max_gen = 100,000` | ~20 min | ~8 GB |
| Fig 5(B) KM survival | `fig5b_km_survival` | 8,000 at `max_gen = 100,000` | ~40 min | ~25 GB |
| Fig 8(a) finite-size LT peak | `fig8a_finite_size_lt_peak` | 8,400 at `max_gen = 100,000` (4 grids) | ~50 min | varies with grid size, < 30 GB |
| SI gap=1..8 | `si_gap_sweep_1to8` | ~1.9 M total | ~50 min (8 sequential runs) | ~8 GB per gap |

## Memory considerations

The dominant memory cost in long-running scripts is the **cycle-detection
hash buffer** of shape `(cycle_buffer, B)` (uint64), where `cycle_buffer`
is `max_gen + 1` and `B` is the batch size:

| `max_gen` | `cycle_buffer` | `B` | Buffer alone | Practical total at the field-update step |
|---|---|---|---|---|
| 2,000 | 2,001 | 100,000 | 1.6 GB | ~13 GB |
| 2,000 | 2,001 | 60,000 | 1.0 GB | ~8 GB |
| 100,000 | 100,001 | 8,000 | 6.4 GB | ~25 GB |
| 100,000 | 100,001 | 1,000 | 0.8 GB | ~5 GB |

If you hit OOM, the lever to pull is `CHUNK_SIZE` in the script.
Halving `CHUNK_SIZE` halves the peak memory of the corresponding step.

## Skipping the simulation when iterating on the plot

All scripts cache their raw outcomes / trajectories to CSV / NPZ / JSON
next to `run_and_plot.py`.  On a second invocation they detect the cache
and skip straight to the plotting code.  Delete the cache file to force
a re-simulation.

## Multi-GPU sharding

`fig3a_width_phase_diagram` uses `multiprocessing` with one CuPy worker
per visible GPU; with `CUDA_VISIBLE_DEVICES=0,1,2,3` it scales linearly.
The other figures are single-batch and would need to be re-organized into
a worker pool for multi-GPU use.

## Determinism

Every script fixes a deterministic seed recipe of the form
`f(parameter_indices, run_id)`.  Re-running a script with the same code
on the same hardware produces bit-identical outcome labels (up to
CuPy reduction non-determinism, which has not affected any reported
share to within the figure's stated standard error).
