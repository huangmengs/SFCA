# Separable Field Cellular Automaton (SFCA)

Code accompanying the manuscript
**"Boundary-Mediated Metastable Dynamics in a Separable Field Cellular Automaton."**
This repository contains the GPU-batched simulator and the scripts that
reproduce every numerical figure in the paper.

> The SFCA replaces the local Moore neighbourhood of a binary cellular
> automaton with a rank-1 *separable field* $n_t(j,i)=R_j(t)\,C_i(t)$ built
> from row and column occupancies.  Cells update against the
> $n_{\max}$-normalised field $q = n / n_{\max}$ through two threshold
> intervals $S = [S_{\text{low}}, S_{\text{high}}]$ (survive) and
> $B = [B_{\text{low}}, B_{\text{high}}]$ (birth).

## Repository layout

```
sfca/                       core simulator (CuPy-batched)
  rule.py                     SFCA update step
  simulate.py                 batched run with outcome detection and
                              optional trajectory recording
figures/                    one folder per paper figure / panel group
  fig2a_snapshots/                 Fig 2(a)  4-outcome snapshot grid
  fig2b_outcome_distribution/      Fig 2(b)  canonical-rule outcome bars
  fig3a_width_phase_diagram/       Fig 3(a)  width-only phase diagram
  fig3b_overlap_geometry/          Fig 3(b)  Δ_low = 5 overlap geometry
  fig4_initial_density/            Fig 4     initial-density robustness
  fig5a_6_fine_transition/         Figs 5(A), 6(A), 6(B)  fine S_w scan
                                            (outcome shares, cycle
                                            density+change-rate, stripe
                                            score)
  fig5b_km_survival/               Fig 5(B)  KM-style survival curves
  fig8a_finite_size_lt_peak/       Fig 8(a)  LT peak vs grid size
  fig8b_normed_std_trajectories/   Fig 8(b)  normed-std trajectories
  si_gap_sweep_1to8/               SI        Δ_low = 1..8 phase-diagram set
docs/
  reproduce.md              expected runtimes / hardware per figure
LICENSE
requirements.txt
```

Each `figures/.../run_and_plot.py` is **self-contained**: it imports
`sfca.simulate.simulate_batch`, runs the simulations the figure needs,
caches the outputs in its own folder, and writes `figure.png` / `figure.pdf`.
No pre-computed data is required.

## Installation

```bash
git clone https://github.com/<your-org>/sfca.git
cd sfca

# CuPy: pick the wheel that matches your CUDA toolkit
pip install cupy-cuda12x   # or cupy-cuda11x

pip install -r requirements.txt
```

CuPy needs a working NVIDIA GPU with CUDA 11.x or 12.x.  All figures were
generated on NVIDIA H100 80 GB GPUs, but a single 24–40 GB consumer GPU is
enough for almost every figure (only `fig5a_6_fine_transition` and
`fig8a_finite_size_lt_peak` use `max_gen = 100,000` with `cycle_buffer`
proportional to it, so they need either an 80 GB GPU or batch-size tuning;
see [docs/reproduce.md](docs/reproduce.md)).

## Reproducing a figure

```bash
cd figures/fig2b_outcome_distribution
python run_and_plot.py
```

Output goes to `figure.png` / `figure.pdf` and a small `*.json` / `*.csv`
cache next to the script.  Reruns are deterministic for a given seed set.

The most time-consuming figures use `max_gen = 100,000` with full-history
cycle detection (`cycle_buffer = 100,001`); see the table below.

| Paper figure | Folder | Sims | Approx. wall time (single H100) |
|---|---|---|---|
| Fig 2(a) | `fig2a_snapshots` | 500 (selection scan) | < 10 s |
| Fig 2(b) | `fig2b_outcome_distribution` | 100,000 | ~3 min |
| Fig 3(a) | `fig3a_width_phase_diagram` | 2,612,000 (4-GPU LPT) | ~20 min on 4 H100; ~80 min single |
| Fig 3(b) | `fig3b_overlap_geometry` | 208,000 | ~6 min |
| Fig 4 | `fig4_initial_density` | 7,600 | ~15 s |
| Fig 5(A) / 6 | `fig5a_6_fine_transition` | 3,100 at `max_gen=100k` | ~20 min |
| Fig 5(B) | `fig5b_km_survival` | 8,000 at `max_gen=100k` | ~40 min |
| Fig 8(a) | `fig8a_finite_size_lt_peak` | 8,400 at `max_gen=100k`, 4 grid sizes | ~50 min |
| Fig 8(b) | `fig8b_normed_std_trajectories` | 1,200 | < 1 min |
| SI gap=1..8 | `si_gap_sweep_1to8` | ~1.6M total over 8 panels | ~40 min |

`docs/reproduce.md` lists hardware requirements, expected memory, and
practical batch-size knobs for each figure.

## The simulator (`sfca/simulate.py`)

```python
from sfca.simulate import simulate_batch

res = simulate_batch(
    s_low, s_high, b_low, b_high,   # (B,) float arrays
    seeds,                          # (B,) int64
    height, width,                  # lattice size
    max_generations,
    initial_density=0.25,           # float OR (B,) float array
    cycle_buffer=2001,              # rolling-history buffer; set to
                                    # max_gen+1 for full-history detection
    record_change_rate_first_n=0,
    record_density_first_n=0,
    snapshot_gens=None,
)

res["outcome"]           # (B,) str: extinction / fixed_point / cycle /
                         # long_transient / saturation
res["stop_generation"]   # (B,) int64
res["cycle_period"]      # (B,) int64; 0 if not a cycle
res["final_density"]     # (B,) float64
res["mean_density"]      # (B,) float64
res["change_rate_mean"]  # (B,) float64
res["change_rate_std"]   # (B,) float64
# optional:
res["change_rate_trajectory"]
res["density_trajectory"]
res["snapshots"]         # dict gen -> (B, H, W) uint8
```

The default saturation threshold is `1.01` so that saturation can never
trigger (the density never exceeds 1).  Lower it (e.g. `0.90`) to make
saturation an early-stopping condition; the simulator then folds saturation
into `fixed_point` for any 4-cat aggregation.

## Citing

If you use this code, please cite the paper (BibTeX placeholder will be
filled once the preprint DOI is assigned).

```bibtex
@misc{TODO_sfca_2026,
  title  = {Boundary-Mediated Metastable Dynamics in a Separable Field
            Cellular Automaton},
  author = {TODO},
  year   = {2026},
  note   = {Preprint},
}
```

## License

MIT.  See `LICENSE`.
