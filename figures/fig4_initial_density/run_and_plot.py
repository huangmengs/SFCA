"""Figure 4: outcome composition vs initial density rho_0, for 4 anchor rules.

Each anchor is a (S, B) configuration where one of the four 4-cat outcomes
dominates at rho_0 = 0.25:
    extinction      :  S=[5,10]/18,   B=[8,11]/18      (99% extinction)
    fixed_point     :  S=[3,12]/18,   B=[5, 7]/18      (97% fp, density ~0.33)
    cycle           :  S=[10,63]/180, B=[60,160]/180   (period-2 type)
    long_transient  :  S=[10,69]/180, B=[60,160]/180

For each anchor we sweep rho_0 from 0.05 to 0.95 (step 0.05) and run
N_RUNS_PER_CELL = 100 independent runs per cell.  The plot shows stacked
outcome composition as a function of rho_0.

Grid 75 x 100, max_gen = 2000, full-history cycle detection.
Total sims = 4 anchors x 19 rho_0 x 100 = 7600  (single GPU, ~15s).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
# Anchors and sweep grid
# ============================================================
ANCHORS = [
    {"name": "extinction",
     "s_low":  5/18, "s_high": 10/18, "b_low": 8/18,  "b_high": 11/18},
    {"name": "fixed_point",
     "s_low":  3/18, "s_high": 12/18, "b_low": 5/18,  "b_high":  7/18},
    {"name": "cycle",
     "s_low": 10/180, "s_high": 63/180, "b_low": 60/180, "b_high": 160/180},
    {"name": "long_transient",
     "s_low": 10/180, "s_high": 69/180, "b_low": 60/180, "b_high": 160/180},
]
FIG_ANCHORS = ["extinction", "fixed_point", "cycle", "long_transient"]

DENSITY_VALUES = np.round(np.arange(0.05, 0.96, 0.05), 2)
GRID_HEIGHT = 75
GRID_WIDTH = 100
MAX_GEN = 2_000
N_RUNS_PER_CELL = 100
CYCLE_BUFFER = MAX_GEN + 1

OUTCOME_ORDER = ["extinction", "fixed_point", "cycle", "long_transient"]
OUTCOME_COLORS = {
    "extinction":      "#FA7F6F",
    "cycle":           "#8ECFC9",
    "long_transient":  "#FFBE7A",
    "fixed_point":     "#82B0D2",
}
REF_DENSITY = 0.25


def run_simulations():
    n_anchors = len(ANCHORS)
    n_density = len(DENSITY_VALUES)
    total_sims = n_anchors * n_density * N_RUNS_PER_CELL

    anchor_idx_grid, density_grid, run_grid = np.meshgrid(
        np.arange(n_anchors),
        DENSITY_VALUES,
        np.arange(N_RUNS_PER_CELL),
        indexing="ij",
    )
    anchor_idx = anchor_idx_grid.reshape(-1).astype(np.int32)
    density    = density_grid.reshape(-1).astype(np.float32)
    run_id     = run_grid.reshape(-1).astype(np.int32)

    s_low_arr  = np.array([a["s_low"]  for a in ANCHORS], dtype=np.float32)[anchor_idx]
    s_high_arr = np.array([a["s_high"] for a in ANCHORS], dtype=np.float32)[anchor_idx]
    b_low_arr  = np.array([a["b_low"]  for a in ANCHORS], dtype=np.float32)[anchor_idx]
    b_high_arr = np.array([a["b_high"] for a in ANCHORS], dtype=np.float32)[anchor_idx]
    anchor_names = np.array([a["name"] for a in ANCHORS], dtype=object)[anchor_idx]

    density_units = np.round(density * 100).astype(np.int64)
    seeds = ((anchor_idx.astype(np.int64) + 1) * np.int64(1_000_000_007)
             + density_units * np.int64(1_000_003)
             + run_id.astype(np.int64))

    print(f"Running {total_sims} sims "
          f"({n_anchors} anchors x {n_density} rho_0 x {N_RUNS_PER_CELL} runs)")
    t0 = time.time()
    res = simulate_batch(
        s_low=s_low_arr, s_high=s_high_arr,
        b_low=b_low_arr, b_high=b_high_arr,
        seeds=seeds,
        height=GRID_HEIGHT, width=GRID_WIDTH,
        max_generations=MAX_GEN,
        initial_density=density,
        cycle_buffer=CYCLE_BUFFER,
    )
    print(f"Sim done in {time.time() - t0:.1f}s")

    outcomes = res["outcome"]
    outcomes_4cat = np.where(outcomes == "saturation", "fixed_point", outcomes)
    df = pd.DataFrame({
        "anchor_name":     anchor_names,
        "initial_density": density,
        "outcome_4cat":    outcomes_4cat,
    })
    return df


def composition_per_anchor(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["rho_0"] = df["initial_density"].astype(float).round(2)
    out = {}
    for name, sub in df.groupby("anchor_name"):
        comp = (sub.groupby("rho_0")["outcome_4cat"]
                  .value_counts(normalize=True).unstack(fill_value=0)
                  .reindex(columns=OUTCOME_ORDER, fill_value=0)
                  .mul(100)
                  .sort_index())
        out[name] = comp
    return out


def plot_stacked_bars(comp_by_anchor: dict):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
    handles_for_legend = None
    for i, anchor in enumerate(FIG_ANCHORS):
        ax = axes[i]
        comp = comp_by_anchor[anchor]
        x = comp.index.values
        bottom = np.zeros_like(x, dtype=float)
        bar_handles = []
        for oc in OUTCOME_ORDER:
            bars = ax.bar(x, comp[oc].values, bottom=bottom, width=0.04,
                          label=oc.replace("_", " "),
                          color=OUTCOME_COLORS[oc],
                          edgecolor="white", linewidth=0.3)
            bottom += comp[oc].values
            bar_handles.append(bars)
        ax.axvline(REF_DENSITY, color="black", linestyle=":", linewidth=1)
        ax.set_title(anchor.replace("_", " "))
        ax.set_xlabel("initial density")
        if i == 0:
            ax.set_ylabel("outcome share (%)")
            handles_for_legend = bar_handles
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
    fig.legend(handles=handles_for_legend,
               labels=[oc.replace("_", " ") for oc in OUTCOME_ORDER],
               loc="upper center", ncol=len(OUTCOME_ORDER),
               bbox_to_anchor=(0.5, 1.02), fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = THIS_DIR / "figure.png"
    out_pdf = THIS_DIR / "figure.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}\n       {out_pdf}")


def main():
    df = run_simulations()
    comp = composition_per_anchor(df)
    plot_stacked_bars(comp)


if __name__ == "__main__":
    main()
