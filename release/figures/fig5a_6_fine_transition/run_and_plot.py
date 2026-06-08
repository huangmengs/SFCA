"""Figures 5(A), 6(A), 6(B): outcome share + cycle-internal dynamics
vs S_width on the canonical fine-transition axis.

Setup
-----
denom = 180, S_low = 10/180 (= 1/18), B = [60/180, 160/180] (= [6/18, 16/18]).
Scan S_width = 45..75 (units of 1/180), 31 widths, N=100 runs each at
max_gen = 100,000 with full-history cycle detection.

Single H100 wall time: ~20 minutes.  Results are cached to raw_combined.csv;
subsequent runs only re-make the plots.

Figure 1
--------
long_transient share and cycle share vs S_width.
Both dashed; colors match the project heatmap palette
  long_transient = "#FFBE7A", cycle = "#8ECFC9".

Figure 2
--------
Same long_transient share trace (dashed orange).  cycle share trace is
dropped.  Instead, overlay the two cycle-internal fingerprints on the
S_width ranges where cycle dominates (45..55 and 66..75):
  - mean of per-sim mean_density across cycle-class runs (solid)
  - mean of per-sim change_rate_mean across cycle-class runs (solid)
Both with std-across-runs error bars.

Outputs
-------
- figure1_shares.{png,pdf}
- figure2_cycle_dynamics.{png,pdf}
- data.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent))

from sfca.simulate import simulate_batch  # noqa: E402


# ============================================================
# Parameters (mirror exp03_change_rate/round01)
# ============================================================
DENOMINATOR = 180

GRID_WIDTH = 100
GRID_HEIGHT = 75

MAX_GENERATIONS = 100_000
N_RUNS_PER_POSITION = 100
INITIAL_DENSITY = 0.25
SATURATION_THRESHOLD = 0.90
CYCLE_BUFFER = MAX_GENERATIONS + 1

S_LOW_UNIT = 10
B_LOW_UNIT = 60
B_HIGH_UNIT = 160

S_WIDTH_FULL = list(range(45, 76))                   # 45..75, 31 values
S_WIDTH_TO_RUN = S_WIDTH_FULL                        # run everything from scratch
S_WIDTH_DYNAMICS_LO = list(range(45, 56))            # 45..55
S_WIDTH_DYNAMICS_HI = list(range(66, 76))            # 66..75

# Colors (project palette).
COLOR_LT = "#FFBE7A"     # long_transient
COLOR_CY = "#8ECFC9"     # cycle share (fig 1)
COLOR_CR = "#2C5F5D"     # darker teal for cycle change rate (fig 2)
COLOR_DENS = "#5E3C99"   # dark purple for cycle density (fig 2, legacy)
COLOR_STRIPE = "#8B1A1A" # dark red for cycle stripe_score (fig 2)
COLOR_EXT = "#FA7F6F"    # extinction
COLOR_FP  = "#82B0D2"    # fixed_point

RAW_CACHE = THIS_DIR / "raw_combined.csv"

# Match heatmap_outcomes_vs_widths fonts but bigger.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["legend.fontsize"] = 18


# ============================================================
# Simulation for the missing S_widths
# ============================================================
def run_all_widths(widths):
    if not widths:
        return pd.DataFrame()
    if RAW_CACHE.exists():
        print(f"Loading cached sims: {RAW_CACHE}")
        df = pd.read_csv(RAW_CACHE)
        if set(df["s_width_unit"].unique()) >= set(widths):
            return df[df["s_width_unit"].isin(widths)].copy()
        print(f"  cache incomplete (has {sorted(df['s_width_unit'].unique())}); re-running")
    print(f"Running sims for S_width units: {widths}")
    n_widths = len(widths)
    total = n_widths * N_RUNS_PER_POSITION

    s_width = np.repeat(widths, N_RUNS_PER_POSITION).astype(np.int32)
    s_low_u = np.full(total, S_LOW_UNIT, dtype=np.int32)
    s_high_u = s_low_u + s_width
    b_low_u = np.full(total, B_LOW_UNIT, dtype=np.int32)
    b_high_u = np.full(total, B_HIGH_UNIT, dtype=np.int32)
    run_id = np.tile(np.arange(N_RUNS_PER_POSITION), n_widths).astype(np.int32)
    # Use the same seed recipe as round01 so any overlap (none here, but
    # safety) would be reproducible.
    seeds = (s_width.astype(np.uint64) * np.uint64(1_000_003)
             + run_id.astype(np.uint64)).astype(np.int64)

    s_low_f  = (s_low_u  / DENOMINATOR).astype(np.float32)
    s_high_f = (s_high_u / DENOMINATOR).astype(np.float32)
    b_low_f  = (b_low_u  / DENOMINATOR).astype(np.float32)
    b_high_f = (b_high_u / DENOMINATOR).astype(np.float32)

    t0 = time.time()
    res = simulate_batch(
        s_low=s_low_f, s_high=s_high_f, b_low=b_low_f, b_high=b_high_f,
        seeds=seeds, height=GRID_HEIGHT, width=GRID_WIDTH,
        max_generations=MAX_GENERATIONS, initial_density=INITIAL_DENSITY,
        saturation_threshold=SATURATION_THRESHOLD,
        cycle_buffer=CYCLE_BUFFER,
    )
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({total/dt:.1f} sims/s)")

    df = pd.DataFrame({
        "s_width_unit": s_width,
        "s_low_unit": s_low_u, "s_high_unit": s_high_u,
        "b_low_unit": b_low_u, "b_high_unit": b_high_u,
        "denominator": DENOMINATOR,
        "run_id": run_id, "seed": seeds,
        "outcome": res["outcome"],
        "stop_generation": res["stop_generation"],
        "cycle_period": res["cycle_period"],
        "final_density": res["final_density"],
        "mean_density": res["mean_density"],
        "change_rate_mean": res["change_rate_mean"],
        "change_rate_std": res["change_rate_std"],
    })
    df["outcome_4cat"] = df["outcome"].replace({"saturation": "fixed_point"})
    df.to_csv(RAW_CACHE, index=False)
    print(f"  cached raw data to {RAW_CACHE}")
    return df


# ============================================================
# Aggregation
# ============================================================
def aggregate(df):
    """Per S_width: outcome shares + cycle-only mean/std of density and CR."""
    widths = sorted(int(w) for w in df["s_width_unit"].unique())
    out = {
        "s_width": widths,
        "share_long_transient": [],
        "share_cycle": [],
        "share_extinction": [],
        "share_fixed_point": [],
        "cycle_density_mean": [],
        "cycle_density_std": [],
        "cycle_cr_mean": [],
        "cycle_cr_std": [],
        "n_cycle": [],
        "n_total": [],
    }
    for w in widths:
        sub = df[df["s_width_unit"] == w]
        n = len(sub)
        out["n_total"].append(int(n))
        for cat in ["long_transient", "cycle", "extinction", "fixed_point"]:
            out[f"share_{cat}"].append(float((sub["outcome_4cat"] == cat).mean()))
        cyc = sub[sub["outcome_4cat"] == "cycle"]
        out["n_cycle"].append(int(len(cyc)))
        if len(cyc) > 0:
            out["cycle_density_mean"].append(float(cyc["mean_density"].mean()))
            out["cycle_density_std"].append(float(cyc["mean_density"].std(ddof=0)))
            out["cycle_cr_mean"].append(float(cyc["change_rate_mean"].mean()))
            out["cycle_cr_std"].append(float(cyc["change_rate_mean"].std(ddof=0)))
        else:
            for k in ("cycle_density_mean", "cycle_density_std",
                      "cycle_cr_mean", "cycle_cr_std"):
                out[k].append(float("nan"))
    return out


# ============================================================
# Plotting
# ============================================================
def plot_figure1(agg):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.array(agg["s_width"]) / DENOMINATOR
    ax.plot(x, agg["share_long_transient"], "--",
            color=COLOR_LT, linewidth=2.4, marker="o", markersize=7,
            label="long transient share")
    ax.plot(x, agg["share_cycle"], "--",
            color=COLOR_CY, linewidth=2.4, marker="s", markersize=7,
            label="cycle share")
    ax.plot(x, agg["share_extinction"], "--",
            color=COLOR_EXT, linewidth=2.4, marker="^", markersize=7,
            label="extinction share")
    ax.plot(x, agg["share_fixed_point"], "--",
            color=COLOR_FP, linewidth=2.4, marker="v", markersize=7,
            label="fixed point share")
    ax.set_xlabel("S width (units of 1/180)")
    ax.set_ylabel("outcome share")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(min(x) - 0.5/DENOMINATOR, max(x) + 0.5/DENOMINATOR)
    # Major x ticks every 5 units, minor at every unit.
    major = [w / DENOMINATOR for w in agg["s_width"] if w % 5 == 0]
    ax.set_xticks(major)
    ax.set_xticklabels([f"{w}/180" for w in agg["s_width"] if w % 5 == 0])
    ax.set_xticks(x, minor=True)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = THIS_DIR / f"figure1_shares.{ext}"
        fig.savefig(p, dpi=200)
        print(f"  saved {p}")
    plt.close(fig)


def plot_figure2(agg):
    """LT share + cycle density (purple) + cycle change rate (dark teal)."""
    fig, ax = plt.subplots(figsize=(13, 7.5))
    widths = np.array(agg["s_width"])
    x = widths / DENOMINATOR

    ax.plot(x, agg["share_long_transient"], "--",
            color=COLOR_LT, linewidth=2.4, marker="o", markersize=7,
            label="long transient share")

    dyn_mask_lo = np.isin(widths, S_WIDTH_DYNAMICS_LO)
    dyn_mask_hi = np.isin(widths, S_WIDTH_DYNAMICS_HI)

    dens_mean = np.array(agg["cycle_density_mean"])
    dens_std  = np.array(agg["cycle_density_std"])
    cr_mean   = np.array(agg["cycle_cr_mean"])
    cr_std    = np.array(agg["cycle_cr_std"])

    for mask, lbl_dens, lbl_cr in [
        (dyn_mask_lo, "cycle density (mean ± std)", "cycle change rate (mean ± std)"),
        (dyn_mask_hi, None, None),
    ]:
        if not mask.any():
            continue
        ax.errorbar(x[mask], dens_mean[mask], yerr=dens_std[mask],
                    fmt="-", color=COLOR_DENS, linewidth=2.4,
                    marker="s", markersize=7, capsize=4, capthick=1.2,
                    elinewidth=1.2, label=lbl_dens)
        ax.errorbar(x[mask], cr_mean[mask], yerr=cr_std[mask],
                    fmt="-", color=COLOR_CR, linewidth=2.4,
                    marker="D", markersize=6.5, capsize=4, capthick=1.2,
                    elinewidth=1.2, label=lbl_cr)

    ax.set_xlabel("S width (units of 1/180)")
    ax.set_ylabel("share  /  density  /  change rate")
    ax.set_ylim(-0.05, 1.30)
    ax.set_xlim(min(x) - 0.5/DENOMINATOR, max(x) + 0.5/DENOMINATOR)
    major = [w / DENOMINATOR for w in agg["s_width"] if w % 5 == 0]
    ax.set_xticks(major)
    ax.set_xticklabels([f"{w}/180" for w in agg["s_width"] if w % 5 == 0])
    ax.set_xticks(x, minor=True)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.0), handlelength=2.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = THIS_DIR / f"figure2_cycle_dynamics.{ext}"
        fig.savefig(p, dpi=200)
        print(f"  saved {p}")
    plt.close(fig)


def plot_figure3(agg, stripe_agg):
    """LT share (left axis) + cycle stripe_score (right twin axis, dark red)."""
    fig, ax = plt.subplots(figsize=(13, 7.5))
    widths = np.array(agg["s_width"])
    x = widths / DENOMINATOR

    ax.plot(x, agg["share_long_transient"], "--",
            color=COLOR_LT, linewidth=2.4, marker="o", markersize=7,
            label="long transient share")
    ax.set_xlabel("S width (units of 1/180)")
    ax.set_ylabel("long transient share")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(min(x) - 0.5/DENOMINATOR, max(x) + 0.5/DENOMINATOR)
    major = [w / DENOMINATOR for w in agg["s_width"] if w % 5 == 0]
    ax.set_xticks(major)
    ax.set_xticklabels([f"{w}/180" for w in agg["s_width"] if w % 5 == 0])
    ax.set_xticks(x, minor=True)
    ax.grid(True, which="major", alpha=0.25)

    ax2 = ax.twinx()
    sx = np.array(stripe_agg["s_width"]) / DENOMINATOR
    s_mean = np.array(stripe_agg["stripe_mean"])
    s_std  = np.array(stripe_agg["stripe_std"])
    s_widths = np.array(stripe_agg["s_width"])
    mask_lo = np.isin(s_widths, S_WIDTH_DYNAMICS_LO)
    mask_hi = np.isin(s_widths, S_WIDTH_DYNAMICS_HI)
    for mask, label in [
        (mask_lo, "cycle stripe score (mean ± std)"),
        (mask_hi, None),
    ]:
        if not mask.any():
            continue
        ax2.errorbar(sx[mask], s_mean[mask], yerr=s_std[mask],
                     fmt="-", color=COLOR_STRIPE, linewidth=2.4,
                     marker="s", markersize=7, capsize=4, capthick=1.2,
                     elinewidth=1.2, label=label)
    ax2.set_ylabel("stripe score", color=COLOR_STRIPE)
    ax2.tick_params(axis="y", colors=COLOR_STRIPE)
    finite = (s_mean + s_std)[np.isfinite(s_mean + s_std)]
    if finite.size:
        top = float(finite.max())
        ax2.set_ylim(-0.05 * top, 1.15 * top)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2,
              frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = THIS_DIR / f"figure3_stripe_score.{ext}"
        fig.savefig(p, dpi=200)
        print(f"  saved {p}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    print(f"Target S_width range: {S_WIDTH_FULL[0]}..{S_WIDTH_FULL[-1]} "
          f"(units of 1/{DENOMINATOR})")

    df = run_all_widths(S_WIDTH_TO_RUN)
    df = df.sort_values(["s_width_unit", "run_id"]).reset_index(drop=True)
    print(f"\nCollected {len(df)} sims across {df['s_width_unit'].nunique()} widths")

    agg = aggregate(df)

    with open(THIS_DIR / "data.json", "w") as f:
        json.dump({
            "parameters": {
                "denominator": DENOMINATOR,
                "grid": [GRID_HEIGHT, GRID_WIDTH],
                "max_generations": MAX_GENERATIONS,
                "initial_density": INITIAL_DENSITY,
                "saturation_threshold": SATURATION_THRESHOLD,
                "n_runs_per_swidth": N_RUNS_PER_POSITION,
                "S_low_unit": S_LOW_UNIT,
                "B_low_unit": B_LOW_UNIT,
                "B_high_unit": B_HIGH_UNIT,
                "S_width_units": S_WIDTH_FULL,
                "S_width_dynamics_lo": S_WIDTH_DYNAMICS_LO,
                "S_width_dynamics_hi": S_WIDTH_DYNAMICS_HI,
            },
            "aggregated": agg,
        }, f, indent=2)
    print(f"saved data.json")

    print("\nPlotting figure 1 (LT + cycle shares):")
    plot_figure1(agg)
    print("\nPlotting figure 2 (LT share + cycle density/CR):")
    plot_figure2(agg)


if __name__ == "__main__":
    main()
