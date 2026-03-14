#!/usr/bin/env python3
"""
sensitivity_analysis.py

Sobol Global Sensitivity Analysis for TensorCyberSimulation.

Decomposes the variance of peak-infected count into contributions from
each input parameter (first-order S1) and their interactions (second-order
S2, total-order ST).  Generates two publication-ready figures:

  1. sobol_indices.png    — Bar chart of S1 and ST per parameter
  2. interaction_heatmap.png — Heatmap of pairwise S2 interactions

Requires: SALib, matplotlib, numpy, joblib
Run:      python3 sensitivity_analysis.py
"""

from __future__ import annotations

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from tensor_engine import run_simulation

# ── Problem definition ────────────────────────────────────────────────────

PROBLEM: dict = {
    "num_vars": 4,
    "names": [
        "spread_chance",
        "patch_completion_prob",
        "rewire_rate",
        "patching_rate",
    ],
    "bounds": [
        [0.05, 0.60],    # transmission probability
        [0.10, 0.90],    # queue drain probability (latency)
        [0.01, 0.15],    # edge rewiring fraction (volatility)
        [0.05, 0.20],    # intervention resource
    ],
}

N_SAMPLES: int = 64
PATCHING_STRATEGY: str = "Targeted"
NUM_TICKS: int = 100
SEED_BASE: int = 0
N_NODES: int = 10_000


# ── Evaluation function ──────────────────────────────────────────────────

def evaluate_sample(params: np.ndarray, idx: int) -> float:
    """
    Run one simulation with the given Sobol sample and return the
    peak-infected fraction (0..1).
    """
    sc, pcp, rr, pr = params
    peak: int = run_simulation(
        patching_strategy=PATCHING_STRATEGY,
        patching_rate=float(pr),
        spread_chance=float(sc),
        num_ticks=NUM_TICKS,
        seed=SEED_BASE + idx,
        patch_completion_prob=float(pcp),
        rewire_rate=float(rr),
    )
    return peak / N_NODES


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Sobol Global Sensitivity Analysis")
    print("=" * 60)

    # Generate Saltelli samples: (2D+2)*N_SAMPLES rows
    param_values: np.ndarray = sobol_sample.sample(PROBLEM, N_SAMPLES, calc_second_order=True)
    total_runs: int = param_values.shape[0]
    print(f"  Parameters:      {PROBLEM['names']}")
    print(f"  Saltelli samples: {total_runs} simulation runs")
    print(f"  Network:         N={N_NODES}, BA(m=3)")
    print(f"  Strategy:        {PATCHING_STRATEGY}")
    print()

    # Execute simulations (sequential — each run already uses all GPU/CPU cores for tensor ops)
    Y: np.ndarray = np.zeros(total_runs)
    t0: float = time.time()
    for i in range(total_runs):
        Y[i] = evaluate_sample(param_values[i], i)
        if (i + 1) % 50 == 0 or i == 0 or i == total_runs - 1:
            elapsed: float = time.time() - t0
            rate: float = (i + 1) / elapsed if elapsed > 0 else 0
            eta: float = (total_runs - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>4d}/{total_runs}]  peak_frac={Y[i]:.4f}  "
                  f"({rate:.1f} runs/s, ETA {eta:.0f}s)")

    elapsed_total: float = time.time() - t0
    print(f"\n  All {total_runs} runs completed in {elapsed_total:.1f}s")

    # Sobol analysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=True)

    names: list[str] = PROBLEM["names"]
    S1: np.ndarray = np.array(Si["S1"])
    ST: np.ndarray = np.array(Si["ST"])
    S2: np.ndarray = np.array(Si["S2"])

    # ── Print research summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Research Executive Summary")
    print("=" * 60)
    print(f"\n  {'Parameter':<25s}  {'S1 (First)':>10s}  {'ST (Total)':>10s}  {'Interaction':>12s}")
    print("  " + "-" * 60)
    for j, name in enumerate(names):
        interaction: float = ST[j] - S1[j]
        print(f"  {name:<25s}  {S1[j]:>10.4f}  {ST[j]:>10.4f}  {interaction:>12.4f}")

    dominant_idx: int = int(np.argmax(ST))
    print(f"\n  Dominant Driver of Outbreak Variance: {names[dominant_idx]}")
    print(f"    Total-order index ST = {ST[dominant_idx]:.4f}")
    print(f"    This parameter explains ~{ST[dominant_idx]*100:.1f}% of output variance")
    print(f"    (including interactions with other parameters)")

    # Check for strong interactions
    max_s2_idx = np.unravel_index(np.argmax(np.abs(S2)), S2.shape)
    if abs(S2[max_s2_idx]) > 0.05:
        print(f"\n  Strongest Interaction: {names[max_s2_idx[0]]} x {names[max_s2_idx[1]]}")
        print(f"    Second-order index S2 = {S2[max_s2_idx]:.4f}")

    print("\n  Output range: peak_frac in [{:.4f}, {:.4f}]".format(Y.min(), Y.max()))
    print("=" * 60)

    # ── Figure 1: Sobol Indices Bar Chart ────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x_pos: np.ndarray = np.arange(len(names))
    width: float = 0.35

    bars_s1 = ax1.bar(x_pos - width / 2, S1, width, label="First-order (S1)")
    bars_st = ax1.bar(x_pos + width / 2, ST, width, label="Total-order (ST)")

    ax1.set_ylabel("Sensitivity Index", fontsize=12)
    ax1.set_title("Sobol Sensitivity Indices — Peak Infected Fraction", fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax1.legend(fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.axhline(y=0, color="black", linewidth=0.5)

    for bars in [bars_s1, bars_st]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=9)

    fig1.tight_layout()
    fig1.savefig("sobol_indices.png", dpi=150)
    print(f"\n  Saved: sobol_indices.png")

    # ── Figure 2: Second-Order Interaction Heatmap ───────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 7))

    # S2 is upper-triangular; mirror it for a full heatmap
    S2_full: np.ndarray = S2.copy()
    S2_full = np.nan_to_num(S2_full, nan=0.0)
    S2_sym: np.ndarray = S2_full + S2_full.T
    np.fill_diagonal(S2_sym, S1)

    im = ax2.imshow(S2_sym, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(names)))
    ax2.set_yticks(range(len(names)))
    ax2.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax2.set_yticklabels(names, fontsize=10)
    ax2.set_title("Parameter Interaction Heatmap (S1 diagonal, S2 off-diagonal)", fontsize=12)

    for i_row in range(len(names)):
        for j_col in range(len(names)):
            val = S2_sym[i_row, j_col]
            ax2.text(j_col, i_row, f"{val:.3f}", ha="center", va="center",
                     fontsize=10, color="white" if abs(val) > 0.3 else "black")

    fig2.colorbar(im, ax=ax2, label="Sensitivity Index")
    fig2.tight_layout()
    fig2.savefig("interaction_heatmap.png", dpi=150)
    print(f"  Saved: interaction_heatmap.png")
    print("\nSobol analysis complete.")


if __name__ == "__main__":
    main()
