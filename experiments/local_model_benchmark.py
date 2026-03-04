"""
Local Model Benchmark for ICG-HVRT.

Holds matching fixed (cone geometry, learn_weights=True) and sweeps five
local regression variants across all 7 DGPs from ite_comparison.

Models compared
---------------
ridge   [X|T] -> Ridge(alpha=1.0)        X-adjusted, L2-loss
ols     [X|T] -> OLS                     X-adjusted, L2-loss, no shrinkage
lad     [X|T] -> QuantileRegressor(0.5)  X-adjusted, L1-loss, outlier-robust
mean    T only -> bivariate OLS          no X-adjustment, L2-loss
median  T only -> Theil-Sen              no X-adjustment, outlier-robust

Hypothesis:
  mean/median ~ Ridge when cone matching pre-balances X (Randomised, Geometric).
  Ridge/OLS > mean/median when matching is imperfect (Mean Confounded, TV Confounded).
  lad/median > ridge/ols on Outlier Spike (Y-outlier robustness).

Output: stdout table + experiments/local_model_benchmark.png
"""

import sys
import os
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoite import ICGHVRTEstimator
from experiments.ite_comparison import (
    TESTS,
    N_TRAIN,
    N_TEST,
    N_OBS,  # noqa: F401 — imported for documentation; DGPs use module-level constant
)

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
N_SEEDS  = 5
LOCAL_MODELS = ["ridge", "ols", "lad", "mean", "median"]
GEOMETRY = "cone"  # overridden by --geometry flag

COLORS = {
    "ridge":  "#2ECC71",   # green  (current baseline)
    "ols":    "#3498DB",   # blue
    "lad":    "#E74C3C",   # red
    "mean":   "#F39C12",   # orange
    "median": "#9B59B6",   # purple
}


# ── Metric ─────────────────────────────────────────────────────────────────

def sqrt_pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


# ── Per-model evaluation ───────────────────────────────────────────────────

def _icg_local(local_model: str, X_tr, T_tr, Y_tr, X_te, T_te) -> np.ndarray:
    est = ICGHVRTEstimator(
        k=30,
        learn_weights=True,
        geometry=GEOMETRY,
        local_model=local_model,
    ).fit(X_tr, T_tr, Y_tr)
    return np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])


def evaluate_once(generator, seed: int) -> dict:
    """Run all local_model variants on one train/test split."""
    np.random.seed(seed)
    X_tr, T_tr, Y_tr, _      = generator(N_TRAIN)
    X_te, T_te, Y_te, tau_te = generator(N_TEST)

    results = {}
    for lm in LOCAL_MODELS:
        tau_hat = _icg_local(lm, X_tr, T_tr, Y_tr, X_te, T_te)
        results[lm] = sqrt_pehe(tau_hat, tau_te)
    return results


def evaluate_dgp(name: str, generator) -> dict:
    """Aggregate sqrt-PEHE mean ± std over N_SEEDS for one DGP."""
    seed_results = {lm: [] for lm in LOCAL_MODELS}
    for s in range(N_SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}...", end="", flush=True)
        once = evaluate_once(generator, seed=s)
        for lm in LOCAL_MODELS:
            seed_results[lm].append(once[lm])
        print(" done")

    return {
        lm: (float(np.mean(seed_results[lm])), float(np.std(seed_results[lm])))
        for lm in LOCAL_MODELS
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    all_results = {}   # dgp_name -> {lm -> (mean, std)}

    for name, gen in TESTS:
        print(f"\n[{name}]")
        all_results[name] = evaluate_dgp(name, gen)

    # ── Print table ──────────────────────────────────────────────────────
    col_w = 10
    header = f"{'DGP':<22}" + "".join(f"{lm:>{col_w}}" for lm in LOCAL_MODELS)
    print("\n" + "=" * len(header))
    print("sqrt-PEHE (mean +- std, lower is better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, res in all_results.items():
        best_mean = min(v[0] for v in res.values())
        row = f"{name:<22}"
        for lm in LOCAL_MODELS:
            mean, std = res[lm]
            cell = f"{mean:.3f}+-{std:.3f}"
            marker = "*" if abs(mean - best_mean) < 1e-9 else " "
            row += f"{marker}{cell:>{col_w-1}}"
        print(row)

    print("=" * len(header))
    print("* = best per DGP")

    # ── Bar chart ────────────────────────────────────────────────────────
    dgp_names = list(all_results.keys())
    n_dgp = len(dgp_names)
    n_models = len(LOCAL_MODELS)
    x = np.arange(n_dgp)
    width = 0.15
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, lm in enumerate(LOCAL_MODELS):
        means = [all_results[name][lm][0] for name in dgp_names]
        stds  = [all_results[name][lm][1] / np.sqrt(N_SEEDS) for name in dgp_names]
        ax.bar(
            x + offsets[i], means, width,
            yerr=stds, capsize=3,
            label=lm, color=COLORS[lm], alpha=0.85,
        )

    # Horizontal line at per-DGP minimum
    for j, name in enumerate(dgp_names):
        best = min(all_results[name][lm][0] for lm in LOCAL_MODELS)
        ax.hlines(best, j + offsets[0] - width / 2, j + offsets[-1] + width / 2,
                  colors="black", linewidths=1.0, linestyles="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(dgp_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("sqrt-PEHE")
    geo_label = "ICG-HVRT cone" if GEOMETRY == "cone" else "ICG-HART pyramid"
    ax.set_title(f"Local model benchmark — {geo_label}, learn_weights=True")
    ax.legend(title="local_model", loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    suffix = "cone" if GEOMETRY == "cone" else "pyramid"
    out_path = os.path.join(os.path.dirname(__file__), f"local_model_benchmark_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["cone", "pyramid"], default="cone")
    args = parser.parse_args()
    GEOMETRY = args.geometry
    main()
