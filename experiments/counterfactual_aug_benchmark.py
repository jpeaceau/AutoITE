"""
Counterfactual Augmentation Benchmark.

Tests the HVRT-based counterfactual augmentation on all DGPs:
  flat        -- current ICG-HVRT (no augmentation)
  cf_aug      -- ICG-HVRT + HVRT counterfactual augmentation
  cf_aug_dw   -- ICG-HVRT + counterfactual augmentation + distance weighting
  crn         -- CRN (reference)

For each DGP and seed we report sqrt-PEHE, MAE, Spearman rho.
The key question: on which DGPs does filling in the missing counterfactual arm
(where treatment was never given to geometrically similar patients) help?

Expected:
  Geometric Conf   -- confounding is in T shift (±1); aug should fill opposite arm
  Mean Conf        -- T confounded via U; aug fills low-T region for high-U patients
  Prognostic Conf  -- T is randomised; aug adds noise (no missing arm to fill)
  Hidden Conf      -- negative control; no geometry to leverage
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoite import ICGHVRTEstimator
from experiments.ite_comparison import (
    gen_randomised,
    gen_geometric_confounded,
    gen_mean_confounded,
    gen_sparse_mean_confounded,
    gen_prognostic_confounder,
    gen_hidden_confounded,
    gen_tv_confounded,
    gen_outlier_spike,
    _crn_learner,
    compute_metrics,
    N_TRAIN, N_TEST, N_OBS,
)

N_SEEDS = 10

DGPS = [
    ("Randomised",         gen_randomised),
    ("Geometric Conf",     gen_geometric_confounded),
    ("Mean Conf",          gen_mean_confounded),
    ("Sparse Mean Conf",   gen_sparse_mean_confounded),
    ("Prognostic Conf",    gen_prognostic_confounder),
    ("Hidden Conf",        gen_hidden_confounded),
    ("TV Conf",            gen_tv_confounded),
    ("Outlier Spike",      gen_outlier_spike),
]

METHODS = ["flat", "cf_aug", "cf_aug_dw", "crn"]

_COLORS = {
    "flat":      "#95A5A6",
    "cf_aug":    "#2ECC71",
    "cf_aug_dw": "#1A8A47",
    "crn":       "#E74C3C",
}


def evaluate_seed(generator, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_tr, T_tr, Y_tr, _     = generator(N_TRAIN)
    np.random.seed(seed + 100_000)
    X_te, T_te, Y_te, E_te  = generator(N_TEST)

    results = {}

    # flat ICG-HVRT
    est = ICGHVRTEstimator(k=30).fit(X_tr, T_tr, Y_tr)
    tau = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(N_TEST)])
    results["flat"] = compute_metrics(tau, E_te)

    # counterfactual augmentation
    est_cf = ICGHVRTEstimator(k=30, counterfactual_aug=True,
                               n_synth_per_neighbor=50).fit(X_tr, T_tr, Y_tr)
    tau_cf = np.array([est_cf.predict_effect(X_te[i], T_te[i]) for i in range(N_TEST)])
    results["cf_aug"] = compute_metrics(tau_cf, E_te)

    # counterfactual augmentation + distance weighting
    est_cf_dw = ICGHVRTEstimator(k=30, counterfactual_aug=True,
                                  distance_weighted=True,
                                  n_synth_per_neighbor=50).fit(X_tr, T_tr, Y_tr)
    tau_cf_dw = np.array([est_cf_dw.predict_effect(X_te[i], T_te[i])
                           for i in range(N_TEST)])
    results["cf_aug_dw"] = compute_metrics(tau_cf_dw, E_te)

    # CRN reference
    tau_crn = _crn_learner(X_tr, T_tr, Y_tr, X_te)
    results["crn"] = compute_metrics(tau_crn, E_te)

    return results, E_te


def run_benchmark():
    all_results = {name: {m: [] for m in METHODS} for name, _ in DGPS}

    for dgp_name, generator in DGPS:
        print(f"  {dgp_name}", end="", flush=True)
        for s in range(N_SEEDS):
            seed = s * 137 + 7
            res, _ = evaluate_seed(generator, seed)
            for m in METHODS:
                all_results[dgp_name][m].append(res[m]["pehe"])
            print(".", end="", flush=True)
        print()

    return all_results


def print_table(all_results):
    col_w = 16
    print()
    print(f"  {'Method':<{col_w}}" +
          "".join(f"  {n[:13]:>13}" for n, _ in DGPS) + f"  {'MEAN':>8}")
    print("  " + "-" * (col_w + 15 * (len(DGPS) + 1)))
    for m in METHODS:
        vals = [float(np.mean(all_results[name][m])) for name, _ in DGPS]
        row  = f"  {m:<{col_w}}"
        for v in vals:
            row += f"  {v:>13.4f}"
        row += f"  {np.mean(vals):>8.4f}"
        print(row)

    print()
    print("  Delta cf_aug vs flat  (negative = improvement):")
    print(f"  {'Method':<{col_w}}" +
          "".join(f"  {n[:13]:>13}" for n, _ in DGPS))
    print("  " + "-" * (col_w + 15 * len(DGPS)))
    for m in ["cf_aug", "cf_aug_dw"]:
        deltas = [
            float(np.mean(all_results[name][m])) -
            float(np.mean(all_results[name]["flat"]))
            for name, _ in DGPS
        ]
        row = f"  {m:<{col_w}}"
        for d in deltas:
            marker = " *" if d < -0.005 else ("  " if d < 0.005 else " !")
            row += f"  {d:>+12.4f}{marker}"
        print(row)


def plot_results(all_results, save_path="counterfactual_aug.png"):
    dgp_names = [n for n, _ in DGPS]
    x = np.arange(len(dgp_names))
    width = 0.20

    fig, ax = plt.subplots(figsize=(14, 5))
    offsets = {"flat": -1.5, "cf_aug": -0.5, "cf_aug_dw": 0.5, "crn": 1.5}
    for m, off in offsets.items():
        means = [float(np.mean(all_results[n][m])) for n, _ in DGPS]
        stds  = [float(np.std( all_results[n][m])) / np.sqrt(N_SEEDS) for n, _ in DGPS]
        ax.bar(x + off * width, means, width, label=m,
               color=_COLORS[m], alpha=0.85, yerr=stds, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(dgp_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("sqrt-PEHE  (lower is better)")
    ax.set_title(
        "Counterfactual augmentation: HVRT partition sampling fills missing treatment arms\n"
        "cf_aug: synthetic (X, T_broad, Y_per-patient) augments each k-NN pool"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    print("=" * 72)
    print("  COUNTERFACTUAL AUGMENTATION BENCHMARK")
    print(f"  {N_SEEDS} seeds x {len(DGPS)} DGPs  |"
          f"  {N_TRAIN} train / {N_TEST} test / {N_OBS} obs per patient")
    print(f"  n_synth_per_neighbor=50  |  k=30")
    print("=" * 72 + "\n")

    results = run_benchmark()
    print_table(results)

    print("\n[Generating plot...]")
    plot_results(results, save_path="counterfactual_aug.png")
    print("\nDone.")
