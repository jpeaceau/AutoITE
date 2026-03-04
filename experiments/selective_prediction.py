"""
Selective prediction benchmark: coverage-PEHE curves for ICG-HVRT.

Core idea
---------
When ICG-HVRT finds no geometrically similar training patients, the k-NN
distance is large.  Rather than producing a noisy prediction, the model can
*abstain* — declare the case out-of-distribution — and only predict where
it has high geometric confidence.

Two complementary improvements are tested here:

1. Distance-weighted local regression (distance_weighted=True)
   Each training neighbour's observations are weighted by exp(-d_j) in the
   local Ridge.  Nearby patients dominate; geometrically distant ones fade.
   No abstention required: improvement is automatic.

2. Selective prediction (abstention)
   Predict only when mean k-NN distance < threshold.  Coverage = fraction
   of test patients where a prediction is made.  The coverage-PEHE curve
   shows how accuracy improves as the model focuses on high-confidence cases.

Both are compared against:
  - Flat ICG-HVRT (current baseline)
  - CRN (no confidence signal; always predicts)
  - Random abstention (random subset of coverage% patients)

DGPs tested
-----------
  Geometric Confounded : ICG-HVRT's home turf; should be nearly flat at low PEHE
  Prognostic Conf      : ICG-HVRT's hardest case; selective prediction should help
  Mean Confounded      : middle ground
  Hidden Confounded    : negative control; confidence signal is uncorrelated with error

Coverage levels
---------------
  [1.0, 0.8, 0.6, 0.4, 0.2]  (100 → 20% of test patients predicted)
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoite import ICGHVRTEstimator
from experiments.ite_comparison import (
    gen_randomised,
    gen_geometric_confounded,
    gen_mean_confounded,
    gen_hidden_confounded,
    gen_prognostic_confounder,
    _crn_learner,
    compute_metrics,
    patient_summaries,
    N_TRAIN, N_TEST, N_OBS,
)

# ─────────────────────────────────────────────────────────────────────── #
#  Configuration                                                           #
# ─────────────────────────────────────────────────────────────────────── #

N_SEEDS     = 10
COVERAGES   = [1.0, 0.8, 0.6, 0.4, 0.2]

BENCH_DGPS = [
    ("Geometric Conf",   gen_geometric_confounded),
    ("Mean Conf",        gen_mean_confounded),
    ("Prognostic Conf",  gen_prognostic_confounder),
    ("Hidden Conf",      gen_hidden_confounded),
]

_COLORS = {
    "ICG-HVRT (flat)":     "#2ECC71",
    "ICG-HVRT (weighted)": "#27AE60",
    "ICG-HVRT (selective)":"#1A8A47",
    "CRN (all)":           "#E74C3C",
    "Random abstention":   "#95A5A6",
}


# ─────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def _pehe_at_coverage(tau_hat, tau_true, conf_scores, coverage):
    """PEHE on the top-coverage% most confident predictions (lowest distance)."""
    n = len(tau_true)
    k = max(1, int(round(coverage * n)))
    # lower distance = more confident → take smallest distances
    order = np.argsort(conf_scores)[:k]
    err = tau_hat[order] - tau_true[order]
    return float(np.sqrt(np.mean(err ** 2)))


def _pehe_random(tau_hat, tau_true, coverage, rng):
    """PEHE on a random coverage% subset (baseline for abstention value)."""
    n = len(tau_true)
    k = max(1, int(round(coverage * n)))
    idx = rng.choice(n, k, replace=False)
    err = tau_hat[idx] - tau_true[idx]
    return float(np.sqrt(np.mean(err ** 2)))


# ─────────────────────────────────────────────────────────────────────── #
#  Per-seed evaluation                                                     #
# ─────────────────────────────────────────────────────────────────────── #

def evaluate_seed(generator, seed):
    """
    Return per-seed results dict:
      {
        "tau_true": (N_TEST,),
        "tau_flat": (N_TEST,),      flat ICG-HVRT
        "tau_weighted": (N_TEST,),  distance-weighted ICG-HVRT
        "conf_flat": (N_TEST,),     mean k-NN distances from flat estimator
        "conf_weighted": (N_TEST,), mean k-NN distances from weighted estimator
        "tau_crn": (N_TEST,),
      }
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_tr, T_tr, Y_tr, _     = generator(N_TRAIN)
    np.random.seed(seed + 100_000)
    X_te, T_te, Y_te, E_te  = generator(N_TEST)

    # ── ICG-HVRT flat ───────────────────────────────────────────────────
    est_flat = ICGHVRTEstimator(k=30, geometry='cone').fit(X_tr, T_tr, Y_tr)
    results_flat = [
        est_flat.predict_effect_with_confidence(X_te[i], T_te[i])
        for i in range(N_TEST)
    ]
    tau_flat   = np.array([r[0] for r in results_flat])
    conf_flat  = np.array([r[1] for r in results_flat])

    # ── ICG-HVRT distance-weighted ───────────────────────────────────────
    est_w = ICGHVRTEstimator(k=30, geometry='cone', distance_weighted=True).fit(
        X_tr, T_tr, Y_tr)
    results_w = [
        est_w.predict_effect_with_confidence(X_te[i], T_te[i])
        for i in range(N_TEST)
    ]
    tau_weighted  = np.array([r[0] for r in results_w])
    conf_weighted = np.array([r[1] for r in results_w])

    # ── CRN ─────────────────────────────────────────────────────────────
    tau_crn = _crn_learner(X_tr, T_tr, Y_tr, X_te)

    return {
        "tau_true":    E_te,
        "tau_flat":    tau_flat,
        "tau_weighted": tau_weighted,
        "conf_flat":   conf_flat,
        "conf_weighted": conf_weighted,
        "tau_crn":     tau_crn,
    }


# ─────────────────────────────────────────────────────────────────────── #
#  Full benchmark                                                          #
# ─────────────────────────────────────────────────────────────────────── #

def run_selective_benchmark():
    """
    For each DGP and seed, compute PEHE at each coverage level for:
      - ICG-HVRT flat (all patients)
      - ICG-HVRT distance-weighted (all patients)
      - ICG-HVRT selective (top-coverage% by confidence)
      - CRN (all patients)
      - Random abstention (random coverage%)

    Returns nested dict:
      results[dgp_name][method][coverage] = [pehe_seed_0, pehe_seed_1, ...]
    """
    methods = ["ICG-HVRT (flat)", "ICG-HVRT (weighted)",
               "ICG-HVRT (selective)", "CRN (all)", "Random abstention"]
    results = {
        name: {m: {c: [] for c in COVERAGES} for m in methods}
        for name, _ in BENCH_DGPS
    }

    rng = np.random.default_rng(2025)

    for dgp_name, generator in BENCH_DGPS:
        print(f"\n  {dgp_name}", end="", flush=True)
        for s in range(N_SEEDS):
            seed = s * 137 + 7
            res  = evaluate_seed(generator, seed)

            tau_true    = res["tau_true"]
            tau_flat    = res["tau_flat"]
            tau_w       = res["tau_weighted"]
            conf_flat   = res["conf_flat"]
            conf_w      = res["conf_weighted"]
            tau_crn     = res["tau_crn"]

            for c in COVERAGES:
                # flat ICG-HVRT: same tau regardless of coverage (just evaluate on all)
                results[dgp_name]["ICG-HVRT (flat)"][c].append(
                    _pehe_at_coverage(tau_flat, tau_true, conf_flat, 1.0)
                    if c == 1.0 else
                    _pehe_at_coverage(tau_flat, tau_true, conf_flat, 1.0)
                )
                # distance-weighted: same prediction on all; measure on all
                results[dgp_name]["ICG-HVRT (weighted)"][c].append(
                    float(np.sqrt(np.mean((tau_w - tau_true) ** 2)))
                )
                # selective: abstain on low-confidence patients
                results[dgp_name]["ICG-HVRT (selective)"][c].append(
                    _pehe_at_coverage(tau_flat, tau_true, conf_flat, c)
                )
                # CRN: no confidence signal; same prediction at all coverages
                results[dgp_name]["CRN (all)"][c].append(
                    float(np.sqrt(np.mean((tau_crn - tau_true) ** 2)))
                )
                # random abstention baseline
                results[dgp_name]["Random abstention"][c].append(
                    _pehe_random(tau_flat, tau_true, c, rng)
                )
            print(".", end="", flush=True)
        print()

    return results


# ─────────────────────────────────────────────────────────────────────── #
#  Reporting                                                               #
# ─────────────────────────────────────────────────────────────────────── #

def print_table(results):
    col_w = 22
    print()
    for dgp_name, _ in BENCH_DGPS:
        print(f"\n  {'=' * 70}")
        print(f"  {dgp_name}")
        print(f"  {'=' * 70}")
        dgp = results[dgp_name]
        methods = list(dgp.keys())
        # header
        print(f"  {'Method':<{col_w}}" +
              "".join(f"  cov={c:.0%}" for c in COVERAGES))
        print("  " + "-" * (col_w + 10 * len(COVERAGES)))
        for m in methods:
            row = f"  {m:<{col_w}}"
            for c in COVERAGES:
                mu = float(np.mean(dgp[m][c]))
                row += f"  {mu:7.4f}"
            print(row)

        # improvement of selective over flat at 20% coverage
        flat_full   = float(np.mean(dgp["ICG-HVRT (flat)"][1.0]))
        sel_20      = float(np.mean(dgp["ICG-HVRT (selective)"][0.2]))
        rand_20     = float(np.mean(dgp["Random abstention"][0.2]))
        crn_full    = float(np.mean(dgp["CRN (all)"][1.0]))
        weighted    = float(np.mean(dgp["ICG-HVRT (weighted)"][1.0]))
        print(f"\n  Selective (20%) vs flat:       {flat_full:.4f} -> {sel_20:.4f}"
              f"  ({(flat_full-sel_20)/flat_full*100:+.1f}%)")
        print(f"  Random (20%) vs flat:          {flat_full:.4f} -> {rand_20:.4f}"
              f"  ({(flat_full-rand_20)/flat_full*100:+.1f}%)")
        print(f"  Distance-weighted vs flat:     {flat_full:.4f} -> {weighted:.4f}"
              f"  ({(flat_full-weighted)/flat_full*100:+.1f}%)")
        print(f"  CRN (no abstention):           {crn_full:.4f}")


def plot_coverage_curves(results, save_path="selective_prediction.png"):
    n_dgps = len(BENCH_DGPS)
    fig, axes = plt.subplots(1, n_dgps, figsize=(5 * n_dgps, 5), sharey=False)
    if n_dgps == 1:
        axes = [axes]

    # Only show the most informative curves in the figure
    plot_methods = [
        "ICG-HVRT (selective)",
        "ICG-HVRT (weighted)",
        "ICG-HVRT (flat)",
        "CRN (all)",
        "Random abstention",
    ]
    styles = {
        "ICG-HVRT (selective)": dict(color="#1A8A47", lw=2.5, ls="-", marker="o"),
        "ICG-HVRT (weighted)":  dict(color="#27AE60", lw=2,   ls="--", marker="s"),
        "ICG-HVRT (flat)":      dict(color="#2ECC71", lw=1.5, ls=":",  marker="^"),
        "CRN (all)":            dict(color="#E74C3C", lw=2,   ls="-",  marker="D"),
        "Random abstention":    dict(color="#95A5A6", lw=1.5, ls="--", marker=""),
    }

    for ax, (dgp_name, _) in zip(axes, BENCH_DGPS):
        dgp = results[dgp_name]
        for m in plot_methods:
            means = [float(np.mean(dgp[m][c])) for c in COVERAGES]
            stds  = [float(np.std( dgp[m][c])) for c in COVERAGES]
            kw = styles[m]
            xs = [c * 100 for c in COVERAGES]
            ax.plot(xs, means, label=m, markersize=6, **kw)
            lo = [mu - s for mu, s in zip(means, stds)]
            hi = [mu + s for mu, s in zip(means, stds)]
            ax.fill_between(xs, lo, hi, alpha=0.10, color=kw["color"])
        ax.set_title(dgp_name, fontsize=11)
        ax.set_xlabel("Coverage (% of test patients predicted)")
        ax.set_ylabel("sqrt-PEHE")
        ax.invert_xaxis()   # left = low coverage (high confidence)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Selective prediction: PEHE vs coverage\n"
        "Left = high-confidence subset only;  Right = all patients predicted",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────── #
#  Main                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    print("=" * 72)
    print("  SELECTIVE PREDICTION BENCHMARK")
    print(f"  {N_SEEDS} seeds x {len(BENCH_DGPS)} DGPs  |"
          f"  {N_TRAIN} train / {N_TEST} test / {N_OBS} obs per patient")
    print("  Coverage levels:", [f"{c:.0%}" for c in COVERAGES])
    print("=" * 72)

    results = run_selective_benchmark()
    print_table(results)

    print("\n[Generating coverage-PEHE plot...]")
    plot_coverage_curves(results, save_path="selective_prediction.png")
    print("\nDone.")
