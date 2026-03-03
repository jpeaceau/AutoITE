"""
Comprehensive Benchmark: ICG-HVRT vs Competitors
=================================================
Three evaluation panels:

1. PEHE Comparison
   Precision in Estimation of Heterogeneous Effects:
     sqrt-PEHE = sqrt(E[(tau_hat - tau_true)^2])
   Across 6 DGPs x 10 seeds.  Lower is better.

2. Policy Regret Comparison
   n_regret = E[|tau_true_i| * I(sign(tau_hat_i) != sign(tau_true_i))] / E[|tau_true_i|]
   Fraction of total treatable effect mass sent to the wrong treatment arm.
   In [0, 1] and DGP-agnostic; each patient's contribution is directly interpretable.
   Lower is better.

3. Interpretability Comparison
   Beyond point estimates, ICG-HVRT's cooperative cone distance provides
   three interpretability layers unavailable in CRN, RMSN, S/R-Learner:

   (a) Ranking quality  (all methods): Spearman(tau_hat, tau_true)
       "Does the method correctly identify who benefits most?"

   (b) Direction accuracy (all methods): P(sign(tau_hat) == sign(tau_true))
       "Does the method make correct binary treat/control decisions?"

   (c) Confidence calibration (ICG-HVRT only):
       Triage confidence (high/medium/low/uncertain) vs actual |error|.
       Well-calibrated uncertainty improves clinical trust.
       Uncertainty proxy: mean_identity_distance across k-NN set.
       Uncertainty-error Spearman: does identity incompatibility predict error?

   (d) Component attribution (ICG-HVRT only):
       Learned normalised effective weight per distance component per DGP.
       Reveals WHICH geometry dimension drives heterogeneity:
         axis + orient. high  -> covariance shape is the effect modifier
         levels high          -> mean position drives heterogeneity
         occup. / dynamics    -> manifold trajectory matters

DGPs
----
Randomised         T randomised; geometric structure drives effect.
Geometric Conf.    U -> Sigma_i AND E[T]. ICG-HVRT cone immune to this.
Mean Conf.         U -> E[X] AND E[T]. Both d_mu matching and R-Learner capture.
Sparse Mean Conf.  U concentrated in K=5 spikes. Pooled-mean methods weaker.
Hidden Conf.       U invisible in X. Negative control: all methods fail.
TV Conf.           X_t -> T_t within-patient confounding.

Methods
-------
S-Learner    Ridge on [F, T_bar, T_bar x F]; confounding-blind.
R-Learner    Robinson double-debiasing; handles mean-observable confounding.
CRN          GRU + gradient reversal (Bica et al., ICLR 2020).
RMSN         GRU + IPW weights (Lim et al., NeurIPS 2018).
ICG-HVRT     Eight-component cooperative cone matching + local Ridge.
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import DGPs and method implementations from ite_comparison (no duplication)
from ite_comparison import (
    gen_randomised,
    gen_geometric_confounded,
    gen_mean_confounded,
    gen_sparse_mean_confounded,
    gen_hidden_confounded,
    gen_tv_confounded,
    gen_outlier_spike,
    patient_summaries,
    compute_metrics,
    _s_learner,
    _r_learner,
    _crn_learner,
    _rmsn_learner,
    N_TRAIN,
    N_TEST,
    _COLORS,
)
from autoite import ICGHVRTEstimator, ICGHVRTMatcher

warnings.filterwarnings("ignore")
torch.set_num_threads(4)


# ============================================================================ #
#  Configuration                                                               #
# ============================================================================ #

N_SEEDS      = 10
COMP_METHODS = ["S-Learner", "R-Learner", "CRN", "RMSN", "ICG-HVRT", "ICG-HART"]
CONF_ORDER   = ["high", "medium", "low", "uncertain"]
COMP_COLORS  = {m: _COLORS[m] for m in COMP_METHODS}

# 8-component names matching ICGHVRTMatcher attribute order
WEIGHT_LABELS = [
    "axis", "opening", "eccent.", "orient.",
    "levels", "lev-perp", "occup.", "dynamics",
]

DGPS = [
    ("Randomised",        gen_randomised),
    ("Geometric Conf.",   gen_geometric_confounded),
    ("Mean Conf.",        gen_mean_confounded),
    ("Sparse Mean Conf.", gen_sparse_mean_confounded),
    ("Hidden Conf.",      gen_hidden_confounded),
    ("TV Conf.",          gen_tv_confounded),
    ("Outlier Spike",     gen_outlier_spike),
]

_METRIC_KEYS = ["pehe", "mae", "nmae", "bias", "spearman", "sign_acc", "n_regret"]


# ============================================================================ #
#  ICG-HVRT with interpretability output                                       #
# ============================================================================ #

def _icg_hvrt_full(X_tr, T_tr, Y_tr, X_te, T_te, k=30, n_partitions=8):
    """Fit ICG-HVRT and return (tau_hat, fitted_estimator)."""
    est = ICGHVRTEstimator(
        matcher=ICGHVRTMatcher(auto_calibrate=True, gamma_levels_perp=0.25),
        k=k, n_partitions=n_partitions, learn_weights=True,
    )
    est.fit(X_tr, T_tr, Y_tr)
    tau_hat = np.array([
        est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))
    ])
    return tau_hat, est


def _icg_hart_full(X_tr, T_tr, Y_tr, X_te, T_te, k=30, n_partitions=8):
    """Fit ICG-HART (MAD pyramid geometry) and return (tau_hat, fitted_estimator)."""
    est = ICGHVRTEstimator(
        matcher=ICGHVRTMatcher(auto_calibrate=True, gamma_levels_perp=0.25),
        k=k, n_partitions=n_partitions, learn_weights=True,
        geometry='pyramid',
    )
    est.fit(X_tr, T_tr, Y_tr)
    tau_hat = np.array([
        est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))
    ])
    return tau_hat, est


def _get_eff_weights(est):
    """
    Normalised effective weight per distance component.

    effective_weight_k = (beta_k or gamma_k) / calibration_scale_k

    Normalising to sum-to-1 shows the *relative* contribution of each
    geometry dimension to the overall distance -- analogous to feature
    importance for a structured metric.
    """
    m = est.matcher
    raw = np.array([
        m.beta_axis, m.beta_opening, m.beta_eccentricity, m.beta_orientation,
        m.gamma_levels, m.gamma_levels_perp, m.gamma_occupation, m.gamma_dynamics,
    ])
    scales = np.array([
        m._scale_axis, m._scale_opening, m._scale_eccentricity, m._scale_orientation,
        m._scale_levels, m._scale_levels_perp, m._scale_occ, m._scale_dyn,
    ])
    eff = raw / (scales + 1e-12)
    total = eff.sum()
    return eff / (total + 1e-12) if total > 0 else np.zeros_like(eff)


# ============================================================================ #
#  Per-seed evaluation                                                         #
# ============================================================================ #

def evaluate_once_full(generator, seed):
    """
    Single-seed evaluation: all methods + ICG-HVRT interpretability.

    Returns
    -------
    metrics : dict[method] -> dict[metric] -> float
    interp  : {
        eff_weights  : (8,) normalised effective weights (component attribution)
        errors       : (n_te,) absolute errors |tau_hat - tau_true|
        confidences  : list[str]   triage confidence level per test patient
        id_distances : list[float] mean_identity_distance per test patient
        unc_spearman : float  Spearman(mean_identity_distance, |error|)
    }
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_tr, T_tr, Y_tr, _ = generator(N_TRAIN)
    np.random.seed(seed + 100_000)
    X_te, T_te, Y_te, E_te = generator(N_TEST)

    F_tr, Tbar_tr, Ybar_tr = patient_summaries(X_tr, T_tr, Y_tr)
    F_te, _, _              = patient_summaries(X_te, T_te, Y_te)

    metrics = {}
    metrics["S-Learner"] = compute_metrics(_s_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)
    metrics["R-Learner"] = compute_metrics(_r_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)
    metrics["CRN"]       = compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te), E_te)
    metrics["RMSN"]      = compute_metrics(_rmsn_learner(X_tr, T_tr, Y_tr, X_te), E_te)

    tau_hvrt, est       = _icg_hvrt_full(X_tr, T_tr, Y_tr, X_te, T_te)
    metrics["ICG-HVRT"] = compute_metrics(tau_hvrt, E_te)

    tau_hart, _         = _icg_hart_full(X_tr, T_tr, Y_tr, X_te, T_te)
    metrics["ICG-HART"] = compute_metrics(tau_hart, E_te)

    # -- Interpretability data (ICG-HVRT only) --------------------------------
    eff_w    = _get_eff_weights(est)
    errors   = np.abs(tau_hvrt - E_te)
    triage   = est.triage_report(X_te)
    confs    = [r["confidence"]             for r in triage]
    id_dists = [r["mean_identity_distance"] for r in triage]

    rho, _  = spearmanr(id_dists, errors)
    unc_spr = 0.0 if (np.isnan(rho) or np.isinf(rho)) else float(rho)

    return metrics, {
        "eff_weights":  eff_w,
        "errors":       errors,
        "confidences":  confs,
        "id_distances": id_dists,
        "unc_spearman": unc_spr,
    }


# ============================================================================ #
#  Multi-seed DGP evaluation                                                   #
# ============================================================================ #

def evaluate_dgp_full(name, generator):
    """Run N_SEEDS seeds, print per-DGP table, return aggregated results."""
    print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}")

    pool = {m: {k: [] for k in _METRIC_KEYS} for m in COMP_METHODS}
    interp_pool = {
        "eff_weights":  [],
        "unc_spearman": [],
        "conf_errors":  {c: [] for c in CONF_ORDER},
    }

    for s in range(N_SEEDS):
        print(f"  seed {s + 1:2d}/{N_SEEDS}\r", end="", flush=True)
        mets, interp = evaluate_once_full(generator, seed=s * 137 + 7)
        for m, vals in mets.items():
            for k, v in vals.items():
                if k in pool.get(m, {}):
                    pool[m][k].append(v)
        interp_pool["eff_weights"].append(interp["eff_weights"])
        interp_pool["unc_spearman"].append(interp["unc_spearman"])
        for conf, err in zip(interp["confidences"], interp["errors"]):
            interp_pool["conf_errors"][conf].append(float(err))

    print()

    agg = {
        m: {k: (float(np.nanmean(vs)), float(np.nanstd(vs))) for k, vs in vals.items()}
        for m, vals in pool.items()
    }

    # -- Console table --------------------------------------------------------
    print(
        f"\n  {'Method':<18}  {'sqrt-PEHE':>13}  {'Spearman':>9}"
        f"  {'Sign%':>7}  {'N-Regret':>9}"
    )
    print("  " + "-" * 72)
    best_nr = min(agg[m]["n_regret"][0] for m in COMP_METHODS
                  if not np.isnan(agg[m]["n_regret"][0]))
    for m in COMP_METHODS:
        mu_p, sd_p = agg[m]["pehe"]
        nr   = agg[m]["n_regret"][0]
        star = " *" if (not np.isnan(nr) and abs(nr - best_nr) < 1e-9) else "  "
        sa   = agg[m]["sign_acc"][0]
        sa_s = f"{sa:6.2%}" if not np.isnan(sa) else "   nan"
        nr_s = f"{nr:8.4f}" if not np.isnan(nr) else "     nan"
        print(
            f"  {m:<18}  {mu_p:.4f}+/-{sd_p:.4f}"
            f"  {agg[m]['spearman'][0]:>9.4f}"
            f"  {sa_s}  {nr_s}{star}"
        )

    # -- ICG-HVRT interpretability summary ------------------------------------
    mean_ew  = (
        np.mean(interp_pool["eff_weights"], axis=0)
        if interp_pool["eff_weights"] else np.zeros(8)
    )
    mean_usp = float(np.nanmean(interp_pool["unc_spearman"]))

    # Dominant component
    dom_idx = int(np.argmax(mean_ew))
    print(
        f"\n  [ICG-HVRT] Unc-error Spearman={mean_usp:.3f}"
        f"  |  Dominant component: {WEIGHT_LABELS[dom_idx]} ({mean_ew[dom_idx]:.2f})"
    )

    conf_stats = {}
    for c in CONF_ORDER:
        errs = interp_pool["conf_errors"][c]
        conf_stats[c] = (
            (float(np.mean(errs)), float(np.std(errs)), len(errs))
            if errs else (float("nan"), float("nan"), 0)
        )

    return agg, {
        "mean_eff_weights": mean_ew,
        "unc_spearman":     mean_usp,
        "conf_stats":       conf_stats,
    }


# ============================================================================ #
#  Shared grouped-bar helper                                                   #
# ============================================================================ #

def _bar_chart(all_agg, dgp_names, metric, ylabel, title, ax, best_is_low=True):
    n_m     = len(COMP_METHODS)
    x       = np.arange(len(dgp_names))
    w       = 0.80 / n_m
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * w

    for i, m in enumerate(COMP_METHODS):
        means = [all_agg[n][m][metric][0] for n in dgp_names]
        stds  = [all_agg[n][m][metric][1] for n in dgp_names]
        # Replace NaN with 0 so matplotlib doesn't error; gaps are invisible
        m_safe = [v if not np.isnan(v) else 0.0 for v in means]
        s_safe = [v if not np.isnan(v) else 0.0 for v in stds]
        ax.bar(
            x + offsets[i], m_safe, w, yerr=s_safe, capsize=2,
            label=m, color=COMP_COLORS[m], alpha=0.85,
        )

    # Mark best per DGP with a black triangle
    for xi, n in enumerate(dgp_names):
        valid = {
            m: all_agg[n][m][metric][0] for m in COMP_METHODS
            if not np.isnan(all_agg[n][m][metric][0])
        }
        if valid:
            best_m = (min if best_is_low else max)(valid, key=valid.get)
            bi     = COMP_METHODS.index(best_m)
            ax.plot(xi + offsets[bi], valid[best_m], "k^", markersize=5, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([n[:16] for n in dgp_names], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    if metric == "spearman":
        ax.axhline(0, color="black", lw=0.7, ls="--")


# ============================================================================ #
#  Figure 1: PEHE                                                              #
# ============================================================================ #

def plot_pehe_comparison(all_agg, dgp_names, save_path="bench_pehe.png"):
    fig, ax = plt.subplots(figsize=(13, 5))
    _bar_chart(
        all_agg, dgp_names, "pehe",
        "sqrt-PEHE  (lower is better)",
        f"PEHE Comparison: ICG-HVRT vs Competitors  ({N_SEEDS} seeds)\n"
        "sqrt( E[(tau_hat - tau_true)^2] )  --  black triangle = best per DGP",
        ax, best_is_low=True,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================ #
#  Figure 2: Policy Regret                                                     #
# ============================================================================ #

def plot_regret_comparison(all_agg, dgp_names, save_path="bench_regret.png"):
    """
    Normalised policy regret: fraction of total |tau_true| assigned to the
    wrong treatment arm.  n_regret in [0, 1]; 0 = perfect decisions.

    Each patient's contribution is |tau_i| * I(sign(tau_hat) != sign(tau_true)),
    so the aggregate is directly mappable to individual misclassification costs.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    _bar_chart(
        all_agg, dgp_names, "n_regret",
        "Normalised policy regret  (lower is better)",
        f"Policy Regret: ICG-HVRT vs Competitors  ({N_SEEDS} seeds)\n"
        "n_regret = E[|tau_true| * wrong_arm] / E[|tau_true|]  in [0, 1]"
        "  --  fraction of total effect mass misrouted",
        ax, best_is_low=True,
    )
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================ #
#  Figure 3: Interpretability (4-panel)                                        #
# ============================================================================ #

def plot_interpretability(all_agg, all_interp, dgp_names, save_path="bench_interp.png"):
    """
    2x2 figure:
      (a) Spearman rho        -- ranking quality, all methods
      (b) Sign accuracy       -- direction accuracy, all methods
      (c) Confidence calib.   -- ICG-HVRT triage vs actual error
      (d) Component heatmap   -- ICG-HVRT learned attribution per DGP
    """
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.38)

    # -- (a) Spearman rho ---------------------------------------------------- #
    ax_a = fig.add_subplot(gs[0, 0])
    _bar_chart(
        all_agg, dgp_names, "spearman",
        "Spearman rho  (higher is better)",
        "(a) Ranking Quality: Spearman(tau_hat, tau_true)\n"
        "How well does the method rank patients by effect size?",
        ax_a, best_is_low=False,
    )

    # -- (b) Sign accuracy --------------------------------------------------- #
    ax_b = fig.add_subplot(gs[0, 1])
    _bar_chart(
        all_agg, dgp_names, "sign_acc",
        "P(sign correct)  (higher is better)",
        "(b) Direction Accuracy: P(sign(tau_hat) == sign(tau_true))\n"
        "Correct binary treat/control decisions  [|tau_true| > 0.1]",
        ax_b, best_is_low=False,
    )

    # -- (c) ICG-HVRT confidence calibration --------------------------------- #
    ax_c = fig.add_subplot(gs[1, 0])

    # Weighted average across DGPs (weighted by patient count per bucket)
    pooled = {c: [] for c in CONF_ORDER}
    for n in dgp_names:
        for c in CONF_ORDER:
            mu, sd, cnt = all_interp[n]["conf_stats"][c]
            if cnt > 0 and not np.isnan(mu):
                pooled[c].append((mu, cnt))

    bar_heights, bar_ns = [], []
    for c in CONF_ORDER:
        entries = pooled[c]
        if entries:
            total = sum(cnt for _, cnt in entries)
            wmean = sum(mu * cnt for mu, cnt in entries) / total
            bar_heights.append(wmean)
            bar_ns.append(total)
        else:
            bar_heights.append(float("nan"))
            bar_ns.append(0)

    conf_colors = ["#27AE60", "#F1C40F", "#E67E22", "#C0392B"]
    x_conf = np.arange(len(CONF_ORDER))
    for xi, (h, col) in enumerate(zip(bar_heights, conf_colors)):
        if not np.isnan(h):
            ax_c.bar(xi, h, color=col, alpha=0.87, edgecolor="white", linewidth=0.8)
    for xi, (h, n_c) in enumerate(zip(bar_heights, bar_ns)):
        if n_c > 0 and not np.isnan(h):
            ax_c.text(
                xi, h + 0.003, f"n={n_c}",
                ha="center", va="bottom", fontsize=7,
            )

    mean_usp = float(np.nanmean([all_interp[n]["unc_spearman"] for n in dgp_names]))
    ax_c.set_xticks(x_conf)
    ax_c.set_xticklabels(CONF_ORDER, fontsize=9)
    ax_c.set_ylabel("Mean |tau_hat - tau_true|  (lower is better)", fontsize=9)
    ax_c.set_title(
        "(c) ICG-HVRT Confidence Calibration\n"
        f"Pooled across {len(dgp_names)} DGPs x {N_SEEDS} seeds"
        f"  |  Uncertainty-error Spearman rho = {mean_usp:.3f}",
        fontsize=9, fontweight="bold",
    )
    ax_c.grid(axis="y", alpha=0.3)

    # Add monotonicity annotation
    if not any(np.isnan(h) for h in bar_heights):
        is_monotone = all(
            bar_heights[i] <= bar_heights[i + 1] for i in range(len(bar_heights) - 1)
        )
        note = "Monotone: yes" if is_monotone else "Monotone: partial"
        ax_c.text(
            0.97, 0.96, note,
            transform=ax_c.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    # -- (d) Component attribution heatmap ----------------------------------- #
    ax_d = fig.add_subplot(gs[1, 1])
    weight_mat = np.array([all_interp[n]["mean_eff_weights"] for n in dgp_names])
    # (n_dgp, 8) -- normalised effective weights

    vmax = weight_mat.max() if weight_mat.max() > 0 else 1.0
    im   = ax_d.imshow(weight_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

    ax_d.set_xticks(range(len(WEIGHT_LABELS)))
    ax_d.set_xticklabels(WEIGHT_LABELS, rotation=35, ha="right", fontsize=8)
    ax_d.set_yticks(range(len(dgp_names)))
    ax_d.set_yticklabels([n[:16] for n in dgp_names], fontsize=8)

    # Annotate each cell with its value
    for r in range(len(dgp_names)):
        for c in range(len(WEIGHT_LABELS)):
            v     = weight_mat[r, c]
            txt_c = "white" if v > 0.55 * vmax else "black"
            ax_d.text(c, r, f"{v:.2f}", ha="center", va="center",
                      fontsize=7, color=txt_c)

    plt.colorbar(im, ax=ax_d, fraction=0.04, pad=0.04,
                 label="Normalised effective weight")
    ax_d.set_title(
        "(d) ICG-HVRT Component Attribution per DGP\n"
        "Effective weight = (beta/gamma) / calibration_scale  --  sum to 1 per DGP",
        fontsize=9, fontweight="bold",
    )

    # Vertical separator between identity (0-3) and state (4-7) blocks
    ax_d.axvline(3.5, color="white", lw=2, ls="--", alpha=0.7)
    ax_d.text(1.5, -0.65, "Identity components",
              ha="center", va="center", fontsize=7,
              transform=ax_d.transData, color="#555555")
    ax_d.text(5.5, -0.65, "State components",
              ha="center", va="center", fontsize=7,
              transform=ax_d.transData, color="#555555")

    fig.suptitle(
        "Interpretability Comparison: ICG-HVRT vs Black-Box Competitors\n"
        "(a)(b): all 5 methods  |  (c)(d): ICG-HVRT geometry interpretability",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================ #
#  Console summary table                                                       #
# ============================================================================ #

def print_summary_table(all_agg, all_interp, dgp_names):
    print("\n" + "=" * 100)
    print("  COMPREHENSIVE BENCHMARK SUMMARY  (mean over seeds, * = best per column)")
    print("=" * 100)

    col_w = 18
    val_w = 12

    for metric, label, best_low in [
        ("pehe",     "sqrt-PEHE", True),
        ("n_regret", "N-Regret",  True),
        ("spearman", "Spearman",  False),
        ("sign_acc", "Sign %",    False),
    ]:
        print(f"\n  -- {label} --")
        header = f"  {'Method':<{col_w}}"
        for n in dgp_names:
            header += f"{n[:10]:>{val_w}}"
        print(header)
        print("  " + "-" * (col_w + val_w * len(dgp_names)))

        # Best per column
        col_best = {}
        for n in dgp_names:
            valid = {
                m: all_agg[n][m][metric][0] for m in COMP_METHODS
                if not np.isnan(all_agg[n][m][metric][0])
            }
            if valid:
                col_best[n] = (min if best_low else max)(valid, key=valid.get)

        for m in COMP_METHODS:
            row = f"  {m:<{col_w}}"
            for n in dgp_names:
                v    = all_agg[n][m][metric][0]
                mark = "*" if col_best.get(n) == m else " "
                row += f"{v:>{val_w - 1}.4f}{mark}" if not np.isnan(v) else f"{'nan':>{val_w}}"
            print(row)

    # ICG-HVRT interpretability rows
    print(f"\n  -- ICG-HVRT Interpretability --")
    header3 = f"  {'Metric':<22}"
    for n in dgp_names:
        header3 += f"{n[:10]:>{val_w}}"
    print(header3)
    print("  " + "-" * (22 + val_w * len(dgp_names)))

    conf_row_labels = {
        "high":      f"  {'MAE@high-conf.':<22}",
        "low":       f"  {'MAE@low-conf.':<22}",
        "uncertain": f"  {'MAE@uncertain':<22}",
    }
    row_usp = f"  {'Unc-Error Spear.':<22}"
    row_dom = f"  {'Dominant component':<22}"

    for n in dgp_names:
        ia = all_interp[n]
        row_usp += f"{ia['unc_spearman']:>{val_w}.3f}"
        for key in ("high", "low", "uncertain"):
            mu, _, _ = ia["conf_stats"][key]
            conf_row_labels[key] += (
                f"{mu:>{val_w}.4f}" if not np.isnan(mu) else f"{'nan':>{val_w}}"
            )
        dom = int(np.argmax(ia["mean_eff_weights"]))
        row_dom += f"{WEIGHT_LABELS[dom]:>{val_w}}"

    print(row_usp)
    for key in ("high", "low", "uncertain"):
        print(conf_row_labels[key])
    print(row_dom)

    # -- Primary ranking: mean normalised policy regret across DGPs
    #    n_regret in [0,1] for every DGP → mean is meaningful and DGP-agnostic.
    print(f"\n  -- Mean normalised policy regret (mean across {len(dgp_names)} DGPs) --")
    print(f"  n_regret = E[|tau_true| * wrong_arm] / E[|tau_true|]  in [0, 1]")
    print(f"           = fraction of total treatable effect mass sent to wrong arm")
    nr_means = {}
    for m in COMP_METHODS:
        vals = [all_agg[n][m]["n_regret"][0] for n in dgp_names
                if not np.isnan(all_agg[n][m]["n_regret"][0])]
        nr_means[m] = float(np.mean(vals)) if vals else float("nan")
    ranked_nr = sorted((m for m in COMP_METHODS if not np.isnan(nr_means[m])),
                       key=nr_means.get)
    for rank, m in enumerate(ranked_nr, 1):
        marker = " <-- WINNER" if rank == 1 else ""
        print(f"  {rank}. {m:<18} mean n_regret = {nr_means[m]:.4f}{marker}")

    # -- Secondary: total PEHE (retained for completeness)
    print(f"\n  -- Total sqrt-PEHE (sum across {len(dgp_names)} DGPs, secondary) --")
    totals = {}
    for m in COMP_METHODS:
        totals[m] = sum(all_agg[n][m]["pehe"][0] for n in dgp_names)
    for rank, m in enumerate(sorted(totals, key=totals.get), 1):
        marker = " <-- WINNER" if rank == 1 else ""
        print(f"  {rank}. {m:<18} total PEHE = {totals[m]:.4f}{marker}")


# ============================================================================ #
#  Main                                                                        #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 72)
    print("  COMPREHENSIVE BENCHMARK: PEHE + NMAE + INTERPRETABILITY")
    print(f"  {len(DGPS)} DGPs  x  {N_SEEDS} seeds  x  {len(COMP_METHODS)} methods")
    print(f"  Train={N_TRAIN}  Test={N_TEST}")
    print("=" * 72)

    all_agg    = {}
    all_interp = {}
    dgp_names  = [n for n, _ in DGPS]

    for name, generator in DGPS:
        agg, interp_agg = evaluate_dgp_full(name, generator)
        all_agg[name]    = agg
        all_interp[name] = interp_agg

    print_summary_table(all_agg, all_interp, dgp_names)

    print("\n[Generating figures...]")
    plot_pehe_comparison(all_agg, dgp_names, "bench_pehe.png")
    plot_regret_comparison(all_agg, dgp_names, "bench_regret.png")
    plot_interpretability(all_agg, all_interp, dgp_names, "bench_interp.png")

    print("\nDone. Output files:")
    print("  bench_pehe.png   -- PEHE comparison (Figure 1)")
    print("  bench_regret.png -- Policy regret comparison (Figure 2)")
    print("  bench_interp.png -- Interpretability analysis (Figure 3)")
    print("\nKey interpretability outputs (ICG-HVRT only):")
    print("  (c) Confidence calibration: does triage confidence predict actual error?")
    print("  (d) Component attribution: which geometry dimension drives each DGP?")
