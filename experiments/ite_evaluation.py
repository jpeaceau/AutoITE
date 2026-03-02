"""
ITE Evaluation: measuring ICG-HVRT's ability to estimate individual treatment effects.

Protocol
--------
  Seeds         : 20 independent random seeds
  Train/Test    : 300 / 50 patients x 100 observations each
  DGPs          : 7 stress tests (Noise Swamp through Dynamics Gate)

Metrics  (vs ground-truth tau from DGP)
----------------------------------------
  sqrt-PEHE   sqrt(E[(tau_hat - tau)^2])  -- primary, equivalent to RMSE
  MAE         mean |tau_hat - tau|
  Bias        mean (tau_hat - tau)        -- signed; near-zero is unbiased
  Spearman    rank correlation rho(tau_hat, tau_true)
  Sign Acc.   P(sign(tau_hat) == sign(tau_true))

Methods
-------
  Global Mean      predict population-mean(tau_train) for every test patient
  Global Linear    Ridge on patient-level (mean, var) feature summary
  Random Forest    RF on patient-level feature summary
  Within-Patient   Ridge fit on the test patient's own 100 observations only
  ICG-HVRT         five-component cooperative geometry JIT estimator
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autoite import ICGHVRTEstimator, ICGHVRTMatcher

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════ #
#  Configuration                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

N_SEEDS = 20
N_TRAIN = 300
N_TEST  = 50
N_OBS   = 100   # observations per patient (same as unified_benchmark)

METHODS = ["Global Mean", "Global Linear", "Random Forest",
           "Within-Patient", "ICG-HVRT"]
METRICS = ["pehe", "mae", "bias", "spearman", "sign_acc"]


# ═══════════════════════════════════════════════════════════════════════ #
#  DGP generators  (mirrors unified_benchmark, self-contained)            #
# ═══════════════════════════════════════════════════════════════════════ #

def gen_noise_swamp(n_units):
    """tau in {-1, +1} — signal in mean, buried in large variance."""
    X, T, Y, E = [], [], [], []
    for _ in range(n_units):
        U   = np.random.choice([-1, 1])
        eff = float(U)
        x   = np.random.normal(U, 5.0, (N_OBS, 1))
        t   = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(eff * t.flatten() + np.random.normal(0, 5.0, N_OBS))
        E.append(eff)
    return X, T, Y, np.array(E)


def gen_sinusoidal(n_units):
    """tau = 2 sin(x_loc) — continuous non-linear heterogeneity."""
    X, T, Y, E = [], [], [], []
    for _ in range(n_units):
        x_loc = np.random.uniform(-np.pi, np.pi)
        eff   = np.sin(x_loc) * 2.0
        x     = np.random.normal(x_loc, 0.1, (N_OBS, 1))
        t     = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(eff * t.flatten() + 0.5 * x.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(eff)
    return X, T, Y, np.array(E)


def gen_interaction_gate(n_units):
    """tau in {0, 2} — signal in covariance structure."""
    X, T, Y, E = [], [], [], []
    for _ in range(n_units):
        regime = np.random.choice(["coupled", "decoupled"])
        cov    = [[1.0, 0.9], [0.9, 1.0]] if regime == "coupled" else [[1.0, 0.0], [0.0, 1.0]]
        eff    = 2.0 if regime == "coupled" else 0.0
        x      = np.random.multivariate_normal([0.0, 0.0], cov, N_OBS)
        t      = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(eff * t.flatten() + np.sum(x, axis=1) + np.random.normal(0, 0.5, N_OBS))
        E.append(eff)
    return X, T, Y, np.array(E)


def gen_direction_gate(n_units):
    """tau = 3 cos(angle(w_i, e_1)) — signal in cooperative direction."""
    d  = 4
    e1 = np.eye(d)[0]
    X, T, Y, E = [], [], [], []
    for _ in range(n_units):
        A   = np.random.randn(d, d) * 0.5
        cov = A @ A.T + np.eye(d)
        eig, vec = np.linalg.eigh(cov)
        inv_sqrt = vec @ np.diag(1.0 / np.sqrt(np.maximum(eig, 1e-6))) @ vec.T
        w   = inv_sqrt @ np.ones(d)
        eff = 3.0 * float(w @ e1) / (np.linalg.norm(w) + 1e-12)
        mu  = np.random.randn(d) * 0.5
        x   = np.random.multivariate_normal(mu, cov, N_OBS)
        t   = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(eff * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(eff)
    return X, T, Y, np.array(E)


def gen_curvature_gate(n_units):
    """tau in {0, 2} — signal in manifold coupling (rho > 0.4)."""
    d = 4
    X, T, Y, E = [], [], [], []
    for _ in range(n_units):
        rho = np.random.uniform(0.0, 0.85)
        cov = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        eff = 2.0 * float(rho > 0.4)
        x   = np.random.multivariate_normal(np.zeros(d), cov, N_OBS)
        t   = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(eff * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(eff)
    return X, T, Y, np.array(E)


def gen_occupation_gate(n_units):
    """tau = 4 * p_coop - 2 — signal in HVRT occupation fraction."""
    d, rho = 4, 0.7
    Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
    Sa = np.eye(d)
    X, T, Y, E = [], [], [], []
    for _ in range(n_units):
        p   = np.random.uniform(0.1, 0.9)
        eff = 4.0 * p - 2.0
        mask = np.random.rand(N_OBS) < p
        x    = np.zeros((N_OBS, d))
        nc   = mask.sum()
        if nc:           x[mask]  = np.random.multivariate_normal(np.zeros(d), Sc, nc)
        if N_OBS - nc:   x[~mask] = np.random.multivariate_normal(np.zeros(d), Sa, N_OBS - nc)
        t    = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(eff * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(eff)
    return X, T, Y, np.array(E)


def gen_dynamics_gate(n_units):
    """tau = 3 * persistence - 0.5 — signal in regime-transition dynamics."""
    d, rho = 4, 0.8
    Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
    Sa = np.eye(d)
    X, T, Y, E = [], [], [], []
    for _ in range(n_units):
        p   = np.random.uniform(0.3, 0.95)
        eff = 3.0 * p - 0.5
        state = np.random.randint(2)
        x     = np.zeros((N_OBS, d))
        for step in range(N_OBS):
            x[step] = np.random.multivariate_normal(
                np.zeros(d), Sc if state == 0 else Sa
            )
            if np.random.rand() > p:
                state = 1 - state
        t = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(eff * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(eff)
    return X, T, Y, np.array(E)


TESTS = [
    ("Noise Swamp",      gen_noise_swamp),
    ("Sinusoidal Trap",  gen_sinusoidal),
    ("Interaction Gate", gen_interaction_gate),
    ("Direction Gate",   gen_direction_gate),
    ("Curvature Gate",   gen_curvature_gate),
    ("Occupation Gate",  gen_occupation_gate),
    ("Dynamics Gate",    gen_dynamics_gate),
]


# ═══════════════════════════════════════════════════════════════════════ #
#  Metrics                                                                #
# ═══════════════════════════════════════════════════════════════════════ #

def compute_metrics(tau_hat: np.ndarray, tau_true: np.ndarray) -> dict:
    """Compute all ITE quality metrics."""
    err  = tau_hat - tau_true
    pehe = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    rho, _ = spearmanr(tau_hat, tau_true)
    # Sign accuracy: exclude near-zero true effects to avoid 0/0
    nz   = np.abs(tau_true) > 1e-6
    sacc = float(np.mean(np.sign(tau_hat[nz]) == np.sign(tau_true[nz]))) if nz.any() else float("nan")
    return {
        "pehe": pehe,
        "mae":  mae,
        "bias": bias,
        "spearman": 0.0 if np.isnan(rho) else float(rho),
        "sign_acc": sacc,
    }


# ═══════════════════════════════════════════════════════════════════════ #
#  Feature extractor for summary-statistic baselines                      #
# ═══════════════════════════════════════════════════════════════════════ #

def patient_features(X_list):
    """Concatenate per-feature mean and variance — (n_patients, 2d)."""
    return np.array([
        np.concatenate([np.mean(x, axis=0), np.var(x, axis=0)])
        for x in X_list
    ])


# ═══════════════════════════════════════════════════════════════════════ #
#  Single-seed evaluation                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

def evaluate_once(generator, seed: int) -> dict:
    """Run one train/test split and return metrics for every method."""
    np.random.seed(seed)
    X_tr, T_tr, Y_tr, E_tr = generator(N_TRAIN)
    np.random.seed(seed + 100_000)
    X_te, T_te, Y_te, E_te = generator(N_TEST)

    F_tr = patient_features(X_tr)
    F_te = patient_features(X_te)

    out = {}

    # ── Global Mean ──────────────────────────────────────────────────
    mu_tau = float(np.mean(E_tr))
    out["Global Mean"] = compute_metrics(np.full(N_TEST, mu_tau), E_te)

    # ── Global Linear (Ridge on feature summary) ─────────────────────
    pred_gl = Ridge(alpha=1.0).fit(F_tr, E_tr).predict(F_te)
    out["Global Linear"] = compute_metrics(pred_gl, E_te)

    # ── Random Forest ─────────────────────────────────────────────────
    pred_rf = RandomForestRegressor(100, random_state=42).fit(F_tr, E_tr).predict(F_te)
    out["Random Forest"] = compute_metrics(pred_rf, E_te)

    # ── Within-Patient (Ridge on own observations, uses Y_te) ─────────
    pred_wp = []
    for i in range(N_TEST):
        Xi  = np.atleast_2d(X_te[i])
        Ti  = np.atleast_2d(T_te[i]).reshape(-1, 1)
        Yi  = np.asarray(Y_te[i]).ravel()
        m   = Ridge(alpha=1.0).fit(np.hstack([Xi, Ti]), Yi)
        pred_wp.append(float(m.coef_[-1]))
    out["Within-Patient"] = compute_metrics(np.array(pred_wp), E_te)

    # ── ICG-HVRT ──────────────────────────────────────────────────────
    est = ICGHVRTEstimator(
        matcher=ICGHVRTMatcher(auto_calibrate=True),
        k=30, n_partitions=8,
    )
    est.fit(X_tr, T_tr, Y_tr)
    pred_hvrt = np.array([
        est.predict_effect(X_te[i], T_te[i]) for i in range(N_TEST)
    ])
    out["ICG-HVRT"] = compute_metrics(pred_hvrt, E_te)

    return out


# ═══════════════════════════════════════════════════════════════════════ #
#  Multi-seed evaluation for one DGP                                      #
# ═══════════════════════════════════════════════════════════════════════ #

def evaluate_dgp(name: str, generator) -> tuple:
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")

    pool = {m: {k: [] for k in METRICS} for m in METHODS}

    for s in range(N_SEEDS):
        print(f"  seed {s+1:2d}/{N_SEEDS}\r", end="", flush=True)
        res = evaluate_once(generator, seed=s * 137 + 7)
        for m, mets in res.items():
            for k, v in mets.items():
                pool[m][k].append(v)

    print()

    # Aggregate to mean ± std
    agg = {
        m: {k: (float(np.mean(vs)), float(np.std(vs))) for k, vs in mets.items()}
        for m, mets in pool.items()
    }

    # Print per-DGP table
    print(f"\n  {'Method':<22}  {'sqrt-PEHE':>14}  {'MAE':>9}  {'Bias':>8}  {'Spearman':>9}  {'Sign%':>7}")
    print("  " + "-" * 77)
    best_pehe = min(agg[m]["pehe"][0] for m in METHODS)
    for m in METHODS:
        p_mu, p_sd = agg[m]["pehe"]
        star = " *" if abs(p_mu - best_pehe) < 1e-9 else "  "
        print(
            f"  {m:<22}"
            f"  {p_mu:6.4f}+/-{p_sd:.4f}"
            f"  {agg[m]['mae'][0]:>9.4f}"
            f"  {agg[m]['bias'][0]:>8.4f}"
            f"  {agg[m]['spearman'][0]:>9.4f}"
            f"  {agg[m]['sign_acc'][0]:>7.2%}"
            f"{star}"
        )

    # Wilcoxon signed-rank test: H1 = baseline PEHE > ICG-HVRT PEHE
    print(f"\n  Wilcoxon (H1: baseline PEHE > ICG-HVRT, one-sided):")
    for m in METHODS:
        if m == "ICG-HVRT":
            continue
        try:
            _, p = wilcoxon(
                pool[m]["pehe"], pool["ICG-HVRT"]["pehe"], alternative="greater"
            )
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        except Exception:
            p, sig = float("nan"), "err"
        print(f"    vs {m:<22}  p={p:.3f}  {sig}")

    return agg, pool


# ═══════════════════════════════════════════════════════════════════════ #
#  Visualisation                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def plot_summary(all_agg: dict, save_path: str = "ite_evaluation.png"):
    dgp_names = [n for n, _ in TESTS]
    n_dgp     = len(dgp_names)
    x         = np.arange(n_dgp)
    w         = 0.14
    offsets   = np.linspace(-2, 2, len(METHODS)) * w
    short     = [n[:11] for n in dgp_names]

    colors = {
        "Global Mean":    "#BDC3C7",
        "Global Linear":  "#7F8C8D",
        "Random Forest":  "#E74C3C",
        "Within-Patient": "#F39C12",
        "ICG-HVRT":       "#2ECC71",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: sqrt-PEHE
    ax = axes[0]
    for i, m in enumerate(METHODS):
        means = [all_agg[n][m]["pehe"][0] for n in dgp_names]
        stds  = [all_agg[n][m]["pehe"][1] for n in dgp_names]
        ax.bar(x + offsets[i], means, w, yerr=stds, capsize=3,
               label=m, color=colors[m], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("sqrt-PEHE (lower is better)")
    ax.set_title("sqrt-PEHE by DGP")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Panel 2: Spearman rho
    ax = axes[1]
    for i, m in enumerate(METHODS):
        means = [all_agg[n][m]["spearman"][0] for n in dgp_names]
        stds  = [all_agg[n][m]["spearman"][1] for n in dgp_names]
        ax.bar(x + offsets[i], means, w, yerr=stds, capsize=3,
               label=m, color=colors[m], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Spearman rho (higher is better)")
    ax.set_title("Rank Correlation with True tau")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Panel 3: Sign accuracy
    ax = axes[2]
    for i, m in enumerate(METHODS):
        means = [all_agg[n][m]["sign_acc"][0] for n in dgp_names]
        stds  = [all_agg[n][m]["sign_acc"][1] for n in dgp_names]
        ax.bar(x + offsets[i], means, w, yerr=stds, capsize=3,
               label=m, color=colors[m], alpha=0.85)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", label="random")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Sign Accuracy (higher is better)")
    ax.set_title("Treatment Direction Accuracy")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.suptitle(
        f"ICG-HVRT ITE Evaluation  ({N_SEEDS} seeds x 7 DGPs,"
        f" {N_TRAIN} train / {N_TEST} test / {N_OBS} obs)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Main                                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("=" * 72)
    print("  ICG-HVRT  ITE  EVALUATION")
    print(f"  Protocol: {N_SEEDS} seeds  |  {N_TRAIN} train / {N_TEST} test / {N_OBS} obs")
    print("=" * 72)

    all_agg  = {}
    all_pool = {}

    for name, gen in TESTS:
        agg, pool   = evaluate_dgp(name, gen)
        all_agg[name]  = agg
        all_pool[name] = pool

    dgp_names = [n for n, _ in TESTS]

    # ── Summary table: sqrt-PEHE (mean) ──────────────────────────────
    col_w = 22
    val_w = 12
    print("\n" + "=" * 72)
    print("  SUMMARY  --  sqrt-PEHE mean across seeds  (lower is better)")
    print("=" * 72)
    hdr = f"\n{'Method':<{col_w}}"
    for n in dgp_names:
        hdr += f"{n[:9]:>{val_w}}"
    hdr += f"{'MEAN':>{val_w}}"
    print(hdr)
    print("-" * (col_w + val_w * (len(dgp_names) + 1)))
    for m in METHODS:
        row  = f"{m:<{col_w}}"
        vals = [all_agg[n][m]["pehe"][0] for n in dgp_names]
        for v in vals:
            row += f"{v:>{val_w}.4f}"
        row += f"{np.mean(vals):>{val_w}.4f}"
        print(row)

    # ── Summary table: Spearman rho (mean) ───────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY  --  Spearman rho mean across seeds  (higher is better)")
    print("=" * 72)
    hdr = f"\n{'Method':<{col_w}}"
    for n in dgp_names:
        hdr += f"{n[:9]:>{val_w}}"
    hdr += f"{'MEAN':>{val_w}}"
    print(hdr)
    print("-" * (col_w + val_w * (len(dgp_names) + 1)))
    for m in METHODS:
        row  = f"{m:<{col_w}}"
        vals = [all_agg[n][m]["spearman"][0] for n in dgp_names]
        for v in vals:
            row += f"{v:>{val_w}.4f}"
        row += f"{np.mean(vals):>{val_w}.4f}"
        print(row)

    # ── Per-DGP winners ───────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  BEST METHOD PER DGP  (sqrt-PEHE):")
    for n in dgp_names:
        best = min(METHODS, key=lambda m: all_agg[n][m]["pehe"][0])
        bval = all_agg[n][best]["pehe"][0]
        rank = sorted(METHODS, key=lambda m: all_agg[n][m]["pehe"][0]).index("ICG-HVRT") + 1
        print(f"  {n:<22}  {best:<22}  ({bval:.4f})   ICG-HVRT rank {rank}/{len(METHODS)}")

    # ── ICG-HVRT summary statistics ───────────────────────────────────
    hvrt_pehes = [all_agg[n]["ICG-HVRT"]["pehe"][0] for n in dgp_names]
    hvrt_rhos  = [all_agg[n]["ICG-HVRT"]["spearman"][0] for n in dgp_names]
    hvrt_signs = [all_agg[n]["ICG-HVRT"]["sign_acc"][0] for n in dgp_names]
    print(f"\n  ICG-HVRT overall: "
          f"mean sqrt-PEHE={np.mean(hvrt_pehes):.4f}  "
          f"mean Spearman={np.mean(hvrt_rhos):.4f}  "
          f"mean Sign Acc={np.mean(hvrt_signs):.2%}")

    # ── Visualisation ─────────────────────────────────────────────────
    print("\n[Generating plot...]")
    plot_summary(all_agg, save_path="ite_evaluation.png")
    print("\nDone.")
