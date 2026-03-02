"""
Model Diagnostics: can ICG-HVRT explain its own residuals?

ICG-HVRT produces internal diagnostics alongside every prediction:
  nearest_dist      -- geometric distance to the closest training patient
  gate_pass_rate    -- fraction of k neighbours where direction gate passes
  neighbor_tau_std  -- spread of true tau among matched neighbours*
  local_r2          -- R² of the local Ridge on the neighbour pool

If these correlate with |tau_hat - tau_true|, the model has genuine
self-diagnostic value: it signals uncertainty when it is wrong.

* neighbor_tau_std uses ground-truth tau from the DGP; this is available
  in simulation and approximated in practice by cross-validated residuals.

Runs 5 seeds x 7 DGPs = 350 test patients per analysis.
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autoite import ICGHVRTEstimator, ICGHVRTMatcher
from autoite.profile import CooperativeGeometryProfile

warnings.filterwarnings("ignore")

N_SEEDS = 5
N_TRAIN = 300
N_TEST  = 50
N_OBS   = 100


# ═══════════════════════════════════════════════════════════════════════ #
#  DGP generators (same as ite_evaluation, self-contained)               #
# ═══════════════════════════════════════════════════════════════════════ #

def gen_noise_swamp(n):
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        U = np.random.choice([-1, 1])
        x = np.random.normal(U, 5.0, (N_OBS, 1)); t = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(float(U) * t.flatten() + np.random.normal(0, 5.0, N_OBS)); E.append(float(U))
    return X, T, Y, np.array(E)

def gen_sinusoidal(n):
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        loc = np.random.uniform(-np.pi, np.pi); eff = np.sin(loc) * 2.0
        x = np.random.normal(loc, 0.1, (N_OBS, 1)); t = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(eff * t.flatten() + 0.5 * x.flatten() + np.random.normal(0, 0.5, N_OBS)); E.append(eff)
    return X, T, Y, np.array(E)

def gen_interaction_gate(n):
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        c = np.random.choice(["coupled","decoupled"])
        cov = [[1.,0.9],[0.9,1.]] if c == "coupled" else [[1.,0.],[0.,1.]]
        eff = 2. if c == "coupled" else 0.
        x = np.random.multivariate_normal([0.,0.], cov, N_OBS); t = np.random.normal(0,1,(N_OBS,1))
        X.append(x); T.append(t)
        Y.append(eff*t.flatten() + np.sum(x,1) + np.random.normal(0,0.5,N_OBS)); E.append(eff)
    return X, T, Y, np.array(E)

def gen_direction_gate(n):
    d = 4; e1 = np.eye(d)[0]
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        A = np.random.randn(d,d)*0.5; cov = A@A.T + np.eye(d)
        eig, vec = np.linalg.eigh(cov)
        inv_sqrt = vec @ np.diag(1./np.sqrt(np.maximum(eig,1e-6))) @ vec.T
        w = inv_sqrt @ np.ones(d)
        eff = 3.*float(w@e1)/(np.linalg.norm(w)+1e-12)
        x = np.random.multivariate_normal(np.random.randn(d)*.5, cov, N_OBS)
        t = np.random.normal(0,1,(N_OBS,1))
        X.append(x); T.append(t)
        Y.append(eff*t.flatten() + np.random.normal(0,.5,N_OBS)); E.append(eff)
    return X, T, Y, np.array(E)

def gen_curvature_gate(n):
    d = 4
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        rho = np.random.uniform(0., 0.85)
        cov = (1-rho)*np.eye(d) + rho*np.ones((d,d)) + 1e-4*np.eye(d)
        eff = 2.*float(rho > 0.4)
        x = np.random.multivariate_normal(np.zeros(d), cov, N_OBS); t = np.random.normal(0,1,(N_OBS,1))
        X.append(x); T.append(t)
        Y.append(eff*t.flatten() + np.random.normal(0,.5,N_OBS)); E.append(eff)
    return X, T, Y, np.array(E)

def gen_occupation_gate(n):
    d, rho = 4, 0.7
    Sc = (1-rho)*np.eye(d)+rho*np.ones((d,d))+1e-4*np.eye(d); Sa = np.eye(d)
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        p = np.random.uniform(0.1, 0.9); eff = 4.*p - 2.
        mask = np.random.rand(N_OBS) < p
        x = np.zeros((N_OBS,d)); nc = mask.sum()
        if nc: x[mask] = np.random.multivariate_normal(np.zeros(d),Sc,nc)
        if N_OBS-nc: x[~mask] = np.random.multivariate_normal(np.zeros(d),Sa,N_OBS-nc)
        t = np.random.normal(0,1,(N_OBS,1))
        X.append(x); T.append(t)
        Y.append(eff*t.flatten() + np.random.normal(0,.5,N_OBS)); E.append(eff)
    return X, T, Y, np.array(E)

def gen_dynamics_gate(n):
    d, rho = 4, 0.8
    Sc = (1-rho)*np.eye(d)+rho*np.ones((d,d))+1e-4*np.eye(d); Sa = np.eye(d)
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        p = np.random.uniform(0.3, 0.95); eff = 3.*p - 0.5
        state = np.random.randint(2); x = np.zeros((N_OBS,d))
        for step in range(N_OBS):
            x[step] = np.random.multivariate_normal(np.zeros(d), Sc if state==0 else Sa)
            if np.random.rand() > p: state = 1-state
        t = np.random.normal(0,1,(N_OBS,1))
        X.append(x); T.append(t)
        Y.append(eff*t.flatten() + np.random.normal(0,.5,N_OBS)); E.append(eff)
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
#  Prediction with full diagnostics                                       #
# ═══════════════════════════════════════════════════════════════════════ #

def predict_with_diagnostics(est: ICGHVRTEstimator, X_new, T_new, E_tr) -> dict:
    """
    Run one prediction and return tau_hat plus all internal diagnostics.

    Parameters
    ----------
    est   : fitted ICGHVRTEstimator
    X_new : (N, d) feature observations for the query patient
    T_new : (N,)  treatment observations
    E_tr  : (n_train,) true ITE values for training patients
    """
    qp = CooperativeGeometryProfile.from_longitudinal(
        np.atleast_2d(X_new), shared_hvrt=est._shared_hvrt
    )
    T_2d = np.atleast_2d(T_new).reshape(-1, 1)
    k    = min(est.k, len(est._profiles))
    idx  = est.matcher.find_neighbours(qp, est._profiles, k=k)

    # Geometric distance to nearest neighbour
    nearest_dist = float(est.matcher.distance(qp, est._profiles[idx[0]]))

    # Direction gate pass rate over all k neighbours
    gate_passes = sum(
        1 for j in idx
        if est.matcher.distance_components(qp, est._profiles[j])["direction_gate_passed"]
    )
    gate_pass_rate = gate_passes / k

    # Distance components to nearest neighbour (breakdown)
    dc = est.matcher.distance_components(qp, est._profiles[idx[0]])

    # Mean pairwise distance to neighbours (neighbourhood density)
    mean_dist = float(np.mean([est.matcher.distance(qp, est._profiles[j]) for j in idx]))

    # Spread of true tau in the matched neighbourhood
    neighbor_taus    = np.array([E_tr[j] for j in idx])
    neighbor_tau_std = float(np.std(neighbor_taus))
    neighbor_tau_mean = float(np.mean(neighbor_taus))

    # Local Ridge fit: stack neighbour observations, get R²
    X_loc  = np.vstack([est._obs_list[j] for j in idx])
    T_loc  = np.vstack([est._T_list[j]   for j in idx])
    Y_loc  = np.hstack([est._Y_list[j]   for j in idx])
    XT_loc = np.hstack([X_loc, T_loc])
    ridge  = Ridge(alpha=est.alpha_local).fit(XT_loc, Y_loc)
    tau_hat = float(ridge.coef_[-1])
    Y_pred  = ridge.predict(XT_loc)
    ss_res  = float(np.sum((Y_loc - Y_pred) ** 2))
    ss_tot  = float(np.sum((Y_loc - Y_loc.mean()) ** 2))
    local_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")

    return {
        "tau_hat":          tau_hat,
        "neighbor_tau_mean": neighbor_tau_mean,
        "nearest_dist":     nearest_dist,
        "mean_dist":        mean_dist,
        "gate_pass_rate":   gate_pass_rate,
        "neighbor_tau_std": neighbor_tau_std,
        "local_r2":         local_r2,
        "d_levels":         float(dc["levels"]),
        "d_direction":      float(dc["direction"]),
        "d_shape":          float(dc["shape"]),
        "d_occupation":     float(dc["occupation"]),
        "d_dynamics":       float(dc["dynamics"]),
    }


# ═══════════════════════════════════════════════════════════════════════ #
#  Single-seed collection                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

def collect_once(generator, seed: int):
    np.random.seed(seed)
    X_tr, T_tr, Y_tr, E_tr = generator(N_TRAIN)
    np.random.seed(seed + 100_000)
    X_te, T_te, Y_te, E_te = generator(N_TEST)

    est = ICGHVRTEstimator(
        matcher=ICGHVRTMatcher(auto_calibrate=True), k=30, n_partitions=8
    )
    est.fit(X_tr, T_tr, Y_tr)

    rows = []
    for i in range(N_TEST):
        d = predict_with_diagnostics(est, X_te[i], T_te[i], E_tr)
        d["tau_true"]  = float(E_te[i])
        d["abs_error"] = abs(d["tau_hat"] - d["tau_true"])
        rows.append(d)
    return rows


# ═══════════════════════════════════════════════════════════════════════ #
#  Correlation analysis                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

DIAGNOSTIC_COLS = [
    ("nearest_dist",     "Nearest distance",           "positive"),
    ("mean_dist",        "Mean neighbour distance",    "positive"),
    ("gate_pass_rate",   "Gate pass rate",             "negative"),
    ("neighbor_tau_std", "Neighbour tau std (oracle)", "positive"),
    ("local_r2",         "Local Ridge R²",             "negative"),
]

COMPONENT_COLS = [
    ("d_levels",    "d_mu   (levels)"),
    ("d_direction", "d_w    (direction)"),
    ("d_shape",     "d_sigma (shape)"),
    ("d_occupation","d_occ  (occupation)"),
    ("d_dynamics",  "d_dyn  (dynamics)"),
]


def analyse(rows: list, name: str) -> dict:
    """Compute Spearman correlations of diagnostics with abs_error."""
    abs_err = np.array([r["abs_error"] for r in rows])

    print(f"\n  {name}  (n={len(rows)},  PEHE={np.sqrt(np.mean(abs_err**2)):.4f})")
    print(f"  {'Diagnostic':<32}  {'Spearman rho':>13}  {'p-value':>10}  {'direction':>10}")
    print("  " + "-" * 72)

    results = {}
    for col, label, expected in DIAGNOSTIC_COLS:
        vals = np.array([r[col] for r in rows])
        if np.isnan(vals).any():
            vals = np.where(np.isnan(vals), np.nanmedian(vals), vals)
        rho, p = spearmanr(vals, abs_err)
        match = "ok" if (expected == "positive" and rho > 0) or (expected == "negative" and rho < 0) else "!"
        print(f"  {label:<32}  {rho:>13.4f}  {p:>10.4f}  {match:>10}")
        results[col] = (rho, p)

    print(f"\n  Distance component correlations with abs_error:")
    print(f"  {'Component':<32}  {'Spearman rho':>13}  {'p-value':>10}")
    print("  " + "-" * 58)
    for col, label in COMPONENT_COLS:
        vals = np.array([r[col] for r in rows])
        rho, p = spearmanr(vals, abs_err)
        print(f"  {label:<32}  {rho:>13.4f}  {p:>10.4f}")
        results[col] = (rho, p)

    return results


# ═══════════════════════════════════════════════════════════════════════ #
#  Visualisation                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def plot_diagnostics(all_rows: dict, save_path: str = "model_diagnostics.png"):
    dgp_names = list(all_rows.keys())
    n_diag = len(DIAGNOSTIC_COLS)

    # Panel layout: one row per DGP, columns = diagnostic scatter plots
    # Limit to 4 DGPs to keep the figure readable
    show = ["Noise Swamp", "Interaction Gate", "Occupation Gate", "Dynamics Gate"]
    show = [n for n in show if n in all_rows]

    fig, axes = plt.subplots(
        len(show), n_diag,
        figsize=(n_diag * 3.5, len(show) * 3.2),
        squeeze=False,
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(show)))

    for row_i, name in enumerate(show):
        rows   = all_rows[name]
        abs_err = np.array([r["abs_error"] for r in rows])
        color   = colors[row_i]

        for col_i, (col, label, _) in enumerate(DIAGNOSTIC_COLS):
            ax   = axes[row_i][col_i]
            vals = np.array([r[col] for r in rows])
            if np.isnan(vals).any():
                vals = np.where(np.isnan(vals), np.nanmedian(vals), vals)

            rho, p = spearmanr(vals, abs_err)
            ax.scatter(vals, abs_err, alpha=0.4, s=18, color=color)

            # Trend line
            z = np.polyfit(vals, abs_err, 1)
            xr = np.linspace(vals.min(), vals.max(), 80)
            ax.plot(xr, np.polyval(z, xr), "k-", lw=1.2, alpha=0.7)

            ax.set_xlabel(label, fontsize=8)
            if col_i == 0:
                short = name[:14]
                ax.set_ylabel(f"{short}\n|error|", fontsize=8)
            if row_i == 0:
                ax.set_title(label, fontsize=8, fontweight="bold")
            ax.text(
                0.97, 0.95,
                f"rho={rho:.2f}\np={p:.3f}",
                transform=ax.transAxes,
                fontsize=7, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
            ax.grid(alpha=0.25)

    plt.suptitle(
        "ICG-HVRT residual diagnostics: do internal signals predict error?",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")


def plot_component_heatmap(all_corr: dict, save_path: str = "component_heatmap.png"):
    """
    Heatmap: Spearman rho between each distance component and abs_error,
    across DGPs. Reveals which component drives matching quality where.
    """
    dgp_names  = list(all_corr.keys())
    comp_names = [label for _, label in COMPONENT_COLS]
    comp_keys  = [key for key, _ in COMPONENT_COLS]

    mat = np.array([
        [all_corr[n].get(k, (0,))[0] for k in comp_keys]
        for n in dgp_names
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman rho (positive = component predicts error)")

    ax.set_xticks(range(len(comp_names)))
    ax.set_xticklabels(comp_names, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(dgp_names)))
    ax.set_yticklabels(dgp_names, fontsize=9)

    for i in range(len(dgp_names)):
        for j in range(len(comp_keys)):
            rho = mat[i, j]
            ax.text(j, i, f"{rho:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(rho) > 0.35 else "black")

    ax.set_title(
        "Distance component vs. |error| correlation by DGP\n"
        "(positive = larger component -> larger error -> component detects hard cases)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Main                                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("=" * 72)
    print("  ICG-HVRT MODEL DIAGNOSTICS")
    print(f"  {N_SEEDS} seeds x 7 DGPs x {N_TEST} patients  =  {N_SEEDS*7*N_TEST} predictions")
    print("=" * 72)

    all_rows = {}
    all_corr = {}

    for name, gen in TESTS:
        print(f"\n[{name}]  collecting seeds...", end="", flush=True)
        rows = []
        for s in range(N_SEEDS):
            rows.extend(collect_once(gen, seed=s * 137 + 7))
            print(".", end="", flush=True)
        print()
        all_rows[name] = rows
        all_corr[name] = analyse(rows, name)

    # ── Global summary: which diagnostics are reliably predictive? ────
    print("\n" + "=" * 72)
    print("  GLOBAL SUMMARY: mean |Spearman rho| across all DGPs")
    print("=" * 72)
    print(f"\n  {'Diagnostic':<32}  {'mean |rho|':>11}  {'min rho':>9}  {'max rho':>9}  {'consistent?':>12}")
    print("  " + "-" * 80)
    for col, label, expected in DIAGNOSTIC_COLS:
        rhos = [all_corr[n].get(col, (0,))[0] for n in all_rows]
        mean_abs = float(np.mean(np.abs(rhos)))
        consistent = all(
            (r > 0 and expected == "positive") or (r < 0 and expected == "negative")
            for r in rhos
        )
        sign_str = "YES" if consistent else f"{sum(1 for r in rhos if (r>0)==(expected=='positive'))}/{len(rhos)}"
        print(f"  {label:<32}  {mean_abs:>11.4f}  {min(rhos):>9.4f}  {max(rhos):>9.4f}  {sign_str:>12}")

    print(f"\n  Component correlations with |error|:")
    print(f"  {'Component':<32}  {'mean rho':>10}  {'interpretation':>30}")
    print("  " + "-" * 76)
    for col, label in COMPONENT_COLS:
        rhos = [all_corr[n].get(col, (0,))[0] for n in all_rows]
        mean_rho = float(np.mean(rhos))
        interp = (
            "large component -> hard to match" if mean_rho > 0.1
            else "large component -> easier match" if mean_rho < -0.1
            else "weak signal"
        )
        print(f"  {label:<32}  {mean_rho:>10.4f}  {interp:>30}")

    # ── Neighbourhood quality vs error: is the oracle tau-std useful? ─
    print("\n" + "=" * 72)
    print("  NEIGHBOURHOOD ORACLE: does neighbour tau spread predict error?")
    print("=" * 72)
    print()
    print("  If neighbor_tau_std (true tau spread in matched neighbourhood)")
    print("  strongly predicts |error|, the model can self-diagnose uncertainty")
    print("  in practice by using cross-validated neighbour effect estimates.")
    print()
    for name in all_rows:
        rows = all_rows[name]
        rho = all_corr[name].get("neighbor_tau_std", (0,))[0]
        p   = all_corr[name].get("neighbor_tau_std", (0,0))[1]
        pehe = float(np.sqrt(np.mean([r["abs_error"]**2 for r in rows])))
        print(f"  {name:<22}  rho={rho:+.4f}  p={p:.4f}  (PEHE={pehe:.4f})")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n[Generating plots...]")
    plot_diagnostics(all_rows, save_path="model_diagnostics.png")
    plot_component_heatmap(all_corr, save_path="component_heatmap.png")
    print("\nDone.")
