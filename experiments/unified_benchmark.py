"""
ICG-HVRT Unified Benchmark — Seven Stress Tests
================================================
Compares three estimators across seven data-generating processes designed to
expose distinct failure modes:

Tests 1–3  (original ICG stress suite)
  1. Noise Swamp       — signal in mean, buried in variance
  2. Sinusoidal Trap   — nonlinear signal in mean position
  3. Interaction Gate  — signal in covariance structure (T-axis)

Tests 4–7  (new ICG-HVRT stress suite)
  4. Direction Gate    — effect depends on cooperative direction alignment
  5. Curvature Gate    — effect depends on manifold coupling strength
  6. Occupation Gate   — effect depends on manifold occupation pattern
  7. Dynamics Gate     — effect depends on cooperative stability

Expected performance:
  | Test      | Random Forest | ICG (Legacy) | ICG-HVRT |
  |-----------|--------------|--------------|----------|
  | 1–2       | Good         | Good         | Good     |
  | 3         | Fails        | Good         | Good     |
  | 4–5       | Fails        | Partial      | Good     |
  | 6–7       | Fails        | Fails        | Good     |
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoite import ICGHVRTEstimator, ICGHVRTMatcher
from autoite.jit import IntrinsicJIT

warnings.filterwarnings("ignore")
np.random.seed(2025)


# ═══════════════════════════════════════════════════════════════════════ #
#  Data generators                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def gen_noise_swamp(n_units: int):
    """Test 1: Signal in Mean, buried in Variance."""
    X, T, Y, Eff = [], [], [], []
    for _ in range(n_units):
        N = 100
        U = np.random.choice([-1, 1])
        X.append(np.random.normal(U, 5.0, (N, 1)))
        T.append(np.random.normal(0, 1, (N, 1)))
        eff = float(U)
        Y.append(eff * T[-1].flatten() + np.random.normal(0, 5.0, N))
        Eff.append(eff)
    return X, T, Y, np.array(Eff)


def gen_sinusoidal(n_units: int):
    """Test 2: Signal in Position (Mean), Non-Linear."""
    X, T, Y, Eff = [], [], [], []
    for _ in range(n_units):
        N = 100
        x_loc = np.random.uniform(-np.pi, np.pi)
        X.append(np.random.normal(x_loc, 0.1, (N, 1)))
        T.append(np.random.normal(0, 1, (N, 1)))
        eff = np.sin(x_loc) * 2.0
        Y.append(eff * T[-1].flatten() + 0.5 * X[-1].flatten() + np.random.normal(0, 0.5, N))
        Eff.append(eff)
    return X, T, Y, np.array(Eff)


def gen_interaction_gate(n_units: int):
    """Test 3: Signal in Covariance (Structure / T-axis)."""
    X, T, Y, Eff = [], [], [], []
    for _ in range(n_units):
        N = 100
        regime = np.random.choice(["coupled", "decoupled"])
        if regime == "coupled":
            cov, eff = [[1.0, 0.9], [0.9, 1.0]], 2.0
        else:
            cov, eff = [[1.0, 0.0], [0.0, 1.0]], 0.0
        X_raw = np.random.multivariate_normal([0, 0], cov, N)
        X.append(X_raw)
        T.append(np.random.normal(0, 1, (N, 1)))
        Y.append(eff * T[-1].flatten() + np.sum(X_raw, axis=1) + np.random.normal(0, 0.5, N))
        Eff.append(eff)
    return X, T, Y, np.array(Eff)


def gen_direction_gate(n_units: int):
    """
    Test 4: Treatment effect depends on cooperative direction alignment.

    tau = 3 * cos(angle(w_i, e_1))

    Patients whose cooperative axis points along the first feature dimension
    benefit; patients whose cooperative geometry is orthogonal do not.
    Tests d_w (cooperative direction component).
    """
    X, T, Y, Eff = [], [], [], []
    d = 4
    e1 = np.zeros(d)
    e1[0] = 1.0
    for _ in range(n_units):
        N = 100
        # Random SPD covariance
        A = np.random.randn(d, d) * 0.5
        cov = A @ A.T + np.eye(d)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        w = inv_sqrt @ np.ones(d)
        # Alignment with first axis
        cos_angle = float(w @ e1) / (np.linalg.norm(w) + 1e-12)
        eff = 3.0 * cos_angle
        mu = np.random.randn(d) * 0.5
        X_raw = np.random.multivariate_normal(mu, cov, N)
        T_raw = np.random.normal(0, 1, (N, 1))
        Y_raw = eff * T_raw.flatten() + np.random.normal(0, 0.5, N)
        X.append(X_raw)
        T.append(T_raw)
        Y.append(Y_raw)
        Eff.append(eff)
    return X, T, Y, np.array(Eff)


def gen_curvature_gate(n_units: int):
    """
    Test 5: Treatment effect depends on manifold coupling strength.

    Tightly coupled features (high off-diagonal correlation) → tau = 2.
    Loosely coupled features (near-identity covariance) → tau = 0.

    Tests d_sigma (Log-Euclidean manifold shape component).
    """
    X, T, Y, Eff = [], [], [], []
    d = 4
    for _ in range(n_units):
        N = 100
        rho = np.random.uniform(0.0, 0.85)
        # Equicorrelated covariance: Sigma = (1-rho)*I + rho*11^T
        cov = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        eff = 2.0 * float(rho > 0.4)   # tightly coupled patients benefit
        X_raw = np.random.multivariate_normal(np.zeros(d), cov, N)
        T_raw = np.random.normal(0, 1, (N, 1))
        Y_raw = eff * T_raw.flatten() + np.random.normal(0, 0.5, N)
        X.append(X_raw)
        T.append(T_raw)
        Y.append(Y_raw)
        Eff.append(eff)
    return X, T, Y, np.array(Eff)


def gen_occupation_gate(n_units: int):
    """
    Test 6: Treatment effect depends on manifold occupation fraction.

    Each patient is a Bernoulli mixture of two states:
      Cooperative (rho=0.7): all features correlated, high T → benefits
      Anti-cooperative (rho=0):  independent features, low T → harms

    tau = 4 * p_coop - 2,  where p_coop is the fraction of observations in
    the cooperative state.

    Mean is zero for all patients (d_mu blind). d_sigma detects a signal via
    the marginal covariance, but d_occ provides a more direct and sensitive
    measurement of the cooperative fraction — especially when n_obs is limited.
    Tests d_occ (occupation component).
    """
    d = 4
    rho = 0.7
    Sigma_coop = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
    Sigma_anti = np.eye(d)
    X, T, Y, Eff = [], [], [], []
    for _ in range(n_units):
        N = 100
        p_coop = np.random.uniform(0.1, 0.9)
        eff = 4.0 * p_coop - 2.0
        mask = np.random.rand(N) < p_coop
        X_raw = np.zeros((N, d))
        n_c = mask.sum()
        if n_c > 0:
            X_raw[mask]  = np.random.multivariate_normal(np.zeros(d), Sigma_coop, n_c)
        if N - n_c > 0:
            X_raw[~mask] = np.random.multivariate_normal(np.zeros(d), Sigma_anti, N - n_c)
        T_raw = np.random.normal(0, 1, (N, 1))
        Y_raw = eff * T_raw.flatten() + np.random.normal(0, 0.5, N)
        X.append(X_raw)
        T.append(T_raw)
        Y.append(Y_raw)
        Eff.append(eff)
    return X, T, Y, np.array(Eff)


def gen_dynamics_gate(n_units: int):
    """
    Test 7: Treatment effect depends on cooperative stability.

    Each patient follows a two-regime Markov chain:
      Cooperative regime:     all features correlated (high T)
      Anti-cooperative regime: features independent (low T)

    Persistence controls how long the patient stays in each regime.
    tau = 3 * persistence - 0.5   (range ~0.4 to 2.2)

    The marginal covariance is CONSTANT across patients (equal stationary
    distribution 50/50), so d_mu and d_sigma are both blind to persistence.
    Only d_dyn (transition matrix Frobenius distance) carries the signal.
    Tests d_dyn (dynamics component).
    """
    d = 4
    rho = 0.8
    Sigma_coop = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
    Sigma_anti = np.eye(d)
    X, T, Y, Eff = [], [], [], []
    for _ in range(n_units):
        N = 100
        persistence = np.random.uniform(0.3, 0.95)
        eff = 3.0 * persistence - 0.5
        # Markov regime-switching with symmetric transition probabilities
        # → stationary distribution is 50/50 → constant marginal covariance
        state = np.random.randint(2)
        X_raw = np.zeros((N, d))
        for t in range(N):
            if state == 0:
                X_raw[t] = np.random.multivariate_normal(np.zeros(d), Sigma_coop)
            else:
                X_raw[t] = np.random.multivariate_normal(np.zeros(d), Sigma_anti)
            if np.random.rand() > persistence:
                state = 1 - state
        T_raw = np.random.normal(0, 1, (N, 1))
        Y_raw = eff * T_raw.flatten() + np.random.normal(0, 0.5, N)
        X.append(X_raw)
        T.append(T_raw)
        Y.append(Y_raw)
        Eff.append(eff)
    return X, T, Y, np.array(Eff)


# ═══════════════════════════════════════════════════════════════════════ #
#  Feature extractor (for RF and Global Linear baselines)                 #
# ═══════════════════════════════════════════════════════════════════════ #

def extract_features(X_list):
    """Mean and variance of each feature across observations."""
    return np.array([
        np.concatenate([np.mean(x, axis=0), np.var(x, axis=0)])
        for x in X_list
    ])


# ═══════════════════════════════════════════════════════════════════════ #
#  Per-test runner                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def run_test(name: str, generator, n_train: int = 500, n_test: int = 100):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    X_tr, T_tr, Y_tr, E_tr = generator(n_train)
    X_te, T_te, Y_te, E_te = generator(n_test)

    feats_tr = extract_features(X_tr)
    feats_te = extract_features(X_te)

    # ── Baseline: Random Forest ──────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(feats_tr, E_tr)
    pred_rf = rf.predict(feats_te)

    # ── Baseline: Global Linear ──────────────────────────────────────
    gl = LinearRegression()
    gl.fit(feats_tr, E_tr)
    pred_gl = gl.predict(feats_te)

    # ── ICG Legacy (IntrinsicJIT, augmented + ridge) ─────────────────
    jit = IntrinsicJIT(k=30, geometry="augmented", local_model="ridge")
    jit.fit(X_tr, T_tr, Y_tr)
    print("  Running ICG Legacy predictions...")
    pred_jit = np.array([jit.predict_effect(X_te[i], T_te[i]) for i in range(n_test)])

    # ── ICG-HVRT ─────────────────────────────────────────────────────
    matcher = ICGHVRTMatcher(
        alpha_levels=1.0,
        alpha_direction=2.0,
        alpha_shape=1.0,
        alpha_occupation=1.5,
        alpha_dynamics=1.0,
        direction_gate=np.pi / 4,
        auto_calibrate=True,
    )
    icghvrt = ICGHVRTEstimator(matcher=matcher, k=30, alpha_local=1.0, n_partitions=8)
    icghvrt.fit(X_tr, T_tr, Y_tr)
    print("  Running ICG-HVRT predictions...")
    pred_hvrt = np.array([icghvrt.predict_effect(X_te[i], T_te[i]) for i in range(n_test)])

    # ── Metrics ──────────────────────────────────────────────────────
    models = {
        "Global Linear":   pred_gl,
        "Random Forest":   pred_rf,
        "ICG (Legacy)":    pred_jit,
        "ICG-HVRT":        pred_hvrt,
    }

    mae_results = {m: float(np.mean(np.abs(pred - E_te))) for m, pred in models.items()}
    std_results = {m: float(np.std(pred - E_te))          for m, pred in models.items()}
    mse_results = {m: float(np.mean((pred - E_te) ** 2))  for m, pred in models.items()}

    print("\n  MAE Results:")
    print("  " + "-" * 55)
    for m, mae in mae_results.items():
        marker = " <-- BEST" if mae == min(mae_results.values()) else ""
        print(f"  {m:<22} MAE: {mae:.4f}  (±{std_results[m]:.4f}){marker}")

    return {
        "mae": mae_results,
        "std": std_results,
        "mse": mse_results,
    }, {m: pred for m, pred in models.items()}, E_te


# ═══════════════════════════════════════════════════════════════════════ #
#  Visualisation                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def plot_results(all_results, all_predictions, all_true, save_path="unified_benchmark.png"):
    test_names  = list(all_results.keys())
    model_names = list(all_results[test_names[0]]["mae"].keys())

    colors = {
        "Global Linear": "#95A5A6",
        "Random Forest": "#E74C3C",
        "ICG (Legacy)":  "#9B59B6",
        "ICG-HVRT":      "#2ECC71",
    }

    n_tests  = len(test_names)
    n_models = len(model_names)
    fig, axes = plt.subplots(n_tests, n_models, figsize=(n_models * 4, n_tests * 3.5))
    if n_tests == 1:
        axes = axes[np.newaxis, :]

    for row, tname in enumerate(test_names):
        true  = all_true[tname]
        preds = all_predictions[tname]
        maes  = all_results[tname]["mae"]

        for col, mname in enumerate(model_names):
            ax   = axes[row, col]
            pred = preds[mname]
            mae  = maes[mname]

            lo = min(true.min(), pred.min()) - 0.2
            hi = max(true.max(), pred.max()) + 0.2
            ax.scatter(true, pred, alpha=0.5, c=colors[mname], s=20)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)

            if row == 0:
                ax.set_title(mname, fontsize=9, fontweight="bold")
            ax.text(0.05, 0.95, f"MAE={mae:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            if col == 0:
                short = tname.split(":")[1].strip() if ":" in tname else tname
                ax.set_ylabel(f"{short[:18]}\nPredicted", fontsize=8)
            if row == n_tests - 1:
                ax.set_xlabel("True Effect", fontsize=8)

    plt.suptitle("ICG-HVRT Unified Benchmark (7 Stress Tests)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"\n  Saved plot: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Main                                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

TESTS = [
    ("Test 1: Noise Swamp",      gen_noise_swamp),
    ("Test 2: Sinusoidal Trap",  gen_sinusoidal),
    ("Test 3: Interaction Gate", gen_interaction_gate),
    ("Test 4: Direction Gate",   gen_direction_gate),
    ("Test 5: Curvature Gate",   gen_curvature_gate),
    ("Test 6: Occupation Gate",  gen_occupation_gate),
    ("Test 7: Dynamics Gate",    gen_dynamics_gate),
]

if __name__ == "__main__":
    print("=" * 70)
    print("  ICG-HVRT UNIFIED BENCHMARK — Seven Stress Tests")
    print("=" * 70)

    all_results: dict     = {}
    all_predictions: dict = {}
    all_true: dict        = {}

    for test_name, generator in TESTS:
        res, preds, true = run_test(test_name, generator)
        all_results[test_name]     = res
        all_predictions[test_name] = preds
        all_true[test_name]        = true

    # ── Summary table ────────────────────────────────────────────────
    model_names = list(all_results[TESTS[0][0]]["mae"].keys())

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY TABLE  (MAE — lower is better)")
    print("=" * 70)

    col_w   = 22
    val_w   = 11
    header  = f"\n{'Model':<{col_w}}"
    for name, _ in TESTS:
        short = name.split(":")[1].strip()[:9]
        header += f"{short:>{val_w}}"
    header += f"{'TOTAL':>{val_w}}"
    print(header)
    print("-" * (col_w + val_w * (len(TESTS) + 1)))

    totals = {m: 0.0 for m in model_names}
    for m in model_names:
        row = f"{m:<{col_w}}"
        for tname, _ in TESTS:
            mae = all_results[tname]["mae"][m]
            totals[m] += mae
            row += f"{mae:>{val_w}.4f}"
        row += f"{totals[m]:>{val_w}.4f}"
        print(row)

    # Best model per test
    print("\n" + "-" * 70)
    print("  BEST MODEL PER TEST:")
    for tname, _ in TESTS:
        mae_d    = all_results[tname]["mae"]
        best_m   = min(mae_d, key=mae_d.get)
        best_mae = mae_d[best_m]
        short    = tname.split(":")[1].strip()
        print(f"  {short:<22}  ->  {best_m}  (MAE={best_mae:.4f})")

    # Overall winner
    best_overall = min(totals, key=totals.get)
    print(f"\n  OVERALL WINNER: {best_overall}  (total MAE={totals[best_overall]:.4f})")

    # ICG-HVRT vs RF improvement
    if "Random Forest" in totals and "ICG-HVRT" in totals:
        improvement = (totals["Random Forest"] - totals["ICG-HVRT"]) / totals["Random Forest"] * 100
        print(f"  ICG-HVRT vs RF total MAE: {improvement:+.1f}%")

    # ── Visualisation ───────────────────────────────────────────────
    print("\n[Generating visualisation...]")
    plot_results(all_results, all_predictions, all_true, save_path="unified_benchmark.png")
