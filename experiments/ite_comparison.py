"""
ITE Comparison: ICG-HVRT vs causal inference methods under confounding.

Each patient has N_OBS time-series observations (X_{i,t}, T_{i,t}, Y_{i,t}).
True ITE tau_i = dE[Y]/dT for patient i is known from the DGP.

DGPs (increasing confounding severity)
--------------------------------------
Randomised          T randomised; no selection confounding. Baseline comparison.
Geometric Confound  U_i -> covariance Sigma_i (coupled vs decoupled) AND E[T_{i,t}].
                    tau_i depends on Sigma_i.  ICG-HVRT d_sigma captures it.
Mean Confound       U_i ~ Uniform(-1,1) -> E[X_{i,t}] AND E[T_{i,t}].  tau_i = U_i.
                    Confound in the mean -- both d_mu matching and R-Learner capture it.
Sparse Mean Conf    Like Mean Confound but U_i appears in only K=5 individual observations
                    (spike_amp=6x) rather than uniformly across all N_OBS.
                    mean(X_i) signal is 3x weaker; spike pattern is strong per observation.
                    Favours methods that use individual-sample structure vs pooled mean.
Hidden Confound     U_i ~ N(0,1) unobserved in X.  tau_i = U_i.  X perp U.
                    Negative control: PEHE ~ std(tau_i) ~ 1 for all pre-treatment methods.
TV Confound         T_{i,t} = 0.5*mean(X_{i,t}) + noise.  Within-patient X->T confounding.
                    tau_i depends on covariance regime.  RMSN/CRN were designed for this.
                    Adversarial to ICG-HVRT: local Ridge sees pooled X but not X_t->T_t link.
Prognostic Conf     U_i ~ N(0,1) leaks into E[X] and directly affects Y, but T perp U.
                    No selection confounding -- U is a pure effect modifier observable via X.
                    Realistic clinical setting: a biomarker predicts treatment response
                    and is measured in baseline covariates, but treatment is randomised.
Partial Leak Sweep  Hidden confound with U partially leaking into E[X] via rho_leak.
                    Tests how quickly each method picks up the signal as rho_leak rises.

Methods
-------
Within-Patient   Per-patient OLS tau_hat. Uses test-Y -- unfair reference.
S-Learner        Ridge on [F, T_bar, T_bar x F]. Confounding-blind.
R-Learner        Robinson (1988) double-debiasing. Handles confounding observable in X.
CRN              Counterfactual Recurrent Network (Bica et al., ICLR 2020).
                 GRU over X_{1:N} -> patient embedding H_i.  Adversarial gradient reversal
                 removes T_bar correlation from H.  tau_hat_i = tau_net(H_i) trained on
                 per-timestep (T_{i,t}, Y_{i,t}) signal.
RMSN             Recurrent Marginal Structural Network (Lim et al., NeurIPS 2018).
                 Two-head GRU: propensity head trains P(T_t|X_{1:t}); IPW weights from
                 stabilized residuals; outcome head trains on IPW-weighted Y.
                 Different inductive bias from CRN: IPW vs adversarial gradient reversal.
ICG-HVRT         Five-component cooperative geometry matching + local Ridge.
                 Pre-treatment: only X_i observations at test time.
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
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autoite import ICGHVRTEstimator, ICGHVRTMatcher

warnings.filterwarnings("ignore")
torch.set_num_threads(4)      # cap PyTorch CPU threads for predictable timing

N_SEEDS      = 10
N_TRAIN      = 300
N_TEST       = 50
N_OBS        = 100
LEAK_RHOS    = [0.0, 0.1, 0.2, 0.5, 1.0]
N_SEEDS_LEAK = 5


# ═══════════════════════════════════════════════════════════════════════ #
#  DGP generators                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

def gen_randomised(n):
    """tau in {0, 2} depending on covariance regime; T randomised (no confounding)."""
    d = 4; rho = 0.9
    Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
    Sa = np.eye(d)
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        coupled = np.random.rand() > 0.5
        Sig = Sc if coupled else Sa
        tau = 2.0 if coupled else 0.0
        x = np.random.multivariate_normal(np.zeros(d), Sig, N_OBS)
        t = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(tau)
    return X, T, Y, np.array(E)


def gen_geometric_confounded(n):
    """U_i -> Sigma_i AND E[T_{i,t}]. Confound lives in the covariance structure."""
    d = 4; rho = 0.9
    Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
    Sa = np.eye(d)
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        coupled = np.random.rand() > 0.5
        Sig = Sc if coupled else Sa
        tau = 2.0 if coupled else 0.0
        t_shift = 1.0 if coupled else -1.0
        x = np.random.multivariate_normal(np.zeros(d), Sig, N_OBS)
        t = np.random.normal(t_shift, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(tau)
    return X, T, Y, np.array(E)


def gen_mean_confounded(n):
    """U_i ~ Uniform(-1,1) -> E[X_{i,t}] AND E[T_{i,t}]. tau_i = U_i."""
    d = 2
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        U = np.random.uniform(-1.0, 1.0)
        tau = float(U)
        x = np.random.normal(U * np.ones(d), 1.0, (N_OBS, d))
        t = np.random.normal(0.5 * U, 1.0, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(tau * t.flatten() + 0.2 * x.sum(axis=1) + np.random.normal(0, 0.5, N_OBS))
        E.append(tau)
    return X, T, Y, np.array(E)


def gen_sparse_mean_confounded(n):
    """
    Like Mean Confounded but U_i is concentrated in K=5 individual observations
    rather than spread uniformly across all N_OBS.

    Each patient has exactly K spike observations: X_{i,spike} += U_i * spike_amp.
    All other observations: X_{i,t} ~ N(0, 1).

    Aggregate mean(X_i) ~ U_i * spike_amp * K/N_OBS = 0.30 * U_i  (3x weaker
    than gen_mean_confounded where mean(X_i) ~ U_i * 1.0).

    Individual spike observations carry a strong signal: X_spike ~ N(6*U_i, 1).
    tau_i = U_i; T confounding identical to Mean Confounded.

    Methods that exploit individual-observation patterns (GRU sequence models)
    should outperform those that rely on pooled mean(X_i) patient summaries.
    """
    d = 2; K = 5; spike_amp = 6.0
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        U         = np.random.uniform(-1.0, 1.0)
        tau       = float(U)
        x         = np.random.normal(0, 1.0, (N_OBS, d))
        spike_idx = np.random.choice(N_OBS, K, replace=False)
        x[spike_idx] += U * spike_amp          # K obs carry strong U signal
        t = np.random.normal(0.5 * U, 1.0, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(tau * t.flatten() + 0.2 * x.sum(axis=1) + np.random.normal(0, 0.5, N_OBS))
        E.append(tau)
    return X, T, Y, np.array(E)


def gen_hidden_confounded(n):
    """U_i ~ N(0,1) completely unobserved in X. tau_i = U_i. Negative control."""
    d = 2
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        U = np.random.normal(0, 1)
        tau = float(U)
        x = np.random.normal(0, 1, (N_OBS, d))
        t = np.random.normal(0.5 * U, 1.0, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(tau)
    return X, T, Y, np.array(E)


def gen_tv_confounded(n):
    """
    Time-varying confounding: T_{i,t} = 0.5*mean(X_{i,t}) + N(0,0.5).

    tau_i = 2.0 * (rho_i > 0.4)  [coupling-dependent, as in Curvature Gate].
    E[T_i] = 0 for all i  (no between-patient T bias; confounding is within-patient).
    Y_{i,t} = tau_i * T_{i,t} + 0.3 * mean(X_{i,t}) + N(0, 0.5).

    This DGP is adversarial to ICG-HVRT: the X_t -> T_t link is within-patient
    and time-varying, so pooled matching does not deconfound it.
    RMSN's IPW weights down-weight high-confounding timesteps.
    CRN's adversarial targets T_bar ~ 0 -> is idle; reduces to GRU regression.
    """
    d = 4
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        rho  = np.random.uniform(0.0, 0.9)
        cov  = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        tau  = 2.0 * float(rho > 0.4)
        x    = np.random.multivariate_normal(np.zeros(d), cov, N_OBS)
        t    = 0.5 * x.mean(axis=1, keepdims=True) + np.random.normal(0, 0.5, (N_OBS, 1))
        y    = tau * t.flatten() + 0.3 * x.mean(axis=1) + np.random.normal(0, 0.5, N_OBS)
        X.append(x); T.append(t); Y.append(y); E.append(tau)
    return X, T, Y, np.array(E)


def gen_outlier_spike(n):
    """
    Geometric confounding + sparse single-feature outlier spikes.

    Cooperative structure (tau = 0 vs 2) is determined by the covariance
    regime, identical to gen_randomised.  Additionally, ~5% of each patient's
    observations have a single random feature set to ±20 sigma.

    Per Proposition 1 (PyramidHART properties, §2.3 main.pdf):
      - A 50σ spike inflates T by O(50√d) but leaves A unchanged
        (exact single-feature outlier cancellation).
    So:
      ICG-HVRT: T-based partitions get corrupted by spikes → poor patient
                matching → high PEHE.
      ICG-HART: A-based partitions are immune → robust matching → low PEHE.
    T is randomised (no selection confounding) so GRU methods provide no
    advantage; the only differentiator is spike-robustness of the partition.
    """
    d = 4; rho = 0.9; spike_frac = 0.05; spike_mag = 20.0
    Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
    Sa = np.eye(d)
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        coupled = np.random.rand() > 0.5
        Sig     = Sc if coupled else Sa
        tau     = 2.0 if coupled else 0.0
        x       = np.random.multivariate_normal(np.zeros(d), Sig, N_OBS)
        # Single-feature spikes: |z_k| >> sum_{j≠k} |z_j|
        n_spikes   = max(1, int(N_OBS * spike_frac))
        spike_rows = np.random.choice(N_OBS, n_spikes, replace=False)
        spike_cols = np.random.randint(0, d, n_spikes)
        spike_sgns = np.random.choice([-1.0, 1.0], n_spikes)
        for r, c, s in zip(spike_rows, spike_cols, spike_sgns):
            x[r, c] += s * spike_mag
        t = np.random.normal(0, 1, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(tau)
    return X, T, Y, np.array(E)


def gen_individual_covariate_leak(n, K=3, leak_amp=8.0):
    """
    U_i leaks into a SINGLE covariate (feature 0) at exactly K individual
    observations.  All other observations and features have no U signal.

    Design (d=4, K=3, N_OBS=100):
      X_{i,t,0} += U_i * leak_amp  at K randomly chosen t.
      X_{i,t,1:4} ~ N(0, 1)  for all t  (pure noise features).
      mean(X_i[:,0]) = U_i * leak_amp * K/N_OBS = 0.24 * U_i  (very weak aggregate).
      Spike obs: X_{i,spike,0} ~ N(8*U_i, 1)  (strong per-observation signal).

    Contrast with gen_sparse_mean_confounded (K=5, all features, d=2):
      -- More feature-selective: only dim 0 carries signal.
      -- Even weaker aggregate: mean ≈ 0.24*U_i (vs 0.30*U_i over 2 dims).

    Scientific hypothesis:
      S-Learner/R-Learner: use mean(X[:,0]) ≈ 0.24*U_i -> very noisy.
      CRN/RMSN: GRU processes individual obs -> detects spike directly.
      ICG-HVRT (cone): spike distorts sample cov -> cone signal available.
      ICG-HART (pyramid): MAD-whitening robust to spikes -> spike IGNORED.
    """
    d = 4
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        U         = np.random.uniform(-1.0, 1.0)
        tau       = float(U)
        x         = np.random.normal(0, 1.0, (N_OBS, d))
        spike_idx = np.random.choice(N_OBS, K, replace=False)
        x[spike_idx, 0] += U * leak_amp     # only feature 0 at K obs
        t = np.random.normal(0.5 * U, 1.0, (N_OBS, 1))
        X.append(x); T.append(t)
        Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
        E.append(tau)
    return X, T, Y, np.array(E)


def gen_sparse_k_leak(K: int, leak_amp: float = 8.0):
    """
    Factory: DGP where U_i leaks into feature 0 at exactly K individual obs.
    Used by run_sparse_k_sweep to vary sparsity (K=1 to K=N_OBS).

    At K=N_OBS: equivalent to single-feature mean confounding (full aggregate).
    At K=1: spike affects only 1 obs; aggregate mean(X[:,0]) = 0.08*U_i.
    """
    d = 4
    def _gen(n):
        X, T, Y, E = [], [], [], []
        for _ in range(n):
            U         = np.random.uniform(-1.0, 1.0)
            tau       = float(U)
            x         = np.random.normal(0, 1.0, (N_OBS, d))
            spike_idx = np.random.choice(N_OBS, min(K, N_OBS), replace=False)
            x[spike_idx, 0] += U * leak_amp
            t = np.random.normal(0.5 * U, 1.0, (N_OBS, 1))
            X.append(x); T.append(t)
            Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
            E.append(tau)
        return X, T, Y, np.array(E)
    return _gen


def gen_partial_leak(rho_leak):
    """
    Return a DGP generator for the partial-leak confounded setting.

    U_i ~ N(0,1) drives both tau_i and E[T_{i,t}].  U leaks into X via:
      E[X_{i,t}] = rho_leak * U_i * ones(d)

    rho_leak = 0  ->  X perp U  (pure hidden confounding, all methods fail)
    rho_leak = 1  ->  E[X] = U * ones  (equivalent to Mean Confounded)

    ICG-HVRT exploits the leak via d_mu (and higher-order geometry at larger rho).
    R-Learner exploits it via mean(X) in F.
    CRN exploits it via the GRU embedding of the X sequence.
    S-Learner is confounding-blind but partially exploits via T_bar correlation.
    """
    d = 2
    def _gen(n):
        X, T, Y, E = [], [], [], []
        for _ in range(n):
            U   = np.random.normal(0, 1)
            tau = float(U)
            x   = np.random.normal(rho_leak * U * np.ones(d), 1.0, (N_OBS, d))
            t   = np.random.normal(0.5 * U, 1.0, (N_OBS, 1))
            X.append(x); T.append(t)
            Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
            E.append(tau)
        return X, T, Y, np.array(E)
    return _gen


def gen_multi_leak(K: int, rho: float = 0.3):
    """
    K independent confounders, each leaking weakly into one X dimension.

    tau_i = (1/sqrt(K)) * sum_k(U_k_i),  std(tau) ~ 1 regardless of K.
    E[X_{i,t,k}] = rho * U_k_i  for k=1..K;  remaining dims are noise.
    d = max(K, 2) keeps ICG-HVRT non-degenerate.

    As K grows:
      - ICG-HVRT d_mu aggregates K independent mean signals -> improves with K.
      - CRN's adversarial removes more from H as H grows more informative
        about T_bar = 0.5*tau -> adversarial component increasingly hurts.
      - GRU-NoAdv (lam_adv=0) serves as the ablation control.
    """
    d = max(K, 2)
    def _gen(n):
        X, T, Y, E = [], [], [], []
        for _ in range(n):
            U   = np.random.normal(0, 1, K)
            tau = float(U.sum() / np.sqrt(K))
            mu_x        = np.zeros(d)
            mu_x[:K]    = rho * U
            x = np.random.normal(mu_x, 1.0, (N_OBS, d))
            t = np.random.normal(0.5 * tau, 1.0, (N_OBS, 1))
            X.append(x); T.append(t)
            Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
            E.append(tau)
        return X, T, Y, np.array(E)
    return _gen


def gen_prognostic_confounder(n, rho_x=0.5, gamma_prog=0.5):
    """
    Prognostic confounder: U modifies tau AND outcome baseline, but T perp U.

    U_i ~ N(0,1)              -- unobserved prognostic factor
    tau_i = U_i               -- U drives treatment effect heterogeneity
    E[X_{i,t}] = rho_x * U_i -- U leaks into ALL covariates (dense, mean-level)
    T_{i,t} ~ N(0, 1)         -- fully randomised; T perp U (no selection bias)
    Y_{i,t} = tau_i * T_{i,t} + gamma_prog * U_i + N(0, 0.5)

    The 'realistic clinical trial' scenario:
      - A latent biomarker U predicts both who benefits from treatment (tau_i)
        and the patient's baseline outcome level (direct gamma_prog effect).
      - U is partially observable via routine measurements X.
      - Treatment is randomised, so U does not drive selection -- there is
        NO confounding bias in the standard causal sense.
      - The challenge is pure effect-modification / heterogeneity recovery:
        methods that exploit X to estimate U will recover tau_i;
        methods that deconfound U->T (e.g. CRN adversarial) gain nothing
        since T_bar ~ 0 for all patients (T already randomised).

    Contrast with gen_hidden_confounded (U->T, X perp U -- all methods fail)
    and gen_mean_confounded (U->T AND U->X -- deconfounding methods win).
    Here: U->X (observable) but NOT U->T -- deconfounding is idle;
    heterogeneity recovery via X is the only signal.
    """
    d = 4
    X, T, Y, E = [], [], [], []
    for _ in range(n):
        U   = np.random.normal(0, 1)
        tau = float(U)
        x   = np.random.normal(rho_x * U * np.ones(d), 1.0, (N_OBS, d))
        t   = np.random.normal(0, 1.0, (N_OBS, 1))               # T perp U
        y   = tau * t.flatten() + gamma_prog * U + np.random.normal(0, 0.5, N_OBS)
        X.append(x); T.append(t)
        Y.append(y)
        E.append(tau)
    return X, T, Y, np.array(E)


def gen_prognostic_rho(rho_x: float, gamma_prog: float = 0.5):
    """
    Factory: prognostic confounder with variable leak strength rho_x in [0, 1].

    rho_x=0  -> X perp U; all methods fail (heterogeneity fully hidden)
    rho_x=0.5 -> moderate leak; methods exploiting mean(X) should improve
    rho_x=1.0 -> strong leak; mean(X) ~ U_i; near-full heterogeneity recovery
    """
    def _gen(n):
        return gen_prognostic_confounder(n, rho_x=rho_x, gamma_prog=gamma_prog)
    _gen.__name__ = f"prog_rho_{rho_x:.1f}"
    return _gen


TESTS = [
    ("Randomised",           gen_randomised),
    ("Geometric Confounded", gen_geometric_confounded),
    ("Mean Confounded",      gen_mean_confounded),
    ("Sparse Mean Conf",     gen_sparse_mean_confounded),
    ("Indiv Feature Leak",   gen_individual_covariate_leak),
    ("Prognostic Conf",      gen_prognostic_confounder),
    ("Hidden Confounded",    gen_hidden_confounded),
    ("TV Confounded",        gen_tv_confounded),
    ("Outlier Spike",        gen_outlier_spike),
]

METHODS            = ["Within-Patient", "S-Learner", "R-Learner", "CRN", "RMSN", "ICG-HVRT", "ICG-HART", "ICG-Synergy"]
LEAK_METHODS       = ["S-Learner", "R-Learner", "CRN", "RMSN", "ICG-HVRT"]
SPARSE_K_VALUES    = [1, 3, 10, 30, 100]
N_SEEDS_SPARSE     = 5
SPARSE_LEAK_METHODS = ["S-Learner", "R-Learner", "CRN", "ICG-HVRT", "ICG-HART"]
PROG_RHO_VALUES    = [0.0, 0.1, 0.25, 0.5, 1.0]
N_SEEDS_PROG       = 5
PROG_METHODS       = ["S-Learner", "R-Learner", "CRN", "ICG-HVRT", "ICG-HART"]
MULTI_LEAK_K       = [1, 2, 4, 8]
MULTI_LEAK_RHO     = 0.3
MULTI_LEAK_METHODS = ["S-Learner", "R-Learner", "CRN", "GRU-NoAdv", "ICG-HVRT"]
N_SEEDS_MULTI      = 5
NOISE_SIGMAS       = [0.5, 1.0, 2.0, 4.0, 8.0]   # observation noise on X
N_SEEDS_NOISE      = 5
METRICS            = ["pehe", "mae", "nmae", "bias", "spearman", "sign_acc", "regret", "n_regret"]


# ═══════════════════════════════════════════════════════════════════════ #
#  Patient summary features (for cross-sectional methods)                 #
# ═══════════════════════════════════════════════════════════════════════ #

def patient_summaries(X_list, T_list, Y_list):
    """F = [mean(X), upper_tri(cov(X))]; T_bar = mean(T); Y_bar = mean(Y)."""
    F_rows, T_bar, Y_bar = [], [], []
    for Xi, Ti, Yi in zip(X_list, T_list, Y_list):
        Xi      = np.atleast_2d(Xi)
        Ti_flat = np.asarray(Ti).ravel()
        Yi_flat = np.asarray(Yi).ravel()
        mu_x    = Xi.mean(axis=0)
        cov_x   = np.array([[float(np.var(Xi.ravel()))]]) if Xi.shape[1] == 1 else np.cov(Xi.T)
        iu      = np.triu_indices(cov_x.shape[0])
        F_rows.append(np.concatenate([mu_x, cov_x[iu]]))
        T_bar.append(float(Ti_flat.mean()))
        Y_bar.append(float(Yi_flat.mean()))
    return np.array(F_rows), np.array(T_bar), np.array(Y_bar)


# ═══════════════════════════════════════════════════════════════════════ #
#  Method implementations                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

def _within_patient(X_list, T_list, Y_list):
    """Per-patient Ridge(Y ~ [X, T]) slope.  UNFAIR -- uses test-patient Y."""
    taus = []
    for Xi, Ti, Yi in zip(X_list, T_list, Y_list):
        Ti_flat = np.asarray(Ti).ravel()
        Yi_flat = np.asarray(Yi).ravel()
        XTi = np.column_stack([np.atleast_2d(Xi), Ti_flat[:, None]])
        taus.append(float(Ridge(alpha=1.0).fit(XTi, Yi_flat).coef_[-1]))
    return np.array(taus)


def _s_learner(F_tr, T_tr_bar, Y_tr_bar, F_te, alpha=10.0):
    """Ridge on [F, T_bar, T_bar x F]; tau_hat(F) = d_Yhat / d_T_bar."""
    def _D(F, Tb):
        Tc = Tb[:, None]
        return np.hstack([F, Tc, Tc * F])
    model   = Ridge(alpha=alpha).fit(_D(F_tr, T_tr_bar), Y_tr_bar)
    d       = F_te.shape[1]
    return model.coef_[d] + F_te @ model.coef_[d + 1:]


def _r_learner(F_tr, T_tr_bar, Y_tr_bar, F_te, alpha=10.0, n_folds=5):
    """Robinson (1988) double-debiasing for continuous T."""
    n  = len(F_tr)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    m_resid = np.zeros(n)
    t_resid = np.zeros(n)
    for tr_idx, va_idx in kf.split(F_tr):
        m = Ridge(alpha=alpha).fit(F_tr[tr_idx], Y_tr_bar[tr_idx])
        e = Ridge(alpha=alpha).fit(F_tr[tr_idx], T_tr_bar[tr_idx])
        m_resid[va_idx] = Y_tr_bar[va_idx] - m.predict(F_tr[va_idx])
        t_resid[va_idx] = T_tr_bar[va_idx] - e.predict(F_tr[va_idx])
    D_tr      = np.column_stack([t_resid, t_resid[:, None] * F_tr])
    tau_model = Ridge(alpha=alpha, fit_intercept=False).fit(D_tr, m_resid)
    return tau_model.coef_[0] + F_te @ tau_model.coef_[1:]


# ─── CRN (Counterfactual Recurrent Network) ────────────────────────────

class _GradReverse(torch.autograd.Function):
    """Gradient reversal layer: identity forward, negated gradient backward."""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class _CRNNet(nn.Module):
    """
    Minimal CRN adapted for continuous-T panel ITE estimation.

    Architecture
    ------------
    Encoder   GRU over X_{i,1:N} -> mean-pooled H_i in R^{d_hidden}
    Outcome   Y_{i,t} = alpha(H_i) + T_{i,t} * (tau_net(H_i) + gamma)
              Trained on per-timestep pairs: captures within-patient T variation.
    Adversary Gradient reversal on T_bar_i prediction forces H_i to be
              balanced w.r.t. between-patient treatment selection.

    At test time: tau_hat_i = tau_net(H_i) + gamma  (scalar per patient).

    Note: using per-timestep signal means CRN works even when T_bar ~ 0
    (randomised T), unlike cross-sectional S/R-Learner.
    The adversarial component may hurt when the confounder IS the effect
    modifier (since removing T_bar correlation also removes tau signal).
    """

    def __init__(self, d_in: int, d_hidden: int = 32, lam_adv: float = 0.5):
        super().__init__()
        self.gru     = nn.GRU(d_in, d_hidden, batch_first=True)
        self.alpha   = nn.Linear(d_hidden, 1)   # patient baseline outcome
        self.tau_net = nn.Linear(d_hidden, 1)   # heterogeneous treatment slope
        self.gamma   = nn.Parameter(torch.zeros(1))  # global slope offset
        self.adv     = nn.Linear(d_hidden, 1)   # adversarial T_bar predictor
        self.lam_adv = lam_adv

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)       # (B, T, d_hidden)
        return out.mean(dim=1)     # mean pool -> (B, d_hidden)

    def forward(self, x, T_seq, T_bar):
        H       = self.encode(x)                                    # (B, d_hidden)
        alpha_i = self.alpha(H).squeeze(-1)                         # (B,)
        tau_i   = self.tau_net(H).squeeze(-1) + self.gamma          # (B,)
        y_hat   = alpha_i[:, None] + T_seq * tau_i[:, None]         # (B, T)
        T_hat   = self.adv(_GradReverse.apply(H, self.lam_adv)).squeeze(-1)  # (B,)
        return y_hat, T_hat


def _crn_learner(X_tr, T_tr, Y_tr, X_te,
                 n_epochs: int = 150, d_hidden: int = 32,
                 lam_adv: float = 0.5, lr: float = 1e-3):
    """
    CRN (Bica et al., ICLR 2020) adapted for continuous-T panel data.

    Cross-sectional S/R-Learner compress (X, T, Y) to patient summaries and rely
    on between-patient T_bar variation.  CRN instead:
      (1) encodes the full X_{i,1:N} sequence via GRU -> H_i
      (2) trains on per-timestep (T_{i,t}, Y_{i,t}) -> exploits within-patient T std
      (3) adversarially removes T_bar-H correlation to deconfound matching

    This allows CRN to estimate tau even when T_bar ~ 0 (randomised setting),
    directly paralleling ICG-HVRT's use of within-patient T variation.
    """
    N  = N_OBS
    d  = np.atleast_2d(X_tr[0]).shape[1]

    X_arr = np.stack([np.atleast_2d(x)[:N] for x in X_tr])           # (n, N, d)
    T_arr = np.stack([np.asarray(t).ravel()[:N] for t in T_tr])       # (n, N)
    Y_arr = np.stack([np.asarray(y).ravel()[:N] for y in Y_tr])       # (n, N)
    T_bar = T_arr.mean(axis=1)                                         # (n,)

    X_t  = torch.tensor(X_arr, dtype=torch.float32)
    T_t  = torch.tensor(T_arr, dtype=torch.float32)
    Y_t  = torch.tensor(Y_arr, dtype=torch.float32)
    Tb_t = torch.tensor(T_bar, dtype=torch.float32)

    model = _CRNNet(d, d_hidden=d_hidden, lam_adv=lam_adv)
    opt   = optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        opt.zero_grad()
        y_hat, tb_hat = model(X_t, T_t, Tb_t)
        loss = F.mse_loss(y_hat, Y_t) + F.mse_loss(tb_hat, Tb_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        X_te_arr = np.stack([np.atleast_2d(x)[:N] for x in X_te])
        H_te     = model.encode(torch.tensor(X_te_arr, dtype=torch.float32))
        tau_hat  = (model.tau_net(H_te).squeeze(-1) + model.gamma).numpy()
    return tau_hat


# ─── RMSN (Recurrent Marginal Structural Network) ──────────────────────

class _RMSNNet(nn.Module):
    """
    Two-headed GRU: separate propensity head and outcome head.

    Stage 1: propensity head learns P(T_t | X_{1:t}) via MSE on T.
    Stage 2: outcome head trains on IPW-weighted Y (stabilized weights from
             stage-1 residuals).  Inductive bias: down-weight high-confounding
             timesteps rather than adversarially removing T_bar from H.
    """

    def __init__(self, d_in: int, d_hidden: int = 32):
        super().__init__()
        self.prop_gru  = nn.GRU(d_in, d_hidden, batch_first=True)
        self.prop_head = nn.Linear(d_hidden, 1)   # predict T_t
        self.out_gru   = nn.GRU(d_in, d_hidden, batch_first=True)
        self.tau_head  = nn.Linear(d_hidden, 1)   # patient treatment slope
        self.mu_head   = nn.Linear(d_hidden, 1)   # patient baseline outcome

    def propensity_forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> T_hat: (B, T)."""
        h, _ = self.prop_gru(x)       # (B, T, d_hidden)
        return self.prop_head(h).squeeze(-1)   # (B, T)

    def outcome_forward(self, x: torch.Tensor):
        """x: (B, T, d_in) -> (tau_i, mu_i): each (B,)."""
        h, _ = self.out_gru(x)        # (B, T, d_hidden)
        H    = h.mean(dim=1)          # mean-pool -> (B, d_hidden)
        return self.tau_head(H).squeeze(-1), self.mu_head(H).squeeze(-1)


def _rmsn_learner(X_tr, T_tr, Y_tr, X_te,
                  n_prop: int = 60, n_out: int = 90,
                  d_hidden: int = 32, lr: float = 1e-3):
    """
    RMSN (Lim et al., NeurIPS 2018) adapted for continuous-T panel ITE.

    Stage 1 (n_prop epochs): train propensity GRU to predict T_{i,t} from X_{i,1:t}.
    Between stages: compute stabilized IPW weights from propensity residuals.
      log_w_it = -T_it^2/(2*sigma_marg^2) + eps_it^2/(2*sigma_eps^2)
      w_it = exp(log_w_it), clipped to [0.1, 10], normalized per patient.
    Stage 2 (n_out epochs): train outcome GRU on IPW-weighted Y.
      Loss = mean_i[ mean_t[ w_it * (Y_it - mu_i - tau_i * T_it)^2 ] ]

    Test time: tau_hat_i = tau_head(mean_pool(out_gru(X_te_i))).
    """
    N = N_OBS
    d = np.atleast_2d(X_tr[0]).shape[1]

    X_arr = np.stack([np.atleast_2d(x)[:N] for x in X_tr])    # (n, N, d)
    T_arr = np.stack([np.asarray(t).ravel()[:N] for t in T_tr])  # (n, N)
    Y_arr = np.stack([np.asarray(y).ravel()[:N] for y in Y_tr])  # (n, N)

    X_t = torch.tensor(X_arr, dtype=torch.float32)
    T_t = torch.tensor(T_arr, dtype=torch.float32)
    Y_t = torch.tensor(Y_arr, dtype=torch.float32)

    model = _RMSNNet(d, d_hidden=d_hidden)
    opt   = optim.Adam(model.parameters(), lr=lr)

    # ── Stage 1: propensity ─────────────────────────────────────────── #
    for _ in range(n_prop):
        opt.zero_grad()
        T_hat = model.propensity_forward(X_t)     # (n, N)
        loss  = F.mse_loss(T_hat, T_t)
        loss.backward()
        opt.step()

    # ── Compute stabilized IPW weights (no_grad, numpy) ─────────────── #
    model.eval()
    with torch.no_grad():
        T_hat_np = model.propensity_forward(X_t).numpy()   # (n, N)
    eps_it   = T_arr - T_hat_np                             # (n, N)
    sigma_eps  = float(eps_it.std()) + 1e-8
    sigma_marg = float(T_arr.std())  + 1e-8
    log_w      = -T_arr**2 / (2 * sigma_marg**2) + eps_it**2 / (2 * sigma_eps**2)
    w          = np.exp(log_w)
    w          = np.clip(w, 0.1, 10.0)
    w          = w / (w.mean(axis=1, keepdims=True) + 1e-8)  # normalize per patient
    W_t        = torch.tensor(w, dtype=torch.float32)         # (n, N)

    # ── Stage 2: IPW-weighted outcome ────────────────────────────────── #
    model.train()
    opt2 = optim.Adam(
        list(model.out_gru.parameters()) +
        list(model.tau_head.parameters()) +
        list(model.mu_head.parameters()),
        lr=lr,
    )
    for _ in range(n_out):
        opt2.zero_grad()
        tau_i, mu_i = model.outcome_forward(X_t)             # (n,), (n,)
        Y_hat = mu_i[:, None] + tau_i[:, None] * T_t         # (n, N)
        loss  = (W_t * (Y_t - Y_hat) ** 2).mean()
        loss.backward()
        opt2.step()

    # ── Test-time prediction ─────────────────────────────────────────── #
    model.eval()
    with torch.no_grad():
        X_te_arr = np.stack([np.atleast_2d(x)[:N] for x in X_te])
        X_te_t   = torch.tensor(X_te_arr, dtype=torch.float32)
        tau_hat, _ = model.outcome_forward(X_te_t)
    return tau_hat.numpy()


def _icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te, k=30, n_partitions=8,
              gamma_levels_perp=0.25):
    """ICG-HVRT: cooperative geometry matching + local Ridge (pre-treatment)."""
    est = ICGHVRTEstimator(
        matcher=ICGHVRTMatcher(auto_calibrate=True,
                               gamma_levels_perp=gamma_levels_perp),
        k=k, n_partitions=n_partitions, learn_weights=True,
    )
    est.fit(X_tr, T_tr, Y_tr)
    return np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])


def _icg_hart(X_tr, T_tr, Y_tr, X_te, T_te, k=30, n_partitions=8,
              gamma_levels_perp=0.25):
    """ICG-HART: MAD-based pyramid geometry + lad local model (pre-treatment)."""
    est = ICGHVRTEstimator(
        matcher=ICGHVRTMatcher(auto_calibrate=True,
                               gamma_levels_perp=gamma_levels_perp),
        k=k, n_partitions=n_partitions, learn_weights=True,
        geometry='pyramid',
    )
    est.fit(X_tr, T_tr, Y_tr)
    return np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])


def _icg_synergy(X_tr, T_tr, Y_tr, X_te, T_te, k=30, n_partitions=8):
    """ICG-Synergy: PyramidHART spike detection → HVRT on clean bulk + lad local model.

    Two-stage pipeline per the cooperation design (Peace 2026, §4.3):
      Stage 1 — PyramidHART identifies spike samples via |A|/‖z‖₁ (Prop 1.3).
      Stage 2 — HVRT fitted on spike-free bulk; Theorem 3 noise invariance holds.
    Cone geometry (ConeIdentity) uses full MAD stats (already spike-robust).
    """
    est = ICGHVRTEstimator(
        matcher=ICGHVRTMatcher(auto_calibrate=True),
        k=k, n_partitions=n_partitions, learn_weights=True,
        geometry='synergy',
    )
    est.fit(X_tr, T_tr, Y_tr)
    return np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])


# ═══════════════════════════════════════════════════════════════════════ #
#  Metrics                                                                #
# ═══════════════════════════════════════════════════════════════════════ #

def compute_metrics(tau_hat, tau_true):
    err  = tau_hat - tau_true
    pehe = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    nmae = mae / (float(np.std(tau_true)) + 1e-8)   # normalised by std(tau): scale-free
    bias = float(np.mean(err))
    rho, _ = spearmanr(tau_hat, tau_true)
    rho  = 0.0 if np.isnan(rho) else float(rho)
    nz   = np.abs(tau_true) > 0.1
    sacc = float(np.mean(np.sign(tau_hat[nz]) == np.sign(tau_true[nz]))) if nz.any() else float("nan")
    # Policy regret: effect mass misrouted to the wrong treatment decision.
    #   regret_i = |tau_true_i| * I(sign(tau_hat_i) != sign(tau_true_i))
    #   n_regret = E[regret_i] / E[|tau_true_i|]  in [0, 1]
    #            = fraction of total absolute effect assigned to the wrong arm.
    wrong    = np.sign(tau_hat) != np.sign(tau_true)
    regret   = float(np.mean(np.abs(tau_true) * wrong))
    n_regret = regret / (float(np.mean(np.abs(tau_true))) + 1e-8)
    return dict(pehe=pehe, mae=mae, nmae=nmae, bias=bias, spearman=rho, sign_acc=sacc,
                regret=regret, n_regret=n_regret)


# ═══════════════════════════════════════════════════════════════════════ #
#  Single-seed evaluation                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

def evaluate_once(generator, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_tr, T_tr, Y_tr, E_tr = generator(N_TRAIN)
    np.random.seed(seed + 100_000)
    X_te, T_te, Y_te, E_te = generator(N_TEST)

    F_tr, Tbar_tr, Ybar_tr = patient_summaries(X_tr, T_tr, Y_tr)
    F_te, _,       _        = patient_summaries(X_te, T_te, Y_te)

    out = {}
    out["Within-Patient"] = compute_metrics(_within_patient(X_te, T_te, Y_te), E_te)
    out["S-Learner"]      = compute_metrics(_s_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)
    out["R-Learner"]      = compute_metrics(_r_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)
    out["CRN"]            = compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te), E_te)
    out["RMSN"]           = compute_metrics(_rmsn_learner(X_tr, T_tr, Y_tr, X_te), E_te)
    out["ICG-HVRT"]       = compute_metrics(_icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te), E_te)
    out["ICG-HART"]       = compute_metrics(_icg_hart(X_tr, T_tr, Y_tr, X_te, T_te), E_te)  # PyramidHART
    out["ICG-Synergy"]    = compute_metrics(_icg_synergy(X_tr, T_tr, Y_tr, X_te, T_te), E_te)
    return out


# ═══════════════════════════════════════════════════════════════════════ #
#  Multi-seed DGP evaluation                                              #
# ═══════════════════════════════════════════════════════════════════════ #

def evaluate_dgp(name, generator):
    print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}")
    pool = {m: {k: [] for k in METRICS} for m in METHODS}

    for s in range(N_SEEDS):
        print(f"  seed {s + 1:2d}/{N_SEEDS}\r", end="", flush=True)
        res = evaluate_once(generator, seed=s * 137 + 7)
        for m, mets in res.items():
            for k, v in mets.items():
                pool[m][k].append(v)
    print()

    agg = {
        m: {k: (float(np.nanmean(vs)), float(np.nanstd(vs))) for k, vs in mets.items()}
        for m, mets in pool.items()
    }

    print(f"\n  {'Method':<18}  {'sqrt-PEHE':>14}  {'MAE':>9}  {'NMAE':>7}  {'Bias':>8}  {'Spearman':>9}  {'Sign%':>7}")
    print("  " + "-" * 84)
    best_pehe = min(agg[m]["pehe"][0] for m in METHODS)
    for m in METHODS:
        mu_p, sd_p = agg[m]["pehe"]
        star     = " *" if abs(mu_p - best_pehe) < 1e-9 else "  "
        sacc_str = f"{agg[m]['sign_acc'][0]:>7.2%}" if not np.isnan(agg[m]['sign_acc'][0]) else "     nan"
        print(
            f"  {m:<18}"
            f"  {mu_p:6.4f}+/-{sd_p:.4f}"
            f"  {agg[m]['mae'][0]:>9.4f}"
            f"  {agg[m]['nmae'][0]:>7.4f}"
            f"  {agg[m]['bias'][0]:>8.4f}"
            f"  {agg[m]['spearman'][0]:>9.4f}"
            f"  {sacc_str}"
            f"{star}"
        )
    return agg, pool


# ═══════════════════════════════════════════════════════════════════════ #
#  Visualisation -- main comparison                                       #
# ═══════════════════════════════════════════════════════════════════════ #

_COLORS = {
    "Within-Patient": "#BDC3C7",
    "S-Learner":      "#E74C3C",
    "R-Learner":      "#3498DB",
    "ICG-HART":       "#1ABC9C",  # teal — distinguishable from HVRT green
    "ICG-Synergy":    "#D35400",  # burnt orange — two-stage pipeline
    "CRN":            "#9B59B6",
    "RMSN":           "#F39C12",
    "GRU-NoAdv":      "#E67E22",
    "ICG-HVRT":       "#2ECC71",
}


def plot_comparison(all_agg, save_path="ite_comparison.png"):
    dgp_names = [n for n, _ in TESTS]
    x = np.arange(len(dgp_names))
    w = 0.13
    offsets = np.linspace(-2.5, 2.5, len(METHODS)) * w
    short   = [n[:13] for n in dgp_names]

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    for ax_i, (metric, ylabel, title) in enumerate([
        ("pehe",     "sqrt-PEHE (lower better)",    "sqrt-PEHE by DGP"),
        ("spearman", "Spearman rho (higher better)", "Rank Correlation with True tau"),
        ("bias",     "|Bias| (lower better)",        "Absolute Bias"),
    ]):
        ax = axes[ax_i]
        for i, m in enumerate(METHODS):
            if metric == "bias":
                means = [abs(all_agg[n][m]["bias"][0]) for n in dgp_names]
                stds  = None
            else:
                means = [all_agg[n][m][metric][0] for n in dgp_names]
                stds  = [all_agg[n][m][metric][1] for n in dgp_names]
            ax.bar(x + offsets[i], means, w, yerr=stds, capsize=3,
                   label=m, color=_COLORS[m], alpha=0.85)
        if metric == "spearman":
            ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"ITE Comparison: ICG-HVRT vs S/R-Learner vs CRN vs RMSN under confounding\n"
        f"({N_SEEDS} seeds x {len(TESTS)} DGPs,  {N_TRAIN} train / {N_TEST} test / {N_OBS} obs)\n"
        f"Within-Patient uses test-patient Y (unfair reference).",
        fontsize=10, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Partial leak sweep                                                     #
# ═══════════════════════════════════════════════════════════════════════ #

def run_leak_sweep():
    """
    Evaluate LEAK_METHODS as rho_leak varies from 0 (U invisible in X) to 1
    (U fully leaks into E[X]).  For each rho, run N_SEEDS_LEAK seeds and record
    the mean sqrt-PEHE.

    Expected behaviour:
      rho=0  -- all methods at ~ std(tau) ~ 1.0  (negative control)
      rho>0  -- ICG-HVRT and CRN should improve faster than R/S-Learner
                because they use the full X sequence (not just mean(X) summary)
    """
    results = {m: {rho: [] for rho in LEAK_RHOS} for m in LEAK_METHODS}

    for rho in LEAK_RHOS:
        gen = gen_partial_leak(rho)
        print(f"  rho_leak = {rho:.1f}", end="", flush=True)
        for s in range(N_SEEDS_LEAK):
            np.random.seed(s * 137 + 7)
            torch.manual_seed(s * 137 + 7)
            X_tr, T_tr, Y_tr, _ = gen(N_TRAIN)
            np.random.seed(s * 137 + 7 + 100_000)
            X_te, T_te, Y_te, E_te = gen(N_TEST)

            F_tr, Tbar_tr, Ybar_tr = patient_summaries(X_tr, T_tr, Y_tr)
            F_te, _,       _        = patient_summaries(X_te, T_te, Y_te)

            results["S-Learner"][rho].append(
                compute_metrics(_s_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["R-Learner"][rho].append(
                compute_metrics(_r_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["CRN"][rho].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te), E_te)["pehe"])
            results["RMSN"][rho].append(
                compute_metrics(_rmsn_learner(X_tr, T_tr, Y_tr, X_te), E_te)["pehe"])
            results["ICG-HVRT"][rho].append(
                compute_metrics(_icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te), E_te)["pehe"])
            print(".", end="", flush=True)
        print()

    return results


def plot_leak_sensitivity(results, save_path="leak_sensitivity.png"):
    """
    Line plot: sqrt-PEHE vs rho_leak for each method, with +/-1 std band.

    Shows how quickly each method picks up the partial confounder signal
    as U leaks progressively more into E[X_{i,t}].
    """
    leak_colors = {m: _COLORS[m] for m in LEAK_METHODS}

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in LEAK_METHODS:
        means = [float(np.mean(results[m][rho])) for rho in LEAK_RHOS]
        stds  = [float(np.std(results[m][rho]))  for rho in LEAK_RHOS]
        ax.plot(LEAK_RHOS, means, "o-", label=m, color=leak_colors[m], lw=2, markersize=7)
        lo = [m_ - s for m_, s in zip(means, stds)]
        hi = [m_ + s for m_, s in zip(means, stds)]
        ax.fill_between(LEAK_RHOS, lo, hi, alpha=0.15, color=leak_colors[m])

    ax.set_xlabel("rho_leak  (0 = U invisible in X,  1 = U fully leaks into E[X])")
    ax.set_ylabel("sqrt-PEHE (lower is better)")
    ax.set_title(
        "Sensitivity to partial confounder leak into X\n"
        f"({N_SEEDS_LEAK} seeds,  {N_TRAIN} train / {N_TEST} test / {N_OBS} obs)"
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Many-weak-leaks sweep                                                  #
# ═══════════════════════════════════════════════════════════════════════ #

def run_multi_leak_sweep():
    """
    Evaluate MULTI_LEAK_METHODS as K (number of independent weak confounders) grows.

    Each confounder leaks at rho=MULTI_LEAK_RHO.  tau_i is normalised so std(tau)~1
    regardless of K, making PEHE values comparable across K.

    Expected pattern:
      ICG-HVRT  -- d_mu aggregates K Euclidean signals; PEHE decreases with K.
      GRU-NoAdv -- GRU aggregates but no adversarial loss; should also improve.
      CRN       -- adversarial removes more from H as K grows; may plateau or degrade.
      R-Learner -- deconfounds via mean(X) in F; should also improve with K.
      S-Learner -- confounding-blind; performance roughly flat.
    """
    results = {m: {K: [] for K in MULTI_LEAK_K} for m in MULTI_LEAK_METHODS}

    for K in MULTI_LEAK_K:
        gen = gen_multi_leak(K, rho=MULTI_LEAK_RHO)
        d   = max(K, 2)
        print(f"  K={K} (d={d})", end="", flush=True)
        for s in range(N_SEEDS_MULTI):
            np.random.seed(s * 137 + 7)
            torch.manual_seed(s * 137 + 7)
            X_tr, T_tr, Y_tr, _   = gen(N_TRAIN)
            np.random.seed(s * 137 + 7 + 100_000)
            X_te, T_te, Y_te, E_te = gen(N_TEST)

            F_tr, Tbar_tr, Ybar_tr = patient_summaries(X_tr, T_tr, Y_tr)
            F_te, _,       _        = patient_summaries(X_te, T_te, Y_te)

            results["S-Learner"][K].append(
                compute_metrics(_s_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["R-Learner"][K].append(
                compute_metrics(_r_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["CRN"][K].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te, lam_adv=0.5), E_te)["pehe"])
            results["GRU-NoAdv"][K].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te, lam_adv=0.0), E_te)["pehe"])
            results["ICG-HVRT"][K].append(
                compute_metrics(_icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te), E_te)["pehe"])
            print(".", end="", flush=True)
        print()

    return results


def plot_multi_leak(results, save_path="multi_leak.png"):
    """
    Two-panel figure:
      Left  -- raw sqrt-PEHE vs K for each method.
      Right -- delta sqrt-PEHE vs GRU-NoAdv baseline (isolates adversarial effect).
    """
    multi_colors = {m: _COLORS.get(m, "#95A5A6") for m in MULTI_LEAK_METHODS}
    Ks = MULTI_LEAK_K

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: raw PEHE vs K ──────────────────────────────────────────── #
    ax = axes[0]
    for m in MULTI_LEAK_METHODS:
        means = [float(np.mean(results[m][K])) for K in Ks]
        stds  = [float(np.std( results[m][K])) for K in Ks]
        ls    = "--" if m == "GRU-NoAdv" else "-"
        ax.plot(Ks, means, "o" + ls, label=m, color=multi_colors[m], lw=2, markersize=7)
        lo = [mu - s for mu, s in zip(means, stds)]
        hi = [mu + s for mu, s in zip(means, stds)]
        ax.fill_between(Ks, lo, hi, alpha=0.12, color=multi_colors[m])
    ax.set_xlabel(f"K  (number of independent confounders,  rho={MULTI_LEAK_RHO:.1f} each)")
    ax.set_ylabel("sqrt-PEHE  (lower is better)")
    ax.set_title("Many-weak-leaks: PEHE vs K")
    ax.set_xticks(Ks)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Right: delta PEHE vs GRU-NoAdv (adversarial cost/benefit) ────── #
    ax2 = axes[1]
    noAdv_means = [float(np.mean(results["GRU-NoAdv"][K])) for K in Ks]
    ax2.axhline(0, color=multi_colors["GRU-NoAdv"], lw=1.5, ls="--", label="GRU-NoAdv (baseline)")
    for m in MULTI_LEAK_METHODS:
        if m == "GRU-NoAdv":
            continue
        means = [float(np.mean(results[m][K])) for K in Ks]
        delta = [means[i] - noAdv_means[i] for i in range(len(Ks))]
        ax2.plot(Ks, delta, "o-", label=m, color=multi_colors[m], lw=2, markersize=7)
    ax2.set_xlabel("K  (number of independent confounders)")
    ax2.set_ylabel("delta sqrt-PEHE vs GRU-NoAdv  (positive = worse than ablation)")
    ax2.set_title("Adversarial cost/benefit: delta PEHE vs GRU-NoAdv")
    ax2.set_xticks(Ks)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"Many-weak-leaks  (rho={MULTI_LEAK_RHO:.1f} per confounder, {N_SEEDS_MULTI} seeds)\n"
        "Does CRN adversarial deconfounding help or hurt as K grows?",
        fontsize=10, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Noise resilience sweep                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

def gen_noise_resilience(sigma_x: float):
    """
    Mean-confounded DGP (rho=1.0) with variable observation noise on X.

    U_i ~ N(0,1);  tau_i = U_i;  E[X_{i,t}] = U_i * ones(d).
    X_{i,t} = U_i * ones(d) + sigma_x * eps_t,  eps_t ~ N(0, I_d).
    T_{i,t} ~ N(0.5 * U_i, 1);  Y_{i,t} = tau_i * T_{i,t} + noise.

    At sigma_x=1.0 this is identical to gen_mean_confounded.

    ICG-HVRT estimates mu_i from N=100 obs: std error ~ sigma_x/sqrt(N).
    With N=100,  d_mu is informative while sigma_x << 10*||U_i - U_j|| ~ 14.
    CRN uses GRU mean-pool over noisy X sequences: similar O(sigma_x/sqrt(N))
    averaging, but gradient-based training may overfit noise at high sigma_x.
    Cross-sectional methods use mean(X) in F: also O(sigma_x/sqrt(N)) robust,
    but lose higher-order covariance signal faster.
    """
    d = 2
    def _gen(n):
        X, T, Y, E = [], [], [], []
        for _ in range(n):
            U   = np.random.normal(0, 1)
            tau = float(U)
            x   = np.random.normal(U * np.ones(d), sigma_x, (N_OBS, d))
            t   = np.random.normal(0.5 * U, 1.0, (N_OBS, 1))
            X.append(x); T.append(t)
            Y.append(tau * t.flatten() + np.random.normal(0, 0.5, N_OBS))
            E.append(tau)
        return X, T, Y, np.array(E)
    return _gen


def run_noise_sweep():
    """
    Evaluate MULTI_LEAK_METHODS as observation noise sigma_x on X scales up.

    Expected resilience ordering:
      ICG-HVRT  -- geometry computed from closed-form sufficient statistics;
                   mu estimated with error O(sigma_x/sqrt(N)).
      GRU-NoAdv -- mean-pool over sequence; similar averaging benefit.
      CRN       -- same averaging BUT adversarial gradient may destabilise
                   under high-noise observations.
      R-Learner -- uses mean(X) in F; O(sigma_x/sqrt(N)) robust.
      S-Learner -- similar to R-Learner for noise; both cross-sectional.
    """
    results = {m: {s: [] for s in NOISE_SIGMAS} for m in MULTI_LEAK_METHODS}

    for sig in NOISE_SIGMAS:
        gen = gen_noise_resilience(sig)
        print(f"  sigma_x={sig:.1f}", end="", flush=True)
        for s in range(N_SEEDS_NOISE):
            np.random.seed(s * 137 + 7)
            torch.manual_seed(s * 137 + 7)
            X_tr, T_tr, Y_tr, _   = gen(N_TRAIN)
            np.random.seed(s * 137 + 7 + 100_000)
            X_te, T_te, Y_te, E_te = gen(N_TEST)

            F_tr, Tbar_tr, Ybar_tr = patient_summaries(X_tr, T_tr, Y_tr)
            F_te, _,       _        = patient_summaries(X_te, T_te, Y_te)

            results["S-Learner"][sig].append(
                compute_metrics(_s_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["R-Learner"][sig].append(
                compute_metrics(_r_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["CRN"][sig].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te, lam_adv=0.5), E_te)["pehe"])
            results["GRU-NoAdv"][sig].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te, lam_adv=0.0), E_te)["pehe"])
            results["ICG-HVRT"][sig].append(
                compute_metrics(_icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te), E_te)["pehe"])
            print(".", end="", flush=True)
        print()

    return results


def plot_noise_resilience(results, save_path="noise_resilience.png"):
    """
    Line plot: sqrt-PEHE vs sigma_x (observation noise on X) for each method.
    Flatter lines = more noise-resilient.
    """
    noise_colors = {m: _COLORS.get(m, "#95A5A6") for m in MULTI_LEAK_METHODS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: raw PEHE vs sigma_x ────────────────────────────────────── #
    ax = axes[0]
    for m in MULTI_LEAK_METHODS:
        means = [float(np.mean(results[m][s])) for s in NOISE_SIGMAS]
        stds  = [float(np.std( results[m][s])) for s in NOISE_SIGMAS]
        ls    = "--" if m == "GRU-NoAdv" else "-"
        ax.plot(NOISE_SIGMAS, means, "o" + ls, label=m, color=noise_colors[m], lw=2, markersize=7)
        lo = [mu - sd for mu, sd in zip(means, stds)]
        hi = [mu + sd for mu, sd in zip(means, stds)]
        ax.fill_between(NOISE_SIGMAS, lo, hi, alpha=0.12, color=noise_colors[m])
    ax.set_xlabel("sigma_x  (observation noise on X,  N=100 obs per patient)")
    ax.set_ylabel("sqrt-PEHE  (lower is better)")
    ax.set_title("Noise resilience: PEHE vs observation noise")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Right: PEHE normalised by sigma_x=1 baseline ─────────────────── #
    ax2 = axes[1]
    ref_sig = 1.0
    for m in MULTI_LEAK_METHODS:
        ref   = float(np.mean(results[m][ref_sig]))
        means = [float(np.mean(results[m][s])) / ref for s in NOISE_SIGMAS]
        ls    = "--" if m == "GRU-NoAdv" else "-"
        ax2.plot(NOISE_SIGMAS, means, "o" + ls, label=m, color=noise_colors[m], lw=2, markersize=7)
    ax2.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.6)
    ax2.set_xlabel("sigma_x  (observation noise on X)")
    ax2.set_ylabel("sqrt-PEHE / PEHE(sigma_x=1)  (1.0 = no degradation)")
    ax2.set_title("Relative noise degradation  (flat line = perfectly resilient)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"Noise resilience: observation noise on X  ({N_SEEDS_NOISE} seeds,  "
        f"N={N_OBS} obs/patient)\n"
        "Mean-confounded DGP  (rho=1.0: confounder fully observed in E[X])",
        fontsize=10, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Prognostic confounder rho sweep                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def run_prognostic_rho_sweep():
    """
    Evaluate PROG_METHODS as rho_x (U leak strength into X) varies from 0 to 1.

    At rho_x=0  -> X perp U -> all methods fail (pure hidden heterogeneity).
    At rho_x=1  -> mean(X) ~ U_i -> strong signal; all X-using methods improve.
    T is always randomised (T perp U), so deconfounding methods gain nothing
    from attempting to remove U->T correlation -- they are in 'idle' mode.

    This tests pure heterogeneity recovery via covariate signal, without any
    selection confounding to cloud the picture.
    """
    results = {m: {r: [] for r in PROG_RHO_VALUES} for m in PROG_METHODS}

    for rho_x in PROG_RHO_VALUES:
        gen = gen_prognostic_rho(rho_x)
        print(f"  rho_x={rho_x:.2f}  (mean(X) ~ {rho_x:.2f}*U_i,  std(tau)=1.0)",
              end="", flush=True)
        for s in range(N_SEEDS_PROG):
            np.random.seed(s * 137 + 13)
            torch.manual_seed(s * 137 + 13)
            X_tr, T_tr, Y_tr, _    = gen(N_TRAIN)
            np.random.seed(s * 137 + 13 + 100_000)
            X_te, T_te, Y_te, E_te = gen(N_TEST)

            F_tr, Tbar_tr, Ybar_tr = patient_summaries(X_tr, T_tr, Y_tr)
            F_te, _,       _        = patient_summaries(X_te, T_te, Y_te)

            results["S-Learner"][rho_x].append(
                compute_metrics(_s_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["R-Learner"][rho_x].append(
                compute_metrics(_r_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["CRN"][rho_x].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te), E_te)["pehe"])
            results["ICG-HVRT"][rho_x].append(
                compute_metrics(_icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te), E_te)["pehe"])
            results["ICG-HART"][rho_x].append(
                compute_metrics(_icg_hart(X_tr, T_tr, Y_tr, X_te, T_te), E_te)["pehe"])
            print(".", end="", flush=True)
        print()

    return results


def plot_prognostic_rho_sweep(results, save_path="prognostic_rho_sweep.png"):
    """
    Line plot: sqrt-PEHE vs rho_x for each method.

    rho_x=0 -> all methods converge to std(tau) ~ 1.0 (hidden heterogeneity).
    Rising curves at rho_x>0 indicate a method is exploiting the covariate leak.
    Flat curves indicate the method cannot use X to detect tau heterogeneity.
    """
    prog_colors = {m: _COLORS.get(m, "#95A5A6") for m in PROG_METHODS}
    rhos = PROG_RHO_VALUES

    fig, ax = plt.subplots(figsize=(9, 5))
    for m in PROG_METHODS:
        means = [float(np.mean(results[m][r])) for r in rhos]
        stds  = [float(np.std( results[m][r])) for r in rhos]
        ax.plot(rhos, means, "o-", label=m, color=prog_colors[m], lw=2, markersize=7)
        lo = [mu - s for mu, s in zip(means, stds)]
        hi = [mu + s for mu, s in zip(means, stds)]
        ax.fill_between(rhos, lo, hi, alpha=0.15, color=prog_colors[m])

    ax.set_xlabel(
        "rho_x  (U leak strength into E[X];  T is randomised, T perp U)\n"
        "  rho_x=0: fully hidden heterogeneity        rho_x=1: full mean leak -->"
    )
    ax.set_ylabel("sqrt-PEHE  (lower is better)")
    ax.set_title(
        "Prognostic confounder: heterogeneity recovery vs X-leak strength\n"
        "T randomised -- no selection confounding; challenge is pure effect-modification"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Sparse-K individual feature leak sweep                                 #
# ═══════════════════════════════════════════════════════════════════════ #

def run_sparse_k_sweep():
    """
    Evaluate SPARSE_LEAK_METHODS as K (number of spike obs carrying U signal)
    varies from 1 (extremely sparse) to N_OBS (full aggregate leakage).

    All DGPs use the same single-feature leak (feature 0) with leak_amp=8.0.

    Expected pattern:
      K=N_OBS  -- everyone sees strong mean(X[:,0]) = 8*U_i -> all methods good.
      K small  -- aggregate signal = 8*U_i*K/N_OBS -> S/R-Learner degrade fast.
                  GRU-based (CRN): processes individual timesteps -> more robust.
                  ICG-HVRT (cone): spike distorts sample cov -> partial signal.
                  ICG-HART (pyramid): MAD-robust -> spike invisible -> degrades.

    This reveals which methods exploit individual-observation structure vs
    requiring the confounder to appear in patient-level aggregate statistics.
    """
    results = {m: {K: [] for K in SPARSE_K_VALUES} for m in SPARSE_LEAK_METHODS}

    for K in SPARSE_K_VALUES:
        gen = gen_sparse_k_leak(K, leak_amp=8.0)
        agg_pct = K / N_OBS * 100
        print(f"  K={K:3d}  (mean leak = {8.0*K/N_OBS:.2f}*U_i,  {agg_pct:.0f}% of obs)",
              end="", flush=True)
        for s in range(N_SEEDS_SPARSE):
            np.random.seed(s * 137 + 7)
            torch.manual_seed(s * 137 + 7)
            X_tr, T_tr, Y_tr, _    = gen(N_TRAIN)
            np.random.seed(s * 137 + 7 + 100_000)
            X_te, T_te, Y_te, E_te = gen(N_TEST)

            F_tr, Tbar_tr, Ybar_tr = patient_summaries(X_tr, T_tr, Y_tr)
            F_te, _,       _        = patient_summaries(X_te, T_te, Y_te)

            results["S-Learner"][K].append(
                compute_metrics(_s_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["R-Learner"][K].append(
                compute_metrics(_r_learner(F_tr, Tbar_tr, Ybar_tr, F_te), E_te)["pehe"])
            results["CRN"][K].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te), E_te)["pehe"])
            results["ICG-HVRT"][K].append(
                compute_metrics(_icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te), E_te)["pehe"])
            results["ICG-HART"][K].append(
                compute_metrics(_icg_hart(X_tr, T_tr, Y_tr, X_te, T_te), E_te)["pehe"])
            print(".", end="", flush=True)
        print()

    return results


def plot_sparse_k_sweep(results, save_path="sparse_k_sweep.png"):
    """
    Line plot: sqrt-PEHE vs K (log x-axis) for each method.

    Flatter curves at small K -> method exploits individual-observation signal.
    Sharp increase at small K -> method requires aggregate (mean) signal only.
    """
    sparse_colors = {m: _COLORS.get(m, "#95A5A6") for m in SPARSE_LEAK_METHODS}
    Ks = SPARSE_K_VALUES

    fig, ax = plt.subplots(figsize=(9, 5))
    for m in SPARSE_LEAK_METHODS:
        means = [float(np.mean(results[m][K])) for K in Ks]
        stds  = [float(np.std( results[m][K])) for K in Ks]
        ax.plot(Ks, means, "o-", label=m, color=sparse_colors[m], lw=2, markersize=7)
        lo = [mu - s for mu, s in zip(means, stds)]
        hi = [mu + s for mu, s in zip(means, stds)]
        ax.fill_between(Ks, lo, hi, alpha=0.15, color=sparse_colors[m])

    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlabel(
        f"K  (# obs carrying U signal per patient,  leak_amp=8.0,  N_OBS={N_OBS})\n"
        "  <-- sparse individual-level        aggregate (all obs) -->"
    )
    ax.set_ylabel("sqrt-PEHE  (lower is better)")
    ax.set_title(
        "Individual vs aggregate feature leak: PEHE vs sparsity K\n"
        "Single-feature confound (feature 0 only); ICG-HART uses MAD-robust matching"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Main                                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("=" * 72)
    print("  ITE COMPARISON: ICG-HVRT/HART vs S/R-Learner vs CRN vs RMSN under confounding")
    print(f"  {N_SEEDS} seeds x {len(TESTS)} DGPs  |  {N_TRAIN} train / {N_TEST} test / {N_OBS} obs per patient")
    print("=" * 72)

    all_agg, all_pool = {}, {}
    for name, gen in TESTS:
        agg, pool         = evaluate_dgp(name, gen)
        all_agg[name]     = agg
        all_pool[name]    = pool

    dgp_names    = [n for n, _ in TESTS]
    col_w, val_w = 18, 12

    # ── Summary: sqrt-PEHE ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY  --  sqrt-PEHE mean across seeds  (lower is better)")
    print("=" * 72 + "\n")
    hdr = f"{'Method':<{col_w}}" + "".join(f"{n[:9]:>{val_w}}" for n in dgp_names)
    print(hdr + f"{'MEAN':>{val_w}}")
    print("-" * (col_w + val_w * (len(dgp_names) + 1)))
    for m in METHODS:
        vals = [all_agg[n][m]["pehe"][0] for n in dgp_names]
        print(f"{m:<{col_w}}" + "".join(f"{v:>{val_w}.4f}" for v in vals)
              + f"{np.mean(vals):>{val_w}.4f}")

    # ── Summary: NMAE ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY  --  NMAE = MAE / std(tau_true)  (scale-free, lower is better)")
    print("=" * 72 + "\n")
    hdr = f"{'Method':<{col_w}}" + "".join(f"{n[:9]:>{val_w}}" for n in dgp_names)
    print(hdr + f"{'MEAN':>{val_w}}")
    print("-" * (col_w + val_w * (len(dgp_names) + 1)))
    for m in METHODS:
        vals = [all_agg[n][m]["nmae"][0] for n in dgp_names]
        print(f"{m:<{col_w}}" + "".join(f"{v:>{val_w}.4f}" for v in vals)
              + f"{np.mean(vals):>{val_w}.4f}")

    # ── Summary: Spearman rho ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY  --  Spearman rho mean across seeds  (higher is better)")
    print("=" * 72 + "\n")
    hdr = f"{'Method':<{col_w}}" + "".join(f"{n[:9]:>{val_w}}" for n in dgp_names)
    print(hdr + f"{'MEAN':>{val_w}}")
    print("-" * (col_w + val_w * (len(dgp_names) + 1)))
    for m in METHODS:
        vals = [all_agg[n][m]["spearman"][0] for n in dgp_names]
        print(f"{m:<{col_w}}" + "".join(f"{v:>{val_w}.4f}" for v in vals)
              + f"{np.mean(vals):>{val_w}.4f}")

    # ── Summary: |Bias| ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY  --  |Bias| mean across seeds  (lower is better)")
    print("=" * 72 + "\n")
    hdr = f"{'Method':<{col_w}}" + "".join(f"{n[:9]:>{val_w}}" for n in dgp_names)
    print(hdr + f"{'MEAN':>{val_w}}")
    print("-" * (col_w + val_w * (len(dgp_names) + 1)))
    for m in METHODS:
        vals = [abs(all_agg[n][m]["bias"][0]) for n in dgp_names]
        print(f"{m:<{col_w}}" + "".join(f"{v:>{val_w}.4f}" for v in vals)
              + f"{np.mean(vals):>{val_w}.4f}")

    # ── Confounding impact ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CONFOUNDING IMPACT  --  delta sqrt-PEHE vs Randomised (positive = hurt)")
    print("=" * 72 + "\n")
    hdr = f"{'Method':<{col_w}}" + "".join(f"{n[:13]:>{val_w}}" for n in dgp_names[1:])
    print(hdr)
    print("-" * (col_w + val_w * len(dgp_names[1:])))
    for m in METHODS:
        base = all_agg["Randomised"][m]["pehe"][0]
        print(f"{m:<{col_w}}"
              + "".join(f"{all_agg[n][m]['pehe'][0] - base:>+{val_w}.4f}"
                        for n in dgp_names[1:]))

    print("\n  Within-Patient uses test-patient Y -- not valid pre-treatment.")
    print("  S/R-Learner: features F = [mean(X), cov(X)]. Operate on patient summaries.")
    print("  CRN: GRU over full X sequence + adversarial T_bar balancing.")
    print("  RMSN: two-stage GRU with IPW weighting (propensity residuals -> stabilized weights).")
    print("  ICG-HVRT: cooperative geometry profile matching + local Ridge.")

    # ── Main comparison plot ─────────────────────────────────────────
    print("\n[Generating comparison plot...]")
    plot_comparison(all_agg, save_path="ite_comparison.png")

    # ── Partial leak sweep ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  PARTIAL LEAK SWEEP  --  rho_leak in {LEAK_RHOS}")
    print(f"  {N_SEEDS_LEAK} seeds x {len(LEAK_RHOS)} rho values x {len(LEAK_METHODS)} methods")
    print("=" * 72 + "\n")
    leak_results = run_leak_sweep()

    print("\n  sqrt-PEHE by rho_leak:")
    rho_hdr = f"{'Method':<{col_w}}" + "".join(f"  rho={r:.1f}" for r in LEAK_RHOS)
    print(rho_hdr)
    print("-" * (col_w + 9 * len(LEAK_RHOS)))
    for m in LEAK_METHODS:
        row = f"{m:<{col_w}}" + "".join(
            f"  {float(np.mean(leak_results[m][r])):>6.4f}" for r in LEAK_RHOS)
        print(row)

    print("\n[Generating leak sensitivity plot...]")
    plot_leak_sensitivity(leak_results, save_path="leak_sensitivity.png")

    # ── Many-weak-leaks sweep ────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  MANY-WEAK-LEAKS SWEEP  --  K in {MULTI_LEAK_K},  rho={MULTI_LEAK_RHO:.1f} each")
    print(f"  {N_SEEDS_MULTI} seeds x {len(MULTI_LEAK_K)} K values x {len(MULTI_LEAK_METHODS)} methods")
    print("=" * 72 + "\n")
    multi_results = run_multi_leak_sweep()

    print("\n  sqrt-PEHE by K:")
    k_hdr = f"{'Method':<{col_w}}" + "".join(f"      K={K}" for K in MULTI_LEAK_K)
    print(k_hdr)
    print("-" * (col_w + 9 * len(MULTI_LEAK_K)))
    for m in MULTI_LEAK_METHODS:
        row = f"{m:<{col_w}}" + "".join(
            f"  {float(np.mean(multi_results[m][K])):>6.4f}" for K in MULTI_LEAK_K)
        print(row)

    print("\n  delta vs GRU-NoAdv (positive = worse than no-adversarial ablation):")
    print(k_hdr)
    print("-" * (col_w + 9 * len(MULTI_LEAK_K)))
    noAdv_means = {K: float(np.mean(multi_results["GRU-NoAdv"][K])) for K in MULTI_LEAK_K}
    for m in MULTI_LEAK_METHODS:
        if m == "GRU-NoAdv":
            continue
        row = f"{m:<{col_w}}" + "".join(
            f"  {float(np.mean(multi_results[m][K])) - noAdv_means[K]:>+6.4f}"
            for K in MULTI_LEAK_K)
        print(row)

    print("\n[Generating many-weak-leaks plot...]")
    plot_multi_leak(multi_results, save_path="multi_leak.png")

    # ── Noise resilience sweep ───────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  NOISE RESILIENCE SWEEP  --  sigma_x in {NOISE_SIGMAS}")
    print(f"  {N_SEEDS_NOISE} seeds x {len(NOISE_SIGMAS)} sigma values x {len(MULTI_LEAK_METHODS)} methods")
    print("=" * 72 + "\n")
    noise_results = run_noise_sweep()

    print("\n  sqrt-PEHE by sigma_x:")
    s_hdr = f"{'Method':<{col_w}}" + "".join(f"  sig={s:.1f}" for s in NOISE_SIGMAS)
    print(s_hdr)
    print("-" * (col_w + 9 * len(NOISE_SIGMAS)))
    for m in MULTI_LEAK_METHODS:
        row = f"{m:<{col_w}}" + "".join(
            f"  {float(np.mean(noise_results[m][s])):>6.4f}" for s in NOISE_SIGMAS)
        print(row)

    print("\n[Generating noise resilience plot...]")
    plot_noise_resilience(noise_results, save_path="noise_resilience.png")

    # ── Many-weak-leaks re-run: d_mu_coop vs d_mu_L2 ─────────────────
    print("\n" + "=" * 72)
    print("  COOPERATIVE MEAN DISTANCE VALIDATION")
    print("  Re-run many-weak-leaks with ICG-HVRT-v2 (d_mu_coop) vs v1 (L2).")
    print("  Theoretical prediction: v2 PEHE constant w.r.t. K; v1 degrades.")
    print("=" * 72 + "\n")

    coop_methods = ["ICG-HVRT-v1", "ICG-HVRT-v2", "CRN", "GRU-NoAdv"]
    coop_results = {m: {K: [] for K in MULTI_LEAK_K} for m in coop_methods}

    for K in MULTI_LEAK_K:
        gen = gen_multi_leak(K, rho=MULTI_LEAK_RHO)
        print(f"  K={K} (d={max(K,2)})", end="", flush=True)
        for s in range(N_SEEDS_MULTI):
            np.random.seed(s * 137 + 7)
            torch.manual_seed(s * 137 + 7)
            X_tr, T_tr, Y_tr, _   = gen(N_TRAIN)
            np.random.seed(s * 137 + 7 + 100_000)
            X_te, T_te, Y_te, E_te = gen(N_TEST)

            # ICG-HVRT v1: alpha_levels_perp=0 -> pure d_mu_coop-weighted,
            # but to really simulate the old L2 we set perp weight = 1.0 (equal)
            coop_results["ICG-HVRT-v1"][K].append(
                compute_metrics(_icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te,
                                          gamma_levels_perp=1.0), E_te)["pehe"])
            coop_results["ICG-HVRT-v2"][K].append(
                compute_metrics(_icg_hvrt(X_tr, T_tr, Y_tr, X_te, T_te,
                                          gamma_levels_perp=0.0), E_te)["pehe"])
            coop_results["CRN"][K].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te, lam_adv=0.5), E_te)["pehe"])
            coop_results["GRU-NoAdv"][K].append(
                compute_metrics(_crn_learner(X_tr, T_tr, Y_tr, X_te, lam_adv=0.0), E_te)["pehe"])
            print(".", end="", flush=True)
        print()

    print("\n  sqrt-PEHE by K (ICG-HVRT-v1 = equal perp weight; v2 = coop-only):")
    k_hdr2 = f"{'Method':<18}" + "".join(f"      K={K}" for K in MULTI_LEAK_K)
    print(k_hdr2)
    print("-" * (18 + 9 * len(MULTI_LEAK_K)))
    for m in coop_methods:
        row = f"{m:<18}" + "".join(
            f"  {float(np.mean(coop_results[m][K])):>6.4f}" for K in MULTI_LEAK_K)
        print(row)

    # Visualise the comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    coop_colors = {
        "ICG-HVRT-v1": "#95A5A6",   # grey (old)
        "ICG-HVRT-v2": "#2ECC71",   # green (new)
        "CRN":         "#9B59B6",
        "GRU-NoAdv":   "#E67E22",
    }
    for m in coop_methods:
        means = [float(np.mean(coop_results[m][K])) for K in MULTI_LEAK_K]
        stds  = [float(np.std( coop_results[m][K])) for K in MULTI_LEAK_K]
        ls    = "--" if m == "ICG-HVRT-v1" else "-"
        ax.plot(MULTI_LEAK_K, means, "o" + ls, label=m,
                color=coop_colors[m], lw=2.2, markersize=8)
        lo = [mu - s for mu, s in zip(means, stds)]
        hi = [mu + s for mu, s in zip(means, stds)]
        ax.fill_between(MULTI_LEAK_K, lo, hi, alpha=0.12, color=coop_colors[m])
    ax.set_xlabel(f"K  (number of independent confounders,  rho={MULTI_LEAK_RHO:.1f})")
    ax.set_ylabel("sqrt-PEHE  (lower is better)")
    ax.set_title(
        "d_mu_coop vs d_mu (L2): many-weak-leaks experiment\n"
        "v1 = equal perp weight (old L2 behaviour);  "
        "v2 = cooperative projection only (tau-correlated)",
        fontsize=9,
    )
    ax.set_xticks(MULTI_LEAK_K)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("coop_distance_validation.png", dpi=130, bbox_inches="tight")
    print("\n  Saved: coop_distance_validation.png")

    # ── Sparse-K individual feature leak sweep ───────────────────────
    print("\n" + "=" * 72)
    print(f"  SPARSE-K INDIVIDUAL FEATURE LEAK SWEEP  --  K in {SPARSE_K_VALUES}")
    print(f"  {N_SEEDS_SPARSE} seeds x {len(SPARSE_K_VALUES)} K values x {len(SPARSE_LEAK_METHODS)} methods")
    print("  Feature 0 only; leak_amp=8.0.  K=1 -> nearly hidden; K=100 -> full mean leak.")
    print("=" * 72 + "\n")
    sparse_results = run_sparse_k_sweep()

    print("\n  sqrt-PEHE by K (lower is better):")
    sk_hdr = f"{'Method':<{col_w}}" + "".join(f"     K={K}" for K in SPARSE_K_VALUES)
    print(sk_hdr)
    print("-" * (col_w + 9 * len(SPARSE_K_VALUES)))
    for m in SPARSE_LEAK_METHODS:
        row = f"{m:<{col_w}}" + "".join(
            f"  {float(np.mean(sparse_results[m][K])):>6.4f}" for K in SPARSE_K_VALUES)
        print(row)

    print("\n[Generating sparse-K sweep plot...]")
    plot_sparse_k_sweep(sparse_results, save_path="sparse_k_sweep.png")

    # ── Prognostic confounder rho sweep ──────────────────────────────
    print("\n" + "=" * 72)
    print(f"  PROGNOSTIC CONFOUNDER SWEEP  --  rho_x in {PROG_RHO_VALUES}")
    print(f"  {N_SEEDS_PROG} seeds x {len(PROG_RHO_VALUES)} rho values x {len(PROG_METHODS)} methods")
    print("  T perp U (randomised); U leaks into mean(X) and directly affects Y.")
    print("  Tests pure heterogeneity recovery via covariate signal.")
    print("=" * 72 + "\n")
    prog_results = run_prognostic_rho_sweep()

    print("\n  sqrt-PEHE by rho_x (lower is better):")
    pr_hdr = f"{'Method':<{col_w}}" + "".join(f"  r={r:.2f}" for r in PROG_RHO_VALUES)
    print(pr_hdr)
    print("-" * (col_w + 9 * len(PROG_RHO_VALUES)))
    for m in PROG_METHODS:
        row = f"{m:<{col_w}}" + "".join(
            f"  {float(np.mean(prog_results[m][r])):>6.4f}" for r in PROG_RHO_VALUES)
        print(row)

    print("\n[Generating prognostic rho sweep plot...]")
    plot_prognostic_rho_sweep(prog_results, save_path="prognostic_rho_sweep.png")

    print("\nDone.")
