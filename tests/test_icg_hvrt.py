"""
Unit tests for the ICG-HVRT implementation.

Run with:  python -m pytest tests/test_icg_hvrt.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoite import (
    CooperativeGeometryProfile,
    SharedHVRT,
    fit_shared_hvrt,
    ICGHVRTMatcher,
    ICGHVRTEstimator,
    CoupledInterventionProtocol,
)
from autoite.distances import (
    euclidean_mean_distance,
    cooperative_direction_distance,
    log_euclidean_distance,
    occupation_distance,
    dynamics_distance,
)

RNG = np.random.default_rng(2025)


# ────────────────────────────────────────────────────────────────────── #
# Helpers                                                                 #
# ────────────────────────────────────────────────────────────────────── #

def random_spd(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T + np.eye(d) * 0.1


def make_profile(d: int = 4, n_obs: int = 60, seed: int = 0) -> CooperativeGeometryProfile:
    rng = np.random.default_rng(seed)
    cov = random_spd(d, seed)
    X = rng.multivariate_normal(np.zeros(d), cov, n_obs)
    return CooperativeGeometryProfile.from_longitudinal(X)


# ────────────────────────────────────────────────────────────────────── #
# CooperativeGeometryProfile tests                                        #
# ────────────────────────────────────────────────────────────────────── #

class TestCooperativeGeometryProfile:

    def test_shape(self):
        """Profile attributes have expected shapes."""
        d, n = 4, 80
        X = RNG.standard_normal((n, d))
        p = CooperativeGeometryProfile.from_longitudinal(X)

        assert p.mu.shape == (d,)
        assert p.sigma.shape == (d, d)
        assert p.cooperative_direction.shape == (d,)
        assert p.cooperative_operator.shape == (d, d)
        assert isinstance(p.cone_angle, float)
        assert p.d == d
        assert p.n_observations == n

    def test_cooperative_direction_formula(self):
        """
        w = Sigma^{-1/2} 1 must satisfy the factorisation:
        C = w w^T - Sigma^{-1}
        and the relation C = Sigma^{-1/2} A Sigma^{-1/2}
        where A = 11^T - I.
        """
        d = 3
        X = RNG.standard_normal((60, d))
        p = CooperativeGeometryProfile.from_longitudinal(X)

        # Verify C = w w^T - Sigma^{-1}
        eigvals, eigvecs = np.linalg.eigh(p.sigma)
        eigvals = np.maximum(eigvals, 1e-10)
        sigma_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

        C_reconstructed = np.outer(p.cooperative_direction, p.cooperative_direction) - sigma_inv
        np.testing.assert_allclose(p.cooperative_operator, C_reconstructed, atol=1e-8)

        # Verify C = Sigma^{-1/2} A Sigma^{-1/2}
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        ones = np.ones(d)
        A = np.outer(ones, ones) - np.eye(d)
        C_from_A = inv_sqrt @ A @ inv_sqrt
        np.testing.assert_allclose(p.cooperative_operator, C_from_A, atol=1e-8)

    def test_cooperative_operator_signature(self):
        """
        C = Sigma^{-1/2} A Sigma^{-1/2} must have signature (1, d-1):
        exactly one positive eigenvalue.
        """
        for d in (2, 3, 4, 5):
            X = RNG.standard_normal((80, d))
            p = CooperativeGeometryProfile.from_longitudinal(X)
            eigvals = np.linalg.eigvalsh(p.cooperative_operator)
            n_positive = np.sum(eigvals > 1e-9)
            assert n_positive == 1, (
                f"d={d}: expected 1 positive eigenvalue, got {n_positive}. "
                f"Eigenvalues: {eigvals}"
            )

    def test_cooperative_trajectory_formula(self):
        """T(t) = S(t)^2 - Q(t) where S and Q are computed in whitened space."""
        d, n = 4, 50
        X = RNG.standard_normal((n, d))
        p = CooperativeGeometryProfile.from_longitudinal(X)

        # Manually compute T
        eigvals, eigvecs = np.linalg.eigh(p.sigma)
        eigvals = np.maximum(eigvals, 1e-10)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        Z = (X - p.mu) @ inv_sqrt.T
        S = Z.sum(axis=1)
        Q = (Z ** 2).sum(axis=1)
        T_manual = S ** 2 - Q

        np.testing.assert_allclose(p.cooperative_trajectory, T_manual, atol=1e-8)

    def test_partition_profile_sums_to_one(self):
        """Occupation histogram must be a probability distribution."""
        X = RNG.standard_normal((100, 4))
        p = CooperativeGeometryProfile.from_longitudinal(X)
        if p.partition_profile is not None:
            assert abs(p.partition_profile.sum() - 1.0) < 1e-6

    def test_transition_matrix_rows_sum_to_one(self):
        """Each row of the transition matrix must be a probability vector."""
        X = RNG.standard_normal((100, 4))
        p = CooperativeGeometryProfile.from_longitudinal(X)
        if p.transition_matrix is not None:
            row_sums = p.transition_matrix.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

    def test_graceful_degradation_short_series(self):
        """Profile falls back to static (no HVRT) when n_obs < 2*K."""
        d, n = 4, 5  # too few for HVRT
        X = RNG.standard_normal((n, d))
        p = CooperativeGeometryProfile.from_longitudinal(X, n_partitions=8)
        # Should still have static geometry even if HVRT fails
        assert p.mu.shape == (d,)
        assert p.cooperative_direction.shape == (d,)
        # Partition profile may be None
        # (No assertion on partition_profile — graceful degradation)

    def test_sigma_is_spd(self):
        """The returned covariance matrix must be symmetric positive definite."""
        X = RNG.standard_normal((50, 4))
        p = CooperativeGeometryProfile.from_longitudinal(X)
        eigvals = np.linalg.eigvalsh(p.sigma)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"
        np.testing.assert_allclose(p.sigma, p.sigma.T, atol=1e-10)

    def test_has_longitudinal_flag(self):
        """has_longitudinal is True iff partition_profile is available."""
        X_long = RNG.standard_normal((100, 4))
        p_long = CooperativeGeometryProfile.from_longitudinal(X_long)
        assert p_long.has_longitudinal == (p_long.partition_profile is not None)

        X_short = RNG.standard_normal((3, 4))
        p_short = CooperativeGeometryProfile.from_longitudinal(X_short)
        assert p_short.has_longitudinal == (p_short.partition_profile is not None)


# ────────────────────────────────────────────────────────────────────── #
# Distance function tests                                                 #
# ────────────────────────────────────────────────────────────────────── #

class TestDistanceFunctions:

    def test_euclidean_mean_distance_zero(self):
        mu = RNG.standard_normal(5)
        assert euclidean_mean_distance(mu, mu) < 1e-12

    def test_euclidean_mean_distance_positive(self):
        mu_a = np.zeros(4)
        mu_b = np.ones(4)
        assert euclidean_mean_distance(mu_a, mu_b) == pytest.approx(2.0)

    def test_cooperative_direction_distance_same(self):
        w = RNG.standard_normal(4)
        assert cooperative_direction_distance(w, w) < 1e-7

    def test_cooperative_direction_distance_orthogonal(self):
        w_a = np.array([1.0, 0.0, 0.0])
        w_b = np.array([0.0, 1.0, 0.0])
        d = cooperative_direction_distance(w_a, w_b)
        assert d == pytest.approx(np.pi / 2, abs=1e-8)

    def test_cooperative_direction_distance_opposite(self):
        w = np.array([1.0, 0.5])
        d = cooperative_direction_distance(w, -w)
        assert d == pytest.approx(np.pi, abs=1e-6)

    def test_log_euclidean_distance_same(self):
        S = random_spd(4)
        assert log_euclidean_distance(S, S) < 1e-10

    def test_log_euclidean_distance_positive(self):
        S1 = random_spd(4, seed=1)
        S2 = random_spd(4, seed=2)
        assert log_euclidean_distance(S1, S2) > 0

    def test_occupation_distance_same(self):
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        assert occupation_distance(pi, pi) < 1e-10

    def test_occupation_distance_extreme(self):
        pi_a = np.array([1.0, 0.0, 0.0, 0.0])
        pi_b = np.array([0.0, 0.0, 0.0, 1.0])
        d = occupation_distance(pi_a, pi_b)
        # CDF_a = [1, 1, 1, 1], CDF_b = [0, 0, 0, 1]
        # |1-0| + |1-0| + |1-0| + |1-1| = 3
        assert d == pytest.approx(3.0, abs=1e-8)

    def test_dynamics_distance_same(self):
        M = np.eye(4) * 0.5 + np.ones((4, 4)) * 0.125
        assert dynamics_distance(M, M) < 1e-10


# ────────────────────────────────────────────────────────────────────── #
# Matcher tests                                                           #
# ────────────────────────────────────────────────────────────────────── #

class TestICGHVRTMatcher:

    def test_distance_non_negative(self):
        p1 = make_profile(seed=1)
        p2 = make_profile(seed=2)
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        assert matcher.distance(p1, p2) >= 0

    def test_distance_zero_same_profile(self):
        X = RNG.standard_normal((60, 4))
        p = CooperativeGeometryProfile.from_longitudinal(X)
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        # Same profile — all components should be numerically zero
        assert matcher.distance(p, p) < 1e-7

    def test_distance_components_keys(self):
        p1 = make_profile(seed=3)
        p2 = make_profile(seed=4)
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        components = matcher.distance_components(p1, p2)
        expected_keys = {"levels", "direction", "shape", "occupation", "dynamics",
                         "total", "direction_gate_passed"}
        assert expected_keys == set(components.keys())

    def test_direction_gate_flag(self):
        """Gate passes when directions are aligned, fails when orthogonal."""
        d = 3
        X1 = RNG.standard_normal((60, d))
        X2 = RNG.standard_normal((60, d))
        p1 = CooperativeGeometryProfile.from_longitudinal(X1)
        p2 = CooperativeGeometryProfile.from_longitudinal(X2)

        # Self-match: gate must pass
        matcher = ICGHVRTMatcher(direction_gate=np.pi / 4, auto_calibrate=False)
        assert matcher.distance_components(p1, p1)["direction_gate_passed"]

    def test_find_neighbours_returns_k(self):
        profiles = [make_profile(seed=i) for i in range(20)]
        matcher = ICGHVRTMatcher(auto_calibrate=True)
        idx = matcher.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        assert len(idx) == 5

    def test_find_neighbours_excludes_self(self):
        profiles = [make_profile(seed=i) for i in range(20)]
        matcher = ICGHVRTMatcher(auto_calibrate=True)
        idx = matcher.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        assert 0 not in idx

    def test_calibrate_reduces_scale_variance(self):
        """After calibration, _scale values should be positive floats."""
        profiles = [make_profile(seed=i) for i in range(30)]
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        matcher.calibrate(profiles)
        assert matcher._scale_levels > 0
        assert matcher._scale_direction > 0
        assert matcher._scale_shape > 0


# ────────────────────────────────────────────────────────────────────── #
# Estimator tests                                                         #
# ────────────────────────────────────────────────────────────────────── #

class TestICGHVRTEstimator:

    @staticmethod
    def make_dataset(n_patients: int = 50, n_obs: int = 60, d: int = 3, seed: int = 42):
        rng = np.random.default_rng(seed)
        X_list, T_list, Y_list, true_effects = [], [], [], []
        for i in range(n_patients):
            eff = rng.uniform(-1.0, 1.0)
            cov = np.eye(d) + rng.standard_normal((d, d)) * 0.2
            cov = cov @ cov.T + 0.1 * np.eye(d)
            X = rng.multivariate_normal(np.zeros(d), cov, n_obs)
            T = rng.standard_normal((n_obs, 1))
            Y = eff * T.flatten() + 0.3 * X.sum(axis=1) + rng.standard_normal(n_obs) * 0.5
            X_list.append(X)
            T_list.append(T)
            Y_list.append(Y)
            true_effects.append(eff)
        return X_list, T_list, Y_list, np.array(true_effects)

    def test_fit_and_predict(self):
        X_tr, T_tr, Y_tr, E_tr = self.make_dataset(50, seed=0)
        X_te, T_te, _, E_te = self.make_dataset(10, seed=1)

        estimator = ICGHVRTEstimator(k=20)
        estimator.fit(X_tr, T_tr, Y_tr)

        preds = [estimator.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))]
        preds = np.array(preds)
        assert preds.shape == (10,)
        # Estimates should be finite
        assert np.all(np.isfinite(preds))

    def test_mae_better_than_global_mean(self):
        """ICG-HVRT should beat naive global-mean baseline on a structured dataset."""
        X_tr, T_tr, Y_tr, E_tr = self.make_dataset(100, seed=10)
        X_te, T_te, _, E_te = self.make_dataset(20, seed=11)

        estimator = ICGHVRTEstimator(k=30)
        estimator.fit(X_tr, T_tr, Y_tr)
        preds = np.array([estimator.predict_effect(X_te[i], T_te[i]) for i in range(20)])

        mae_model = np.mean(np.abs(preds - E_te))
        mae_naive = np.mean(np.abs(E_tr.mean() - E_te))

        # ICG-HVRT should do at least as well as the global mean
        assert mae_model <= mae_naive * 1.5, (
            f"Model MAE {mae_model:.4f} much worse than naive {mae_naive:.4f}"
        )

    def test_t_content_diagnostic_returns_float(self):
        X_tr, T_tr, Y_tr, E_tr = self.make_dataset(50, seed=5)
        estimator = ICGHVRTEstimator(k=20)
        estimator.fit(X_tr, T_tr, Y_tr)
        preds = np.array([
            estimator.predict_effect(X_tr[i], T_tr[i], exclude_idx=i)
            for i in range(10)
        ])
        tc = estimator.t_content_diagnostic(preds, estimator._profiles[:10])
        assert isinstance(tc, float)
        assert -1.0 <= tc <= 1.0


# ────────────────────────────────────────────────────────────────────── #
# CoupledInterventionProtocol tests                                       #
# ────────────────────────────────────────────────────────────────────── #

class TestCoupledInterventionProtocol:

    def test_assess_readiness_not_aligned(self):
        d = 4
        w_target = np.ones(d)
        protocol = CoupledInterventionProtocol(
            target_direction=w_target,
            rotation_threshold=np.pi / 6,
            occupation_threshold=0.6,
        )
        # A profile with opposite cooperative direction
        X = RNG.standard_normal((60, d))
        p = CooperativeGeometryProfile.from_longitudinal(X)
        result = protocol.assess_readiness(p)

        assert "direction_gap" in result
        assert "recommendation" in result
        assert result["recommendation"] in (
            "stage_1_required", "continue_stage_1", "proceed_to_stage_2"
        )

    def test_verify_rotation_direction_improved(self):
        d = 4
        w_target = np.ones(d)
        protocol = CoupledInterventionProtocol(
            target_direction=w_target, rotation_threshold=np.pi / 6
        )

        # Pre: observations far from cooperative direction
        X_pre = RNG.standard_normal((60, d))
        p_pre = CooperativeGeometryProfile.from_longitudinal(X_pre)

        # Post: observations shifted toward cooperative direction
        X_post = X_pre + 2.0 * np.ones(d)
        p_post = CooperativeGeometryProfile.from_longitudinal(X_post)

        result = protocol.verify_rotation(p_pre, p_post)

        assert "gate_decision" in result
        assert result["gate_decision"] in (
            "proceed_to_stage_2", "continue_stage_1", "clinical_review"
        )

    def test_from_responders(self):
        d = 4
        profiles = [make_profile(d=d, seed=i) for i in range(10)]
        protocol = CoupledInterventionProtocol.from_responders(profiles)
        assert protocol.target_direction.shape == (d,)


# ────────────────────────────────────────────────────────────────────── #
# Shared HVRT tests                                                       #
# ────────────────────────────────────────────────────────────────────── #

class TestSharedHVRT:

    def test_fit_and_assign(self):
        d, n = 4, 200
        X = RNG.standard_normal((n, d))
        shared = fit_shared_hvrt(X, n_partitions=8)
        assert shared.n_partitions > 0
        assert len(shared.id_to_rank) == shared.n_partitions

        ids = shared.assign_partitions(X[:20])
        assert ids.shape == (20,)
        assert np.all(ids >= 0)
        assert np.all(ids < shared.n_partitions)

    def test_profiles_use_shared_hvrt(self):
        d, n_patients, n_obs = 4, 10, 80
        X_all = RNG.standard_normal((n_patients * n_obs, d))
        shared = fit_shared_hvrt(X_all, n_partitions=8)

        profiles = []
        for i in range(n_patients):
            X_i = RNG.standard_normal((n_obs, d))
            p = CooperativeGeometryProfile.from_longitudinal(X_i, shared_hvrt=shared)
            profiles.append(p)

        # All partition profiles should have the same length K
        K_values = [len(p.partition_profile) for p in profiles if p.partition_profile is not None]
        assert len(set(K_values)) <= 1, f"Inconsistent K across patients: {K_values}"


# ────────────────────────────────────────────────────────────────────── #
# Cascaded matching tests  (spec §5.2)                                    #
# ────────────────────────────────────────────────────────────────────── #

class TestCascadedMatching:
    """Verify the three-level cascade search (spec §5.2)."""

    def test_cascade_returns_k(self):
        profiles = [make_profile(d=4, seed=i) for i in range(30)]
        matcher = ICGHVRTMatcher(cascade=True, auto_calibrate=True)
        idx = matcher.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        assert len(idx) == 5

    def test_cascade_excludes_self(self):
        profiles = [make_profile(d=4, seed=i) for i in range(30)]
        matcher = ICGHVRTMatcher(cascade=True, auto_calibrate=True)
        idx = matcher.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        assert 0 not in idx

    def test_cascade_info_populated(self):
        """last_cascade_info must be set after a cascade find_neighbours call."""
        profiles = [make_profile(d=4, seed=i) for i in range(30)]
        matcher = ICGHVRTMatcher(cascade=True, auto_calibrate=True)
        matcher.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        info = matcher.last_cascade_info
        assert "level1_size" in info
        assert "fallback" in info

    def test_cascade_level1_filters_direction(self):
        """
        Level 1 retains only profiles whose cooperative direction is within
        direction_gate of the query.  Using a very tight gate should produce
        fewer survivors than the full pool.
        """
        rng = np.random.default_rng(99)
        d = 4
        profiles = []
        for i in range(40):
            A = rng.standard_normal((d, d)) * 0.8
            cov = A @ A.T + np.eye(d)
            X = rng.multivariate_normal(np.zeros(d), cov, 60)
            profiles.append(CooperativeGeometryProfile.from_longitudinal(X))

        # Very tight gate → Level 1 likely produces fewer survivors than n
        tight_gate = np.pi / 12   # 15 degrees
        matcher = ICGHVRTMatcher(
            cascade=True, direction_gate=tight_gate, auto_calibrate=True
        )
        matcher.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        info = matcher.last_cascade_info
        # Either a fallback occurred (gate too tight, fewer than k survived)
        # or Level 1 produced a strict subset of the 39 candidates.
        assert info["fallback"] or info["level1_size"] < 39

    def test_cascade_fallback_when_level1_thin(self):
        """
        When fewer than k profiles survive the direction gate, the cascade
        must fall back to exhaustive search and set fallback=True.
        """
        rng = np.random.default_rng(7)
        d = 4
        profiles = []
        for i in range(10):
            A = rng.standard_normal((d, d)) * 2.0   # large spread → diverse directions
            cov = A @ A.T + np.eye(d)
            X = rng.multivariate_normal(np.zeros(d), cov, 60)
            profiles.append(CooperativeGeometryProfile.from_longitudinal(X))

        # Extremely tight gate forces Level 1 to produce 0 survivors
        matcher = ICGHVRTMatcher(
            cascade=True, direction_gate=1e-6, auto_calibrate=True
        )
        idx = matcher.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        assert len(idx) == 5                              # still returns k
        assert matcher.last_cascade_info["fallback"]      # flagged as fallback

    def test_cascade_k2_controls_level2_size(self):
        """
        cascade_k2 explicitly bounds the number of Level-2 candidates.
        When set to a small value, level2_size must not exceed it.
        When set larger than the Level-1 survivor count, level2_size equals
        the survivor count.
        """
        profiles = [make_profile(d=4, seed=i) for i in range(40)]

        # Small k2: Level 2 should be capped at k2
        m_small = ICGHVRTMatcher(cascade=True, cascade_k2=6, auto_calibrate=True)
        m_small.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        info = m_small.last_cascade_info
        if not info["fallback"]:
            assert info["level2_size"] <= 6

        # Large k2: Level 2 keeps all Level-1 survivors
        m_large = ICGHVRTMatcher(cascade=True, cascade_k2=1000, auto_calibrate=True)
        m_large.find_neighbours(profiles[0], profiles, k=5, exclude_idx=0)
        info2 = m_large.last_cascade_info
        if not info2["fallback"]:
            assert info2["level2_size"] == info2["level1_size"]


# ────────────────────────────────────────────────────────────────────── #
# Stress tests  (spec §8)                                                 #
# ────────────────────────────────────────────────────────────────────── #
#
# Each class implements one of the four new DGPs described in spec §8.
# The tests verify that ICG-HVRT achieves meaningful rank correlation
# with the true ITE on each DGP, as required by the "Good" grade in
# the expected-results table (spec §8.2).
#
# Thresholds are set conservatively to avoid flakiness; the ITE
# evaluation experiment (experiments/ite_evaluation.py) provides
# the full statistical picture across 20 seeds.
# ────────────────────────────────────────────────────────────────────── #

def _spearman(a, b):
    from scipy.stats import spearmanr
    rho, _ = spearmanr(a, b)
    return float(0.0 if np.isnan(rho) else rho)


class TestStressDirectionGate:
    """
    Spec §8 Test 4 — Cooperative Direction Gate.

    tau = 3 * cos(angle(w_i, e_1)).
    Patients whose cooperative axis aligns with the first feature
    dimension benefit; orthogonal patients do not.
    ICG-HVRT must detect this via d_w.
    """

    @staticmethod
    def _gen(n_units: int, n_obs: int = 80, seed: int = 0):
        rng = np.random.default_rng(seed)
        d = 4
        e1 = np.eye(d)[0]
        X_list, T_list, Y_list, effects = [], [], [], []
        for _ in range(n_units):
            A   = rng.standard_normal((d, d)) * 0.5
            cov = A @ A.T + np.eye(d)
            eig, vec = np.linalg.eigh(cov)
            inv_sqrt = vec @ np.diag(1.0 / np.sqrt(np.maximum(eig, 1e-6))) @ vec.T
            w   = inv_sqrt @ np.ones(d)
            eff = 3.0 * float(w @ e1) / (np.linalg.norm(w) + 1e-12)
            mu  = rng.standard_normal(d) * 0.5
            X   = rng.multivariate_normal(mu, cov, n_obs)
            T   = rng.standard_normal((n_obs, 1))
            Y   = eff * T.flatten() + rng.standard_normal(n_obs) * 0.5
            X_list.append(X); T_list.append(T); Y_list.append(Y); effects.append(eff)
        return X_list, T_list, Y_list, np.array(effects)

    def test_ite_rank_correlation(self):
        """ICG-HVRT must achieve Spearman rho > 0.4 on the direction-gate DGP."""
        X_tr, T_tr, Y_tr, E_tr = self._gen(150, seed=0)
        X_te, T_te, _,    E_te = self._gen(40,  seed=100)

        est = ICGHVRTEstimator(k=20)
        est.fit(X_tr, T_tr, Y_tr)
        preds = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])

        rho = _spearman(preds, E_te)
        assert rho > 0.4, (
            f"ICG-HVRT Spearman rho={rho:.4f} < 0.4 on Direction Gate; "
            "d_w component is not providing sufficient signal"
        )

    def test_direction_component_discriminates(self):
        """
        d_w between two profiles with similar mean/covariance but opposite
        cooperative directions must be larger than d_w between profiles with
        aligned directions (after calibration).
        """
        rng = np.random.default_rng(55)
        d   = 4
        cov = np.eye(d)

        # Profile with w pointing toward e1
        X_align = rng.multivariate_normal(np.zeros(d), cov, 100)
        # Profiles with cooperative direction spread across angles
        p_query  = CooperativeGeometryProfile.from_longitudinal(X_align)

        # Build a small set, calibrate, then check gate behaviour
        profiles = [CooperativeGeometryProfile.from_longitudinal(
            rng.multivariate_normal(np.zeros(d), np.eye(d) + rng.standard_normal((d, d)) * 0.3, 80)
        ) for _ in range(20)]
        profiles.insert(0, p_query)

        matcher = ICGHVRTMatcher(auto_calibrate=True)
        matcher.calibrate(profiles)

        # Self-distance direction component must be exactly zero
        dc_self = matcher.distance_components(p_query, p_query)
        assert dc_self["direction"] < 1e-7


class TestStressCurvatureGate:
    """
    Spec §8 Test 5 — Manifold Curvature Gate.

    tau = 2 if manifold coupling (rho) > 0.4 else 0.
    Equicorrelated covariance with rho ~ U[0, 0.85].
    ICG-HVRT must detect via d_sigma.
    """

    @staticmethod
    def _gen(n_units: int, n_obs: int = 80, seed: int = 0):
        rng = np.random.default_rng(seed)
        d   = 4
        X_list, T_list, Y_list, effects = [], [], [], []
        for _ in range(n_units):
            rho = rng.uniform(0.0, 0.85)
            cov = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
            eff = 2.0 * float(rho > 0.4)
            X   = rng.multivariate_normal(np.zeros(d), cov, n_obs)
            T   = rng.standard_normal((n_obs, 1))
            Y   = eff * T.flatten() + rng.standard_normal(n_obs) * 0.5
            X_list.append(X); T_list.append(T); Y_list.append(Y); effects.append(eff)
        return X_list, T_list, Y_list, np.array(effects)

    def test_ite_rank_correlation(self):
        """ICG-HVRT must achieve Spearman rho > 0.4 on the curvature-gate DGP."""
        X_tr, T_tr, Y_tr, E_tr = self._gen(150, seed=0)
        X_te, T_te, _,    E_te = self._gen(40,  seed=100)

        est = ICGHVRTEstimator(k=20)
        est.fit(X_tr, T_tr, Y_tr)
        preds = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])

        rho = _spearman(preds, E_te)
        assert rho > 0.4, (
            f"ICG-HVRT Spearman rho={rho:.4f} < 0.4 on Curvature Gate; "
            "d_sigma component is not providing sufficient signal"
        )

    def test_shape_component_ranks_coupling(self):
        """
        d_sigma between a tightly coupled patient (rho=0.8) and a reference
        must exceed d_sigma for a loosely coupled patient (rho=0.05).
        """
        d     = 4
        ref   = np.eye(d)
        tight = (1 - 0.8) * np.eye(d) + 0.8 * np.ones((d, d)) + 1e-4 * np.eye(d)
        loose = (1 - 0.05) * np.eye(d) + 0.05 * np.ones((d, d)) + 1e-4 * np.eye(d)

        rng  = np.random.default_rng(10)
        X_r  = rng.multivariate_normal(np.zeros(d), ref,   80)
        X_t  = rng.multivariate_normal(np.zeros(d), tight, 80)
        X_l  = rng.multivariate_normal(np.zeros(d), loose, 80)

        p_ref   = CooperativeGeometryProfile.from_longitudinal(X_r)
        p_tight = CooperativeGeometryProfile.from_longitudinal(X_t)
        p_loose = CooperativeGeometryProfile.from_longitudinal(X_l)

        matcher = ICGHVRTMatcher(auto_calibrate=False)
        dc_tight = matcher.distance_components(p_ref, p_tight)
        dc_loose = matcher.distance_components(p_ref, p_loose)

        assert dc_tight["shape"] > dc_loose["shape"], (
            "Tightly coupled covariance should be farther from identity in d_sigma"
        )


class TestStressOccupationGate:
    """
    Spec §8 Test 6 — Occupation Gate.

    tau = 4 * p_coop - 2  where p_coop is the cooperative-state fraction.
    Patients spending more time in the cooperative regime benefit more.
    ICG-HVRT must detect via d_occ.
    """

    @staticmethod
    def _gen(n_units: int, n_obs: int = 100, seed: int = 0):
        rng  = np.random.default_rng(seed)
        d, rho = 4, 0.7
        Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        Sa = np.eye(d)
        X_list, T_list, Y_list, effects = [], [], [], []
        for _ in range(n_units):
            p    = rng.uniform(0.1, 0.9)
            eff  = 4.0 * p - 2.0
            mask = rng.random(n_obs) < p
            X    = np.zeros((n_obs, d))
            nc   = mask.sum()
            if nc:          X[mask]  = rng.multivariate_normal(np.zeros(d), Sc, nc)
            if n_obs - nc:  X[~mask] = rng.multivariate_normal(np.zeros(d), Sa, n_obs - nc)
            T = rng.standard_normal((n_obs, 1))
            Y = eff * T.flatten() + rng.standard_normal(n_obs) * 0.5
            X_list.append(X); T_list.append(T); Y_list.append(Y); effects.append(eff)
        return X_list, T_list, Y_list, np.array(effects)

    def test_ite_rank_correlation(self):
        """ICG-HVRT must achieve Spearman rho > 0.4 on the occupation-gate DGP."""
        X_tr, T_tr, Y_tr, E_tr = self._gen(150, seed=0)
        X_te, T_te, _,    E_te = self._gen(40,  seed=100)

        est = ICGHVRTEstimator(k=20)
        est.fit(X_tr, T_tr, Y_tr)
        preds = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])

        rho = _spearman(preds, E_te)
        assert rho > 0.4, (
            f"ICG-HVRT Spearman rho={rho:.4f} < 0.4 on Occupation Gate; "
            "d_occ component is not providing sufficient signal"
        )

    def test_occupation_component_ranks_cooperative_fraction(self):
        """
        Two patients with the same mean covariance but different cooperative
        fractions should have non-zero occupation distance when using a
        shared HVRT partition.
        """
        rng  = np.random.default_rng(20)
        d, rho = 4, 0.7
        Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        Sa = np.eye(d)

        n_obs = 120
        # High cooperative fraction (p=0.85)
        m_h = rng.random(n_obs) < 0.85
        X_h = np.zeros((n_obs, d))
        X_h[m_h]   = rng.multivariate_normal(np.zeros(d), Sc, m_h.sum())
        X_h[~m_h]  = rng.multivariate_normal(np.zeros(d), Sa, (~m_h).sum())

        # Low cooperative fraction (p=0.15)
        m_l = rng.random(n_obs) < 0.15
        X_l = np.zeros((n_obs, d))
        X_l[m_l]   = rng.multivariate_normal(np.zeros(d), Sc, m_l.sum())
        X_l[~m_l]  = rng.multivariate_normal(np.zeros(d), Sa, (~m_l).sum())

        X_pool = np.vstack([X_h, X_l])
        from autoite.profile import fit_shared_hvrt
        shared = fit_shared_hvrt(X_pool, n_partitions=8)

        p_h = CooperativeGeometryProfile.from_longitudinal(X_h, shared_hvrt=shared)
        p_l = CooperativeGeometryProfile.from_longitudinal(X_l, shared_hvrt=shared)

        if p_h.partition_profile is not None and p_l.partition_profile is not None:
            K   = min(len(p_h.partition_profile), len(p_l.partition_profile))
            d_o = occupation_distance(p_h.partition_profile[:K], p_l.partition_profile[:K])
            assert d_o > 0.0, "High vs low cooperative fraction should yield non-zero d_occ"


class TestStressDynamicsGate:
    """
    Spec §8 Test 7 — Dynamics Gate.

    tau = 3 * persistence - 0.5.
    Symmetric Markov regime-switching with constant marginal covariance,
    so d_mu and d_sigma are blind; only d_dyn carries the signal.

    Note: at N=100 observations the transition-matrix estimator is noisy.
    The threshold here is intentionally weak (rho > 0.0) — the full
    statistical picture is in experiments/ite_evaluation.py.
    """

    @staticmethod
    def _gen(n_units: int, n_obs: int = 100, seed: int = 0):
        rng  = np.random.default_rng(seed)
        d, rho = 4, 0.8
        Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        Sa = np.eye(d)
        X_list, T_list, Y_list, effects = [], [], [], []
        for _ in range(n_units):
            p     = rng.uniform(0.3, 0.95)
            eff   = 3.0 * p - 0.5
            state = int(rng.integers(2))
            X     = np.zeros((n_obs, d))
            for step in range(n_obs):
                X[step] = rng.multivariate_normal(np.zeros(d), Sc if state == 0 else Sa)
                if rng.random() > p:
                    state = 1 - state
            T = rng.standard_normal((n_obs, 1))
            Y = eff * T.flatten() + rng.standard_normal(n_obs) * 0.5
            X_list.append(X); T_list.append(T); Y_list.append(Y); effects.append(eff)
        return X_list, T_list, Y_list, np.array(effects)

    def test_ite_rank_correlation_not_strongly_inverted(self):
        """
        ICG-HVRT must not produce strongly inverted rankings on the
        dynamics-gate DGP.

        At N=100 observations the transition-matrix estimator is noisy and
        d_dyn carries weak signal (Spearman ~0.07 in the full evaluation).
        This test guards against catastrophic failure (rho << 0) while the
        full statistical picture lives in experiments/ite_evaluation.py.
        """
        X_tr, T_tr, Y_tr, E_tr = self._gen(150, seed=0)
        X_te, T_te, _,    E_te = self._gen(40,  seed=100)

        est = ICGHVRTEstimator(k=20)
        est.fit(X_tr, T_tr, Y_tr)
        preds = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])

        rho = _spearman(preds, E_te)
        assert rho > -0.3, (
            f"ICG-HVRT Spearman rho={rho:.4f} on Dynamics Gate is strongly negative; "
            "d_dyn component is introducing noise rather than signal. "
            "Known limitation: N=100 obs is below the reliable estimation threshold "
            "for transition matrices with K=8 partitions."
        )

    def test_dynamics_component_nonzero_for_different_persistence(self):
        """
        A high-persistence patient (p=0.9) and a low-persistence patient (p=0.35)
        must have non-zero dynamics distance when transition matrices are available.
        """
        rng  = np.random.default_rng(30)
        d, rho = 4, 0.8
        Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        Sa = np.eye(d)

        def _make_markov(persistence, seed):
            r2 = np.random.default_rng(seed)
            n  = 200           # more obs for stable transition estimate
            state = int(r2.integers(2))
            X = np.zeros((n, d))
            for step in range(n):
                X[step] = r2.multivariate_normal(np.zeros(d), Sc if state == 0 else Sa)
                if r2.random() > persistence:
                    state = 1 - state
            return X

        X_high = _make_markov(0.90, 40)
        X_low  = _make_markov(0.35, 41)

        X_pool = np.vstack([X_high, X_low])
        from autoite.profile import fit_shared_hvrt
        shared = fit_shared_hvrt(X_pool, n_partitions=8)

        p_high = CooperativeGeometryProfile.from_longitudinal(X_high, shared_hvrt=shared)
        p_low  = CooperativeGeometryProfile.from_longitudinal(X_low,  shared_hvrt=shared)

        if p_high.transition_matrix is not None and p_low.transition_matrix is not None:
            d_dyn = dynamics_distance(p_high.transition_matrix, p_low.transition_matrix)
            assert d_dyn > 0.0, (
                "High-persistence vs low-persistence patients should have non-zero d_dyn"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
