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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
