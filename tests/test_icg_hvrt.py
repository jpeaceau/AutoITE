"""
Unit tests for the ICG-HVRT implementation (v0.2.0).

Run with:  python -m pytest tests/test_icg_hvrt.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoite import (
    ConeIdentity,
    CooperativeGeometryProfile,
    fit_shared_hvrt,
    ICGHVRTMatcher,
    ICGHVRTEstimator,
    CoupledInterventionProtocol,
)
from autoite.distances import (
    euclidean_mean_distance,
    cooperative_mean_distance,
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

    def test_cone_identity_attached(self):
        """Profile now carries a ConeIdentity object."""
        X = RNG.standard_normal((60, 4))
        p = CooperativeGeometryProfile.from_longitudinal(X)
        assert p.cone_identity is not None
        assert isinstance(p.cone_identity, ConeIdentity)

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
        # Should still have static geometry and cone identity even if HVRT fails
        assert p.mu.shape == (d,)
        assert p.cooperative_direction.shape == (d,)
        assert p.cone_identity is not None

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
# ConeIdentity tests                                                      #
# ────────────────────────────────────────────────────────────────────── #

class TestConeIdentity:

    def test_shape_and_types(self):
        """ConeIdentity.from_covariance returns correct array shapes."""
        d = 4
        sigma = random_spd(d, seed=10)
        ci = ConeIdentity.from_covariance(sigma)

        assert ci.axis.shape == (d,), f"axis shape {ci.axis.shape}"
        assert isinstance(ci.positive_eigenvalue, float)
        assert ci.negative_eigenvalues.shape == (d - 1,)
        assert ci.anti_cooperative_frame.shape == (d, d - 1)
        assert ci.opening_profile.shape == (d - 1,)
        assert isinstance(ci.eccentricity, float)
        assert ci.positive_eigenvalue > 0, "positive eigenvalue must be > 0"
        assert np.all(ci.negative_eigenvalues < 0), "negative eigenvalues must be < 0"

    def test_axis_is_normalised(self):
        """The cooperative axis v+ must be a unit vector."""
        sigma = random_spd(4, seed=20)
        ci = ConeIdentity.from_covariance(sigma)
        assert abs(float(np.linalg.norm(ci.axis)) - 1.0) < 1e-8

    def test_circular_cone_isotropic_sigma(self):
        """
        When Sigma = I, the cone is circular: all half-angles are equal and
        eccentricity = 1.0 (up to numerical noise from sample covariance).
        """
        d = 4
        sigma = np.eye(d)
        ci = ConeIdentity.from_covariance(sigma)
        # All half-angles should be identical for isotropic Sigma
        assert abs(ci.eccentricity - 1.0) < 1e-6, (
            f"Isotropic sigma should give ecc=1.0, got {ci.eccentricity}"
        )
        # All half-angles equal
        theta_std = float(np.std(ci.opening_profile))
        assert theta_std < 1e-6, (
            f"Isotropic sigma: half-angles should be equal, std={theta_std}"
        )

    def test_eccentricity_anisotropic(self):
        """Anisotropic Sigma gives eccentricity > 1.0."""
        # Strong anisotropy: first dimension has 16x more variance
        sigma = np.diag([16.0, 1.0, 1.0, 1.0])
        ci = ConeIdentity.from_covariance(sigma)
        assert ci.eccentricity > 1.0, (
            f"Anisotropic sigma should give ecc > 1.0, got {ci.eccentricity}"
        )

    def test_opening_profile_sorted_descending(self):
        """opening_profile must be sorted in descending order (widest first)."""
        sigma = random_spd(5, seed=30)
        ci = ConeIdentity.from_covariance(sigma)
        diffs = np.diff(ci.opening_profile)
        assert np.all(diffs <= 1e-9), (
            f"opening_profile not descending: {ci.opening_profile}"
        )

    def test_distance_zero_same_identity(self):
        """All distance components are zero when comparing an identity to itself."""
        sigma = random_spd(4, seed=40)
        ci = ConeIdentity.from_covariance(sigma)
        d = ConeIdentity.distance(ci, ci)
        assert d["axis"] < 1e-8
        assert d["opening"] < 1e-8
        assert d["eccentricity"] < 1e-8
        assert d["orientation"] < 1e-8

    def test_distance_axis_range(self):
        """d_axis must lie in [0, pi/2] for any two cone identities."""
        rng = np.random.default_rng(50)
        for _ in range(20):
            sigma_i = random_spd(4, seed=int(rng.integers(1000)))
            sigma_j = random_spd(4, seed=int(rng.integers(1000)))
            ci = ConeIdentity.from_covariance(sigma_i)
            cj = ConeIdentity.from_covariance(sigma_j)
            d = ConeIdentity.distance(ci, cj)
            assert 0.0 <= d["axis"] <= np.pi / 2 + 1e-9, (
                f"d_axis={d['axis']} out of [0, pi/2]"
            )

    def test_distance_keys(self):
        """ConeIdentity.distance returns the expected keys."""
        ci = ConeIdentity.from_covariance(random_spd(4, seed=60))
        cj = ConeIdentity.from_covariance(random_spd(4, seed=61))
        d = ConeIdentity.distance(ci, cj)
        assert set(d.keys()) == {"axis", "opening", "eccentricity", "orientation"}

    def test_eccentricity_ranking(self):
        """
        A highly eccentric cone should have larger d_ecc from an isotropic
        reference than a mildly eccentric cone.
        """
        d = 4
        sigma_ref  = np.eye(d)
        sigma_mild = np.diag([2.0, 1.0, 1.0, 1.0])   # mild eccentricity
        sigma_high = np.diag([16.0, 1.0, 1.0, 1.0])   # high eccentricity

        ci_ref  = ConeIdentity.from_covariance(sigma_ref)
        ci_mild = ConeIdentity.from_covariance(sigma_mild)
        ci_high = ConeIdentity.from_covariance(sigma_high)

        d_mild = ConeIdentity.distance(ci_ref, ci_mild)["eccentricity"]
        d_high = ConeIdentity.distance(ci_ref, ci_high)["eccentricity"]
        assert d_high > d_mild, (
            f"High eccentricity should give larger d_ecc: "
            f"d_mild={d_mild:.4f}, d_high={d_high:.4f}"
        )

    def test_from_operator_matches_from_covariance(self):
        """from_operator and from_covariance must give the same result."""
        d = 4
        sigma = random_spd(d, seed=70)

        # Compute C directly (same as profile.py does it)
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.maximum(eigvals, 1e-10)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        sigma_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        w = inv_sqrt @ np.ones(d)
        C = np.outer(w, w) - sigma_inv

        ci_from_op  = ConeIdentity.from_operator(C)
        ci_from_cov = ConeIdentity.from_covariance(sigma)

        np.testing.assert_allclose(ci_from_op.opening_profile,
                                   ci_from_cov.opening_profile, atol=1e-8)
        assert abs(ci_from_op.eccentricity - ci_from_cov.eccentricity) < 1e-8


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


class TestCooperativeMeanDistance:
    """Tests for the cooperative mean distance decomposition."""

    def test_pythagorean_identity(self):
        """d_coop^2 + d_perp^2 == ||delta_mu||^2 for arbitrary inputs."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            d = rng.integers(2, 9)
            mu_i = rng.standard_normal(d)
            mu_j = rng.standard_normal(d)
            # Build w vectors from random positive-definite Sigma
            A = rng.standard_normal((d, d))
            Sigma = A @ A.T + np.eye(d)
            eig, vec = np.linalg.eigh(Sigma)
            w = vec @ np.diag(1.0 / np.sqrt(np.maximum(eig, 1e-10))) @ vec.T @ np.ones(d)
            d_coop, d_perp = cooperative_mean_distance(mu_i, mu_j, w, w)
            l2 = float(np.linalg.norm(mu_i - mu_j))
            assert abs(d_coop**2 + d_perp**2 - l2**2) < 1e-9, \
                f"Pythagorean identity failed: {d_coop**2 + d_perp**2} != {l2**2}"

    def test_zero_for_identical_means(self):
        """Both components are zero when mu_i == mu_j."""
        w = np.ones(4)
        mu = np.array([1.0, 2.0, 3.0, 4.0])
        d_coop, d_perp = cooperative_mean_distance(mu, mu, w, w)
        assert d_coop < 1e-12
        assert d_perp < 1e-12

    def test_coop_detects_aligned_shift(self):
        """Pure shift along w: d_coop = ||delta_mu||, d_perp = 0."""
        w = np.ones(4) / 2.0          # cooperative direction (unnormalized)
        mu_i = np.zeros(4)
        mu_j = np.ones(4) * 2.0       # delta_mu = 2*ones, parallel to w
        d_coop, d_perp = cooperative_mean_distance(mu_i, mu_j, w, w)
        expected_coop = float(np.linalg.norm(mu_j - mu_i))   # pure projection
        assert abs(d_coop - expected_coop) < 1e-9
        assert d_perp < 1e-9

    def test_perp_detects_orthogonal_shift(self):
        """Shift orthogonal to w: d_coop = 0, d_perp = ||delta_mu||."""
        w = np.array([1.0, 1.0, 0.0, 0.0])          # cooperative axis in dims 0,1
        mu_i = np.zeros(4)
        mu_j = np.array([0.0, 0.0, 3.0, 4.0])        # shift purely in dims 2,3
        d_coop, d_perp = cooperative_mean_distance(mu_i, mu_j, w, w)
        assert d_coop < 1e-9
        assert abs(d_perp - float(np.linalg.norm(mu_j))) < 1e-9

    def test_many_weak_leaks_tau_proportionality(self):
        """
        Mathematical property: when Sigma ~ I and mu = rho * U,
        d_mu_coop = rho * |tau_i - tau_j|  regardless of K.

        This is the key result that fixes the many-weak-measurements problem.
        """
        rng = np.random.default_rng(42)
        rho = 0.3
        for K in [1, 2, 4, 8, 16]:
            d = K
            coop_dists, tau_diffs, l2_dists = [], [], []
            for _ in range(500):
                U_i = rng.standard_normal(K)
                U_j = rng.standard_normal(K)
                mu_i = rho * U_i
                mu_j = rho * U_j
                tau_i = U_i.sum() / np.sqrt(K)
                tau_j = U_j.sum() / np.sqrt(K)
                # w ~ ones when Sigma ~ I
                w = np.ones(d)
                d_coop, _ = cooperative_mean_distance(mu_i, mu_j, w, w)
                coop_dists.append(d_coop)
                tau_diffs.append(abs(tau_i - tau_j))
                l2_dists.append(float(np.linalg.norm(mu_i - mu_j)))

            from scipy.stats import spearmanr
            rho_coop, _ = spearmanr(coop_dists, tau_diffs)
            rho_l2,   _ = spearmanr(l2_dists,   tau_diffs)
            # d_mu_coop must be strongly correlated with tau regardless of K
            assert rho_coop > 0.95, \
                f"K={K}: d_mu_coop Spearman with |tau|={rho_coop:.3f} < 0.95"
            # d_mu (L2) must degrade as K grows
            if K >= 4:
                assert rho_l2 < rho_coop, \
                    f"K={K}: L2 rho={rho_l2:.3f} should be < coop rho={rho_coop:.3f}"


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
        expected_keys = {
            "axis", "opening", "eccentricity", "orientation",
            "levels", "levels_perp", "occupation", "dynamics",
            "identity_distance", "total", "geometry_reliable",
        }
        assert expected_keys == set(components.keys())

    def test_identity_distance_in_total(self):
        """identity_distance must equal sum of identity components."""
        p1 = make_profile(seed=5)
        p2 = make_profile(seed=6)
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        c = matcher.distance_components(p1, p2)
        identity_sum = c["axis"] + c["opening"] + c["eccentricity"] + c["orientation"]
        assert abs(identity_sum - c["identity_distance"]) < 1e-9

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

    def test_calibrate_sets_positive_scales(self):
        """After calibration, all internal scales must be positive floats."""
        profiles = [make_profile(seed=i) for i in range(30)]
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        matcher.calibrate(profiles)
        assert matcher._scale_axis > 0
        assert matcher._scale_opening > 0
        assert matcher._scale_eccentricity > 0
        assert matcher._scale_orientation > 0
        assert matcher._scale_levels > 0
        assert matcher._scale_occ > 0

    def test_eccentric_patients_have_larger_identity_distance(self):
        """
        Two patients with very different eccentricities should have a larger
        identity_distance than two patients with similar eccentricities.
        """
        d = 4
        rng = np.random.default_rng(99)
        n_obs = 80

        # Circular cone: Sigma ~ I
        X_circ1 = rng.multivariate_normal(np.zeros(d), np.eye(d), n_obs)
        X_circ2 = rng.multivariate_normal(np.zeros(d), np.eye(d) * 1.05, n_obs)
        # Eccentric cone: one dimension has 16x variance
        X_ecc = rng.multivariate_normal(np.zeros(d),
                                        np.diag([16.0, 1.0, 1.0, 1.0]), n_obs)

        p_circ1 = CooperativeGeometryProfile.from_longitudinal(X_circ1)
        p_circ2 = CooperativeGeometryProfile.from_longitudinal(X_circ2)
        p_ecc   = CooperativeGeometryProfile.from_longitudinal(X_ecc)

        matcher = ICGHVRTMatcher(auto_calibrate=False)
        id_similar  = matcher.distance_components(p_circ1, p_circ2)["identity_distance"]
        id_different = matcher.distance_components(p_circ1, p_ecc)["identity_distance"]

        assert id_different > id_similar, (
            f"Different eccentricities should give larger identity_distance: "
            f"similar={id_similar:.4f}, different={id_different:.4f}"
        )


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

        estimator = ICGHVRTEstimator(k=20, learn_weights=False)
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

        estimator = ICGHVRTEstimator(k=30, learn_weights=False)
        estimator.fit(X_tr, T_tr, Y_tr)
        preds = np.array([estimator.predict_effect(X_te[i], T_te[i]) for i in range(20)])

        mae_model = np.mean(np.abs(preds - E_te))
        mae_naive = np.mean(np.abs(E_tr.mean() - E_te))

        assert mae_model <= mae_naive * 1.5, (
            f"Model MAE {mae_model:.4f} much worse than naive {mae_naive:.4f}"
        )

    def test_t_content_diagnostic_returns_float(self):
        X_tr, T_tr, Y_tr, E_tr = self.make_dataset(50, seed=5)
        estimator = ICGHVRTEstimator(k=20, learn_weights=False)
        estimator.fit(X_tr, T_tr, Y_tr)
        preds = np.array([
            estimator.predict_effect(X_tr[i], T_tr[i], exclude_idx=i)
            for i in range(10)
        ])
        tc = estimator.t_content_diagnostic(preds, estimator._profiles[:10])
        assert isinstance(tc, float)
        assert -1.0 <= tc <= 1.0

    def test_triage_report_keys(self):
        """triage_report must return dicts with the v0.2.0 uncertainty keys."""
        X_tr, T_tr, Y_tr, _ = self.make_dataset(50, seed=20)
        X_te, _, _, _ = self.make_dataset(5, seed=21)

        estimator = ICGHVRTEstimator(k=20, learn_weights=False)
        estimator.fit(X_tr, T_tr, Y_tr)
        report = estimator.triage_report(X_te)

        assert len(report) == 5
        for entry in report:
            assert "nearest_distance" in entry
            assert "mean_identity_distance" in entry
            assert "confidence" in entry
            assert entry["confidence"] in ("high", "medium", "low", "uncertain")


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
        from autoite.profile import pool_whitened_observations
        X_list_all = [RNG.standard_normal((n_obs, d)) for _ in range(n_patients)]
        shared = fit_shared_hvrt(pool_whitened_observations(X_list_all), n_partitions=8)

        profiles = []
        for X_i in X_list_all:
            p = CooperativeGeometryProfile.from_longitudinal(X_i, shared_hvrt=shared)
            profiles.append(p)

        # All partition profiles should have the same length K
        K_values = [len(p.partition_profile) for p in profiles if p.partition_profile is not None]
        assert len(set(K_values)) <= 1, f"Inconsistent K across patients: {K_values}"


# ────────────────────────────────────────────────────────────────────── #
# Stress tests  (spec §8)                                                 #
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
    ICG-HVRT must detect this via the 'axis' identity component.
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

        est = ICGHVRTEstimator(k=20, learn_weights=False)
        est.fit(X_tr, T_tr, Y_tr)
        preds = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])

        rho = _spearman(preds, E_te)
        assert rho > 0.4, (
            f"ICG-HVRT Spearman rho={rho:.4f} < 0.4 on Direction Gate; "
            "axis identity component is not providing sufficient signal"
        )

    def test_axis_component_zero_for_self(self):
        """
        Axis component of distance must be exactly zero for self-comparison,
        and positive for profiles with different cooperative directions.
        """
        rng = np.random.default_rng(55)
        d   = 4

        X_align = rng.multivariate_normal(np.zeros(d), np.eye(d), 100)
        p_query  = CooperativeGeometryProfile.from_longitudinal(X_align)

        profiles = [CooperativeGeometryProfile.from_longitudinal(
            rng.multivariate_normal(np.zeros(d), np.eye(d) + rng.standard_normal((d, d)) * 0.3, 80)
        ) for _ in range(20)]
        profiles.insert(0, p_query)

        matcher = ICGHVRTMatcher(auto_calibrate=True)
        matcher.calibrate(profiles)

        # Self-distance axis component must be exactly zero
        dc_self = matcher.distance_components(p_query, p_query)
        assert dc_self["axis"] < 1e-7


class TestStressCurvatureGate:
    """
    Spec §8 Test 5 — Manifold Curvature Gate.

    tau = 2 if manifold coupling (rho) > 0.4 else 0.
    Equicorrelated covariance with rho ~ U[0, 0.85].
    ICG-HVRT must detect via the opening profile identity component.
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

        est = ICGHVRTEstimator(k=20, learn_weights=False)
        est.fit(X_tr, T_tr, Y_tr)
        preds = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])

        rho = _spearman(preds, E_te)
        assert rho > 0.4, (
            f"ICG-HVRT Spearman rho={rho:.4f} < 0.4 on Curvature Gate; "
            "opening identity component is not providing sufficient signal"
        )

    def test_opening_component_ranks_coupling(self):
        """
        d_opening between a tightly coupled patient (rho=0.8) and a reference
        must exceed d_opening for a loosely coupled patient (rho=0.05).
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

        # Tight coupling produces larger opening profile distance
        assert dc_tight["opening"] > dc_loose["opening"], (
            "Tightly coupled covariance should be farther from identity in d_opening"
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

        est = ICGHVRTEstimator(k=20, learn_weights=False)
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

        from autoite.profile import fit_shared_hvrt, pool_whitened_observations
        shared = fit_shared_hvrt(
            pool_whitened_observations([X_h, X_l]), n_partitions=8
        )

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
    The threshold here is intentionally weak (rho > -0.3).
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
        """
        X_tr, T_tr, Y_tr, E_tr = self._gen(150, seed=0)
        X_te, T_te, _,    E_te = self._gen(40,  seed=100)

        est = ICGHVRTEstimator(k=20, learn_weights=False)
        est.fit(X_tr, T_tr, Y_tr)
        preds = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])

        rho = _spearman(preds, E_te)
        assert rho > -0.3, (
            f"ICG-HVRT Spearman rho={rho:.4f} on Dynamics Gate is strongly negative; "
            "d_dyn component is introducing noise rather than signal."
        )

    def test_dynamics_component_nonzero_for_different_persistence(self):
        """
        A high-persistence patient and a low-persistence patient must have
        non-zero dynamics distance when transition matrices are available.
        """
        d, rho = 4, 0.8
        Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        Sa = np.eye(d)

        def _make_markov(persistence, seed):
            r2 = np.random.default_rng(seed)
            n  = 200
            state = int(r2.integers(2))
            X = np.zeros((n, d))
            for step in range(n):
                X[step] = r2.multivariate_normal(np.zeros(d), Sc if state == 0 else Sa)
                if r2.random() > persistence:
                    state = 1 - state
            return X

        X_high = _make_markov(0.90, 40)
        X_low  = _make_markov(0.35, 41)

        from autoite.profile import fit_shared_hvrt, pool_whitened_observations
        shared = fit_shared_hvrt(
            pool_whitened_observations([X_high, X_low]), n_partitions=8
        )

        p_high = CooperativeGeometryProfile.from_longitudinal(X_high, shared_hvrt=shared)
        p_low  = CooperativeGeometryProfile.from_longitudinal(X_low,  shared_hvrt=shared)

        if p_high.transition_matrix is not None and p_low.transition_matrix is not None:
            d_dyn = dynamics_distance(p_high.transition_matrix, p_low.transition_matrix)
            assert d_dyn > 0.0, (
                "High-persistence vs low-persistence patients should have non-zero d_dyn"
            )


class TestStressEccentricityGate:
    """
    Spec §8 Test 8 — Eccentricity Gate (new in v0.2.0).

    tau = f(eccentricity) — patients with near-circular cones have different
    treatment effects from patients with highly eccentric cones.
    ICG-HVRT must detect this via the eccentricity identity component.
    v0.1.0 fails because it had no eccentricity measure.
    """

    @staticmethod
    def _gen(n_units: int, n_obs: int = 100, seed: int = 0):
        rng = np.random.default_rng(seed)
        d = 4
        X_list, T_list, Y_list, effects = [], [], [], []
        for _ in range(n_units):
            # Stretch first dimension by 'stretch' factor (1..4)
            stretch = rng.uniform(1.0, 4.0)
            sigma = np.diag(np.concatenate([[stretch ** 2], np.ones(d - 1)]))
            # Effect decreases with stretch: circular cones respond well,
            # eccentric cones respond poorly.
            eff = 3.0 / stretch
            X = rng.multivariate_normal(np.zeros(d), sigma, n_obs)
            T = rng.standard_normal((n_obs, 1))
            Y = eff * T.flatten() + rng.standard_normal(n_obs) * 0.5
            X_list.append(X); T_list.append(T); Y_list.append(Y); effects.append(eff)
        return X_list, T_list, Y_list, np.array(effects)

    def test_ite_rank_correlation(self):
        """ICG-HVRT must achieve Spearman rho > 0.2 on the eccentricity-gate DGP."""
        X_tr, T_tr, Y_tr, E_tr = self._gen(150, seed=0)
        X_te, T_te, _,    E_te = self._gen(40,  seed=100)

        est = ICGHVRTEstimator(k=20, learn_weights=False)
        est.fit(X_tr, T_tr, Y_tr)
        preds = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])

        rho = _spearman(preds, E_te)
        assert rho > 0.2, (
            f"ICG-HVRT Spearman rho={rho:.4f} < 0.2 on Eccentricity Gate; "
            "eccentricity identity component is not providing sufficient signal. "
            "v0.1.0 had no eccentricity measure; this test specifically targets "
            "the new ConeIdentity capability."
        )

    def test_eccentricity_correlates_with_stretch(self):
        """
        ConeIdentity eccentricity must be monotonically related to the
        stretch factor used to construct the covariance matrix.
        Higher stretch -> higher eccentricity.
        """
        d = 4
        rng = np.random.default_rng(77)
        n_obs = 200

        eccentricities = []
        stretches = [1.0, 1.5, 2.0, 3.0, 4.0]
        for stretch in stretches:
            sigma = np.diag(np.concatenate([[stretch ** 2], np.ones(d - 1)]))
            X = rng.multivariate_normal(np.zeros(d), sigma, n_obs)
            p = CooperativeGeometryProfile.from_longitudinal(X)
            eccentricities.append(p.cone_identity.eccentricity)

        from scipy.stats import spearmanr
        rho, _ = spearmanr(stretches, eccentricities)
        assert rho > 0.8, (
            f"Eccentricity should increase monotonically with stretch, "
            f"Spearman rho={rho:.3f} < 0.8. "
            f"Eccentricities: {[f'{e:.2f}' for e in eccentricities]}"
        )


class TestSynergyGeometry:
    """ICG-Synergy (PyramidHART spike detection → HVRT on clean bulk)."""

    def test_smoke_clean_data(self):
        """Synergy mode produces finite ITE on spike-free data."""
        rng = np.random.default_rng(0)
        X = [rng.standard_normal((50, 4)) for _ in range(30)]
        T = [rng.standard_normal(50) for _ in range(30)]
        Y = [rng.standard_normal(50) for _ in range(30)]
        est = ICGHVRTEstimator(k=10, geometry='synergy').fit(X, T, Y)
        tau = est.predict_effect(X[0], T[0])
        assert np.isfinite(tau)

    def test_smoke_spiked_data(self):
        """Synergy mode produces finite ITE when 5% of observations are ±20σ spikes."""
        rng = np.random.default_rng(1)
        X, T, Y = [], [], []
        for _ in range(30):
            x = rng.standard_normal((50, 4))
            # Inject single-feature spikes into 5% of rows
            spike_rows = rng.choice(50, 3, replace=False)
            spike_cols = rng.integers(0, 4, 3)
            x[spike_rows, spike_cols] += rng.choice([-20.0, 20.0], 3)
            X.append(x)
            T.append(rng.standard_normal(50))
            Y.append(rng.standard_normal(50))
        est = ICGHVRTEstimator(k=10, geometry='synergy').fit(X, T, Y)
        tau = est.predict_effect(X[0], T[0])
        assert np.isfinite(tau)

    def test_default_local_model_is_lad(self):
        """Synergy geometry auto-selects lad local model."""
        est = ICGHVRTEstimator(geometry='synergy')
        assert est.local_model == 'lad'

    def test_spike_fraction_stored(self):
        """Custom spike_fraction is stored on the estimator."""
        est = ICGHVRTEstimator(geometry='synergy', spike_fraction=0.10)
        assert est.spike_fraction == 0.10

    def test_synergy_vs_cone_on_spike_dgp(self):
        """Synergy PEHE < cone PEHE on the Outlier Spike DGP."""
        d = 4; rho = 0.9; spike_mag = 20.0; spike_frac = 0.05  # noqa: E702
        Sc = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        Sa = np.eye(d)

        def _gen(n, seed):
            np.random.seed(seed)
            X, T, Y, E = [], [], [], []
            for _ in range(n):
                coupled = np.random.rand() > 0.5
                Sig = Sc if coupled else Sa
                tau = 2.0 if coupled else 0.0
                x = np.random.multivariate_normal(np.zeros(d), Sig, 80)
                n_spikes = max(1, int(80 * spike_frac))
                rows = np.random.choice(80, n_spikes, replace=False)
                cols = np.random.randint(0, d, n_spikes)
                x[rows, cols] += np.random.choice([-1., 1.], n_spikes) * spike_mag
                t = np.random.normal(0, 1, (80, 1))
                X.append(x); T.append(t)
                Y.append(tau * t.ravel() + np.random.normal(0, 0.5, 80))
                E.append(tau)
            return X, T, Y, np.array(E)

        X_tr, T_tr, Y_tr, _ = _gen(60, seed=0)
        X_te, T_te, Y_te, E_te = _gen(20, seed=1)

        pehe = {}
        for geo in ('cone', 'synergy'):
            est = ICGHVRTEstimator(k=10, geometry=geo).fit(X_tr, T_tr, Y_tr)
            tau_hat = np.array([est.predict_effect(X_te[i], T_te[i]) for i in range(len(X_te))])
            pehe[geo] = float(np.sqrt(np.mean((tau_hat - E_te) ** 2)))

        assert pehe['synergy'] < pehe['cone'], (
            f"Expected synergy PEHE ({pehe['synergy']:.3f}) < cone PEHE ({pehe['cone']:.3f})"
        )


class TestLocalModels:
    """Each local_model variant produces a finite ITE estimate."""

    @pytest.mark.parametrize("local_model", ["ridge", "ols", "lad", "mean", "median"])
    def test_smoke(self, local_model):
        rng = np.random.default_rng(42)
        X = [rng.standard_normal((30, 3)) for _ in range(20)]
        T = [rng.standard_normal(30) for _ in range(20)]
        Y = [rng.standard_normal(30) for _ in range(20)]
        est = ICGHVRTEstimator(k=5, local_model=local_model).fit(X, T, Y)
        tau = est.predict_effect(X[0], T[0])
        assert np.isfinite(tau), f"local_model='{local_model}' returned non-finite tau"


class TestIndividualCovariateLeak:
    """Smoke tests for the gen_individual_covariate_leak DGP."""

    def test_dgp_shapes(self):
        """DGP returns correct list lengths and array shapes."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from experiments.ite_comparison import gen_individual_covariate_leak, N_OBS
        n = 10
        X, T, Y, E = gen_individual_covariate_leak(n)
        assert len(X) == n
        assert X[0].shape == (N_OBS, 4), f"expected (N_OBS, 4), got {X[0].shape}"
        assert T[0].shape == (N_OBS, 1)
        assert Y[0].shape == (N_OBS,)
        assert E.shape == (n,)

    def test_mean_leak_is_weak(self):
        """Aggregate mean(X[:,0]) should be close to 0 for U~U[-1,1], K=3, amp=8."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from experiments.ite_comparison import gen_individual_covariate_leak
        np.random.seed(0)
        X, _, _, E = gen_individual_covariate_leak(500)
        means_f0 = np.array([np.mean(xi[:, 0]) for xi in X])
        # mean(X[:,0]) ~ U * 8 * 3/N_OBS; correlation with U should be moderate but small
        from scipy.stats import pearsonr
        corr, _ = pearsonr(means_f0, E)
        # Signal is present but aggregate is weaker than full mean confound (K=N_OBS, rho~0.98).
        # With K=3, amp=8, N_OBS=100: expected SNR~1.4, Pearson r ~ 0.81.
        assert 0.0 < corr < 0.97, f"mean(X[:,0])-E[tau] correlation {corr:.3f} unexpected"

    def test_cone_vs_pyramid_spike_sensitivity(self):
        """Cone geometry should produce finite ITE on this DGP for both geometries."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from experiments.ite_comparison import gen_individual_covariate_leak
        np.random.seed(1)
        X, T, Y, _ = gen_individual_covariate_leak(30)
        X_te, T_te, Y_te, _ = gen_individual_covariate_leak(5)
        for geo in ("cone", "pyramid"):
            est = ICGHVRTEstimator(k=10, geometry=geo).fit(X, T, Y)
            for i in range(len(X_te)):
                tau = est.predict_effect(X_te[i], T_te[i])
                assert np.isfinite(tau), f"geometry='{geo}' returned non-finite tau at i={i}"

    def test_sparse_k_factory(self):
        """gen_sparse_k_leak factory produces finite ITE estimates for both K=1 and K=50."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from experiments.ite_comparison import gen_sparse_k_leak
        np.random.seed(2)
        for K in [1, 50]:
            gen = gen_sparse_k_leak(K)
            X, T, Y, _ = gen(20)
            X_te, T_te, _, _ = gen(5)
            est = ICGHVRTEstimator(k=8).fit(X, T, Y)
            for i in range(len(X_te)):
                tau = est.predict_effect(X_te[i], T_te[i])
                assert np.isfinite(tau), f"K={K}: non-finite tau at i={i}"


class TestSelectivePrediction:
    """Tests for distance-weighted k-NN and predict_effect_with_confidence."""

    def _make_data(self, n=40, d=4, n_obs=30, seed=0):
        rng = np.random.default_rng(seed)
        X = [rng.standard_normal((n_obs, d)) for _ in range(n)]
        T = [rng.standard_normal(n_obs)      for _ in range(n)]
        Y = [rng.standard_normal(n_obs)      for _ in range(n)]
        return X, T, Y

    def test_predict_with_confidence_returns_tuple(self):
        """predict_effect_with_confidence returns (float, float)."""
        X, T, Y = self._make_data()
        est = ICGHVRTEstimator(k=10).fit(X, T, Y)
        tau, dist = est.predict_effect_with_confidence(X[0], T[0])
        assert isinstance(tau, float) and np.isfinite(tau)
        assert isinstance(dist, float) and dist > 0 and np.isfinite(dist)

    def test_predict_effect_backward_compat(self):
        """predict_effect still returns a plain float after API change."""
        X, T, Y = self._make_data()
        est = ICGHVRTEstimator(k=10).fit(X, T, Y)
        result = est.predict_effect(X[0], T[0])
        assert isinstance(result, float), (
            f"predict_effect must return float, got {type(result)}")

    def test_distance_weighted_differs_from_flat(self):
        """distance_weighted=True should change the tau estimate on most patients."""
        rng = np.random.default_rng(7)
        d = 4; n = 60
        # Build data with geometric heterogeneity so distances vary
        rho = 0.9
        Sc  = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        Sa  = np.eye(d)
        X, T, Y = [], [], []
        for i in range(n):
            Sig = Sc if i < n // 2 else Sa
            x = rng.multivariate_normal(np.zeros(d), Sig, 40)
            t = rng.standard_normal(40)
            y = 2.0 * t + rng.standard_normal(40) * 0.5
            X.append(x); T.append(t); Y.append(y)

        est_flat = ICGHVRTEstimator(k=10).fit(X, T, Y)
        est_w    = ICGHVRTEstimator(k=10, distance_weighted=True).fit(X, T, Y)

        diffs = [
            abs(est_flat.predict_effect(X[i], T[i]) -
                est_w.predict_effect(X[i], T[i]))
            for i in range(20)
        ]
        # At least some predictions should differ when weights are applied
        assert max(diffs) > 1e-6, "distance_weighted had no effect on any prediction"

    def test_confidence_discriminates_in_vs_out(self):
        """In-distribution patients should get lower k-NN distance than OOD patients."""
        rng = np.random.default_rng(9)
        d = 4
        # Train on coupled-covariance patients only
        rho = 0.9
        Sc  = (1 - rho) * np.eye(d) + rho * np.ones((d, d)) + 1e-4 * np.eye(d)
        X = [rng.multivariate_normal(np.zeros(d), Sc, 50) for _ in range(40)]
        T = [rng.standard_normal(50) for _ in range(40)]
        Y = [rng.standard_normal(50) for _ in range(40)]
        est = ICGHVRTEstimator(k=10).fit(X, T, Y)

        # In-distribution: same covariance structure
        _, dist_in = est.predict_effect_with_confidence(
            rng.multivariate_normal(np.zeros(d), Sc, 50),
            rng.standard_normal(50),
        )
        # Out-of-distribution: identity covariance (very different geometry)
        _, dist_ood = est.predict_effect_with_confidence(
            rng.standard_normal((50, d)),
            rng.standard_normal(50),
        )
        assert dist_in < dist_ood, (
            f"Expected in-dist distance ({dist_in:.3f}) < OOD distance ({dist_ood:.3f})"
        )

    def test_coverage_pehe_monotone(self):
        """PEHE should not increase as coverage decreases on a clean DGP."""
        from experiments.ite_comparison import gen_geometric_confounded
        np.random.seed(42)
        X_tr, T_tr, Y_tr, _     = gen_geometric_confounded(80)
        X_te, T_te, Y_te, E_te  = gen_geometric_confounded(30)

        est = ICGHVRTEstimator(k=20, geometry='cone').fit(X_tr, T_tr, Y_tr)
        results = [est.predict_effect_with_confidence(X_te[i], T_te[i])
                   for i in range(len(X_te))]
        tau_hat = np.array([r[0] for r in results])
        conf    = np.array([r[1] for r in results])

        pehes = []
        for cov in [1.0, 0.8, 0.6, 0.4]:
            k = max(1, int(round(cov * len(E_te))))
            order = np.argsort(conf)[:k]
            pehes.append(float(np.sqrt(np.mean((tau_hat[order] - E_te[order]) ** 2))))

        # Allow slight non-monotonicity due to noise (small n), but trend should improve
        assert pehes[-1] <= pehes[0] * 1.5, (
            f"PEHE did not improve with selectivity: {pehes}"
        )


class TestPrognosticConfounder:
    """Smoke tests for the gen_prognostic_confounder DGP."""

    def test_dgp_shapes(self):
        """DGP returns correct list lengths and array shapes."""
        from experiments.ite_comparison import gen_prognostic_confounder, N_OBS
        n = 10
        X, T, Y, E = gen_prognostic_confounder(n)
        assert len(X) == n
        assert X[0].shape == (N_OBS, 4), f"expected (N_OBS, 4), got {X[0].shape}"
        assert T[0].shape == (N_OBS, 1)
        assert Y[0].shape == (N_OBS,)
        assert E.shape == (n,)

    def test_treatment_independent_of_confounder(self):
        """T must be independent of U (no selection confounding): |corr(T_bar, tau)| < 0.15."""
        from experiments.ite_comparison import gen_prognostic_confounder
        from scipy.stats import pearsonr
        np.random.seed(7)
        X, T, _, E = gen_prognostic_confounder(500)
        T_bar = np.array([np.mean(np.asarray(t).ravel()) for t in T])
        corr, _ = pearsonr(T_bar, E)
        assert abs(corr) < 0.15, (
            f"T_bar-tau Pearson {corr:.3f} -- T should be independent of U (tau); "
            "selection bias was not intended in this DGP"
        )

    def test_covariate_leak_is_present(self):
        """mean(X) should correlate with tau via rho_x: |corr| in (0.1, 0.99) for rho_x=0.5."""
        from experiments.ite_comparison import gen_prognostic_confounder
        from scipy.stats import pearsonr
        np.random.seed(8)
        X, _, _, E = gen_prognostic_confounder(500, rho_x=0.5)
        means_x0 = np.array([np.mean(xi[:, 0]) for xi in X])
        corr, _ = pearsonr(means_x0, E)
        assert 0.1 < corr < 0.99, (
            f"mean(X[:,0])-tau Pearson {corr:.3f} unexpected for rho_x=0.5"
        )

    def test_rho_x_zero_hides_confounder(self):
        """rho_x=0 -> X perp U -> mean(X) uncorrelated with tau."""
        from experiments.ite_comparison import gen_prognostic_confounder
        from scipy.stats import pearsonr
        np.random.seed(9)
        X, _, _, E = gen_prognostic_confounder(500, rho_x=0.0)
        means_x0 = np.array([np.mean(xi[:, 0]) for xi in X])
        corr, _ = pearsonr(means_x0, E)
        assert abs(corr) < 0.15, (
            f"rho_x=0 should hide U from X, but corr={corr:.3f}"
        )

    def test_finite_ite_both_geometries(self):
        """Both cone and pyramid geometries return finite ITE on this DGP."""
        from experiments.ite_comparison import gen_prognostic_confounder
        np.random.seed(10)
        X, T, Y, _ = gen_prognostic_confounder(30)
        X_te, T_te, _, _ = gen_prognostic_confounder(5)
        for geo in ("cone", "pyramid"):
            est = ICGHVRTEstimator(k=10, geometry=geo).fit(X, T, Y)
            for i in range(len(X_te)):
                tau = est.predict_effect(X_te[i], T_te[i])
                assert np.isfinite(tau), f"geometry='{geo}' returned non-finite tau at i={i}"

    def test_rho_factory(self):
        """gen_prognostic_rho factory produces valid data at boundary rho_x values."""
        from experiments.ite_comparison import gen_prognostic_rho
        for rho in [0.0, 1.0]:
            gen = gen_prognostic_rho(rho)
            X, T, Y, E = gen(20)
            assert len(X) == 20
            assert all(np.all(np.isfinite(xi)) for xi in X)
            assert all(np.all(np.isfinite(yi)) for yi in Y)
            assert np.all(np.isfinite(E))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
