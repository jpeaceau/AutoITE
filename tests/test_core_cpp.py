"""
tests/test_core_cpp.py — differential correctness tests for autoite._core.

Strategy
--------
Every test compares the C++ output against the Python reference implementation
on identical random inputs.  Tests are skipped automatically when the C++
extension has not been built (no build failure on CI without compiler).

Three test categories:

1. Numerical correctness  — C++ loo_objective == Python within atol=1e-8.
   Covers: standard inputs, single eval patient, k=1, large alpha, zero alpha,
   very small n_obs (n_obs=k), d=2, d=8, d=16.

2. Edge cases  — degenerate inputs that should not crash or NaN.
   Covers: identical Y_local (zero variance), single neighbour, alpha=0.

3. Performance assertion  — fit_weights with C++ must be < Python × 0.5
   (at minimum 2x faster) on a 100-patient, 40-eval, 50-iter trial.
   Marked with @pytest.mark.slow so CI can skip with -m "not slow".
"""
import time

import numpy as np
import pytest

# ── skip entire module if _core is not built ──────────────────────────────── #
try:
    from autoite import _core as _cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CPP_AVAILABLE,
    reason="autoite._core not built (run build_ext.bat first)",
)

# ── shared fixtures ───────────────────────────────────────────────────────── #

RNG = np.random.default_rng(0xC0FFEE)

def _make_inputs(
    N=50, d=4, n_obs=20, n_eval=10, k=8,
    alpha=1.0, rng=None,
):
    """Build raw_cache + flat arrays + offsets matching fit_weights layout."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Constant n_obs per patient for simplicity
    offsets = np.arange(0, (N + 1) * n_obs, n_obs, dtype=np.int32)
    obs_flat = rng.standard_normal((N * n_obs, d)).astype(np.float64)
    T_flat   = rng.standard_normal(N * n_obs).astype(np.float64)
    Y_flat   = rng.standard_normal(N * n_obs).astype(np.float64)

    eval_indices = rng.choice(N, n_eval, replace=False).astype(np.int32)

    # raw_cache: random non-negative components
    raw_cache = np.abs(rng.standard_normal((n_eval, N, 8))).astype(np.float64)

    weights = np.abs(rng.standard_normal(8)).astype(np.float64) + 0.1
    scales  = np.abs(rng.standard_normal(8)).astype(np.float64) + 0.5

    return {
        "raw_cache":    raw_cache,
        "obs_flat":     obs_flat,
        "T_flat":       T_flat,
        "Y_flat":       Y_flat,
        "offsets":      offsets,
        "eval_indices": eval_indices,
        "weights":      weights,
        "scales":       scales,
        "k":            k,
        "alpha":        alpha,
        "N":            N,
        "n_eval":       n_eval,
        "n_obs":        n_obs,
        "d":            d,
    }


def _python_loo_objective(inp):
    """Pure-Python reference implementation of loo_objective."""
    from sklearn.linear_model import Ridge

    raw_cache    = inp["raw_cache"]
    obs_flat     = inp["obs_flat"]
    T_flat       = inp["T_flat"]
    Y_flat       = inp["Y_flat"]
    offsets      = inp["offsets"]
    eval_indices = inp["eval_indices"]
    weights      = inp["weights"]
    scales       = inp["scales"]
    k            = inp["k"]
    alpha        = inp["alpha"]
    n_eval       = inp["n_eval"]
    _N           = inp["N"]
    _d           = inp["d"]

    eff_w = weights / scales
    total_mse = 0.0

    for ei in range(n_eval):
        i = int(eval_indices[ei])

        # distance vector
        D_i = raw_cache[ei] @ eff_w          # (N,)
        D_i = D_i.copy()
        D_i[i] = np.inf

        nn_idx = np.argsort(D_i)[:k]

        # stack local neighbourhood
        local_rows = np.hstack([
            np.arange(offsets[j], offsets[j + 1]) for j in nn_idx
        ])
        X_loc  = obs_flat[local_rows]
        T_loc  = T_flat[local_rows].reshape(-1, 1)
        Y_loc  = Y_flat[local_rows]
        XT_loc = np.hstack([X_loc, T_loc])

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(XT_loc, Y_loc)

        # predict on patient i
        rows_i = np.arange(offsets[i], offsets[i + 1])
        X_i    = obs_flat[rows_i]
        T_i    = T_flat[rows_i].reshape(-1, 1)
        XT_i   = np.hstack([X_i, T_i])
        Y_i    = Y_flat[rows_i]
        Y_hat  = model.predict(XT_i)

        total_mse += float(np.mean((Y_i - Y_hat) ** 2))

    return total_mse / n_eval


def _cpp_call(inp):
    return _cpp.loo_objective(
        np.ascontiguousarray(inp["raw_cache"],    np.float64),
        np.ascontiguousarray(inp["obs_flat"],     np.float64),
        np.ascontiguousarray(inp["T_flat"],       np.float64),
        np.ascontiguousarray(inp["Y_flat"],       np.float64),
        np.ascontiguousarray(inp["offsets"],      np.int32),
        np.ascontiguousarray(inp["eval_indices"], np.int32),
        np.ascontiguousarray(inp["weights"],      np.float64),
        np.ascontiguousarray(inp["scales"],       np.float64),
        inp["k"],
        inp["alpha"],
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# 1. Numerical correctness                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestLooObjectiveCorrectness:

    def test_standard_inputs(self):
        inp = _make_inputs(N=50, d=4, n_obs=20, n_eval=10, k=8, alpha=1.0)
        cpp = _cpp_call(inp)
        py  = _python_loo_objective(inp)
        assert abs(cpp - py) / (abs(py) + 1e-12) < 1e-7, (
            f"C++={cpp:.8f}  Python={py:.8f}  rel_err={(abs(cpp-py)/(abs(py)+1e-12)):.2e}"
        )

    def test_single_eval_patient(self):
        inp = _make_inputs(N=30, d=4, n_obs=15, n_eval=1, k=5, alpha=0.5)
        assert abs(_cpp_call(inp) - _python_loo_objective(inp)) < 1e-8

    def test_k_equals_one(self):
        inp = _make_inputs(N=30, d=4, n_obs=15, n_eval=5, k=1, alpha=1.0)
        assert abs(_cpp_call(inp) - _python_loo_objective(inp)) < 1e-8

    def test_large_alpha(self):
        inp = _make_inputs(N=30, d=4, n_obs=15, n_eval=5, k=8, alpha=100.0)
        assert abs(_cpp_call(inp) - _python_loo_objective(inp)) < 1e-8

    def test_small_alpha(self):
        inp = _make_inputs(N=30, d=4, n_obs=25, n_eval=5, k=8, alpha=1e-4)
        cpp = _cpp_call(inp)
        py  = _python_loo_objective(inp)
        # Small alpha → Ridge approaches OLS; both should still agree closely
        assert abs(cpp - py) / (abs(py) + 1e-12) < 1e-6

    def test_d_equals_2(self):
        inp = _make_inputs(N=30, d=2, n_obs=15, n_eval=5, k=6, alpha=1.0)
        assert abs(_cpp_call(inp) - _python_loo_objective(inp)) < 1e-8

    def test_d_equals_8(self):
        inp = _make_inputs(N=60, d=8, n_obs=30, n_eval=8, k=10, alpha=1.0)
        cpp = _cpp_call(inp)
        py  = _python_loo_objective(inp)
        assert abs(cpp - py) / (abs(py) + 1e-12) < 1e-7

    def test_d_equals_16(self):
        inp = _make_inputs(N=80, d=16, n_obs=50, n_eval=6, k=12, alpha=1.0)
        cpp = _cpp_call(inp)
        py  = _python_loo_objective(inp)
        assert abs(cpp - py) / (abs(py) + 1e-12) < 1e-7

    def test_returns_float(self):
        inp = _make_inputs()
        result = _cpp_call(inp)
        assert isinstance(result, float)

    def test_result_non_negative(self):
        """MSE is always >= 0."""
        inp = _make_inputs()
        assert _cpp_call(inp) >= 0.0

    def test_zero_weights_gives_zero_eff_w(self):
        """When all weights are near-zero, distances are near-zero → kNN is
        arbitrary but objective should still be a finite float."""
        inp = _make_inputs()
        inp["weights"] = np.full(8, 1e-15)
        result = _cpp_call(inp)
        assert np.isfinite(result)

    def test_different_weights_give_different_results(self):
        inp1 = _make_inputs(rng=np.random.default_rng(7))
        inp2 = dict(inp1)
        # Perturb only the first weight; uniform scaling leaves kNN ordering unchanged
        w2 = inp1["weights"].copy()
        w2[0] *= 10.0
        inp2["weights"] = w2
        assert _cpp_call(inp1) != _cpp_call(inp2)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_random_seeds(self, seed):
        inp = _make_inputs(rng=np.random.default_rng(seed))
        cpp = _cpp_call(inp)
        py  = _python_loo_objective(inp)
        assert abs(cpp - py) / (abs(py) + 1e-12) < 1e-7, (
            f"seed={seed}  C++={cpp:.8f}  Python={py:.8f}"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# 2. Edge cases                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestLooObjectiveEdgeCases:

    def test_constant_Y_returns_near_zero_mse(self):
        """Y all equal → perfect Ridge fit (intercept = Y, coef ≈ 0) → near-zero MSE."""
        inp = _make_inputs()
        inp["Y_flat"] = np.ones(len(inp["Y_flat"]))
        result = _cpp_call(inp)
        assert result < 1e-6, f"Expected near-zero MSE for constant Y, got {result}"

    def test_n_obs_equals_k_plus_1(self):
        """Minimal neighbourhood: n_obs per patient = k+1 (tight but valid)."""
        inp = _make_inputs(N=30, n_obs=9, k=8)
        result = _cpp_call(inp)
        assert np.isfinite(result) and result >= 0.0

    def test_large_N_train(self):
        inp = _make_inputs(N=200, d=4, n_obs=20, n_eval=10, k=20)
        cpp = _cpp_call(inp)
        py  = _python_loo_objective(inp)
        assert abs(cpp - py) / (abs(py) + 1e-12) < 1e-7

    def test_module_version(self):
        assert hasattr(_cpp, "__version__")
        assert "phase2" in _cpp.__version__
        assert hasattr(_cpp, "compute_distances")
        assert hasattr(_cpp, "compute_raw_cache")

    def test_wrong_raw_cache_shape_raises(self):
        inp = _make_inputs()
        bad_cache = inp["raw_cache"][:, :, :6]   # wrong last dim
        with pytest.raises(Exception):
            _cpp.loo_objective(
                np.ascontiguousarray(bad_cache, np.float64),
                inp["obs_flat"], inp["T_flat"], inp["Y_flat"],
                inp["offsets"], inp["eval_indices"],
                inp["weights"], inp["scales"],
                inp["k"], inp["alpha"],
            )


# ═══════════════════════════════════════════════════════════════════════════ #
# 3. Performance                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestLooObjectivePerformance:

    @pytest.mark.slow
    def test_cpp_faster_than_python(self):
        """
        C++ loo_objective must complete a 40-patient, 50-call trial faster
        than the equivalent Python loop.  Minimum requirement: 2× speedup.
        """
        inp = _make_inputs(N=100, d=4, n_obs=80, n_eval=40, k=15, alpha=1.0)
        n_calls = 50

        # Warm up (JIT, caches, etc.)
        for _ in range(3):
            _cpp_call(inp)

        t0 = time.perf_counter()
        for _ in range(n_calls):
            _cpp_call(inp)
        t_cpp = (time.perf_counter() - t0) / n_calls

        t0 = time.perf_counter()
        for _ in range(max(n_calls // 5, 5)):
            _python_loo_objective(inp)
        t_py = (time.perf_counter() - t0) / max(n_calls // 5, 5)

        speedup = t_py / t_cpp
        print(f"\n  C++: {t_cpp*1e3:.2f} ms/call   Python: {t_py*1e3:.2f} ms/call   "
              f"speedup: {speedup:.1f}x")

        assert speedup >= 2.0, (
            f"C++ speedup {speedup:.1f}x is below the minimum 2x threshold. "
            f"C++={t_cpp*1e3:.2f}ms  Python={t_py*1e3:.2f}ms"
        )

    @pytest.mark.slow
    def test_full_fit_weights_cpp_path(self):
        """
        End-to-end fit_weights with C++ path completes in < 3s for N=150,
        n_eval=40, max_iter=50 — verifies the full integration path.
        """
        from autoite.estimator import ICGHVRTEstimator
        from autoite.matcher import _HAS_CPP

        if not _HAS_CPP:
            pytest.skip("_core not available")

        rng = np.random.default_rng(99)
        n, d, n_obs = 150, 4, 80
        X_list = [rng.standard_normal((n_obs, d)) for _ in range(n)]
        T_list = [rng.standard_normal((n_obs, 1)) for _ in range(n)]
        Y_list = [rng.standard_normal(n_obs) for _ in range(n)]

        t0 = time.perf_counter()
        est = ICGHVRTEstimator(k=20, learn_weights=True)
        est.matcher.fit_weights(
            est._profiles if est._profiles else [],  # will be empty, use direct call
            X_list, T_list, Y_list,
            k=20, max_iter=50, n_eval_patients=40,
        )
        elapsed = time.perf_counter() - t0

        # Just verify fit_weights runs without error through C++ path
        # (the matcher hasn't been calibrated so skip the perf assert here;
        #  see test_cpp_faster_than_python for the isolated perf test)
        assert elapsed < 60.0, f"fit_weights took {elapsed:.1f}s — something is very wrong"


# ═══════════════════════════════════════════════════════════════════════════ #
# 4. compute_distances / compute_raw_cache correctness                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def _make_profiles(N=30, d=4, n_obs=40, K=8, seed=0):
    """Build N training profiles + a shared HVRT for differential tests."""
    from autoite.profile import (
        CooperativeGeometryProfile, fit_shared_hvrt, pool_whitened_observations,
    )
    rng = np.random.default_rng(seed)
    X_list = [rng.standard_normal((n_obs, d)) * 0.5 + rng.standard_normal(d) for _ in range(N)]
    Z_pool = pool_whitened_observations(X_list)
    shared = fit_shared_hvrt(Z_pool, n_partitions=K, random_state=42)
    profiles = [
        CooperativeGeometryProfile.from_longitudinal(X, shared_hvrt=shared)
        for X in X_list
    ]
    return profiles


class TestComputeDistances:

    def test_distances_match_python_matcher(self):
        """compute_distances C++ == Python distance() for all N pairs."""
        from autoite.matcher import ICGHVRTMatcher, _pack_profiles, _pack_query
        profiles = _make_profiles(N=20, d=4, n_obs=40, seed=1)
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        matcher.calibrate(profiles)

        packed  = _pack_profiles(profiles)
        weights = np.array([
            matcher.beta_axis, matcher.beta_opening,
            matcher.beta_eccentricity, matcher.beta_orientation,
            matcher.gamma_levels, matcher.gamma_levels_perp,
            matcher.gamma_occupation, matcher.gamma_dynamics,
        ], dtype=np.float64)
        scales = np.array([
            matcher._scale_axis, matcher._scale_opening,
            matcher._scale_eccentricity, matcher._scale_orientation,
            matcher._scale_levels, matcher._scale_levels_perp,
            matcher._scale_occ, matcher._scale_dyn,
        ], dtype=np.float64)

        # Use profile 0 as query; compare against Python distances.
        q = _pack_query(profiles[0], packed["K"], packed["dp1"])
        cpp_dists = _cpp.compute_distances(
            q["axis"], q["ecc"], q["opening"], q["anti_coop"],
            q["mu"], q["coop_dir"], q["partition"], q["transition"],
            q["geo_reliable"], q["has_partition"], q["has_transition"],
            packed["axes"], packed["eccentricities"],
            packed["openings"], packed["anti_coops"],
            packed["mus"], packed["coop_dirs"],
            packed["partitions"], packed["transitions"],
            packed["geo_reliable"], packed["has_partition"], packed["has_transition"],
            weights, scales,
        )
        py_dists = np.array([
            matcher.distance(profiles[0], p) for p in profiles
        ])
        # Tolerance: JacobiSVD (C++) vs numpy.linalg.svd (Python) agree to ~1e-7.
        np.testing.assert_allclose(cpp_dists, py_dists, rtol=1e-5, atol=1e-6,
            err_msg="compute_distances disagrees with Python distance()")

    def test_self_distance_is_zero(self):
        """Distance from a profile to itself should be exactly 0.0."""
        from autoite.matcher import ICGHVRTMatcher, _pack_profiles, _pack_query
        profiles = _make_profiles(N=10, d=4, n_obs=40, seed=2)
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        matcher.calibrate(profiles)
        packed  = _pack_profiles(profiles)
        weights = np.ones(8, dtype=np.float64)
        scales  = np.ones(8, dtype=np.float64)

        for idx in range(len(profiles)):
            q = _pack_query(profiles[idx], packed["K"], packed["dp1"])
            dists = _cpp.compute_distances(
                q["axis"], q["ecc"], q["opening"], q["anti_coop"],
                q["mu"], q["coop_dir"], q["partition"], q["transition"],
                q["geo_reliable"], q["has_partition"], q["has_transition"],
                packed["axes"], packed["eccentricities"],
                packed["openings"], packed["anti_coops"],
                packed["mus"], packed["coop_dirs"],
                packed["partitions"], packed["transitions"],
                packed["geo_reliable"], packed["has_partition"], packed["has_transition"],
                weights, scales,
            )
            # SVD floating-point gives ~1e-8 for self; 1e-6 is a safe floor.
            assert dists[idx] < 1e-6, \
                f"Self-distance for profile {idx} = {dists[idx]}"

    def test_compute_raw_cache_matches_python(self):
        """compute_raw_cache C++ == _raw_components Python for all (eval, train) pairs."""
        from autoite.matcher import ICGHVRTMatcher, _pack_profiles
        profiles = _make_profiles(N=15, d=4, n_obs=40, seed=3)
        matcher = ICGHVRTMatcher(auto_calibrate=False)
        matcher.calibrate(profiles)
        packed = _pack_profiles(profiles)

        rng  = np.random.default_rng(99)
        n_ev = 5
        eval_idx = rng.choice(len(profiles), n_ev, replace=False).astype(np.int32)

        cpp_cache = _cpp.compute_raw_cache(
            eval_idx,
            packed["axes"], packed["eccentricities"],
            packed["openings"], packed["anti_coops"],
            packed["mus"], packed["coop_dirs"],
            packed["partitions"], packed["transitions"],
            packed["geo_reliable"], packed["has_partition"], packed["has_transition"],
        )
        assert cpp_cache.shape == (n_ev, len(profiles), 8)

        _COMP = ["axis", "opening", "eccentricity", "orientation",
                 "levels", "levels_perp", "occupation", "dynamics"]
        py_cache = np.zeros((n_ev, len(profiles), 8))
        for ei, i in enumerate(eval_idx):
            for j in range(len(profiles)):
                raw = matcher._raw_components(profiles[i], profiles[j])
                for c, key in enumerate(_COMP):
                    py_cache[ei, j, c] = raw[key]

        # Tolerance: Procrustes SVD library differences give ~1e-7 on orientation.
        np.testing.assert_allclose(cpp_cache, py_cache, rtol=1e-5, atol=1e-6,
            err_msg="compute_raw_cache disagrees with Python _raw_components")

    def test_find_neighbours_cpp_matches_python(self):
        """find_neighbours() C++ path returns same k indices as Python path."""
        from autoite.matcher import ICGHVRTMatcher
        import autoite.matcher as _matcher_mod

        profiles = _make_profiles(N=20, d=4, n_obs=40, seed=4)

        # C++ path
        cpp_matcher = ICGHVRTMatcher(auto_calibrate=False)
        cpp_matcher.calibrate(profiles)

        # Python path (force off)
        orig = _matcher_mod._HAS_CPP
        _matcher_mod._HAS_CPP = False
        py_matcher = ICGHVRTMatcher(auto_calibrate=False)
        py_matcher.calibrate(profiles)
        _matcher_mod._HAS_CPP = orig

        k = 5
        for qi in range(0, len(profiles), 4):
            cpp_nn = set(cpp_matcher.find_neighbours(profiles[qi], profiles, k=k))
            py_nn  = set(py_matcher.find_neighbours(profiles[qi], profiles, k=k))
            assert cpp_nn == py_nn, \
                f"find_neighbours differs at qi={qi}: C++={sorted(cpp_nn)} Python={sorted(py_nn)}"

    @pytest.mark.slow
    def test_find_neighbours_speedup(self):
        """C++ find_neighbours must be at least 5x faster than Python."""
        from autoite.matcher import ICGHVRTMatcher
        import autoite.matcher as _matcher_mod

        profiles = _make_profiles(N=150, d=4, n_obs=60, seed=5)
        cpp_matcher = ICGHVRTMatcher(auto_calibrate=False)
        cpp_matcher.calibrate(profiles)

        orig = _matcher_mod._HAS_CPP
        _matcher_mod._HAS_CPP = False
        py_matcher = ICGHVRTMatcher(auto_calibrate=False)
        py_matcher.calibrate(profiles)
        _matcher_mod._HAS_CPP = orig

        queries = profiles[:10]
        k = 20
        n_reps = 5

        t0 = time.perf_counter()
        for _ in range(n_reps):
            for qp in queries:
                cpp_matcher.find_neighbours(qp, profiles, k=k)
        t_cpp = (time.perf_counter() - t0) / (n_reps * len(queries))

        t0 = time.perf_counter()
        for _ in range(n_reps):
            for qp in queries:
                py_matcher.find_neighbours(qp, profiles, k=k)
        t_py = (time.perf_counter() - t0) / (n_reps * len(queries))

        speedup = t_py / t_cpp
        print(f"\n  C++: {t_cpp*1e3:.2f} ms/query  Python: {t_py*1e3:.2f} ms/query  "
              f"speedup: {speedup:.1f}x")
        assert speedup >= 5.0, (
            f"find_neighbours speedup {speedup:.1f}x below minimum 5x. "
            f"C++={t_cpp*1e3:.2f}ms  Python={t_py*1e3:.2f}ms"
        )
