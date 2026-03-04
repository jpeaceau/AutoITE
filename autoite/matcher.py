"""
ICGHVRTMatcher: eight-component cooperative cone distance for patient matching.

Distance structure (ICG-HVRT v0.2.0)
--------------------------------------
All eight components are combined into a single unified weighted distance.
k-nearest-neighbours are selected by this total distance — no hard gate.

Identity components (cone family compatibility):
  d_axis        Angular distance between cooperative axes v+
  d_opening     L2 distance between directional half-angle profiles
  d_eccentricity |log(ecc_i) - log(ecc_j)| — cone circularity difference
  d_orientation  Procrustes distance between anti-cooperative frames V-

State components (where on the cone the patient currently sits):
  d_mu_coop     Mean shift along the shared cooperative axis
  d_mu_perp     Mean shift perpendicular to the cooperative axis
  d_occ         Wasserstein-1 occupation distance
  d_dyn         Frobenius transition-matrix distance

Design rationale
----------------
A hard gate (exclude patients whose identity distance exceeds a threshold)
would impose binary decisions on a continuous geometric signal.  Instead,
large identity components inflate the total distance, naturally pushing
geometrically incompatible matches down the k-NN ranking without exclusion.
The 'identity_distance' value in distance_components() exposes the identity
contribution as an uncertainty signal: high identity_distance among the k
selected neighbours means the best available matches are geometrically poor.

By coupling samples through their cone geometry, every training patient is
automatically weighted by their structural compatibility with the query —
latent and observable geometry are jointly scrutinised in the single ranking.
"""
import numpy as np
from typing import Dict, List, Optional

from .profile import CooperativeGeometryProfile

# Optional C++ extension — loaded once at import time.
# Falls back to pure Python silently if the extension is not built.
try:
    from . import _core as _cpp
    _HAS_CPP = True
except ImportError:
    _cpp = None
    _HAS_CPP = False
from .distances import (
    ConeIdentity,
    cooperative_mean_distance,
    occupation_distance,
    dynamics_distance,
)


def _pack_profiles(profiles: list, K: int = 0) -> dict:
    """
    Pack a list of CooperativeGeometryProfiles into flat C-contiguous numpy
    arrays suitable for the C++ compute_distances / compute_raw_cache kernels.

    Called once after calibrate(); result stored as self._tr_packed.

    Returns a dict with keys:
      axes, eccentricities, openings, anti_coops,
      mus, coop_dirs, partitions, transitions,
      geo_reliable, has_partition, has_transition,
      N, d, dp1, K
    """
    N = len(profiles)
    if N == 0:
        return {}
    d   = profiles[0].d
    dp1 = d - 1

    # Infer K from largest partition profile present.
    if K == 0:
        for p in profiles:
            if p.partition_profile is not None:
                K = max(K, len(p.partition_profile))

    axes           = np.zeros((N, d),           dtype=np.float64)
    eccentricities = np.zeros(N,                dtype=np.float64)
    # dp1 may be 0 for 1-D data; keep at least shape (N,1) for contiguity.
    openings   = np.zeros((N, max(dp1, 1)),      dtype=np.float64)
    anti_coops = np.zeros((N, max(d * dp1, 1)), dtype=np.float64)
    mus        = np.zeros((N, d),               dtype=np.float64)
    coop_dirs  = np.zeros((N, d),               dtype=np.float64)
    partitions  = np.zeros((N, max(K, 1)),      dtype=np.float64)
    transitions = np.zeros((N, max(K * K, 1)),  dtype=np.float64)
    geo_reliable   = np.zeros(N, dtype=np.int32)
    has_partition  = np.zeros(N, dtype=np.int32)
    has_transition = np.zeros(N, dtype=np.int32)

    for i, p in enumerate(profiles):
        mus[i]       = p.mu
        coop_dirs[i] = p.cooperative_direction
        geo_reliable[i] = int(p.geometry_reliable)

        ci = p.cone_identity
        if ci is not None and dp1 > 0:
            axes[i]           = ci.axis
            eccentricities[i] = ci.eccentricity
            n_ang = min(len(ci.opening_profile), dp1)
            if n_ang > 0:
                openings[i, :n_ang] = ci.opening_profile[:n_ang]
            V     = ci.anti_cooperative_frame
            n_col = min(V.shape[1], dp1)
            if n_col > 0:
                # Store (d, n_col) row-major: V[r,c] -> anti_coops[i, r*dp1+c].
                # V is a C-contiguous (d, d-1) array; ravel() gives row-major.
                anti_coops[i, :d * n_col] = V[:, :n_col].ravel()

        if p.partition_profile is not None and K > 0:
            kp = min(len(p.partition_profile), K)
            partitions[i, :kp] = p.partition_profile[:kp]
            has_partition[i]   = 1

        if p.transition_matrix is not None and K > 0:
            km = min(p.transition_matrix.shape[0], K)
            transitions[i, :km * km] = p.transition_matrix[:km, :km].ravel()
            has_transition[i]        = 1

    # Trim to actual dimensions (no padding columns).
    op_arr = np.ascontiguousarray(openings[:, :dp1]    if dp1 > 0 else openings[:, :0])
    ac_arr = np.ascontiguousarray(anti_coops[:, :d*dp1] if dp1 > 0 else anti_coops[:, :0])
    pa_arr = np.ascontiguousarray(partitions[:, :K]     if K  > 0 else partitions[:, :0])
    tr_arr = np.ascontiguousarray(transitions[:, :K*K]  if K  > 0 else transitions[:, :0])

    return dict(
        axes           = np.ascontiguousarray(axes),
        eccentricities = np.ascontiguousarray(eccentricities),
        openings       = op_arr,
        anti_coops     = ac_arr,
        mus            = np.ascontiguousarray(mus),
        coop_dirs      = np.ascontiguousarray(coop_dirs),
        partitions     = pa_arr,
        transitions    = tr_arr,
        geo_reliable   = np.ascontiguousarray(geo_reliable),
        has_partition  = np.ascontiguousarray(has_partition),
        has_transition = np.ascontiguousarray(has_transition),
        N=N, d=d, dp1=dp1, K=K,
    )


def _pack_query(profile: "CooperativeGeometryProfile", K: int, dp1: int) -> dict:
    """
    Pack a single query profile into flat arrays for compute_distances().
    Separated to keep find_neighbours() readable.
    """
    d   = profile.d
    ci  = profile.cone_identity

    q_axis      = np.zeros(d, dtype=np.float64)
    q_ecc       = 1.0
    q_opening   = np.zeros(max(dp1, 1), dtype=np.float64)
    q_anti_coop = np.zeros(max(d * dp1, 1), dtype=np.float64)

    if ci is not None and dp1 > 0:
        q_axis[:] = ci.axis
        q_ecc     = ci.eccentricity
        n_ang = min(len(ci.opening_profile), dp1)
        if n_ang > 0:
            q_opening[:n_ang] = ci.opening_profile[:n_ang]
        V     = ci.anti_cooperative_frame
        n_col = min(V.shape[1], dp1)
        if n_col > 0:
            q_anti_coop[:d * n_col] = V[:, :n_col].ravel()

    q_partition  = np.zeros(max(K, 1), dtype=np.float64)
    q_transition = np.zeros(max(K * K, 1), dtype=np.float64)
    q_has_part   = 0
    q_has_trans  = 0

    if profile.partition_profile is not None and K > 0:
        kp = min(len(profile.partition_profile), K)
        q_partition[:kp] = profile.partition_profile[:kp]
        q_has_part = 1

    if profile.transition_matrix is not None and K > 0:
        km = min(profile.transition_matrix.shape[0], K)
        q_transition[:km * km] = profile.transition_matrix[:km, :km].ravel()
        q_has_trans = 1

    return dict(
        axis       = np.ascontiguousarray(q_axis),
        ecc        = float(q_ecc),
        opening    = np.ascontiguousarray(q_opening[:dp1]   if dp1 > 0 else q_opening[:0]),
        anti_coop  = np.ascontiguousarray(q_anti_coop[:d*dp1] if dp1 > 0 else q_anti_coop[:0]),
        mu         = np.ascontiguousarray(profile.mu, dtype=np.float64),
        coop_dir   = np.ascontiguousarray(profile.cooperative_direction, dtype=np.float64),
        partition  = np.ascontiguousarray(q_partition[:K]   if K > 0 else q_partition[:0]),
        transition = np.ascontiguousarray(q_transition[:K*K] if K > 0 else q_transition[:0]),
        geo_reliable  = int(profile.geometry_reliable),
        has_partition = q_has_part,
        has_transition= q_has_trans,
    )


class ICGHVRTMatcher:
    """
    Eight-component cone distance for matching patients on their personal
    cooperative cones (ICG-HVRT v0.2.0).

    Identity weights (beta_*) — cone family compatibility
    -------------------------------------------------------
    beta_axis         : weight for cooperative axis alignment (d_axis)
    beta_opening      : weight for directional half-angle profile (d_opening)
    beta_eccentricity : weight for cone eccentricity difference (d_ecc)
    beta_orientation  : weight for anti-cooperative frame orientation (d_orient)

    State weights (gamma_*) — cooperative state
    --------------------------------------------
    gamma_levels      : weight for cooperative mean distance d_mu_coop
    gamma_levels_perp : weight for perpendicular mean distance d_mu_perp
    gamma_occupation  : weight for Wasserstein occupation distance (d_occ)
    gamma_dynamics    : weight for Frobenius transition-matrix distance (d_dyn)

    auto_calibrate : if True, call calibrate() on first find_neighbours() call
    """

    def __init__(
        self,
        beta_axis: float = 3.0,
        beta_opening: float = 2.0,
        beta_eccentricity: float = 1.0,
        beta_orientation: float = 1.5,
        gamma_levels: float = 1.0,
        gamma_levels_perp: float = 0.25,
        gamma_occupation: float = 1.5,
        gamma_dynamics: float = 1.0,
        auto_calibrate: bool = True,
    ) -> None:
        self.beta_axis = beta_axis
        self.beta_opening = beta_opening
        self.beta_eccentricity = beta_eccentricity
        self.beta_orientation = beta_orientation
        self.gamma_levels = gamma_levels
        self.gamma_levels_perp = gamma_levels_perp
        self.gamma_occupation = gamma_occupation
        self.gamma_dynamics = gamma_dynamics
        self.auto_calibrate = auto_calibrate

        # Internal calibration scales (set by calibrate())
        self._scale_axis: float = 1.0
        self._scale_opening: float = 1.0
        self._scale_eccentricity: float = 1.0
        self._scale_orientation: float = 1.0
        self._scale_levels: float = 1.0
        self._scale_levels_perp: float = 1.0
        self._scale_occ: float = 1.0
        self._scale_dyn: float = 1.0
        self._calibrated: bool = False
        # Packed training profile arrays for C++ kernels (set by calibrate())
        self._tr_packed: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Calibration                                                          #
    # ------------------------------------------------------------------ #

    def calibrate(self, profiles: List[CooperativeGeometryProfile]) -> None:
        """
        Estimate per-component standard deviations across a reference profile
        set and update internal scales so each component contributes in
        proportion to its empirical discriminative power.

        effective_weight_k = weight_k / scale_k

        where scale_k = std(component_k across calibration pairs).
        Minimum floors prevent noise amplification on low-variation components.
        """
        n = len(profiles)
        if n < 2:
            self._calibrated = True
            return

        rng = np.random.default_rng(42)
        idx = rng.choice(n, min(n, 100), replace=False)
        pairs = [
            (idx[k], idx[m])
            for k in range(len(idx))
            for m in range(k + 1, min(k + 8, len(idx)))
        ]

        ax, op, ec, ori, lv_coop, lv_perp, oc, dy = [], [], [], [], [], [], [], []

        for i, j in pairs:
            pi, pj = profiles[i], profiles[j]

            # Identity components from ConeIdentity
            if pi.cone_identity is not None and pj.cone_identity is not None:
                id_dist = ConeIdentity.distance(pi.cone_identity, pj.cone_identity)
                ax.append(id_dist["axis"])
                op.append(id_dist["opening"])
                ec.append(id_dist["eccentricity"])
                ori.append(id_dist["orientation"])

            # State: mean decomposition
            d_coop, d_perp = cooperative_mean_distance(
                pi.mu, pj.mu, pi.cooperative_direction, pj.cooperative_direction
            )
            lv_coop.append(d_coop)
            lv_perp.append(d_perp)

            # State: longitudinal
            if pi.partition_profile is not None and pj.partition_profile is not None:
                K = min(len(pi.partition_profile), len(pj.partition_profile))
                oc.append(occupation_distance(
                    pi.partition_profile[:K], pj.partition_profile[:K]
                ))
                dy.append(dynamics_distance(pi.transition_matrix, pj.transition_matrix))

        # Minimum "natural" scales prevent amplification of near-zero noise
        _min = {
            "axis":        np.pi / 12,  # ~15 degrees minimum meaningful axis difference
            "opening":     0.05,        # minimum meaningful opening profile difference
            "eccentricity": 0.10,       # minimum meaningful log-eccentricity difference
            "orientation": 0.20,        # minimum meaningful Procrustes frame distance
            "levels_coop": 0.15,
            "levels_perp": 0.15,
            "occ":         0.05,
            "dyn":         0.10,
        }

        def _safe_std(lst: list, key: str) -> float:
            s = float(np.std(lst)) if lst else 0.0
            return max(s, _min[key])

        self._scale_axis        = _safe_std(ax,      "axis")
        self._scale_opening     = _safe_std(op,      "opening")
        self._scale_eccentricity = _safe_std(ec,     "eccentricity")
        self._scale_orientation = _safe_std(ori,     "orientation")
        self._scale_levels      = _safe_std(lv_coop, "levels_coop")
        self._scale_levels_perp = _safe_std(lv_perp, "levels_perp")
        self._scale_occ         = _safe_std(oc,      "occ")
        self._scale_dyn         = _safe_std(dy,      "dyn")
        self._calibrated = True

        # Pack training profiles for C++ kernels (zero cost when _core absent).
        if _HAS_CPP:
            self._tr_packed = _pack_profiles(profiles)

    # ------------------------------------------------------------------ #
    # Distance computation                                                 #
    # ------------------------------------------------------------------ #

    def _raw_components(
        self,
        p_i: CooperativeGeometryProfile,
        p_j: CooperativeGeometryProfile,
    ) -> Dict[str, float]:
        """
        Return the 8 raw (unscaled, unweighted) component values plus a
        ``geo_reliable`` flag.

        Identity components are zeroed when geometry is unreliable (mirrors
        the masking in distance_components).  Used by fit_weights() to
        pre-compute the full component matrix once and reweight cheaply.

        Key order matches the COMP_ORDER constant in fit_weights():
          axis, opening, eccentricity, orientation,
          levels, levels_perp, occupation, dynamics
        """
        geo_reliable = p_i.geometry_reliable and p_j.geometry_reliable

        # Identity components (zeroed when geometry unreliable)
        axis = opening = eccentricity = orientation = 0.0
        if (geo_reliable
                and p_i.cone_identity is not None
                and p_j.cone_identity is not None):
            id_dist = ConeIdentity.distance(p_i.cone_identity, p_j.cone_identity)
            axis        = id_dist["axis"]
            opening     = id_dist["opening"]
            eccentricity = id_dist["eccentricity"]
            orientation  = id_dist["orientation"]

        # State: mean decomposition (levels_perp zeroed when geo unreliable)
        d_coop, d_perp = cooperative_mean_distance(
            p_i.mu, p_j.mu, p_i.cooperative_direction, p_j.cooperative_direction
        )
        if not geo_reliable:
            d_perp = 0.0

        # State: longitudinal
        occ = 0.0
        if p_i.partition_profile is not None and p_j.partition_profile is not None:
            K = min(len(p_i.partition_profile), len(p_j.partition_profile))
            occ = occupation_distance(p_i.partition_profile[:K], p_j.partition_profile[:K])

        dyn = 0.0
        if p_i.transition_matrix is not None and p_j.transition_matrix is not None:
            dyn = dynamics_distance(p_i.transition_matrix, p_j.transition_matrix)

        return {
            "axis":         float(axis),
            "opening":      float(opening),
            "eccentricity": float(eccentricity),
            "orientation":  float(orientation),
            "levels":       float(d_coop),
            "levels_perp":  float(d_perp),
            "occupation":   float(occ),
            "dynamics":     float(dyn),
            "geo_reliable": geo_reliable,
        }

    def distance_components(
        self,
        p_i: CooperativeGeometryProfile,
        p_j: CooperativeGeometryProfile,
    ) -> Dict[str, float]:
        """
        Compute all eight distance components between two profiles.

        Returns a dict with keys:
          'axis', 'opening', 'eccentricity', 'orientation'  -- identity
          'levels', 'levels_perp', 'occupation', 'dynamics' -- state
          'identity_distance'  -- sum of weighted identity components
          'total'              -- sum of all weighted components
          'geometry_reliable'  -- True when covariance geometry is trustworthy

        The 'identity_distance' key exposes how geometrically incompatible
        the two cones are, independent of their states.  Use it as an
        uncertainty signal: when the k-NN set has high average identity_distance,
        the prediction relies on structurally mismatched neighbours.
        """
        raw = self._raw_components(p_i, p_j)
        geo_reliable = raw["geo_reliable"]

        d_axis_w    = self.beta_axis         * raw["axis"]         / self._scale_axis
        d_opening_w = self.beta_opening      * raw["opening"]      / self._scale_opening
        d_ecc_w     = self.beta_eccentricity * raw["eccentricity"] / self._scale_eccentricity
        d_orient_w  = self.beta_orientation  * raw["orientation"]  / self._scale_orientation

        identity_dist = d_axis_w + d_opening_w + d_ecc_w + d_orient_w

        d_mu_coop_w = self.gamma_levels      * raw["levels"]      / self._scale_levels
        d_mu_perp_w = self.gamma_levels_perp * raw["levels_perp"] / self._scale_levels_perp
        d_occ_w     = self.gamma_occupation  * raw["occupation"]  / self._scale_occ
        d_dyn_w     = self.gamma_dynamics    * raw["dynamics"]    / self._scale_dyn

        total = identity_dist + d_mu_coop_w + d_mu_perp_w + d_occ_w + d_dyn_w

        return {
            "axis":              float(d_axis_w),
            "opening":           float(d_opening_w),
            "eccentricity":      float(d_ecc_w),
            "orientation":       float(d_orient_w),
            "levels":            float(d_mu_coop_w),
            "levels_perp":       float(d_mu_perp_w),
            "occupation":        float(d_occ_w),
            "dynamics":          float(d_dyn_w),
            "identity_distance": float(identity_dist),
            "total":             float(total),
            "geometry_reliable": geo_reliable,
        }

    def distance(
        self,
        p_i: CooperativeGeometryProfile,
        p_j: CooperativeGeometryProfile,
    ) -> float:
        """Scalar total distance between two profiles."""
        return self.distance_components(p_i, p_j)["total"]

    # ------------------------------------------------------------------ #
    # Weight learning                                                      #
    # ------------------------------------------------------------------ #

    def fit_weights(
        self,
        profiles: List[CooperativeGeometryProfile],
        obs_list: List[np.ndarray],
        T_list: List[np.ndarray],
        Y_list: List[np.ndarray],
        k: int = 10,
        alpha_local: float = 1.0,
        max_iter: int = 200,
        n_eval_patients: int = 80,
    ) -> None:
        """
        Learn beta/gamma multipliers from training data using LOO Y-prediction
        error as a fully self-supervised proxy (no ITE labels required).

        Principle: good weights → geometrically consistent neighbours →
        local Ridge predicts Y accurately → low LOO MSE.

        Optimises in log-space (exp enforces positivity) with Nelder-Mead.
        Updates self.beta_* and self.gamma_* in-place.

        Parameters
        ----------
        profiles      : training profiles (same list passed to calibrate)
        obs_list      : per-patient (N_i, d) feature observation matrices
        T_list        : per-patient (N_i, 1) treatment observation vectors
        Y_list        : per-patient (N_i,) outcome vectors
        k             : neighbourhood size for LOO proxy (default 10)
        alpha_local   : Ridge regularisation for local models
        max_iter      : Nelder-Mead maximum iterations
        n_eval_patients : number of patients sub-sampled for evaluation
        """
        from scipy.optimize import minimize
        from sklearn.linear_model import Ridge

        N = len(profiles)
        n_eval = min(n_eval_patients, N)

        rng = np.random.default_rng(42)
        eval_idx = rng.choice(N, n_eval, replace=False)

        # Component order (must match scales array below)
        _COMP = [
            "axis", "opening", "eccentricity", "orientation",
            "levels", "levels_perp", "occupation", "dynamics",
        ]

        # Pre-compute (n_eval, N, 8) raw component cache.
        # C++ path: single call replaces the O(n_eval * N) Python loop.
        if _HAS_CPP and self._tr_packed is not None:
            packed = self._tr_packed
            eval_idx_i32 = np.asarray(eval_idx, dtype=np.int32)
            raw_cache = _cpp.compute_raw_cache(
                eval_idx_i32,
                packed["axes"], packed["eccentricities"],
                packed["openings"], packed["anti_coops"],
                packed["mus"], packed["coop_dirs"],
                packed["partitions"], packed["transitions"],
                packed["geo_reliable"], packed["has_partition"],
                packed["has_transition"],
            )
        else:
            raw_cache = np.zeros((n_eval, N, 8))
            for ei, i in enumerate(eval_idx):
                for j in range(N):
                    raw = self._raw_components(profiles[i], profiles[j])
                    for c, key in enumerate(_COMP):
                        raw_cache[ei, j, c] = raw[key]

        scales = np.array([
            self._scale_axis,
            self._scale_opening,
            self._scale_eccentricity,
            self._scale_orientation,
            self._scale_levels,
            self._scale_levels_perp,
            self._scale_occ,
            self._scale_dyn,
        ])

        w0 = np.array([
            self.beta_axis,
            self.beta_opening,
            self.beta_eccentricity,
            self.beta_orientation,
            self.gamma_levels,
            self.gamma_levels_perp,
            self.gamma_occupation,
            self.gamma_dynamics,
        ])
        log_w0 = np.log(np.maximum(w0, 1e-6))

        if _HAS_CPP:
            # ── C++ fast path ─────────────────────────────────────────────
            # Pre-stack all patient observations into flat contiguous arrays.
            # Offsets[j] = start row of patient j; offsets[N] = total rows.
            offsets = np.zeros(N + 1, dtype=np.int32)
            for j in range(N):
                offsets[j + 1] = offsets[j] + len(obs_list[j])
            obs_flat = np.ascontiguousarray(
                np.vstack(obs_list), dtype=np.float64)
            T_flat = np.ascontiguousarray(
                np.vstack(T_list).ravel(), dtype=np.float64)
            Y_flat = np.ascontiguousarray(
                np.hstack(Y_list), dtype=np.float64)
            eval_idx_i32 = np.asarray(eval_idx, dtype=np.int32)
            raw_cache_c  = np.ascontiguousarray(raw_cache, dtype=np.float64)
            scales_c     = np.ascontiguousarray(scales,    dtype=np.float64)

            def _objective(log_weights: np.ndarray) -> float:
                weights_arr = np.ascontiguousarray(
                    np.exp(log_weights), dtype=np.float64)
                return _cpp.loo_objective(
                    raw_cache_c, obs_flat, T_flat, Y_flat,
                    offsets, eval_idx_i32,
                    weights_arr, scales_c,
                    k, alpha_local,
                )
        else:
            # ── Pure-Python fallback ──────────────────────────────────────
            def _objective(log_weights: np.ndarray) -> float:
                weights = np.exp(log_weights)
                eff_w = weights / scales

                D_matrix = raw_cache @ eff_w

                total_mse = 0.0
                for ei, i in enumerate(eval_idx):
                    D_i = D_matrix[ei].copy()
                    D_i[i] = np.inf
                    nn_idx = np.argsort(D_i)[:k]

                    X_local = np.vstack([obs_list[j] for j in nn_idx])
                    T_local = np.vstack([T_list[j] for j in nn_idx])
                    Y_local = np.hstack([Y_list[j] for j in nn_idx])

                    XT_local = np.hstack([X_local, T_local])
                    model = Ridge(alpha=alpha_local)
                    model.fit(XT_local, Y_local)

                    XT_i = np.hstack([obs_list[i], T_list[i]])
                    Y_hat_i = model.predict(XT_i)
                    total_mse += float(np.mean((Y_list[i] - Y_hat_i) ** 2))

                return total_mse / n_eval

        # L-BFGS-B with numerical Jacobian: ~30 gradient steps × 9 FD evals
        # instead of ~500 Nelder-Mead evaluations.  Weights are bounded to
        # [0.001, 50] in original space → [-6.9, 3.9] in log-space.
        bounds = [(-6.9, 3.9)] * 8
        result = minimize(
            _objective,
            log_w0,
            method="L-BFGS-B",
            jac="2-point",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-7, "gtol": 1e-3},
        )

        learned = np.exp(result.x)
        self.beta_axis         = float(learned[0])
        self.beta_opening      = float(learned[1])
        self.beta_eccentricity = float(learned[2])
        self.beta_orientation  = float(learned[3])
        self.gamma_levels      = float(learned[4])
        self.gamma_levels_perp = float(learned[5])
        self.gamma_occupation  = float(learned[6])
        self.gamma_dynamics    = float(learned[7])

    # ------------------------------------------------------------------ #
    # Neighbour search                                                     #
    # ------------------------------------------------------------------ #

    def find_neighbours(
        self,
        query_profile: CooperativeGeometryProfile,
        training_profiles: List[CooperativeGeometryProfile],
        k: int,
        exclude_idx: Optional[int] = None,
        return_distances: bool = False,
    ):
        """
        Return the indices of the k nearest training profiles to the query,
        ranked by the unified eight-component total distance.

        No hard gate is applied.  Geometrically incompatible patients receive
        higher identity_distance, inflating their total distance and pushing
        them down the ranking — the geometry does the weighting implicitly.

        Parameters
        ----------
        query_profile : profile of the test patient
        training_profiles : list of training profiles
        k : number of nearest neighbours to return
        exclude_idx : index to skip (for leave-one-out cross-validation)
        return_distances : if True, return (indices, distances) tuple instead
            of just indices.  Distances are the raw total distance values for
            the k selected neighbours, in ranked order (nearest first).
        """
        if self.auto_calibrate and not self._calibrated:
            self.calibrate(training_profiles)

        n = len(training_profiles)
        k_eff = min(k, n - (1 if exclude_idx is not None else 0))

        # ── C++ fast path ─────────────────────────────────────────────────
        if _HAS_CPP and self._tr_packed is not None:
            packed = self._tr_packed
            dp1    = packed["dp1"]
            K      = packed["K"]
            q      = _pack_query(query_profile, K, dp1)
            weights = np.array([
                self.beta_axis, self.beta_opening,
                self.beta_eccentricity, self.beta_orientation,
                self.gamma_levels, self.gamma_levels_perp,
                self.gamma_occupation, self.gamma_dynamics,
            ], dtype=np.float64)
            scales = np.array([
                self._scale_axis, self._scale_opening,
                self._scale_eccentricity, self._scale_orientation,
                self._scale_levels, self._scale_levels_perp,
                self._scale_occ, self._scale_dyn,
            ], dtype=np.float64)
            dists = _cpp.compute_distances(
                q["axis"], q["ecc"], q["opening"], q["anti_coop"],
                q["mu"], q["coop_dir"], q["partition"], q["transition"],
                q["geo_reliable"], q["has_partition"], q["has_transition"],
                packed["axes"], packed["eccentricities"],
                packed["openings"], packed["anti_coops"],
                packed["mus"], packed["coop_dirs"],
                packed["partitions"], packed["transitions"],
                packed["geo_reliable"], packed["has_partition"],
                packed["has_transition"],
                weights, scales,
            )
            if exclude_idx is not None:
                dists[exclude_idx] = np.inf
            order = np.argsort(dists)[:k_eff]
            if return_distances:
                return order, dists[order]
            return order

        # ── Pure-Python fallback ──────────────────────────────────────────
        dists = np.full(n, np.inf)
        for j, p_j in enumerate(training_profiles):
            if j == exclude_idx:
                continue
            dists[j] = self.distance(query_profile, p_j)
        order = np.argsort(dists)[:k_eff]
        if return_distances:
            return order, dists[order]
        return order
