"""
ICGHVRTMatcher: five-component cooperative geometry distance for patient matching.

d(i, j) = alpha_1*d_mu + alpha_2*d_w + alpha_3*d_sigma + alpha_4*d_occ + alpha_5*d_dyn

Matching modes
--------------
cascade=False (default)
    Exhaustive weighted-sum search across all training patients.

cascade=True  (spec §5.2)
    Three-level hierarchical search:
      Level 1 — Direction filter: retain j where d_w(i,j) < direction_gate
      Level 2 — Shape match:     among survivors, keep k2 nearest by d_sigma
      Level 3 — Occ+Dyn match:  among those, keep k nearest by d_occ + d_dyn
    Falls back to exhaustive when Level 1 produces fewer than k survivors.
    Cascade info is stored in last_cascade_info after each call.
"""
import numpy as np
from typing import Dict, List, Optional

from .profile import CooperativeGeometryProfile
from .distances import (
    euclidean_mean_distance,
    cooperative_direction_distance,
    log_euclidean_distance,
    occupation_distance,
    dynamics_distance,
)


class ICGHVRTMatcher:
    """
    Five-component distance for matching patients on their personal cooperative manifolds.

    Parameters
    ----------
    alpha_levels : weight for Euclidean mean distance (d_mu)
    alpha_direction : weight for cooperative direction alignment (d_w)
    alpha_shape : weight for Log-Euclidean manifold curvature (d_sigma)
    alpha_occupation : weight for Wasserstein occupation distance (d_occ)
    alpha_dynamics : weight for Frobenius transition-matrix distance (d_dyn)
    direction_gate : angular threshold (radians) for the cooperative direction gate.
                     When d_w exceeds this threshold the match is flagged as
                     geometrically unsafe.  Default pi/4 (~45 degrees).
    auto_calibrate : if True, call calibrate() before each find_neighbours()
                     unless already calibrated.
    cascade : if True, use the hierarchical three-level cascade (spec §5.2) instead
              of exhaustive weighted-sum search.  Default False.
    cascade_k2 : number of Level-2 (shape-match) candidates to carry into Level 3.
                 None = auto (3*k, capped at the Level-1 survivor count).
    """

    def __init__(
        self,
        alpha_levels: float = 1.0,
        alpha_direction: float = 2.0,
        alpha_shape: float = 1.0,
        alpha_occupation: float = 1.5,
        alpha_dynamics: float = 1.0,
        direction_gate: float = np.pi / 4,
        auto_calibrate: bool = True,
        cascade: bool = False,
        cascade_k2: Optional[int] = None,
    ) -> None:
        self.alpha_levels = alpha_levels
        self.alpha_direction = alpha_direction
        self.alpha_shape = alpha_shape
        self.alpha_occupation = alpha_occupation
        self.alpha_dynamics = alpha_dynamics
        self.direction_gate = direction_gate
        self.auto_calibrate = auto_calibrate
        self.cascade = cascade
        self.cascade_k2 = cascade_k2

        # Internal calibration scales (set by calibrate())
        self._scale_levels: float = 1.0
        self._scale_direction: float = 1.0
        self._scale_shape: float = 1.0
        self._scale_occ: float = 1.0
        self._scale_dyn: float = 1.0
        self._calibrated: bool = False

        # Populated by find_neighbours() when cascade=True
        self.last_cascade_info: Dict = {}

    # ------------------------------------------------------------------ #
    # Calibration                                                          #
    # ------------------------------------------------------------------ #

    def calibrate(self, profiles: List[CooperativeGeometryProfile]) -> None:
        """
        Estimate per-component standard deviations across a reference profile set
        and update internal scales so each component contributes proportionally
        to its discriminative power.

        alpha_k_effective = alpha_k / scale_k

        where scale_k = std(component_k across calibration pairs).
        """
        n = len(profiles)
        if n < 2:
            self._calibrated = True
            return

        rng = np.random.default_rng(42)
        idx = rng.choice(n, min(n, 100), replace=False)
        pairs = [(idx[k], idx[l]) for k in range(len(idx)) for l in range(k + 1, min(k + 8, len(idx)))]

        lv, dw, ds, oc, dy = [], [], [], [], []
        for i, j in pairs:
            pi, pj = profiles[i], profiles[j]
            lv.append(euclidean_mean_distance(pi.mu, pj.mu))
            dw.append(cooperative_direction_distance(pi.cooperative_direction, pj.cooperative_direction))
            ds.append(log_euclidean_distance(pi.sigma, pj.sigma))
            if pi.partition_profile is not None and pj.partition_profile is not None:
                K = min(len(pi.partition_profile), len(pj.partition_profile))
                oc.append(occupation_distance(pi.partition_profile[:K], pj.partition_profile[:K]))
                dy.append(dynamics_distance(pi.transition_matrix, pj.transition_matrix))

        # Minimum "natural" scales prevent amplification of near-zero noise
        # (e.g., d_sigma ≈ 0 for patients with identical covariance structure
        # should not dominate d_mu when all Sigmas are nearly equal).
        _min = {
            "levels":    0.20,       # minimum meaningful Euclidean mean distance
            "direction": np.pi / 8,  # ~22 degrees
            "shape":     0.30,       # minimum meaningful log-Euclidean distance
            "occ":       0.05,       # minimum meaningful W1 occupation distance
            "dyn":       0.10,       # minimum meaningful Frobenius dynamics distance
        }

        def _safe_std(lst: list, key: str) -> float:
            s = float(np.std(lst)) if lst else 0.0
            return max(s, _min[key])

        self._scale_levels    = _safe_std(lv, "levels")
        self._scale_direction = _safe_std(dw, "direction")
        self._scale_shape     = _safe_std(ds, "shape")
        self._scale_occ       = _safe_std(oc, "occ")
        self._scale_dyn       = _safe_std(dy, "dyn")
        self._calibrated = True

    # ------------------------------------------------------------------ #
    # Distance computation                                                 #
    # ------------------------------------------------------------------ #

    def distance_components(
        self,
        p_i: CooperativeGeometryProfile,
        p_j: CooperativeGeometryProfile,
    ) -> Dict[str, float]:
        """
        Compute all five distance components between two profiles.

        Returns a dict with keys:
          'levels', 'direction', 'shape', 'occupation', 'dynamics',
          'total', 'direction_gate_passed'.
        """
        # Normalised raw components
        d_mu = euclidean_mean_distance(p_i.mu, p_j.mu) / self._scale_levels
        d_w_raw = cooperative_direction_distance(
            p_i.cooperative_direction, p_j.cooperative_direction
        )
        d_w = d_w_raw / self._scale_direction
        d_sigma = log_euclidean_distance(p_i.sigma, p_j.sigma) / self._scale_shape

        gate_passed = d_w_raw <= self.direction_gate

        d_occ = 0.0
        if p_i.partition_profile is not None and p_j.partition_profile is not None:
            K = min(len(p_i.partition_profile), len(p_j.partition_profile))
            d_occ = occupation_distance(
                p_i.partition_profile[:K], p_j.partition_profile[:K]
            ) / self._scale_occ

        d_dyn = 0.0
        if p_i.transition_matrix is not None and p_j.transition_matrix is not None:
            d_dyn = dynamics_distance(
                p_i.transition_matrix, p_j.transition_matrix
            ) / self._scale_dyn

        total = (
            self.alpha_levels * d_mu
            + self.alpha_direction * d_w
            + self.alpha_shape * d_sigma
            + self.alpha_occupation * d_occ
            + self.alpha_dynamics * d_dyn
        )

        return {
            "levels": float(self.alpha_levels * d_mu),
            "direction": float(self.alpha_direction * d_w),
            "shape": float(self.alpha_shape * d_sigma),
            "occupation": float(self.alpha_occupation * d_occ),
            "dynamics": float(self.alpha_dynamics * d_dyn),
            "total": float(total),
            "direction_gate_passed": gate_passed,
        }

    def distance(
        self,
        p_i: CooperativeGeometryProfile,
        p_j: CooperativeGeometryProfile,
    ) -> float:
        """Scalar five-component distance between two profiles."""
        return self.distance_components(p_i, p_j)["total"]

    # ------------------------------------------------------------------ #
    # Neighbour search                                                     #
    # ------------------------------------------------------------------ #

    def find_neighbours(
        self,
        query_profile: CooperativeGeometryProfile,
        training_profiles: List[CooperativeGeometryProfile],
        k: int,
        exclude_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return the indices of the k nearest training profiles to the query.

        Parameters
        ----------
        query_profile : profile of the test patient
        training_profiles : list of training profiles
        k : number of nearest neighbours to return
        exclude_idx : index to skip (for leave-one-out cross-validation)

        When cascade=True, self.last_cascade_info is populated with:
          level1_size  -- candidates surviving the direction filter
          level2_size  -- candidates surviving the shape match
          level3_size  -- final neighbourhood size (= k unless fallback)
          fallback     -- True if Level 1 was too thin and exhaustive was used
        """
        if self.auto_calibrate and not self._calibrated:
            self.calibrate(training_profiles)

        n = len(training_profiles)
        k_eff = min(k, n - (1 if exclude_idx is not None else 0))

        if self.cascade:
            return self._cascade_search(query_profile, training_profiles, k_eff, exclude_idx)
        return self._exhaustive_search(query_profile, training_profiles, k_eff, exclude_idx)

    # ------------------------------------------------------------------ #
    # Private search implementations                                       #
    # ------------------------------------------------------------------ #

    def _exhaustive_search(
        self,
        query_profile: CooperativeGeometryProfile,
        training_profiles: List[CooperativeGeometryProfile],
        k: int,
        exclude_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Weighted-sum search across all training patients."""
        n = len(training_profiles)
        dists = np.full(n, np.inf)
        for j, p_j in enumerate(training_profiles):
            if j == exclude_idx:
                continue
            dists[j] = self.distance(query_profile, p_j)
        return np.argsort(dists)[:k]

    def _cascade_search(
        self,
        query_profile: CooperativeGeometryProfile,
        training_profiles: List[CooperativeGeometryProfile],
        k: int,
        exclude_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Hierarchical three-level cascade (spec §5.2).

        Level 1 — Cooperative Direction Filter
            Retain j where d_w(i, j) < direction_gate.
            O(n · d).

        Level 2 — Manifold Shape Match
            Among Level-1 survivors, keep the k2 nearest by d_sigma.
            O(|L1| · d³).

        Level 3 — Occupation + Dynamics Match
            Among Level-2 survivors, keep the k nearest by
            alpha_occupation*d_occ + alpha_dynamics*d_dyn.
            Falls back to full weighted-sum within Level-2 when
            neither partition profiles nor transition matrices are available.
            O(|L2| · K²).
        """
        # ── Level 1: Cooperative Direction Filter ──────────────────── #
        level1: List[int] = []
        for j, p_j in enumerate(training_profiles):
            if j == exclude_idx:
                continue
            d_w_raw = cooperative_direction_distance(
                query_profile.cooperative_direction, p_j.cooperative_direction
            )
            if d_w_raw < self.direction_gate:
                level1.append(j)

        # Fallback when Level 1 is too thin to supply k neighbours
        if len(level1) < k:
            self.last_cascade_info = {
                "level1_size": len(level1),
                "level2_size": 0,
                "level3_size": 0,
                "fallback": True,
            }
            return self._exhaustive_search(query_profile, training_profiles, k, exclude_idx)

        # ── Level 2: Manifold Shape Match ──────────────────────────── #
        k2 = (
            self.cascade_k2
            if self.cascade_k2 is not None
            else min(k * 3, len(level1))
        )
        k2 = min(k2, len(level1))

        shape_dists = sorted(
            (
                (j, log_euclidean_distance(query_profile.sigma, training_profiles[j].sigma)
                 / self._scale_shape)
                for j in level1
            ),
            key=lambda x: x[1],
        )
        level2 = [j for j, _ in shape_dists[:k2]]

        # ── Level 3: Occupation + Dynamics Match ───────────────────── #
        dyn_dists = []
        has_longitudinal = False
        for j in level2:
            p_j = training_profiles[j]
            d_occ = 0.0
            if query_profile.partition_profile is not None and p_j.partition_profile is not None:
                K = min(len(query_profile.partition_profile), len(p_j.partition_profile))
                d_occ = (
                    occupation_distance(
                        query_profile.partition_profile[:K], p_j.partition_profile[:K]
                    )
                    / self._scale_occ
                )
                has_longitudinal = True
            d_dyn = 0.0
            if query_profile.transition_matrix is not None and p_j.transition_matrix is not None:
                d_dyn = (
                    dynamics_distance(query_profile.transition_matrix, p_j.transition_matrix)
                    / self._scale_dyn
                )
                has_longitudinal = True
            dyn_dists.append((j, self.alpha_occupation * d_occ + self.alpha_dynamics * d_dyn))

        # When occ+dyn provide no signal (static-only profiles), fall back to
        # full weighted-sum distance within the Level-2 candidate set so Level 3
        # still provides useful ranking rather than arbitrary ordering.
        if not has_longitudinal:
            dyn_dists = [
                (j, self.distance(query_profile, training_profiles[j]))
                for j in level2
            ]

        dyn_dists.sort(key=lambda x: x[1])
        final_idx = np.array([j for j, _ in dyn_dists[:k]])

        self.last_cascade_info = {
            "level1_size": len(level1),
            "level2_size": len(level2),
            "level3_size": int(len(final_idx)),
            "fallback": False,
        }
        return final_idx
