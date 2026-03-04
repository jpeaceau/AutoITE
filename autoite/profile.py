"""
CooperativeGeometryProfile: per-patient personal cooperative cone.

Each patient is represented as their personal cooperative cone family,
characterised by the eigendecomposition of the cooperative geometry operator
C_i = Sigma_i^{-1/2} A Sigma_i^{-1/2} = w_i w_i^T - Sigma_i^{-1}
where A = 11^T - I is the universal cooperative form.

The cone has exactly one positive eigenvalue (cooperative axis) and d-1
negative eigenvalues (anti-cooperative frame).  In whitened space the
cone is circular; in original space it is generally elliptical.
"""
import numpy as np
from numpy.linalg import eigh
from dataclasses import dataclass
from typing import Optional

from .distances import ConeIdentity

# Consistency factor for MAD → σ estimate (= 1/Φ⁻¹(0.75) ≈ 1.4826).
# Matches the normalisation used by hvrt.HART and hvrt.PyramidHART so that
# manually-whitened Z agrees exactly with the model's internal statistics.
_MAD_SCALE = 1.4826


@dataclass
class SharedHVRT:
    """
    Shared HVRT fitted on a reference population, with partitions ordered by E[T].
    Used to make partition profiles comparable across patients.
    """
    model: object              # HVRT instance
    id_to_rank: dict           # raw leaf-ID → 0-indexed rank ordered by E_T
    n_partitions: int          # number of leaf partitions

    def assign_partitions(self, observations: np.ndarray) -> np.ndarray:
        """Return 0-indexed partition ranks for each observation row."""
        raw_ids = self.model.apply_raw(observations)
        return np.array([self.id_to_rank.get(int(rid), 0) for rid in raw_ids])


def fit_shared_hvrt(
    X_all: np.ndarray,
    n_partitions: int = 8,
    y_weight: float = 0.0,
    random_state: int = 42,
    model_class=None,
) -> SharedHVRT:
    """
    Fit a shared HVRT/HART/PyramidHART on pooled feature data and return a
    SharedHVRT wrapper with partitions ordered by mean cooperative statistic.

    Parameters
    ----------
    X_all : (n_total, d) pooled feature observations from all training patients
    n_partitions : number of HVRT leaf partitions
    y_weight : 0.0 for pure cooperative geometry (causal applications)
    random_state : reproducibility seed
    model_class : HVRT-compatible model class; None defaults to hvrt.HVRT (cone).
                  Pass hvrt.PyramidHART for MAD-based ℓ₁ pyramid geometry.
    """
    import hvrt as _hvrt
    if model_class is None:
        model_class = _hvrt.HVRT
    model = model_class(
        n_partitions=n_partitions,
        y_weight=y_weight,
        auto_tune=False,
        random_state=random_state,
    )
    model.fit(X_all)

    gs = model.geometry_stats(X_all)
    partitions = gs.get("partitions", [])
    if not partitions:
        # Fallback: no geometry_stats partition info, use sorted IDs
        raw_ids = model.apply_raw(X_all)
        unique_ids = sorted(set(raw_ids.tolist()))
        id_to_rank = {v: i for i, v in enumerate(unique_ids)}
        return SharedHVRT(model=model, id_to_rank=id_to_rank, n_partitions=len(unique_ids))

    # PyramidHART partitions by A_mean (less negative = more cooperative);
    # HVRT / HART partitions by E_T (same ordering logic, different statistic).
    _is_pyramid = isinstance(model, getattr(_hvrt, "PyramidHART", type(None)))
    if _is_pyramid:
        def sort_key(p):
            return p.get("A_mean", 0.0)
    else:
        def sort_key(p):
            return p.get("E_T", 0.0)

    partitions_sorted = sorted(partitions, key=sort_key)
    id_to_rank = {p["id"]: i for i, p in enumerate(partitions_sorted)}
    return SharedHVRT(model=model, id_to_rank=id_to_rank, n_partitions=len(partitions_sorted))


def pool_synergy_whitened(
    X_list: list,
    spike_fraction: float = 0.05,
    regularization: float = 1e-4,
) -> np.ndarray:
    """
    Two-stage synergy pooling (PyramidHART → HVRT pipeline).

    Stage 1 — MAD-whiten each patient's observations (spike-robust normalisation,
               matching hvrt.PyramidHART's internal scaling exactly).

    Stage 2 — Detect and remove spike samples globally using the |A|/‖z‖₁ score
               (Proposition 1.3: a single dominant feature cancels from A exactly,
               so high |A|/‖z‖₁ ≈ 1 unmistakably flags sign-divergent outliers).

    The returned Z_bulk is spike-free and suitable for fitting HVRT (not
    PyramidHART), so Theorem 3 (noise invariance of E[T]) holds cleanly on
    the remaining bulk without single-feature spike interference.
    """
    from hvrt import detect_spikes as _detect_spikes

    whitened = []
    for X in X_list:
        obs = np.atleast_2d(X)
        med = np.median(obs, axis=0)
        mad = np.median(np.abs(obs - med), axis=0)
        sigma_hat = np.maximum(_MAD_SCALE * mad, regularization)
        whitened.append((obs - med) / sigma_hat)
    Z_all = np.vstack(whitened)

    spike_mask, _ = _detect_spikes(Z_all, spike_fraction=spike_fraction)
    Z_bulk = Z_all[~spike_mask]

    # Guard: if spike removal was too aggressive, fall back to full Z
    if len(Z_bulk) < max(10, len(Z_all) // 4):
        return Z_all
    return Z_bulk


def pool_whitened_observations(
    X_list: list,
    regularization: float = 1e-4,
    use_mad: bool = False,
) -> np.ndarray:
    """
    Whiten each patient's observations, then pool into a single (sum(N_i), d)
    array for shared HVRT/HART fitting.

    When use_mad=False (default): whitens by personal Sigma^{-1/2} (SD-based,
    ellipsoidal unit ball — cone geometry).

    When use_mad=True: whitens by personal MAD (axis-aligned median/MAD
    rescaling — pyramid/cross-polytope unit ball for HART geometry).

    In either whitened space the shared HVRT/HART partitions form a universal
    reference frame geometrically unbiased across patients.
    """
    whitened = []
    for X in X_list:
        obs = np.atleast_2d(X)
        n_obs, d = obs.shape
        if use_mad:
            med = np.median(obs, axis=0)
            mad = np.median(np.abs(obs - med), axis=0)
            # _MAD_SCALE = 1.4826: consistency factor making MAD a consistent
            # estimator of σ for Gaussian data (= 1/Φ⁻¹(0.75)).  Matches the
            # normalisation used internally by hvrt.HART / hvrt.PyramidHART.
            sigma_hat = np.maximum(_MAD_SCALE * mad, regularization)
            whitened.append((obs - med) / sigma_hat)
        else:
            mu = obs.mean(axis=0)
            if n_obs >= 2:
                cov = (np.cov(obs, rowvar=False) if d > 1
                       else np.array([[float(np.var(obs, ddof=1))]]))
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
            else:
                cov = np.eye(d)
            cov = cov + regularization * np.eye(d)
            eigenvalues, eigenvectors = eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            sigma_inv_sqrt = (eigenvectors
                              @ np.diag(1.0 / np.sqrt(eigenvalues))
                              @ eigenvectors.T)
            whitened.append((obs - mu) @ sigma_inv_sqrt.T)
    return np.vstack(whitened)


@dataclass
class CooperativeGeometryProfile:
    """
    Personal cooperative geometry profile for a single patient.

    Represents the patient as their personal cooperative cone family.
    C_i = w_i w_i^T - Sigma_i^{-1}  (the cooperative geometry operator)

    The cone identity captures the full shape of the cone: axis direction,
    directional half-angles, eccentricity, and anti-cooperative frame.

    Attributes
    ----------
    mu : (d,) mean feature vector
    sigma : (d, d) regularised covariance matrix
    cooperative_direction : (d,) w = Sigma^{-1/2} 1
    cone_angle : scalar theta in radians (mean opening angle; invariant for fixed d)
    cooperative_operator : (d, d) C = w w^T - Sigma^{-1}
    cone_identity : ConeIdentity — full cone eigendecomposition (axis, opening
                    profile, eccentricity, anti-cooperative frame)
    partition_profile : (K,) occupation histogram — None if insufficient data
    transition_matrix : (K, K) Markov transition probabilities — None if insufficient data
    cooperative_trajectory : (n_obs,) T(t) = S(t)^2 - Q(t) — None if insufficient data
    n_partitions : K used for HVRT
    n_observations : number of observations used to build this profile
    cone_degenerate : bool — HVRT flag; True means cooperative cone is malformed
    frac_in_cone : float — fraction of patient's observations inside the cone
                   (nan when geometry_stats unavailable)
    cooperation_ratio : float — HVRT cooperation strength (nan when unavailable)
    sigma_condition_number : float — max/min eigenvalue of estimated Sigma;
                             high (>100) indicates unreliable covariance geometry
    """
    mu: np.ndarray
    sigma: np.ndarray
    cooperative_direction: np.ndarray
    cone_angle: float
    cooperative_operator: np.ndarray
    cone_identity: Optional[ConeIdentity] = None
    partition_profile: Optional[np.ndarray] = None
    transition_matrix: Optional[np.ndarray] = None
    cooperative_trajectory: Optional[np.ndarray] = None
    n_partitions: Optional[int] = None
    n_observations: int = 0
    cone_degenerate: bool = False
    frac_in_cone: float = float("nan")
    cooperation_ratio: float = float("nan")
    sigma_condition_number: float = 1.0

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_longitudinal(
        cls,
        observations: np.ndarray,
        shared_hvrt: Optional[SharedHVRT] = None,
        y_weight: float = 0.0,
        n_partitions: int = 8,
        regularization: float = 1e-4,
        use_mad: bool = False,
        spike_fraction: float = 0.0,
    ) -> "CooperativeGeometryProfile":
        """
        Build a cooperative geometry profile from longitudinal feature observations.

        Parameters
        ----------
        observations : (n_obs, d) feature matrix (no treatment or outcome columns)
        shared_hvrt : pre-fitted SharedHVRT for standardised partition comparison.
                      If None, a personal HVRT is fitted when n_obs is sufficient.
        y_weight : HVRT y_weight; must be 0.0 for causal applications to prevent
                   outcome leakage into partition structure.
        n_partitions : number of HVRT partitions (used when fitting personal HVRT)
        regularization : covariance regularisation lambda
        use_mad : if True, use MAD-based pyramid geometry (HART) instead of the
                  default SD-based cone geometry (HVRT).
        """
        observations = np.atleast_2d(observations)
        n_obs, d = observations.shape

        if use_mad:
            # ── PyramidHART path: ℓ₁ pyramid geometry ───────────────────
            # Stage 1: Personal MAD dispersion.
            # sigma_hat = 1.4826 * MAD is the consistent σ estimator used
            # by hvrt.HART / hvrt.PyramidHART internally (confirmed numerically).
            med = np.median(observations, axis=0)
            mad = np.median(np.abs(observations - med), axis=0)
            sigma_hat = np.maximum(_MAD_SCALE * mad, regularization)
            mu = med

            # Stage 2: Diagonal precision M^{-1} = diag(1/sigma_hat_j)
            m_inv  = 1.0 / sigma_hat    # (d,)
            m_inv2 = m_inv ** 2         # (d,)

            # Stage 3: Cooperative direction w = M^{-1}·1 / ||M^{-1}·1||
            w = m_inv / np.linalg.norm(m_inv)

            # Condition number proxy: (σ_hat_max / σ_hat_min)²
            sigma_cond = float(m_inv2.max() / (m_inv2.min() + 1e-12))

            # sigma field: store diag(sigma_hat²) as covariance stand-in
            cov = np.diag(sigma_hat ** 2)

            # Stage 4: Cone angle (in whitened space wSw ≈ 1 for all d)
            cone_angle = float(np.arccos(np.clip(1.0 / np.sqrt(max(float(d), 1e-12)), -1.0, 1.0)))

            # Stage 5: Operator C_MAD = ww^T − M^{-2} (cone approximation of
            # the polyhedral pyramid; ConeIdentity still gives meaningful
            # identity distances capturing MAD ellipsoid shape differences).
            sigma_inv = np.diag(m_inv2)
            C = np.outer(w, w) - sigma_inv
            cone_id = ConeIdentity.from_operator(C)

            # Stage 6: Element-wise sigma_hat whitening (matches hvrt.PyramidHART)
            Z = (observations - med) / sigma_hat  # (n_obs, d)

            # Note: spike_fraction is intentionally NOT applied here.
            # Per-patient filtering would remove valid anti-cooperative observations
            # (e.g. uncoupled patients' tau=0 signal) and bias partition profiles toward
            # cooperation. The synergy filtering belongs only at the global pool level
            # in pool_synergy_whitened(), not per-patient.

        else:
            # ── SD path: cone (ellipsoidal) geometry ────────────────────
            # Stage 1: Personal statistics
            mu = observations.mean(axis=0)

            if n_obs >= 2:
                if d == 1:
                    cov = np.array([[float(np.var(observations, ddof=1))]])
                else:
                    cov = np.cov(observations, rowvar=False)
                    if cov.ndim == 0:
                        cov = np.array([[float(cov)]])
            else:
                cov = np.eye(d)

            cov = cov + regularization * np.eye(d)

            # Stage 2: Eigendecomposition for Sigma^{-1/2} and Sigma^{-1}
            eigenvalues, eigenvectors = eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            sigma_cond = float(eigenvalues[-1] / eigenvalues[0])

            inv_sqrt_vals = 1.0 / np.sqrt(eigenvalues)
            sigma_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_vals) @ eigenvectors.T

            inv_vals = 1.0 / eigenvalues
            sigma_inv = eigenvectors @ np.diag(inv_vals) @ eigenvectors.T

            # Stage 3: Cooperative direction: w = Sigma^{-1/2} * 1
            ones = np.ones(d)
            w = sigma_inv_sqrt @ ones

            # Stage 4: Cone angle
            wSw = float(w @ cov @ w)
            cone_angle = float(np.arccos(np.clip(1.0 / np.sqrt(max(wSw, 1e-12)), -1.0, 1.0)))

            # Stage 5: Cooperative operator: C = w w^T - Sigma^{-1}
            C = np.outer(w, w) - sigma_inv
            cone_id = ConeIdentity.from_operator(C)

            # Stage 6: Whitened observations
            Z = (observations - mu) @ sigma_inv_sqrt.T  # (n_obs, d)

        # ── Cooperative trajectory ──────────────────────────────────────
        # Cone (HVRT/HART):    T = S² − Q = 2∑_{i<j} z_i z_j  (Theorem 2)
        # Pyramid (PyramidHART): A = |S| − ‖z‖₁  (Proposition 1, §2.3)
        #   A ≤ 0 always; A = 0 iff all components share the same sign.
        #   Outlier-cancellation: a single dominant spike leaves A unchanged.
        S = Z.sum(axis=1)
        if use_mad:
            Q1 = np.abs(Z).sum(axis=1)          # ℓ₁ norm
            cooperative_trajectory = np.abs(S) - Q1   # A ≤ 0
        else:
            Q = (Z ** 2).sum(axis=1)            # ℓ₂² norm
            cooperative_trajectory = S ** 2 - Q       # T

        # ── HVRT partition profile and transition matrix ─────────────
        partition_profile = None
        transition_matrix = None
        actual_K = n_partitions
        cone_degenerate = False
        frac_in_cone = float("nan")
        cooperation_ratio = float("nan")

        # Require at least 2*K observations for a useful histogram.
        min_for_hvrt = max(n_partitions * 2, 10)

        if n_obs >= min_for_hvrt:
            try:
                if shared_hvrt is not None:
                    ranked_ids = shared_hvrt.assign_partitions(Z)
                    actual_K = shared_hvrt.n_partitions
                    # Per-patient geometry quality from the shared model.
                    # Key names differ between HVRT/HART and PyramidHART.
                    try:
                        _gs = shared_hvrt.model.geometry_stats(Z)
                        if "cone_degenerate" in _gs:
                            # HVRT / HART
                            cone_degenerate   = bool(_gs.get("cone_degenerate", False))
                            frac_in_cone      = float(_gs.get("frac_in_cone", float("nan")))
                        else:
                            # PyramidHART — no cone_degenerate flag; use
                            # sign_consistent_fraction as the frac_in_cone proxy.
                            cone_degenerate   = False
                            frac_in_cone      = float(_gs.get("sign_consistent_fraction", float("nan")))
                        cooperation_ratio = float(_gs.get("cooperation_ratio", float("nan")))
                    except Exception:
                        pass
                else:
                    import hvrt as _hvrt
                    personal_hvrt = _hvrt.HVRT(
                        n_partitions=n_partitions,
                        y_weight=y_weight,
                        auto_tune=False,
                        random_state=42,
                    )
                    personal_hvrt.fit(Z)
                    raw_ids = personal_hvrt.apply_raw(Z)
                    unique_ids = sorted(set(raw_ids.tolist()))
                    id_to_rank = {v: i for i, v in enumerate(unique_ids)}
                    ranked_ids = np.array([id_to_rank[int(r)] for r in raw_ids])
                    actual_K = len(unique_ids)

                # Occupation histogram (normalised)
                counts = np.bincount(ranked_ids, minlength=actual_K).astype(float)
                partition_profile = counts / (counts.sum() + 1e-12)

                # Transition matrix (Markov estimates)
                M = np.zeros((actual_K, actual_K))
                for t in range(len(ranked_ids) - 1):
                    k, m = int(ranked_ids[t]), int(ranked_ids[t + 1])
                    if 0 <= k < actual_K and 0 <= m < actual_K:
                        M[k, m] += 1.0
                row_sums = M.sum(axis=1, keepdims=True)
                transition_matrix = M / np.where(row_sums > 0, row_sums, 1.0)

            except Exception:
                pass  # Fall back to static three-component profile

        return cls(
            mu=mu,
            sigma=cov,
            cooperative_direction=w,
            cone_angle=cone_angle,
            cooperative_operator=C,
            cone_identity=cone_id,
            partition_profile=partition_profile,
            transition_matrix=transition_matrix,
            cooperative_trajectory=cooperative_trajectory,
            n_partitions=actual_K,
            n_observations=n_obs,
            cone_degenerate=cone_degenerate,
            frac_in_cone=frac_in_cone,
            cooperation_ratio=cooperation_ratio,
            sigma_condition_number=sigma_cond,
        )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def has_longitudinal(self) -> bool:
        """True when partition profile and transition matrix are available."""
        return self.partition_profile is not None

    @property
    def geometry_reliable(self) -> bool:
        """
        True when the cooperative geometry components (d_sigma, d_w) are
        trustworthy for matching.

        Returns False when ANY of:
          - cone is flagged degenerate by HVRT
          - Sigma condition number > 100 (poorly estimated covariance; N/d² too small)
          - frac_in_cone < 0.05 (almost no observations inside the cooperative cone)
          - frac_in_cone > 0.95 (cone entirely fills the space; trivially aligned)
        """
        if self.cone_degenerate:
            return False
        if self.sigma_condition_number > 100.0:
            return False
        if not np.isnan(self.frac_in_cone):
            if self.frac_in_cone < 0.05 or self.frac_in_cone > 0.95:
                return False
        return True

    @property
    def d(self) -> int:
        """Feature dimensionality."""
        return len(self.mu)

    @property
    def cooperative_norm(self) -> float:
        """||w|| = sqrt(1^T Sigma^{-1} 1) — patient-specific coupling strength proxy."""
        return float(np.linalg.norm(self.cooperative_direction))

    @property
    def mean_t_value(self) -> float:
        """Mean cooperative statistic across observations (proxy for cooperative fraction)."""
        if self.cooperative_trajectory is not None:
            return float(np.mean(self.cooperative_trajectory))
        return 0.0
