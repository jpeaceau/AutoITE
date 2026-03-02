"""
CooperativeGeometryProfile: per-patient personal quadratic manifold.

Each patient is represented as a personal cooperative geometry operator
C_i = Sigma_i^{-1/2} A Sigma_i^{-1/2} = w_i w_i^T - Sigma_i^{-1}
where A = 11^T - I is the universal cooperative form.
"""
import numpy as np
from numpy.linalg import eigh
from dataclasses import dataclass, field
from typing import Optional
import warnings


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
) -> SharedHVRT:
    """
    Fit a shared HVRT on pooled feature data and return a SharedHVRT wrapper
    with partitions ordered by mean cooperative statistic E[T].

    Parameters
    ----------
    X_all : (n_total, d) pooled feature observations from all training patients
    n_partitions : number of HVRT leaf partitions
    y_weight : 0.0 for pure cooperative geometry (causal applications)
    random_state : reproducibility seed
    """
    import hvrt as _hvrt
    model = _hvrt.HVRT(
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

    partitions_sorted = sorted(partitions, key=lambda p: p.get("E_T", 0.0))
    id_to_rank = {p["id"]: i for i, p in enumerate(partitions_sorted)}
    return SharedHVRT(model=model, id_to_rank=id_to_rank, n_partitions=len(partitions_sorted))


@dataclass
class CooperativeGeometryProfile:
    """
    Personal cooperative geometry profile for a single patient.

    Represents the patient as their personal quadratic manifold
    C_i = w_i w_i^T - Sigma_i^{-1}  (the cooperative geometry operator)

    Attributes
    ----------
    mu : (d,) mean feature vector
    sigma : (d, d) regularised covariance matrix
    cooperative_direction : (d,) w = Sigma^{-1/2} 1
    cone_angle : scalar theta in radians (invariant for fixed d, useful as check)
    cooperative_operator : (d, d) C = w w^T - Sigma^{-1}
    partition_profile : (K,) occupation histogram — None if insufficient data
    transition_matrix : (K, K) Markov transition probabilities — None if insufficient data
    cooperative_trajectory : (n_obs,) T(t) = S(t)^2 - Q(t) — None if insufficient data
    n_partitions : K used for HVRT
    n_observations : number of observations used to build this profile
    """
    mu: np.ndarray
    sigma: np.ndarray
    cooperative_direction: np.ndarray
    cone_angle: float
    cooperative_operator: np.ndarray
    partition_profile: Optional[np.ndarray] = None
    transition_matrix: Optional[np.ndarray] = None
    cooperative_trajectory: Optional[np.ndarray] = None
    n_partitions: Optional[int] = None
    n_observations: int = 0

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
        """
        observations = np.atleast_2d(observations)
        n_obs, d = observations.shape

        # ── 1. Personal statistics ──────────────────────────────────────
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

        # ── 2. Eigendecomposition for Sigma^{-1/2} and Sigma^{-1} ──────
        eigenvalues, eigenvectors = eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        inv_sqrt_vals = 1.0 / np.sqrt(eigenvalues)
        sigma_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_vals) @ eigenvectors.T

        inv_vals = 1.0 / eigenvalues
        sigma_inv = eigenvectors @ np.diag(inv_vals) @ eigenvectors.T

        # ── 3. Cooperative direction: w = Sigma^{-1/2} * 1 ─────────────
        ones = np.ones(d)
        w = sigma_inv_sqrt @ ones

        # ── 4. Cone angle: arccos(1 / sqrt(w^T Sigma w)) = arccos(1/sqrt(d)) ─
        # w^T Sigma w = 1^T Sigma^{-1/2} Sigma Sigma^{-1/2} 1 = 1^T I 1 = d
        # Computed numerically for consistency check / numerical accuracy.
        wSw = float(w @ cov @ w)
        cone_angle = float(np.arccos(np.clip(1.0 / np.sqrt(max(wSw, 1e-12)), -1.0, 1.0)))

        # ── 5. Cooperative operator: C = w w^T - Sigma^{-1} ───────────
        C = np.outer(w, w) - sigma_inv

        # ── 6. Cooperative trajectory T(t) = S^2 - Q ──────────────────
        # In whitened space z_t = Sigma^{-1/2}(x_t - mu):
        #   S_t = sum(z_t),  Q_t = sum(z_t^2)
        Z = (observations - mu) @ sigma_inv_sqrt.T  # (n_obs, d)
        S = Z.sum(axis=1)
        Q = (Z ** 2).sum(axis=1)
        cooperative_trajectory = S ** 2 - Q

        # ── 7. HVRT partition profile and transition matrix ─────────────
        partition_profile = None
        transition_matrix = None
        actual_K = n_partitions

        # Require at least 2*K observations for a useful histogram.
        min_for_hvrt = max(n_partitions * 2, 10)

        if n_obs >= min_for_hvrt:
            try:
                if shared_hvrt is not None:
                    ranked_ids = shared_hvrt.assign_partitions(observations)
                    actual_K = shared_hvrt.n_partitions
                else:
                    import hvrt as _hvrt
                    personal_hvrt = _hvrt.HVRT(
                        n_partitions=n_partitions,
                        y_weight=y_weight,
                        auto_tune=False,
                        random_state=42,
                    )
                    personal_hvrt.fit(observations)
                    raw_ids = personal_hvrt.apply_raw(observations)
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
                    k, l = int(ranked_ids[t]), int(ranked_ids[t + 1])
                    if 0 <= k < actual_K and 0 <= l < actual_K:
                        M[k, l] += 1.0
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
            partition_profile=partition_profile,
            transition_matrix=transition_matrix,
            cooperative_trajectory=cooperative_trajectory,
            n_partitions=actual_K,
            n_observations=n_obs,
        )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def has_longitudinal(self) -> bool:
        """True when partition profile and transition matrix are available."""
        return self.partition_profile is not None

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
