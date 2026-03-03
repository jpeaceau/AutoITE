"""
Distance functions for the eight-component ICG-HVRT metric (v0.2.0).

Total distance = identity_distance + state_distance, unified k-NN ranking.

Identity components (cone family compatibility)
-----------------------------------------------
d_axis      Angular distance between cooperative axes (v+_i, v+_j)
d_opening   L2 distance between directional half-angle profiles
d_ecc       |log(ecc_i) - log(ecc_j)|  cone circularity difference
d_orient    Procrustes distance between anti-cooperative frames V-

State components (where on the compatible cone)
------------------------------------------------
d_mu_coop   Mean shift along the shared cooperative axis
d_mu_perp   Mean shift perpendicular to the cooperative axis
d_occ       Wasserstein-1 occupation distance
d_dyn       Frobenius transition-matrix distance

The identity components capture the *shape* of each patient's personal cone
family.  Two patients with very different cone shapes are poor matches —
their identity components inflate the total distance, pushing them down the
k-NN ranking without any hard gate.  The geometry does the weighting.

The state components capture *where* on the cone each patient currently sits.

Together: k-nearest-neighbours by total distance selects patients who are
both cone-compatible (small identity distance) and in a similar cooperative
state (small state distance).  The identity_distance term exposed in the
distance_components dict acts as an uncertainty signal: high identity_distance
means the best available matches are geometrically poor.

Cooperative mean distance
-------------------------
d_mu_coop = |w_bar^T (mu_i - mu_j)| / ||w_bar||
When Sigma ~ I, w ~ 1, so:
  w_bar^T (mu_i - mu_j) = rho * sum_k (U_{k,i} - U_{k,j})
                        = rho * sqrt(K) * (tau_i - tau_j)
  ||w_bar|| = sqrt(K)
  d_mu_coop = rho * |tau_i - tau_j|  (constant w.r.t. K, perfectly tau-correlated)
"""
import numpy as np
from dataclasses import dataclass
from numpy.linalg import eigh, norm
from typing import Optional


# ── ConeIdentity ─────────────────────────────────────────────────────── #

@dataclass
class ConeIdentity:
    """
    Full cooperative cone identity for a single patient.

    Extracts the complete eigendecomposition of the cooperative geometry
    operator C_i = w_i w_i^T - Sigma_i^{-1} and stores all geometric
    invariants needed for cone identity matching (ICG-HVRT v0.2.0).

    In whitened space (Sigma = I), C = 11^T - I has one positive eigenvalue
    (d-1) and d-1 equal negative eigenvalues (-1), giving a circular cone.
    In original space with Sigma != I, the negative eigenvalues are generally
    unequal, giving an elliptical cone whose eccentricity and orientation
    encode patient-specific cooperative geometry.

    Attributes
    ----------
    axis : (d,) normalised positive eigenvector v+ (cooperative axis).
           Proportional to w/||w||; canonical sign: sum(axis) >= 0.
    positive_eigenvalue : scalar lambda+ > 0
    negative_eigenvalues : (d-1,) negative eigenvalues, sorted by theta
                           descending (widest direction first).
    anti_cooperative_frame : (d, d-1) anti-cooperative eigenvectors V-
                             (column k is the eigenvector for negative_eigenvalues[k]).
    opening_profile : (d-1,) directional half-angles
                      theta_k = arctan(sqrt(lambda+ / |lambda-_k|)),
                      sorted descending (widest to narrowest).
    eccentricity : max(opening_profile) / min(opening_profile).
                   1.0 for circular cones (Sigma = scalar * I or d <= 2).
    """
    axis: np.ndarray
    positive_eigenvalue: float
    negative_eigenvalues: np.ndarray
    anti_cooperative_frame: np.ndarray
    opening_profile: np.ndarray
    eccentricity: float

    @classmethod
    def from_operator(cls, C: np.ndarray) -> "ConeIdentity":
        """
        Build ConeIdentity from the pre-computed cooperative operator C.

        C must be the (d, d) matrix C_i = w_i w_i^T - Sigma_i^{-1}.
        Uses the already-computed C to avoid redundant eigendecomposition.
        """
        d = C.shape[0]

        if d == 1:
            return cls(
                axis=np.array([1.0]),
                positive_eigenvalue=0.0,
                negative_eigenvalues=np.array([]),
                anti_cooperative_frame=np.zeros((1, 0)),
                opening_profile=np.array([]),
                eccentricity=1.0,
            )

        c_eigenvalues, c_eigenvectors = eigh(C)
        # eigh returns ascending order; C has signature (1, d-1) so last is positive
        lam_pos = float(c_eigenvalues[-1])
        v_pos = c_eigenvectors[:, -1].copy()
        # Canonical sign: cooperative axis points in the positive-sum direction
        if float(np.sum(v_pos)) < 0:
            v_pos = -v_pos
        # Ensure exact unit norm (eigh eigenvectors can deviate at machine epsilon)
        v_norm = float(norm(v_pos))
        if v_norm > 1e-12:
            v_pos = v_pos / v_norm

        lam_neg = c_eigenvalues[:-1].copy()   # (d-1,) negative eigenvalues (ascending)
        V_neg = c_eigenvectors[:, :-1].copy() # (d, d-1) anti-cooperative eigenvectors

        # Compute directional half-angles: theta_k = arctan(sqrt(lambda+ / |lambda-_k|))
        theta_raw = np.array([
            float(np.arctan(np.sqrt(
                max(lam_pos, 1e-12) / max(abs(float(lk)), 1e-12)
            )))
            for lk in lam_neg
        ])

        # Sort by theta descending (widest opening first)
        sort_idx = np.argsort(theta_raw)[::-1]
        theta = theta_raw[sort_idx]
        lam_neg = lam_neg[sort_idx]
        V_neg = V_neg[:, sort_idx]

        if len(theta) > 1:
            eccentricity = float(theta[0] / max(theta[-1], 1e-12))
        else:
            eccentricity = 1.0  # single anti-cooperative direction: no eccentricity

        return cls(
            axis=v_pos,
            positive_eigenvalue=lam_pos,
            negative_eigenvalues=lam_neg,
            anti_cooperative_frame=V_neg,
            opening_profile=theta,
            eccentricity=eccentricity,
        )

    @classmethod
    def from_covariance(cls, sigma: np.ndarray) -> "ConeIdentity":
        """
        Build ConeIdentity from the patient's regularised covariance matrix.

        Computes C = w w^T - Sigma^{-1} (where w = Sigma^{-1/2} * 1) and
        delegates to from_operator(C).
        """
        d = sigma.shape[0]

        if d == 1:
            return cls(
                axis=np.array([1.0]),
                positive_eigenvalue=0.0,
                negative_eigenvalues=np.array([]),
                anti_cooperative_frame=np.zeros((1, 0)),
                opening_profile=np.array([]),
                eccentricity=1.0,
            )

        eigenvalues, eigenvectors = eigh(sigma)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        inv_sqrt_vals = 1.0 / np.sqrt(eigenvalues)
        sigma_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_vals) @ eigenvectors.T
        inv_vals = 1.0 / eigenvalues
        sigma_inv = eigenvectors @ np.diag(inv_vals) @ eigenvectors.T

        w = sigma_inv_sqrt @ np.ones(d)
        C = np.outer(w, w) - sigma_inv
        return cls.from_operator(C)

    @staticmethod
    def distance(id_i: "ConeIdentity", id_j: "ConeIdentity") -> dict:
        """
        Compute the four identity distance components between two cone identities.

        Returns
        -------
        dict with keys:
          'axis'          arccos(|v+_i . v+_j|)  in [0, pi/2]
          'opening'       ||theta_i - theta_j||_2 (profile L2)
          'eccentricity'  |log(ecc_i) - log(ecc_j)|
          'orientation'   Procrustes distance between anti-cooperative frames

        Interpretation: all components are 0 when the two identities are
        identical, and increase monotonically with cone dissimilarity.
        A high 'orientation' with low 'axis'/'opening'/'eccentricity' means
        the patients share a cone shape but their anti-cooperative directions
        (the directions of easy deviation) point different ways.
        """
        # d_axis: angle between cooperative axes (sign-invariant).
        # Use the numerically stable half-chord formula:
        #   arccos(v . w) = 2 * arcsin(||v - w|| / 2)
        # Handles sign ambiguity by taking the minimum of d(v+, w+) and d(v+, -w+).
        # Gives exactly 0.0 when both axes are the same array object.
        diff_same = id_i.axis - id_j.axis
        diff_flip = id_i.axis + id_j.axis
        chord_same = float(norm(diff_same)) / 2.0
        chord_flip = float(norm(diff_flip)) / 2.0
        d_axis = 2.0 * float(np.arcsin(np.clip(
            min(chord_same, chord_flip), 0.0, 1.0
        )))

        # d_opening: L2 distance between half-angle profiles (both sorted descending)
        n_ang = min(len(id_i.opening_profile), len(id_j.opening_profile))
        if n_ang > 0:
            d_opening = float(norm(
                id_i.opening_profile[:n_ang] - id_j.opening_profile[:n_ang]
            ))
        else:
            d_opening = 0.0

        # d_ecc: log-ratio of eccentricities
        ecc_i = max(float(id_i.eccentricity), 1.0)
        ecc_j = max(float(id_j.eccentricity), 1.0)
        d_ecc = abs(float(np.log(ecc_i) - np.log(ecc_j)))

        # d_orient: Procrustes distance between anti-cooperative frames
        V_i = id_i.anti_cooperative_frame
        V_j = id_j.anti_cooperative_frame
        if V_i.shape[1] >= 2 and V_j.shape[1] >= 2:
            d_orient = _procrustes_distance(V_i, V_j)
        elif V_i.shape[1] == 1 and V_j.shape[1] == 1:
            # Single anti-cooperative vector: use sign-invariant half-chord formula
            v_i = V_i[:, 0]
            v_j = V_j[:, 0]
            n_i = float(norm(v_i))
            n_j = float(norm(v_j))
            if n_i < 1e-12 or n_j < 1e-12:
                d_orient = 0.0
            else:
                ui = v_i / n_i
                uj = v_j / n_j
                c_same = float(norm(ui - uj)) / 2.0
                c_flip = float(norm(ui + uj)) / 2.0
                d_orient = 2.0 * float(np.arcsin(np.clip(
                    min(c_same, c_flip), 0.0, 1.0
                )))
        else:
            d_orient = 0.0

        return {
            "axis":         d_axis,
            "opening":      d_opening,
            "eccentricity": d_ecc,
            "orientation":  d_orient,
        }


def _procrustes_distance(V_i: np.ndarray, V_j: np.ndarray) -> float:
    """
    Procrustes distance between two (d, p) orthonormal frames.

    Minimises ||V_i - V_j Q||_F over orthogonal Q and returns the minimum.
    Via SVD of M = V_i^T V_j: d = sqrt(max(2p - 2 * sum(singular_values(M)), 0)).

    Handles near-degenerate singular values correctly since it operates on
    the full SVD decomposition, not individual eigenvectors.
    """
    p = min(V_i.shape[1], V_j.shape[1])
    if p == 0:
        return 0.0
    M = V_i[:, :p].T @ V_j[:, :p]
    try:
        sv = np.linalg.svd(M, compute_uv=False)
        sv = np.clip(sv, 0.0, 1.0)
        return float(np.sqrt(max(2.0 * p - 2.0 * float(np.sum(sv)), 0.0)))
    except np.linalg.LinAlgError:
        return float(np.sqrt(2.0 * p))


# ── Static geometry ──────────────────────────────────────────────────── #

def euclidean_mean_distance(mu_i: np.ndarray, mu_j: np.ndarray) -> float:
    """
    Legacy L2 distance between mean feature vectors.

    Kept for backward compatibility and benchmark comparisons.
    Prefer cooperative_mean_distance() which decomposes this into
    tau-correlated and perpendicular components.
    """
    return float(norm(mu_i - mu_j))


def cooperative_mean_distance(
    mu_i: np.ndarray,
    mu_j: np.ndarray,
    w_i: np.ndarray,
    w_j: np.ndarray,
) -> tuple:
    """
    Decompose the mean difference into cooperative and perpendicular components.

    Uses the average cooperative direction w_bar = w_i + w_j (unnormalised sum;
    direction only) as the projection axis.

    Parameters
    ----------
    mu_i, mu_j : (d,) mean feature vectors
    w_i, w_j   : (d,) cooperative directions (Sigma^{-1/2} * 1)

    Returns
    -------
    d_coop : float
        |w_bar^T (mu_i - mu_j)| / ||w_bar||
        Proportional to |tau_i - tau_j| when Sigma ~ I; constant w.r.t. K.
    d_perp : float
        ||delta_mu - proj_{w_bar}(delta_mu)||
        Residual mean shift orthogonal to the cooperative axis.

    Note: d_coop^2 + d_perp^2 = ||mu_i - mu_j||^2  (Pythagorean decomposition).
    """
    delta_mu = mu_i - mu_j
    w_bar = w_i + w_j          # proportional to the average cooperative direction
    w_norm_sq = float(w_bar @ w_bar)
    if w_norm_sq < 1e-12:
        return 0.0, float(norm(delta_mu))

    w_norm = np.sqrt(w_norm_sq)
    proj_scalar = float(w_bar @ delta_mu) / w_norm      # signed projection magnitude
    d_coop = abs(proj_scalar)
    perp_vec = delta_mu - (float(w_bar @ delta_mu) / w_norm_sq) * w_bar
    d_perp = float(norm(perp_vec))
    return d_coop, d_perp


def cooperative_direction_distance(w_i: np.ndarray, w_j: np.ndarray) -> float:
    """
    Angular distance between cooperative directions (legacy; use ConeIdentity.distance
    for v0.2.0 matching which uses the normalised axis v+).

    d_w = arccos( (w_i . w_j) / (||w_i|| ||w_j||) )

    Returns a value in [0, pi].
    """
    norm_i = norm(w_i)
    norm_j = norm(w_j)
    if norm_i < 1e-12 or norm_j < 1e-12:
        return float(np.pi / 2)
    cos_sim = float(w_i @ w_j) / (norm_i * norm_j)
    return float(np.arccos(np.clip(cos_sim, -1.0, 1.0)))


def log_euclidean_distance(Sigma_i: np.ndarray, Sigma_j: np.ndarray) -> float:
    """
    Log-Euclidean distance between SPD covariance matrices.

    d_LE(Sigma_i, Sigma_j) = || log(Sigma_i) - log(Sigma_j) ||_F

    Legacy component; kept for benchmark comparisons.  In v0.2.0 the cone
    shape information is captured by the identity components (opening profile,
    eccentricity, orientation) derived from ConeIdentity.
    """
    log_i = _safe_matrix_log(Sigma_i)
    log_j = _safe_matrix_log(Sigma_j)
    return float(norm(log_i - log_j, "fro"))


def _safe_matrix_log(M: np.ndarray) -> np.ndarray:
    """Matrix logarithm for SPD matrices via eigendecomposition."""
    eigenvalues, eigenvectors = eigh(M)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T


# ── Longitudinal geometry ─────────────────────────────────────────────── #

def occupation_distance(pi_i: np.ndarray, pi_j: np.ndarray) -> float:
    """
    Wasserstein-1 distance between partition occupation profiles.

    W_1 on an ordered 1-D support equals the L1 norm of the difference of CDFs:
        W_1(pi_i, pi_j) = sum_k |CDF_i(k) - CDF_j(k)|

    The ordered support respects cooperative geometry: adjacent partitions have
    similar T-values, so moving mass between them costs less than moving between
    distant partitions.
    """
    cdf_i = np.cumsum(pi_i)
    cdf_j = np.cumsum(pi_j)
    return float(np.sum(np.abs(cdf_i - cdf_j)))


def dynamics_distance(M_i: np.ndarray, M_j: np.ndarray) -> float:
    """
    Frobenius distance between Markov transition matrices.

    d_dyn(i, j) = || M_i - M_j ||_F

    Captures differences in cooperative stability, oscillation frequency,
    and regime persistence that no static covariance summary encodes.
    """
    K = min(M_i.shape[0], M_j.shape[0])
    return float(norm(M_i[:K, :K] - M_j[:K, :K], "fro"))
