"""
Distance functions for the five-component ICG-HVRT metric.

d(i, j) = alpha_1*d_mu + alpha_2*d_w + alpha_3*d_sigma + alpha_4*d_occ + alpha_5*d_dyn

Component    Measures                        Requires
---------    --------                        --------
d_mu         Where is the patient?           mu_i (static)
d_w          Cooperative axis alignment      Sigma_i (static)
d_sigma      Manifold curvature (shape)      Sigma_i (static, Log-Euclidean proxy)
d_occ        Manifold occupation             HVRT partition profile (longitudinal)
d_dyn        Cooperative dynamics            HVRT transition matrix (longitudinal)
"""
import numpy as np
from numpy.linalg import eigh, norm


# ── Static geometry ──────────────────────────────────────────────── #

def euclidean_mean_distance(mu_i: np.ndarray, mu_j: np.ndarray) -> float:
    """Component 1: L2 distance between mean feature vectors."""
    return float(norm(mu_i - mu_j))


def cooperative_direction_distance(w_i: np.ndarray, w_j: np.ndarray) -> float:
    """
    Component 2: Angular distance between cooperative directions.

    d_w = arccos( (w_i . w_j) / (||w_i|| ||w_j||) )

    Returns a value in [0, pi].  0 means perfectly aligned cooperative axes;
    pi/2 means orthogonal (geometrically incommensurable cooperative regimes).
    """
    norm_i = norm(w_i)
    norm_j = norm(w_j)
    if norm_i < 1e-12 or norm_j < 1e-12:
        return float(np.pi / 2)
    cos_sim = float(w_i @ w_j) / (norm_i * norm_j)
    return float(np.arccos(np.clip(cos_sim, -1.0, 1.0)))


def log_euclidean_distance(Sigma_i: np.ndarray, Sigma_j: np.ndarray) -> float:
    """
    Component 3: Log-Euclidean distance between SPD covariance matrices.

    d_LE(Sigma_i, Sigma_j) = || log(Sigma_i) - log(Sigma_j) ||_F

    This is the simplified proxy for the full matrix-pencil manifold distance
    (Spec §3.4 simplified alternative), avoiding the indefinite pencil
    eigenvalue problem while still capturing cooperative curvature differences.
    """
    log_i = _safe_matrix_log(Sigma_i)
    log_j = _safe_matrix_log(Sigma_j)
    return float(norm(log_i - log_j, "fro"))


def _safe_matrix_log(M: np.ndarray) -> np.ndarray:
    """Matrix logarithm for SPD matrices via eigendecomposition."""
    eigenvalues, eigenvectors = eigh(M)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T


# ── Longitudinal geometry ─────────────────────────────────────────── #

def occupation_distance(pi_i: np.ndarray, pi_j: np.ndarray) -> float:
    """
    Component 4: Wasserstein-1 distance between partition occupation profiles.

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
    Component 5: Frobenius distance between Markov transition matrices.

    d_dyn(i, j) = || M_i - M_j ||_F

    Captures differences in cooperative stability, oscillation frequency,
    and regime persistence that no static covariance summary encodes.
    """
    # Pad or truncate to matching size if different (shouldn't happen with shared HVRT)
    K = min(M_i.shape[0], M_j.shape[0])
    return float(norm(M_i[:K, :K] - M_j[:K, :K], "fro"))
