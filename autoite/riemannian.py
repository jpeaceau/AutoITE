import numpy as np
from numpy.linalg import eigh, norm
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PatientData:
    """Container for a single patient's observation data."""
    patient_id: int
    regime: str  # 'A' (Clean), 'B' (Confounded), 'C' (Collider)
    regime_name: str  # Descriptive name
    observations: np.ndarray  # Shape: (N, 4) - [X1, X2, T, Y] only
    true_treatment_effect: float
    mean_treatment: float
    mean_outcome: float
    observed_correlation: float  # T-Y correlation (may be spurious)

class MatrixBuilder:
    """
    Constructs SPD matrices from patient observations.
    Now handles 4D: [X1, X2, T, Y]
    """

    def __init__(self, regularization_lambda: float = 1e-3):
        # Increased regularization for collider-induced near-singularities
        self.regularization_lambda = regularization_lambda

    def build_covariance_matrix(self, observations: np.ndarray) -> np.ndarray:
        """Build a regularized covariance matrix."""
        try:
            lw = LedoitWolf()
            lw.fit(observations)
            cov_matrix = lw.covariance_
        except Exception:
            cov_matrix = np.cov(observations, rowvar=False)

        d = cov_matrix.shape[0]
        cov_matrix = cov_matrix + self.regularization_lambda * np.eye(d)

        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        if np.min(eigenvalues) <= 0:
            min_eig = np.abs(np.min(eigenvalues))
            cov_matrix = cov_matrix + (min_eig + self.regularization_lambda) * np.eye(d)

        return cov_matrix

    def build_matrices_for_patients(self, patients: List[PatientData]) -> np.ndarray:
        """Build covariance matrices for all patients."""
        matrices = []
        for patient in patients:
            cov_mat = self.build_covariance_matrix(patient.observations)
            matrices.append(cov_mat)
        return np.array(matrices)


class RiemannianGeometry:
    """Riemannian geometry operations on the SPD manifold."""

    @staticmethod
    def _safe_matrix_log(matrix: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        log_eigenvalues = np.log(eigenvalues)
        log_matrix = eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.T
        return log_matrix

    @staticmethod
    def _safe_matrix_exp(matrix: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = eigh(matrix)
        exp_eigenvalues = np.exp(eigenvalues)
        exp_matrix = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T
        return exp_matrix

    @classmethod
    def log_euclidean_distance(cls, A: np.ndarray, B: np.ndarray) -> float:
        log_A = cls._safe_matrix_log(A)
        log_B = cls._safe_matrix_log(B)
        return norm(log_A - log_B, 'fro')

    @classmethod
    def compute_frechet_mean(cls, matrices: np.ndarray) -> np.ndarray:
        log_matrices = np.array([cls._safe_matrix_log(m) for m in matrices])
        mean_log = np.mean(log_matrices, axis=0)
        frechet_mean = cls._safe_matrix_exp(mean_log)
        return frechet_mean

    @classmethod
    def compute_global_alignment_scores(
        cls, matrices: np.ndarray, frechet_mean: np.ndarray
    ) -> np.ndarray:
        scores = np.array([
            cls.log_euclidean_distance(m, frechet_mean) for m in matrices
        ])
        return scores


class EnhancedJITSelector:
    """
    Enhanced JIT selector with dual-factor selection.
    Cost: J(H_i) = d_R(Q, H_i) + alpha * |G_Q - G_H_i|
    """

    def __init__(self, k_neighbors: int = 50, alpha: float = 1.0):
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.historical_matrices = None
        self.historical_patients = None
        self.frechet_mean = None
        self.global_scores = None

    def fit(self, matrices: np.ndarray, patients: List[PatientData]):
        self.historical_matrices = matrices
        self.historical_patients = patients
        self.frechet_mean = RiemannianGeometry.compute_frechet_mean(matrices)
        self.global_scores = RiemannianGeometry.compute_global_alignment_scores(
            matrices, self.frechet_mean
        )

    def compute_query_global_score(self, query_matrix: np.ndarray) -> float:
        return RiemannianGeometry.log_euclidean_distance(
            query_matrix, self.frechet_mean
        )

    def select_cohort(
        self, query_matrix: np.ndarray, exclude_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_historical = len(self.historical_matrices)
        G_query = self.compute_query_global_score(query_matrix)

        local_distances = np.zeros(n_historical)
        costs = np.zeros(n_historical)

        for i in range(n_historical):
            if exclude_idx is not None and i == exclude_idx:
                costs[i] = np.inf
                continue

            d_local = RiemannianGeometry.log_euclidean_distance(
                query_matrix, self.historical_matrices[i]
            )
            local_distances[i] = d_local
            d_global = np.abs(G_query - self.global_scores[i])
            costs[i] = d_local + self.alpha * d_global

        cohort_indices = np.argsort(costs)[:self.k_neighbors]
        return cohort_indices, local_distances[cohort_indices], costs[cohort_indices]

class JITRiemannianModel:
    """JIT Riemannian Model: Local regression on geometrically-similar cohort."""

    def __init__(self, matrix_builder: MatrixBuilder, jit_selector: EnhancedJITSelector):
        self.matrix_builder = matrix_builder
        self.jit_selector = jit_selector

    def fit(self, matrices: np.ndarray, patients: List[PatientData]):
        self.jit_selector.fit(matrices, patients)
        self.training_patients = patients
        self.training_matrices = matrices

    def predict_and_estimate(
        self, query_patient: PatientData, query_matrix: np.ndarray,
        query_idx: Optional[int] = None
    ) -> Tuple[float, float, np.ndarray]:
        cohort_indices, _, _ = self.jit_selector.select_cohort(
            query_matrix, exclude_idx=query_idx
        )

        X_local, Y_local = [], []
        for idx in cohort_indices:
            cohort_patient = self.training_patients[idx]
            obs = cohort_patient.observations
            X_mean = obs[:, :3].mean(axis=0)  # X1, X2, T
            Y_mean = obs[:, 3].mean()
            X_local.append(X_mean)
            Y_local.append(Y_mean)

        local_model = LinearRegression()
        local_model.fit(np.array(X_local), np.array(Y_local))

        query_obs = query_patient.observations
        query_X = query_obs[:, :3].mean(axis=0).reshape(1, -1)
        prediction = local_model.predict(query_X)[0]
        local_treatment_effect = local_model.coef_[2]  # T coefficient

        return prediction, local_treatment_effect, cohort_indices
