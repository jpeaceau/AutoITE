"""
ICGHVRTEstimator: Just-in-Time treatment effect estimation with cooperative
geometry matching.

Replaces ICG's Log-Euclidean (mu, Sigma) matching with the full five-component
cooperative manifold distance, while retaining the JIT local regression approach.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge

from .profile import CooperativeGeometryProfile, SharedHVRT, fit_shared_hvrt
from .matcher import ICGHVRTMatcher


class ICGHVRTEstimator:
    """
    JIT treatment effect estimator using five-component cooperative geometry matching.

    Workflow
    --------
    1. fit(X_list, T_list, Y_list) — build per-patient profiles, calibrate matcher.
    2. predict_effect(X_new, T_new) — match query against training, fit local Ridge,
       return treatment coefficient as ITE estimate.

    The API mirrors IntrinsicJIT for direct benchmark comparability.

    Parameters
    ----------
    matcher : ICGHVRTMatcher instance (five-component distance)
    k : number of nearest neighbours for local regression
    alpha_local : Ridge regularisation strength for local model
    n_partitions : number of HVRT partitions for shared HVRT
    y_weight : HVRT y_weight (must be 0.0 for causal applications)
    random_state : reproducibility seed
    """

    def __init__(
        self,
        matcher: Optional[ICGHVRTMatcher] = None,
        k: int = 30,
        alpha_local: float = 1.0,
        n_partitions: int = 8,
        y_weight: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.matcher = matcher or ICGHVRTMatcher()
        self.k = k
        self.alpha_local = alpha_local
        self.n_partitions = n_partitions
        self.y_weight = y_weight
        self.random_state = random_state

        self._profiles: List[CooperativeGeometryProfile] = []
        self._obs_list: List[np.ndarray] = []   # raw (N_i, d) feature observations
        self._T_list: List[np.ndarray] = []     # (N_i, 1) treatment per observation
        self._Y_list: List[np.ndarray] = []     # (N_i,) outcome per observation
        self._shared_hvrt: Optional[SharedHVRT] = None

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_list: List[np.ndarray],
        T_list: List[np.ndarray],
        Y_list: List[np.ndarray],
    ) -> "ICGHVRTEstimator":
        """
        Fit the estimator on training patients.

        Parameters
        ----------
        X_list : list of (N_i, d) feature observation matrices
        T_list : list of (N_i, 1) or (N_i,) treatment observation vectors
        Y_list : list of (N_i,) outcome observation vectors
        """
        # Pool all feature observations to fit shared HVRT
        X_pool = np.vstack(X_list)
        self._shared_hvrt = fit_shared_hvrt(
            X_pool,
            n_partitions=self.n_partitions,
            y_weight=self.y_weight,
            random_state=self.random_state,
        )

        # Build per-patient profiles
        self._profiles = [
            CooperativeGeometryProfile.from_longitudinal(
                X, shared_hvrt=self._shared_hvrt
            )
            for X in X_list
        ]

        self._obs_list = [np.atleast_2d(X) for X in X_list]
        self._T_list = [np.atleast_2d(T).reshape(-1, 1) for T in T_list]
        self._Y_list = [np.asarray(Y).ravel() for Y in Y_list]

        # Calibrate matcher on training profiles
        if self.matcher.auto_calibrate:
            self.matcher.calibrate(self._profiles)
            self.matcher._calibrated = True  # prevent re-calibration per query

        return self

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict_effect(
        self,
        X_new: np.ndarray,
        T_new: np.ndarray,
        exclude_idx: Optional[int] = None,
    ) -> float:
        """
        Estimate the individual treatment effect for a new patient.

        Parameters
        ----------
        X_new : (N, d) feature observations for the query patient
        T_new : (N, 1) or (N,) treatment observations for the query patient
        exclude_idx : training index to exclude (leave-one-out evaluation)

        Returns
        -------
        tau_hat : estimated treatment effect (T coefficient in local Ridge)
        """
        query_profile = CooperativeGeometryProfile.from_longitudinal(
            np.atleast_2d(X_new), shared_hvrt=self._shared_hvrt
        )
        T_new_2d = np.atleast_2d(T_new).reshape(-1, 1)

        neighbour_idx = self.matcher.find_neighbours(
            query_profile, self._profiles, k=self.k, exclude_idx=exclude_idx
        )

        # Stack all per-observation data from neighbours
        X_local = np.vstack([self._obs_list[j] for j in neighbour_idx])
        T_local = np.vstack([self._T_list[j] for j in neighbour_idx])
        Y_local = np.hstack([self._Y_list[j] for j in neighbour_idx])

        XT_local = np.hstack([X_local, T_local])
        model = Ridge(alpha=self.alpha_local)
        model.fit(XT_local, Y_local)

        return float(model.coef_[-1])

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def t_content_diagnostic(
        self,
        tau_hats: np.ndarray,
        profiles: Optional[List[CooperativeGeometryProfile]] = None,
    ) -> float:
        """
        Compute T-content: Spearman(tau_hat, mean_T) where mean_T is the
        mean cooperative statistic across each patient's observations.

        A large positive T-content means the estimated heterogeneity aligns with
        cooperative geometry structure — indicating ICG-HVRT's geometric features
        are relevant and Q-blind methods would miss this heterogeneity.
        """
        from scipy.stats import spearmanr

        profs = profiles if profiles is not None else self._profiles
        T_means = np.array([p.mean_t_value for p in profs])
        corr, _ = spearmanr(tau_hats, T_means)
        return float(corr)

    def triage_report(
        self,
        X_test_list: List[np.ndarray],
    ) -> List[Dict]:
        """
        Generate per-patient triage report for test patients.

        Returns a list of dicts with keys:
          'nearest_distance', 'n_direction_gate_passed', 'confidence'.
        """
        report = []
        for X in X_test_list:
            qp = CooperativeGeometryProfile.from_longitudinal(
                np.atleast_2d(X), shared_hvrt=self._shared_hvrt
            )
            k = min(self.k, len(self._profiles))
            idx = self.matcher.find_neighbours(qp, self._profiles, k=k)
            nearest_dist = self.matcher.distance(qp, self._profiles[idx[0]])
            gate_passes = sum(
                self.matcher.distance_components(qp, self._profiles[j])["direction_gate_passed"]
                for j in idx
            )
            if nearest_dist < 0.5 and gate_passes >= k * 0.8:
                confidence = "high"
            elif nearest_dist < 1.5 and gate_passes >= k * 0.5:
                confidence = "medium"
            elif gate_passes > 0:
                confidence = "low"
            else:
                confidence = "triage"
            report.append({
                "nearest_distance": nearest_dist,
                "n_direction_gate_passed": gate_passes,
                "confidence": confidence,
            })
        return report
