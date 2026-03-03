"""
ICGHVRTEstimator: Just-in-Time treatment effect estimation with cooperative
cone matching.

Uses the eight-component cooperative cone distance (ICG-HVRT v0.2.0) for
k-nearest-neighbour selection, then fits a local Ridge regression on the
matched neighbourhood to estimate the individual treatment effect.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge

from .profile import CooperativeGeometryProfile, SharedHVRT, fit_shared_hvrt, pool_whitened_observations
from .matcher import ICGHVRTMatcher


class ICGHVRTEstimator:
    """
    JIT treatment effect estimator using eight-component cooperative cone matching.

    Workflow
    --------
    1. fit(X_list, T_list, Y_list) — build per-patient profiles, calibrate matcher.
    2. predict_effect(X_new, T_new) — match query against training by unified
       cone distance, fit local Ridge on k-NN neighbourhood, return treatment
       coefficient as ITE estimate.

    The API mirrors IntrinsicJIT for direct benchmark comparability.

    Parameters
    ----------
    matcher : ICGHVRTMatcher instance (eight-component cone distance)
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
        learn_weights: bool = False,
        geometry: str = 'cone',
    ) -> None:
        self.matcher = matcher or ICGHVRTMatcher()
        self.k = k
        self.alpha_local = alpha_local
        self.n_partitions = n_partitions
        self.y_weight = y_weight
        self.random_state = random_state
        self.learn_weights = learn_weights
        self.geometry = geometry  # 'cone' (SD-HVRT) | 'pyramid' (MAD-HART)

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
        # Determine geometry mode
        use_mad = (self.geometry == 'pyramid')
        import hvrt as _hvrt
        # Pyramid → PyramidHART (MAD whitening, A = |S|−‖z‖₁ partitioning).
        # Cone   → HVRT (SD whitening, T = S²−Q partitioning, default).
        model_cls = _hvrt.PyramidHART if use_mad else None  # None -> HVRT default

        # Pool whitened observations to fit shared HVRT/HART in a universal reference frame
        Z_pool = pool_whitened_observations(X_list, use_mad=use_mad)
        self._shared_hvrt = fit_shared_hvrt(
            Z_pool,
            n_partitions=self.n_partitions,
            y_weight=self.y_weight,
            random_state=self.random_state,
            model_class=model_cls,
        )

        # Build per-patient profiles
        self._profiles = [
            CooperativeGeometryProfile.from_longitudinal(
                X, shared_hvrt=self._shared_hvrt, use_mad=use_mad
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

        if self.learn_weights:
            self.matcher.fit_weights(
                self._profiles, self._obs_list, self._T_list, self._Y_list,
                k=self.k, alpha_local=self.alpha_local,
            )

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
        use_mad = (self.geometry == 'pyramid')
        query_profile = CooperativeGeometryProfile.from_longitudinal(
            np.atleast_2d(X_new), shared_hvrt=self._shared_hvrt, use_mad=use_mad
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
        Generate per-patient uncertainty report for test patients.

        Returns a list of dicts with keys:
          'nearest_distance'    -- total distance to the nearest neighbour
          'mean_identity_distance' -- average identity distance across k-NN
                                     (high = geometrically poor matches)
          'confidence'          -- 'high', 'medium', 'low', or 'uncertain'

        Confidence logic: similarity is continuous, not binary.
        'nearest_distance' measures overall match quality; 'mean_identity_distance'
        measures specifically how well the matched cone families agree.
        A prediction backed by geometrically compatible neighbours (low
        mean_identity_distance) is more trustworthy even when the total
        distance is moderate.
        """
        use_mad = (self.geometry == 'pyramid')
        report = []
        for X in X_test_list:
            qp = CooperativeGeometryProfile.from_longitudinal(
                np.atleast_2d(X), shared_hvrt=self._shared_hvrt, use_mad=use_mad
            )
            k = min(self.k, len(self._profiles))
            idx = self.matcher.find_neighbours(qp, self._profiles, k=k)
            nearest_dist = self.matcher.distance(qp, self._profiles[idx[0]])
            id_dists = [
                self.matcher.distance_components(qp, self._profiles[j])["identity_distance"]
                for j in idx
            ]
            mean_id_dist = float(np.mean(id_dists)) if id_dists else 0.0

            if nearest_dist < 0.5 and mean_id_dist < 0.5:
                confidence = "high"
            elif nearest_dist < 1.5 and mean_id_dist < 1.5:
                confidence = "medium"
            elif nearest_dist < 3.0:
                confidence = "low"
            else:
                confidence = "uncertain"
            report.append({
                "nearest_distance":      nearest_dist,
                "mean_identity_distance": mean_id_dist,
                "confidence":            confidence,
            })
        return report
