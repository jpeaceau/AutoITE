"""
ICGHVRTEstimator: Just-in-Time treatment effect estimation with cooperative
cone matching.

Uses the eight-component cooperative cone distance (ICG-HVRT v0.2.0) for
k-nearest-neighbour selection, then fits a local Ridge regression on the
matched neighbourhood to estimate the individual treatment effect.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, LinearRegression

from .profile import CooperativeGeometryProfile, SharedHVRT, fit_shared_hvrt, pool_whitened_observations, pool_synergy_whitened
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
        local_model: Optional[str] = None,  # None=auto | 'ridge' | 'ols' | 'lad' | 'mean' | 'median'
        spike_fraction: float = 0.05,        # fraction flagged as spikes in synergy mode
        distance_weighted: bool = False,     # weight pooled obs by exp(-dist) of their patient
        counterfactual_aug: bool = False,    # augment pool with HVRT-generated counterfactuals
        n_synth_per_neighbor: int = 30,      # synthetic obs to generate per k-NN neighbor
    ) -> None:
        self.matcher = matcher or ICGHVRTMatcher()
        self.k = k
        self.alpha_local = alpha_local
        self.n_partitions = n_partitions
        self.y_weight = y_weight
        self.random_state = random_state
        self.learn_weights = learn_weights
        self.geometry = geometry  # 'cone' (SD-HVRT) | 'pyramid' (MAD-HART) | 'synergy' (PyramidHART→HVRT)
        self.spike_fraction = spike_fraction
        self.distance_weighted = distance_weighted
        self.counterfactual_aug = counterfactual_aug
        self.n_synth_per_neighbor = n_synth_per_neighbor
        # geometry-appropriate default: lad for pyramid/synergy (clean matching → L1 wins),
        # ridge for cone (standard L2 with shrinkage)
        self.local_model = local_model if local_model is not None else (
            'lad' if geometry in ('pyramid', 'synergy') else 'ridge'
        )

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
        import hvrt as _hvrt

        if self.geometry == 'synergy':
            # Two-stage pipeline: PyramidHART spike detection → HVRT on clean bulk.
            # Stage 1 (pool_synergy_whitened): MAD-whiten all obs, remove top
            #   spike_fraction by |A|/‖z‖₁ score (Prop 1.3 single-feature cancellation).
            # Stage 2 (fit_shared_hvrt, model_class=None): HVRT on spike-free bulk so
            #   Theorem 3 (noise invariance of E[T]) holds without spike interference.
            Z_pool = pool_synergy_whitened(X_list, spike_fraction=self.spike_fraction)
            self._shared_hvrt = fit_shared_hvrt(
                Z_pool,
                n_partitions=self.n_partitions,
                y_weight=self.y_weight,
                random_state=self.random_state,
                model_class=None,  # HVRT — T-based, noise-invariant
            )
            self._profiles = [
                CooperativeGeometryProfile.from_longitudinal(
                    X, shared_hvrt=self._shared_hvrt, use_mad=True,
                )
                for X in X_list
            ]
        else:
            use_mad = (self.geometry == 'pyramid')
            # Pyramid → PyramidHART (MAD whitening, A = |S|−‖z‖₁ partitioning).
            # Cone   → HVRT (SD whitening, T = S²−Q partitioning, default).
            model_cls = _hvrt.PyramidHART if use_mad else None  # None -> HVRT default

            Z_pool = pool_whitened_observations(X_list, use_mad=use_mad)
            self._shared_hvrt = fit_shared_hvrt(
                Z_pool,
                n_partitions=self.n_partitions,
                y_weight=self.y_weight,
                random_state=self.random_state,
                model_class=model_cls,
            )
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

        # Pre-compute counterfactual generation context once at fit time.
        # Avoids repeating the expensive KDE-per-partition fitting on every
        # predict_effect call.  Context is built on the shared HVRT's z-scored
        # training data and is valid for any query patient.
        self._cf_ctx = None
        self._cf_T_min = self._cf_T_max = None
        if self.counterfactual_aug and self._shared_hvrt is not None:
            from hvrt.generation_strategies import multivariate_kde as _kde_strat
            hvrt_m = self._shared_hvrt.model
            self._cf_ctx = _kde_strat.prepare(
                hvrt_m.X_z_, hvrt_m.partition_ids_, hvrt_m.unique_partitions_
            )
            T_all = np.concatenate([t.ravel() for t in self._T_list])
            self._cf_T_min = float(T_all.min())
            self._cf_T_max = float(T_all.max())

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
        use_mad = (self.geometry in ('pyramid', 'synergy'))
        query_profile = CooperativeGeometryProfile.from_longitudinal(
            np.atleast_2d(X_new), shared_hvrt=self._shared_hvrt, use_mad=use_mad,
        )
        T_new_2d = np.atleast_2d(T_new).reshape(-1, 1)

        neighbour_idx, neighbour_dists = self.matcher.find_neighbours(
            query_profile, self._profiles, k=self.k,
            exclude_idx=exclude_idx, return_distances=True,
        )

        # Stack all per-observation data from neighbours
        X_local = np.vstack([self._obs_list[j] for j in neighbour_idx])
        T_local = np.vstack([self._T_list[j] for j in neighbour_idx])
        Y_local = np.hstack([self._Y_list[j] for j in neighbour_idx])

        # Distance-weighted pooling: observations from closer patients contribute
        # more to the local regression.  Weight w_j = exp(-d_j); expand per obs.
        sample_weights = None
        if self.distance_weighted:
            w = np.exp(-neighbour_dists)
            w = w / w.sum()
            n_obs = [len(self._obs_list[j]) for j in neighbour_idx]
            sample_weights = np.repeat(w, n_obs)

        # Counterfactual augmentation: for each neighbor j, synthesise
        # (X_synth, T_synth, Y_synth) triplets using HVRT partition sampling.
        # Synthetic X comes from j's cooperative distribution; T is drawn broadly
        # over the full training range; Y is predicted by j's individual model.
        # This fills in treatment arms that were never observed for similar patients,
        # directly targeting the pool-averaging and confounding problems.
        if self.counterfactual_aug and self._cf_ctx is not None:
            X_local, T_local, Y_local = self._augment_with_counterfactuals(
                X_local, T_local, Y_local, neighbour_idx,
            )
            sample_weights = None  # weights are for distance-weighted only; reset

        return self._local_tau(X_local, T_local, Y_local, sample_weights=sample_weights)

    def _augment_with_counterfactuals(
        self,
        X_real: np.ndarray,
        T_real: np.ndarray,
        Y_real: np.ndarray,
        neighbour_idx: np.ndarray,
    ):
        """
        Augment the local pool with HVRT-generated counterfactual observations.

        For each neighbor j:
          1. Fit a within-patient Ridge on (X_j, T_j, Y_j) → individual model m_j.
          2. Sample X_synth from j's HVRT partition distribution (preserves j's
             cooperative geometry without being tied to j's observed T values).
          3. Draw T_synth uniformly over the full training T range (broad coverage,
             fills the treatment arm that confounding may have emptied).
          4. Predict Y_synth = m_j.predict(X_synth, T_synth) — uses j's individual
             tau estimate, not the pooled estimate, so this is not circular.

        The augmented pool then has both real observations (biased T distribution
        from confounding) and synthetic observations (uniform T distribution),
        giving the final local Ridge access to counterfactual data that did not
        exist in the training set.
        """
        from hvrt.generation_strategies import multivariate_kde as _kde_strat
        from hvrt.expand import compute_expansion_budgets as _budgets
        hvrt_m = self._shared_hvrt.model
        n_synth = self.n_synth_per_neighbor
        rng = np.random.default_rng(int(abs(X_real.sum() * 1e4)) % (2**31))

        X_aug_parts, T_aug_parts, Y_aug_parts = [X_real], [T_real], [Y_real]

        for j in neighbour_idx:
            X_j = self._obs_list[j]
            T_j = self._T_list[j]
            Y_j = self._Y_list[j]
            d   = X_j.shape[1]

            # Need at least d+2 obs to fit within-patient Ridge reliably
            if len(X_j) < d + 2:
                continue

            # ── within-patient individual model ──────────────────────────
            m_j = Ridge(alpha=self.alpha_local).fit(
                np.hstack([X_j, T_j]), Y_j
            )

            # ── HVRT partition sampling for neighbor j ────────────────────
            Z_j    = hvrt_m._to_z(X_j)
            pids_j = hvrt_m.apply_raw(Z_j)
            budgets_j = _budgets(
                pids_j, hvrt_m.unique_partitions_, n_synth,
                False, hvrt_m.X_z_,
            )
            X_synth_z = _kde_strat.generate(self._cf_ctx, budgets_j,
                                             random_state=int(rng.integers(2**31)))
            # Clamp to avoid extreme KDE tails
            X_synth_z = np.clip(X_synth_z, -6.0, 6.0)
            X_synth = hvrt_m.scaler_.inverse_transform(X_synth_z)

            # ── broad counterfactual T sample ─────────────────────────────
            T_synth = rng.uniform(
                self._cf_T_min, self._cf_T_max, (len(X_synth), 1)
            )

            # ── predict Y using neighbor j's individual model ─────────────
            Y_synth = m_j.predict(np.hstack([X_synth, T_synth]))

            X_aug_parts.append(X_synth)
            T_aug_parts.append(T_synth)
            Y_aug_parts.append(Y_synth)

        return (
            np.vstack(X_aug_parts),
            np.vstack(T_aug_parts),
            np.hstack(Y_aug_parts),
        )

    def predict_effect_with_confidence(
        self,
        X_new: np.ndarray,
        T_new: np.ndarray,
        exclude_idx: Optional[int] = None,
    ):
        """
        Estimate the individual treatment effect and return a confidence score.

        The confidence score is the mean k-NN total distance: lower = more
        confident.  Patients with no geometrically similar neighbours will
        have high distances (inflated by identity_distance components), giving
        a natural abstention signal.

        Returns
        -------
        tau_hat : float
            Estimated treatment effect.
        mean_knn_dist : float
            Mean total distance across the k nearest neighbours.  Use as an
            abstention threshold: predict only when mean_knn_dist < threshold.
        """
        use_mad = (self.geometry in ('pyramid', 'synergy'))
        query_profile = CooperativeGeometryProfile.from_longitudinal(
            np.atleast_2d(X_new), shared_hvrt=self._shared_hvrt, use_mad=use_mad,
        )

        neighbour_idx, neighbour_dists = self.matcher.find_neighbours(
            query_profile, self._profiles, k=self.k,
            exclude_idx=exclude_idx, return_distances=True,
        )

        X_local = np.vstack([self._obs_list[j] for j in neighbour_idx])
        T_local = np.vstack([self._T_list[j] for j in neighbour_idx])
        Y_local = np.hstack([self._Y_list[j] for j in neighbour_idx])

        sample_weights = None
        if self.distance_weighted:
            w = np.exp(-neighbour_dists)
            w = w / w.sum()
            n_obs = [len(self._obs_list[j]) for j in neighbour_idx]
            sample_weights = np.repeat(w, n_obs)

        tau_hat = self._local_tau(X_local, T_local, Y_local, sample_weights=sample_weights)
        mean_knn_dist = float(np.mean(neighbour_dists))
        return tau_hat, mean_knn_dist

    def _local_tau(
        self,
        X_local: np.ndarray,   # (n_pool, d)
        T_local: np.ndarray,   # (n_pool, 1)
        Y_local: np.ndarray,   # (n_pool,)
        sample_weights: Optional[np.ndarray] = None,  # per-observation weights or None
    ) -> float:
        t = T_local.ravel()
        sw = sample_weights  # shorthand; None = uniform

        if self.local_model == 'ols':
            m = LinearRegression().fit(np.hstack([X_local, T_local]), Y_local,
                                       sample_weight=sw)
            return float(m.coef_[-1])

        if self.local_model == 'lad':
            # QuantileRegressor does not support sample_weight; fall back to
            # Ridge when weights are provided (lad is used for outlier robustness,
            # which distance weighting already partially addresses).
            from sklearn.linear_model import QuantileRegressor
            if sw is None:
                m = QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs')
                m.fit(np.hstack([X_local, T_local]), Y_local)
            else:
                m = Ridge(alpha=self.alpha_local).fit(
                    np.hstack([X_local, T_local]), Y_local, sample_weight=sw)
            return float(m.coef_[-1])

        if self.local_model == 'mean':
            if sw is not None:
                sw_n = sw / sw.sum()
                t_bar = float(np.dot(sw_n, t))
                y_bar = float(np.dot(sw_n, Y_local))
                t_c = t - t_bar
                y_c = Y_local - y_bar
                denom = float(np.dot(sw_n, t_c * t_c))
                return float(np.dot(sw_n, t_c * y_c) / denom) if denom > 1e-10 else 0.0
            t_c = t - t.mean()
            y_c = Y_local - Y_local.mean()
            denom = float(np.dot(t_c, t_c))
            return float(np.dot(t_c, y_c) / denom) if denom > 1e-10 else 0.0

        if self.local_model == 'median':
            from scipy.stats import theilslopes
            # theilslopes has no sample_weight; use weighted Ridge instead
            if sw is not None:
                m = Ridge(alpha=self.alpha_local).fit(
                    T_local, Y_local, sample_weight=sw)
                return float(m.coef_[-1])
            result = theilslopes(Y_local, t)
            return float(result.slope)

        # default: 'ridge'
        m = Ridge(alpha=self.alpha_local).fit(np.hstack([X_local, T_local]), Y_local,
                                              sample_weight=sw)
        return float(m.coef_[-1])

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
        use_mad = (self.geometry in ('pyramid', 'synergy'))
        report = []
        for X in X_test_list:
            qp = CooperativeGeometryProfile.from_longitudinal(
                np.atleast_2d(X), shared_hvrt=self._shared_hvrt, use_mad=use_mad,
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
