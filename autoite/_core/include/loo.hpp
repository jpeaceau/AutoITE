#pragma once
#include <cstdint>

namespace autoite {

/**
 * Compute the LOO-MSE objective for weight optimisation in ICG-HVRT fit_weights.
 *
 * Replaces the inner Python loop in matcher.ICGHVRTMatcher.fit_weights() that:
 *   1. Converts log-weights → effective weights  (eff_w = w / scales)
 *   2. Computes per-eval-patient distance vectors from pre-built raw_cache
 *   3. Finds k nearest neighbours via partial sort
 *   4. Stacks neighbourhood observations, fits local Ridge regression
 *   5. Predicts on the held-out patient, accumulates MSE
 *
 * All Ridge solves use centering (match sklearn Ridge(fit_intercept=True)).
 *
 * @param raw_cache     (n_eval, N_train, 8) C-contiguous float64 array.
 *                      raw_cache[ei, j, c] = unscaled component c of the
 *                      distance between eval patient ei and training patient j.
 * @param obs_flat      (sum_n_obs, d) feature observations, all patients stacked.
 * @param T_flat        (sum_n_obs,)  treatment observations, flattened.
 * @param Y_flat        (sum_n_obs,)  outcome observations, flattened.
 * @param offsets       (N_train+1,) cumulative row offsets into flat arrays.
 *                      Patient j's observations occupy rows [offsets[j], offsets[j+1]).
 * @param eval_indices  (n_eval,) training-set indices of the eval patients.
 *                      eval_indices[ei] is the training-set index of eval patient ei.
 *                      D_i[eval_indices[ei]] is set to inf (leave-one-out exclusion).
 * @param weights       (8,) current weight vector (in original, not log, space).
 * @param scales        (8,) calibration scale vector from matcher.calibrate().
 * @param n_eval        number of evaluation patients.
 * @param N_train       total number of training patients.
 * @param d             feature dimensionality.
 * @param k             neighbourhood size for local Ridge regression.
 * @param alpha_ridge   Ridge regularisation strength.
 * @returns             mean LOO prediction MSE across n_eval patients.
 */
double loo_objective(
    const double*   raw_cache,
    const double*   obs_flat,
    const double*   T_flat,
    const double*   Y_flat,
    const int32_t*  offsets,
    const int32_t*  eval_indices,
    const double*   weights,
    const double*   scales,
    int n_eval,
    int N_train,
    int d,
    int k,
    double alpha_ridge
);

} // namespace autoite
