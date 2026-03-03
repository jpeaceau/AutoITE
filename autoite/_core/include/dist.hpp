#pragma once
#include <cstdint>

namespace autoite {

//
// compute_distances
// -----------------
// Computes (N,) weighted total distances from a single query profile to
// every training profile.  Called once per find_neighbours() query.
//
// Query fields are passed explicitly (the query may be an unseen patient).
// Training fields are flat C-contiguous arrays packed from calibrate().
//
// Component order (matches _COMP in matcher.py / raw_cache layout):
//   0 axis  1 opening  2 eccentricity  3 orientation
//   4 levels (coop)  5 levels_perp  6 occupation  7 dynamics
//
void compute_distances(
    // --- query profile ---
    const double*  q_axis,          // (d,)
    double         q_ecc,
    const double*  q_opening,       // (dp1,)  dp1 = d-1
    const double*  q_anti_coop,     // (d*dp1,) row-major: V[r,c] = q_anti_coop[r*dp1+c]
    const double*  q_mu,            // (d,)
    const double*  q_coop_dir,      // (d,)
    const double*  q_partition,     // (K,)
    const double*  q_transition,    // (K*K,) row-major
    int            q_geo_reliable,
    int            q_has_partition,
    int            q_has_transition,
    // --- training profiles (N patients, packed) ---
    const double*   tr_axes,            // (N, d)
    const double*   tr_eccentricities,  // (N,)
    const double*   tr_openings,        // (N, dp1)
    const double*   tr_anti_coops,      // (N, d*dp1)
    const double*   tr_mus,             // (N, d)
    const double*   tr_coop_dirs,       // (N, d)
    const double*   tr_partitions,      // (N, K)
    const double*   tr_transitions,     // (N, K*K)
    const int32_t*  tr_geo_reliable,    // (N,)
    const int32_t*  tr_has_partition,   // (N,)
    const int32_t*  tr_has_transition,  // (N,)
    // --- weights / scales ---
    const double*  weights,   // (8,)
    const double*  scales,    // (8,)
    // --- dimensions ---
    int N, int d, int dp1, int K,
    // --- output ---
    double* out_distances       // (N,)  pre-allocated by caller
);


//
// compute_raw_cache
// -----------------
// Builds the (n_eval, N, 8) raw-component cache used by the LOO objective.
// Queries are training profiles at eval_indices (LOO: distance to self is
// always placed into out[ei, i, :] without masking — the Python caller sets
// D[i] = inf after weighting, matching the existing raw_cache contract).
//
void compute_raw_cache(
    const int32_t* eval_indices,  // (n_eval,)
    int            n_eval,
    // --- training profiles (same packed layout as compute_distances) ---
    const double*   tr_axes,
    const double*   tr_eccentricities,
    const double*   tr_openings,
    const double*   tr_anti_coops,
    const double*   tr_mus,
    const double*   tr_coop_dirs,
    const double*   tr_partitions,
    const double*   tr_transitions,
    const int32_t*  tr_geo_reliable,
    const int32_t*  tr_has_partition,
    const int32_t*  tr_has_transition,
    // --- dimensions ---
    int N, int d, int dp1, int K,
    // --- output ---
    double* out_raw_cache   // (n_eval, N, 8)  pre-allocated by caller
);

} // namespace autoite
