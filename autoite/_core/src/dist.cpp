#include "dist.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef _OPENMP
#  include <omp.h>
#endif

namespace autoite {

// ── Elementary distance kernels ──────────────────────────────────────────── //

// Angular distance between two unit vectors using the numerically stable
// half-chord formula:  2 * arcsin( min(||a-b||, ||a+b||) / 2 ).
// Handles sign ambiguity (v+ and -v+ are the same axis direction).
static double axis_dist(const double* a, const double* b, int d) {
    double ss = 0.0, sf = 0.0;
    for (int k = 0; k < d; ++k) {
        double diff = a[k] - b[k];
        double sum  = a[k] + b[k];
        ss += diff * diff;
        sf += sum  * sum;
    }
    double chord = 0.5 * std::sqrt(std::min(ss, sf));
    if (chord > 1.0) chord = 1.0;
    return 2.0 * std::asin(chord);
}

// L2 distance between two opening-angle profiles (both sorted descending).
static double opening_dist(const double* a, const double* b, int dp1) {
    double s = 0.0;
    for (int k = 0; k < dp1; ++k) {
        double diff = a[k] - b[k];
        s += diff * diff;
    }
    return std::sqrt(s);
}

// Log-eccentricity distance.
static double ecc_dist(double ea, double eb) {
    if (ea < 1.0) ea = 1.0;
    if (eb < 1.0) eb = 1.0;
    return std::abs(std::log(ea) - std::log(eb));
}

// Procrustes distance between two anti-cooperative frames.
// Vi, Vj are (d, dp1) stored row-major: element [r,c] at Vi[r*dp1+c].
// For dp1==1: uses the sign-invariant half-chord formula on unit vectors.
// For dp1>=2: SVD of M = Vi^T Vj (dp1 x dp1); sqrt(2p - 2*sum_sv), sv clipped [0,1].
static double orient_dist(const double* Vi, const double* Vj, int d, int dp1) {
    if (dp1 == 0) return 0.0;

    if (dp1 == 1) {
        // Scalar case: Vi, Vj are (d,) vectors (one column each).
        double ni_sq = 0.0, nj_sq = 0.0;
        for (int r = 0; r < d; ++r) {
            ni_sq += Vi[r] * Vi[r];
            nj_sq += Vj[r] * Vj[r];
        }
        double ni = std::sqrt(ni_sq), nj = std::sqrt(nj_sq);
        if (ni < 1e-12 || nj < 1e-12) return 0.0;
        double ss = 0.0, sf = 0.0;
        for (int r = 0; r < d; ++r) {
            double ui = Vi[r] / ni, uj = Vj[r] / nj;
            double diff = ui - uj, sum = ui + uj;
            ss += diff * diff;
            sf += sum  * sum;
        }
        double chord = 0.5 * std::sqrt(std::min(ss, sf));
        if (chord > 1.0) chord = 1.0;
        return 2.0 * std::asin(chord);
    }

    // General case: Procrustes via SVD of M = Vi^T Vj (dp1 x dp1).
    Eigen::MatrixXd M(dp1, dp1);
    for (int a = 0; a < dp1; ++a) {
        for (int b = 0; b < dp1; ++b) {
            double s = 0.0;
            for (int r = 0; r < d; ++r)
                s += Vi[r * dp1 + a] * Vj[r * dp1 + b];
            M(a, b) = s;
        }
    }

    // JacobiSVD is exact for small matrices.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
    const auto& sv = svd.singularValues();
    double sum_sv = 0.0;
    for (int i = 0; i < dp1; ++i) {
        double s = sv(i);
        if (s > 1.0) s = 1.0;
        if (s < 0.0) s = 0.0;
        sum_sv += s;
    }
    double val = 2.0 * dp1 - 2.0 * sum_sv;
    return std::sqrt(val > 0.0 ? val : 0.0);
}

// Cooperative + perpendicular mean distance.
// w_bar = w_i + w_j; d_coop = |w_bar.(mu_i-mu_j)| / ||w_bar||
// d_perp = sqrt(||delta_mu||^2 - d_coop^2)  (Pythagorean identity)
static void coop_mean_dist(
    const double* mu_i, const double* mu_j,
    const double* w_i,  const double* w_j,
    int d,
    double& d_coop, double& d_perp
) {
    double wn_sq = 0.0, wp = 0.0, dm_sq = 0.0;
    for (int k = 0; k < d; ++k) {
        double wb = w_i[k] + w_j[k];
        double dm = mu_i[k] - mu_j[k];
        wn_sq += wb * wb;
        wp    += wb * dm;
        dm_sq += dm * dm;
    }
    if (wn_sq < 1e-12) {
        d_coop = 0.0;
        d_perp = std::sqrt(dm_sq);
        return;
    }
    double wn = std::sqrt(wn_sq);
    d_coop = std::abs(wp / wn);
    double perp_sq = dm_sq - (wp * wp / wn_sq);
    d_perp = std::sqrt(perp_sq > 0.0 ? perp_sq : 0.0);
}

// Wasserstein-1 distance: L1 norm of the difference of CDFs.
static double occ_dist(const double* pi_i, const double* pi_j, int K) {
    double cdf = 0.0, result = 0.0;
    for (int k = 0; k < K; ++k) {
        cdf    += pi_i[k] - pi_j[k];   // running CDF difference
        result += std::abs(cdf);
    }
    return result;
}

// Frobenius distance between (K x K) transition matrices (row-major).
static double dyn_dist(const double* M_i, const double* M_j, int K) {
    double s = 0.0;
    for (int k = 0; k < K * K; ++k) {
        double diff = M_i[k] - M_j[k];
        s += diff * diff;
    }
    return std::sqrt(s);
}

// ── Core pair kernel ─────────────────────────────────────────────────────── //

// Compute 8 raw (unscaled, unweighted) distance components between a query
// profile (passed as explicit pointers) and a single training profile at
// offset j inside the packed training arrays.
//
// Output raw[8]:
//   0 axis  1 opening  2 ecc  3 orient  4 coop  5 perp  6 occ  7 dyn
static void pair_raw(
    // query
    const double* q_axis, double q_ecc, const double* q_opening,
    const double* q_anti_coop,
    const double* q_mu, const double* q_coop_dir,
    const double* q_partition, const double* q_transition,
    int q_geo_reliable, int q_has_partition, int q_has_transition,
    // training[j]
    int j,
    const double* tr_axes,          // (N,d)
    const double* tr_eccentricities,// (N,)
    const double* tr_openings,      // (N,dp1)
    const double* tr_anti_coops,    // (N,d*dp1)
    const double* tr_mus,           // (N,d)
    const double* tr_coop_dirs,     // (N,d)
    const double* tr_partitions,    // (N,K)
    const double* tr_transitions,   // (N,K*K)
    const int32_t* tr_geo_reliable,
    const int32_t* tr_has_partition,
    const int32_t* tr_has_transition,
    int d, int dp1, int K,
    double raw[8]
) {
    for (int c = 0; c < 8; ++c) raw[c] = 0.0;

    const bool geo_pair = q_geo_reliable && tr_geo_reliable[j];

    // ── Identity components (only when geometry is reliable for both) ── //
    if (geo_pair && dp1 > 0) {
        raw[0] = axis_dist(q_axis, tr_axes + (int64_t)j * d, d);
        raw[1] = opening_dist(q_opening, tr_openings + (int64_t)j * dp1, dp1);
        raw[2] = ecc_dist(q_ecc, tr_eccentricities[j]);
        raw[3] = orient_dist(q_anti_coop, tr_anti_coops + (int64_t)j * d * dp1, d, dp1);
    }

    // ── State: mean decomposition ─────────────────────────────────────── //
    double d_coop = 0.0, d_perp = 0.0;
    coop_mean_dist(
        q_mu, tr_mus + (int64_t)j * d,
        q_coop_dir, tr_coop_dirs + (int64_t)j * d,
        d, d_coop, d_perp
    );
    raw[4] = d_coop;
    raw[5] = geo_pair ? d_perp : 0.0;

    // ── State: occupation ────────────────────────────────────────────── //
    if (K > 0 && q_has_partition && tr_has_partition[j])
        raw[6] = occ_dist(q_partition, tr_partitions + (int64_t)j * K, K);

    // ── State: dynamics ──────────────────────────────────────────────── //
    if (K > 0 && q_has_transition && tr_has_transition[j])
        raw[7] = dyn_dist(q_transition, tr_transitions + (int64_t)j * K * K, K);
}

// ── Public API ───────────────────────────────────────────────────────────── //

void compute_distances(
    const double*  q_axis,  double q_ecc, const double* q_opening,
    const double*  q_anti_coop, const double* q_mu, const double* q_coop_dir,
    const double*  q_partition, const double* q_transition,
    int            q_geo_reliable, int q_has_partition, int q_has_transition,
    const double*  tr_axes, const double* tr_eccentricities,
    const double*  tr_openings, const double* tr_anti_coops,
    const double*  tr_mus, const double* tr_coop_dirs,
    const double*  tr_partitions, const double* tr_transitions,
    const int32_t* tr_geo_reliable, const int32_t* tr_has_partition,
    const int32_t* tr_has_transition,
    const double*  weights, const double* scales,
    int N, int d, int dp1, int K,
    double* out_distances
) {
    // Pre-compute effective weights once.
    double eff_w[8];
    for (int c = 0; c < 8; ++c) eff_w[c] = weights[c] / scales[c];

    double raw[8];
    for (int j = 0; j < N; ++j) {
        pair_raw(
            q_axis, q_ecc, q_opening, q_anti_coop,
            q_mu, q_coop_dir, q_partition, q_transition,
            q_geo_reliable, q_has_partition, q_has_transition,
            j,
            tr_axes, tr_eccentricities, tr_openings, tr_anti_coops,
            tr_mus, tr_coop_dirs, tr_partitions, tr_transitions,
            tr_geo_reliable, tr_has_partition, tr_has_transition,
            d, dp1, K, raw
        );

        double dist = 0.0;
        for (int c = 0; c < 8; ++c) dist += raw[c] * eff_w[c];
        out_distances[j] = dist;
    }
}

void compute_raw_cache(
    const int32_t* eval_indices, int n_eval,
    const double*  tr_axes, const double* tr_eccentricities,
    const double*  tr_openings, const double* tr_anti_coops,
    const double*  tr_mus, const double* tr_coop_dirs,
    const double*  tr_partitions, const double* tr_transitions,
    const int32_t* tr_geo_reliable, const int32_t* tr_has_partition,
    const int32_t* tr_has_transition,
    int N, int d, int dp1, int K,
    double* out_raw_cache   // (n_eval, N, 8) row-major
) {
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
    for (int ei = 0; ei < n_eval; ++ei) {
        const int i = static_cast<int>(eval_indices[ei]);

        // Extract query from training packed arrays at row i.
        const double*  q_axis      = tr_axes      + (int64_t)i * d;
        const double   q_ecc       = tr_eccentricities[i];
        const double*  q_opening   = (dp1 > 0) ? tr_openings   + (int64_t)i * dp1    : nullptr;
        const double*  q_anti_coop = (dp1 > 0) ? tr_anti_coops + (int64_t)i * d*dp1  : nullptr;
        const double*  q_mu        = tr_mus       + (int64_t)i * d;
        const double*  q_coop_dir  = tr_coop_dirs + (int64_t)i * d;
        const double*  q_partition = (K > 0) ? tr_partitions  + (int64_t)i * K      : nullptr;
        const double*  q_transition= (K > 0) ? tr_transitions + (int64_t)i * K*K    : nullptr;
        const int q_geo_reliable   = tr_geo_reliable[i];
        const int q_has_partition  = (K > 0) ? tr_has_partition[i]  : 0;
        const int q_has_transition = (K > 0) ? tr_has_transition[i] : 0;

        double* row_out = out_raw_cache + (int64_t)ei * N * 8;

        for (int j = 0; j < N; ++j) {
            pair_raw(
                q_axis, q_ecc, q_opening, q_anti_coop,
                q_mu, q_coop_dir, q_partition, q_transition,
                q_geo_reliable, q_has_partition, q_has_transition,
                j,
                tr_axes, tr_eccentricities, tr_openings, tr_anti_coops,
                tr_mus, tr_coop_dirs, tr_partitions, tr_transitions,
                tr_geo_reliable, tr_has_partition, tr_has_transition,
                d, dp1, K,
                row_out + j * 8
            );
        }
    }
}

} // namespace autoite
