#include "loo.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>
#ifdef _OPENMP
#  include <omp.h>
#endif

namespace autoite {

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
) {
    const int dp1 = d + 1;   // feature cols + treatment col in XT

    // Effective weights: eff_w[c] = weights[c] / scales[c]
    double eff_w[8];
    for (int c = 0; c < 8; ++c)
        eff_w[c] = weights[c] / scales[c];

    // Working buffers are declared inside the loop so each OpenMP thread
    // gets its own copy (thread-local by virtue of being loop-local variables).

    double total_mse = 0.0;

#ifdef _OPENMP
#   pragma omp parallel for reduction(+:total_mse) schedule(dynamic, 4)
#endif
    for (int ei = 0; ei < n_eval; ++ei) {
        std::vector<double> D_i(N_train);
        std::vector<int>    sorted_idx(N_train);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        const int       i          = eval_indices[ei];
        const double*   cache_row  = raw_cache + (int64_t)ei * N_train * 8;

        // ── Step 1: distance to every training patient ────────────────────
        for (int j = 0; j < N_train; ++j) {
            const double* comp = cache_row + j * 8;
            double dist = 0.0;
            for (int c = 0; c < 8; ++c)
                dist += comp[c] * eff_w[c];
            D_i[j] = dist;
        }
        D_i[i] = std::numeric_limits<double>::infinity();  // LOO: exclude self

        // ── Step 2: partial sort to find k nearest ────────────────────────
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::partial_sort(
            sorted_idx.begin(), sorted_idx.begin() + k, sorted_idx.end(),
            [&D_i](int a, int b) { return D_i[a] < D_i[b]; }
        );
        // sorted_idx[0..k-1] are now the k nearest training indices

        // ── Step 3: count rows in local neighbourhood ─────────────────────
        int n_local = 0;
        for (int ki = 0; ki < k; ++ki)
            n_local += offsets[sorted_idx[ki] + 1] - offsets[sorted_idx[ki]];

        // ── Step 4: assemble XT_local and Y_local ─────────────────────────
        Eigen::MatrixXd XT_local(n_local, dp1);
        Eigen::VectorXd  Y_local(n_local);
        {
            int row = 0;
            for (int ki = 0; ki < k; ++ki) {
                const int j     = sorted_idx[ki];
                const int start = offsets[j];
                const int end   = offsets[j + 1];
                for (int r = start; r < end; ++r, ++row) {
                    for (int c = 0; c < d; ++c)
                        XT_local(row, c) = obs_flat[(int64_t)r * d + c];
                    XT_local(row, d) = T_flat[r];
                    Y_local(row)     = Y_flat[r];
                }
            }
        }

        // ── Step 5: centred Ridge (matches sklearn Ridge(fit_intercept=True)) ──
        //
        // Center XT and Y using training neighbourhood means, then solve:
        //   (XT_c^T XT_c + alpha*I) coef = XT_c^T Y_c
        //
        // Intercept: b = Y_mean - X_mean^T coef
        // Predict:   y_hat = (XT_test - X_mean) coef + b

        const Eigen::RowVectorXd XT_mean = XT_local.colwise().mean();
        const Eigen::MatrixXd    XT_c    = XT_local.rowwise() - XT_mean;
        const double             Y_mean  = Y_local.mean();
        const Eigen::VectorXd    Y_c     = Y_local.array() - Y_mean;

        Eigen::MatrixXd A = XT_c.transpose() * XT_c;
        A.diagonal().array() += alpha_ridge;
        const Eigen::VectorXd coef = A.ldlt().solve(XT_c.transpose() * Y_c);

        // intercept = Y_mean - X_mean . coef
        double intercept = Y_mean;
        for (int c = 0; c < dp1; ++c)
            intercept -= XT_mean(c) * coef(c);

        // ── Step 6: predict on patient i's own observations ───────────────
        const int start_i = offsets[i];
        const int end_i   = offsets[i + 1];
        const int n_i     = end_i - start_i;

        Eigen::MatrixXd XT_i(n_i, dp1);
        Eigen::VectorXd  Y_i(n_i);
        for (int r = start_i; r < end_i; ++r) {
            const int r_local = r - start_i;
            for (int c = 0; c < d; ++c)
                XT_i(r_local, c) = obs_flat[(int64_t)r * d + c];
            XT_i(r_local, d) = T_flat[r];
            Y_i(r_local)     = Y_flat[r];
        }

        // Predict using raw XT_i.
        // intercept = Y_mean - XT_mean @ coef already encodes the mean offset,
        // so we must NOT subtract XT_mean again here.
        Eigen::VectorXd Y_hat = XT_i * coef;
        Y_hat.array() += intercept;

        const double mse = (Y_i - Y_hat).squaredNorm() / n_i;
        total_mse += mse;
    }

    return total_mse / n_eval;
}

} // namespace autoite
