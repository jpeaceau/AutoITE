#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "loo.hpp"
#include "dist.hpp"

namespace py = pybind11;
using arr_d  = py::array_t<double,  py::array::c_style | py::array::forcecast>;
using arr_i  = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

// ── loo_objective wrapper ────────────────────────────────────────────────── //

static double py_loo_objective(
    arr_d raw_cache,      // (n_eval, N_train, 8)
    arr_d obs_flat,       // (sum_n_obs, d)
    arr_d T_flat,         // (sum_n_obs,)
    arr_d Y_flat,         // (sum_n_obs,)
    arr_i offsets,        // (N_train+1,)
    arr_i eval_indices,   // (n_eval,)
    arr_d weights,        // (8,)
    arr_d scales,         // (8,)
    int   k,
    double alpha_ridge
) {
    auto rc_info  = raw_cache.request();
    auto obs_info = obs_flat.request();

    if (rc_info.ndim != 3 || rc_info.shape[2] != 8)
        throw std::runtime_error("raw_cache must be shape (n_eval, N_train, 8)");
    if (obs_info.ndim != 2)
        throw std::runtime_error("obs_flat must be shape (sum_n_obs, d)");

    const int n_eval  = static_cast<int>(rc_info.shape[0]);
    const int N_train = static_cast<int>(rc_info.shape[1]);
    const int d       = static_cast<int>(obs_info.shape[1]);

    return autoite::loo_objective(
        static_cast<const double*>(rc_info.ptr),
        static_cast<const double*>(obs_info.ptr),
        static_cast<const double*>(T_flat.request().ptr),
        static_cast<const double*>(Y_flat.request().ptr),
        static_cast<const int32_t*>(offsets.request().ptr),
        static_cast<const int32_t*>(eval_indices.request().ptr),
        static_cast<const double*>(weights.request().ptr),
        static_cast<const double*>(scales.request().ptr),
        n_eval, N_train, d, k, alpha_ridge
    );
}

// ── compute_distances wrapper ────────────────────────────────────────────── //

static py::array_t<double> py_compute_distances(
    arr_d q_axis,           // (d,)
    double q_ecc,
    arr_d q_opening,        // (dp1,)
    arr_d q_anti_coop,      // (d*dp1,)
    arr_d q_mu,             // (d,)
    arr_d q_coop_dir,       // (d,)
    arr_d q_partition,      // (K,)
    arr_d q_transition,     // (K*K,)
    int   q_geo_reliable,
    int   q_has_partition,
    int   q_has_transition,
    arr_d tr_axes,              // (N, d)
    arr_d tr_eccentricities,    // (N,)
    arr_d tr_openings,          // (N, dp1)
    arr_d tr_anti_coops,        // (N, d*dp1)
    arr_d tr_mus,               // (N, d)
    arr_d tr_coop_dirs,         // (N, d)
    arr_d tr_partitions,        // (N, K)
    arr_d tr_transitions,       // (N, K*K)
    arr_i tr_geo_reliable,      // (N,)
    arr_i tr_has_partition,     // (N,)
    arr_i tr_has_transition,    // (N,)
    arr_d weights,              // (8,)
    arr_d scales                // (8,)
) {
    auto tr_ax_info = tr_axes.request();
    if (tr_ax_info.ndim != 2)
        throw std::runtime_error("tr_axes must be 2-D (N, d)");

    const int N   = static_cast<int>(tr_ax_info.shape[0]);
    const int d   = static_cast<int>(tr_ax_info.shape[1]);
    const int dp1 = d - 1;

    auto tr_part_info = tr_partitions.request();
    const int K = (tr_part_info.ndim == 2 && tr_part_info.shape[0] == N)
                  ? static_cast<int>(tr_part_info.shape[1]) : 0;

    auto result = py::array_t<double>(N);
    double* out = static_cast<double*>(result.request().ptr);

    // Null-safe data pointers for zero-dimension arrays.
    auto safe_d = [](const arr_d& a) -> const double* {
        auto info = a.request();
        return (info.size > 0) ? static_cast<const double*>(info.ptr) : nullptr;
    };

    autoite::compute_distances(
        safe_d(q_axis), q_ecc, safe_d(q_opening), safe_d(q_anti_coop),
        safe_d(q_mu), safe_d(q_coop_dir), safe_d(q_partition), safe_d(q_transition),
        q_geo_reliable, q_has_partition, q_has_transition,
        safe_d(tr_axes), static_cast<const double*>(tr_eccentricities.request().ptr),
        safe_d(tr_openings), safe_d(tr_anti_coops),
        safe_d(tr_mus), safe_d(tr_coop_dirs),
        safe_d(tr_partitions), safe_d(tr_transitions),
        static_cast<const int32_t*>(tr_geo_reliable.request().ptr),
        static_cast<const int32_t*>(tr_has_partition.request().ptr),
        static_cast<const int32_t*>(tr_has_transition.request().ptr),
        static_cast<const double*>(weights.request().ptr),
        static_cast<const double*>(scales.request().ptr),
        N, d, dp1, K,
        out
    );
    return result;
}

// ── compute_raw_cache wrapper ────────────────────────────────────────────── //

static py::array_t<double> py_compute_raw_cache(
    arr_i eval_indices,         // (n_eval,)
    arr_d tr_axes,              // (N, d)
    arr_d tr_eccentricities,    // (N,)
    arr_d tr_openings,          // (N, dp1)
    arr_d tr_anti_coops,        // (N, d*dp1)
    arr_d tr_mus,               // (N, d)
    arr_d tr_coop_dirs,         // (N, d)
    arr_d tr_partitions,        // (N, K)
    arr_d tr_transitions,       // (N, K*K)
    arr_i tr_geo_reliable,      // (N,)
    arr_i tr_has_partition,     // (N,)
    arr_i tr_has_transition     // (N,)
) {
    auto ev_info  = eval_indices.request();
    auto tr_info  = tr_axes.request();
    if (tr_info.ndim != 2)
        throw std::runtime_error("tr_axes must be 2-D (N, d)");

    const int n_eval = static_cast<int>(ev_info.shape[0]);
    const int N      = static_cast<int>(tr_info.shape[0]);
    const int d      = static_cast<int>(tr_info.shape[1]);
    const int dp1    = d - 1;

    auto tr_part_info = tr_partitions.request();
    const int K = (tr_part_info.ndim == 2 && tr_part_info.shape[0] == N)
                  ? static_cast<int>(tr_part_info.shape[1]) : 0;

    std::vector<py::ssize_t> shape = {n_eval, N, 8};
    auto result = py::array_t<double>(shape);
    double* out = static_cast<double*>(result.request().ptr);

    auto safe_d = [](const arr_d& a) -> const double* {
        auto info = a.request();
        return (info.size > 0) ? static_cast<const double*>(info.ptr) : nullptr;
    };

    autoite::compute_raw_cache(
        static_cast<const int32_t*>(ev_info.ptr), n_eval,
        safe_d(tr_axes), static_cast<const double*>(tr_eccentricities.request().ptr),
        safe_d(tr_openings), safe_d(tr_anti_coops),
        safe_d(tr_mus), safe_d(tr_coop_dirs),
        safe_d(tr_partitions), safe_d(tr_transitions),
        static_cast<const int32_t*>(tr_geo_reliable.request().ptr),
        static_cast<const int32_t*>(tr_has_partition.request().ptr),
        static_cast<const int32_t*>(tr_has_transition.request().ptr),
        N, d, dp1, K,
        out
    );
    return result;
}

// ── module definition ────────────────────────────────────────────────────── //

PYBIND11_MODULE(_core, m) {
    m.doc() = "autoite C++ core — Phase 1+2: distance kernel + fit_weights LOO objective";

    m.def("loo_objective", &py_loo_objective,
        py::arg("raw_cache"), py::arg("obs_flat"), py::arg("T_flat"), py::arg("Y_flat"),
        py::arg("offsets"), py::arg("eval_indices"),
        py::arg("weights"), py::arg("scales"), py::arg("k"), py::arg("alpha_ridge"),
        "LOO-MSE objective for weight optimisation (Phase 2).");

    m.def("compute_distances", &py_compute_distances,
        py::arg("q_axis"), py::arg("q_ecc"), py::arg("q_opening"), py::arg("q_anti_coop"),
        py::arg("q_mu"), py::arg("q_coop_dir"), py::arg("q_partition"), py::arg("q_transition"),
        py::arg("q_geo_reliable"), py::arg("q_has_partition"), py::arg("q_has_transition"),
        py::arg("tr_axes"), py::arg("tr_eccentricities"),
        py::arg("tr_openings"), py::arg("tr_anti_coops"),
        py::arg("tr_mus"), py::arg("tr_coop_dirs"),
        py::arg("tr_partitions"), py::arg("tr_transitions"),
        py::arg("tr_geo_reliable"), py::arg("tr_has_partition"), py::arg("tr_has_transition"),
        py::arg("weights"), py::arg("scales"),
        "Compute (N,) weighted distances from one query to N training profiles (Phase 1).");

    m.def("compute_raw_cache", &py_compute_raw_cache,
        py::arg("eval_indices"),
        py::arg("tr_axes"), py::arg("tr_eccentricities"),
        py::arg("tr_openings"), py::arg("tr_anti_coops"),
        py::arg("tr_mus"), py::arg("tr_coop_dirs"),
        py::arg("tr_partitions"), py::arg("tr_transitions"),
        py::arg("tr_geo_reliable"), py::arg("tr_has_partition"), py::arg("tr_has_transition"),
        "Compute (n_eval, N, 8) raw component cache from training-set eval patients (Phase 1).");

    m.attr("__version__") = "0.2.0-phase1+phase2";
}
