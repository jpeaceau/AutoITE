cmake_minimum_required(VERSION 3.20)

# ── pybind11 extension ───────────────────────────────────────────────────── #

pybind11_add_module(_core
    src/loo.cpp
    src/dist.cpp
    src/bindings.cpp
)

target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIRS}
)

# Compiler flags (OpenMP is included here for each toolchain)
#
# MSVC: /openmp activates OpenMP 2.0 support, defines _OPENMP, and
#        auto-links vcomp.lib — no separate find_package target needed.
# GCC/Clang: -fopenmp covers both compilation and linking.
target_compile_options(_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3 /EHsc /openmp>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native -Wall>
)

# Prevent Eigen's own OpenMP from fighting our explicit parallel regions.
target_compile_definitions(_core PRIVATE EIGEN_DONT_PARALLELIZE)

# OpenMP for GCC/Clang: add -fopenmp link flag via the imported target.
# MSVC handles this implicitly via /openmp above.
if(OpenMP_FOUND AND NOT MSVC)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# C++17 required for structured bindings, if/constexpr etc.
target_compile_features(_core PRIVATE cxx_std_17)

# ── install: place _core.pyd alongside autoite/__init__.py ──────────────── #
# For in-place dev builds: cmake --install _build_ext --prefix .
install(TARGETS _core
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/autoite"
    RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/autoite"
)
