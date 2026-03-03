cmake_minimum_required(VERSION 3.20)
project(autoite_core CXX)

# ── Eigen3 discovery ─────────────────────────────────────────────────────── #
#
# Search order (first match wins):
#   1. EIGEN3_INCLUDE_DIR CMake variable (set by caller or pyproject.toml)
#   2. EIGEN3_INCLUDE_DIR environment variable
#   3. RcppEigen (R package, common on Windows data-science machines)
#   4. HVRT CLion build cache (project-local Eigen download)
#   5. vcpkg x64-windows
#   6. Standard find_package(Eigen3)

if(NOT DEFINED EIGEN3_INCLUDE_DIR OR EIGEN3_INCLUDE_DIR STREQUAL "")
    if(DEFINED ENV{EIGEN3_INCLUDE_DIR} AND NOT "$ENV{EIGEN3_INCLUDE_DIR}" STREQUAL "")
        set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
    endif()
endif()

if(NOT DEFINED EIGEN3_INCLUDE_DIR OR EIGEN3_INCLUDE_DIR STREQUAL "")
    # RcppEigen — ships Eigen headers as part of an R package
    set(_rcppeigen "$ENV{LOCALAPPDATA}/R/win-library/4.4/RcppEigen/include")
    if(EXISTS "${_rcppeigen}/Eigen/Dense")
        set(EIGEN3_INCLUDE_DIR "${_rcppeigen}")
        message(STATUS "Eigen3 found via RcppEigen: ${EIGEN3_INCLUDE_DIR}")
    endif()
endif()

if(NOT DEFINED EIGEN3_INCLUDE_DIR OR EIGEN3_INCLUDE_DIR STREQUAL "")
    # HVRT CLion FetchContent build
    set(_hvrt "$ENV{USERPROFILE}/CLionProjects/HVRT/build_nomp/_deps/eigen-src")
    if(EXISTS "${_hvrt}/Eigen/Dense")
        set(EIGEN3_INCLUDE_DIR "${_hvrt}")
        message(STATUS "Eigen3 found via HVRT build: ${EIGEN3_INCLUDE_DIR}")
    endif()
endif()

if(NOT DEFINED EIGEN3_INCLUDE_DIR OR EIGEN3_INCLUDE_DIR STREQUAL "")
    # vcpkg fallback
    foreach(_arch x64-windows x64-windows-static)
        set(_vcpkg "C:/vcpkg/installed/${_arch}/include")
        if(EXISTS "${_vcpkg}/Eigen/Dense")
            set(EIGEN3_INCLUDE_DIR "${_vcpkg}")
            message(STATUS "Eigen3 found via vcpkg: ${EIGEN3_INCLUDE_DIR}")
            break()
        endif()
    endforeach()
endif()

if(NOT DEFINED EIGEN3_INCLUDE_DIR OR EIGEN3_INCLUDE_DIR STREQUAL "")
    find_package(Eigen3 3.3 QUIET)
    if(Eigen3_FOUND)
        set(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIRS}")
        message(STATUS "Eigen3 found via find_package: ${EIGEN3_INCLUDE_DIR}")
    endif()
endif()

if(NOT DEFINED EIGEN3_INCLUDE_DIR OR EIGEN3_INCLUDE_DIR STREQUAL "")
    message(WARNING
        "Eigen3 not found — skipping C++ extension.\n"
        "Set EIGEN3_INCLUDE_DIR to the directory containing Eigen/Dense.")
    return()
endif()

message(STATUS "Using Eigen3 at: ${EIGEN3_INCLUDE_DIR}")

# ── pybind11 ─────────────────────────────────────────────────────────────── #
#
# scikit-build-core injects pybind11 automatically.
# For manual cmake builds, find it via Python.

if(NOT DEFINED pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT pybind11_DIR)
        # Fallback: ask current python interpreter
        execute_process(
            COMMAND python -c
                "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
endif()

if(pybind11_DIR)
    find_package(pybind11 REQUIRED CONFIG HINTS "${pybind11_DIR}")
else()
    find_package(pybind11 REQUIRED)
endif()

# ── C++ standard ─────────────────────────────────────────────────────────── #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OpenMP (optional; enables parallel LOO and raw-cache loops) ───────────── #
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "OpenMP NOT found — building single-threaded _core")
endif()

# ── subdirectory with the actual extension target ─────────────────────────── #
add_subdirectory(autoite/_core)
