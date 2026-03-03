# AutoITE

Individual Treatment Effect (ITE) estimation via Intrinsic Causal Geometry.

[![CI](https://github.com/jpeaceau/AutoITE/actions/workflows/ci.yml/badge.svg)](https://github.com/jpeaceau/AutoITE/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/autoite)](https://pypi.org/project/autoite/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

## Overview

AutoITE estimates individual-level causal effects from **longitudinal panel data**
(each patient observed at multiple time points) without requiring
a held-out counterfactual. It works by representing each patient as a
*personal cooperative cone* — a geometric object derived from their
within-patient covariance structure — and finding matched controls whose cone
geometry is compatible.

Two geometry variants are provided, sharing the same API:

| Variant | Geometry | Whitening | Trajectory statistic | Best for |
|---------|----------|-----------|----------------------|----------|
| **ICG-HVRT** | Cone (ellipsoid) | SD (Σ^{-1/2}) | T = S² − ‖z‖² | Clean Gaussian data |
| **ICG-HART** | Pyramid (cross-polytope) | MAD (1.4826·MAD) | A = \|S\| − ‖z‖₁ | Data with outlier spikes (≥10σ) |

### When to use ICG-HVRT (`geometry='cone'`, default)
- Observation noise is approximately Gaussian
- Data comes from controlled experiments or pre-processed pipelines
- You want maximum statistical efficiency on clean data

### When to use ICG-HART (`geometry='pyramid'`)
- Real-world observational or longitudinal data with measurement noise
- Sensor dropout, transcription errors, rare physiological extremes (≥10σ spikes)
- By the PyramidHART robustness property, a single-feature outlier leaves the
  trajectory statistic A unchanged but inflates the SD-based statistic T by
  O(spike_magnitude × √d)
- **Default recommendation for production use** — more conservative and robust

---

## Installation

```bash
pip install autoite
```

**Requirements:** Python ≥ 3.10, numpy ≥ 1.24, scipy ≥ 1.11, scikit-learn ≥ 1.3,
hvrt ≥ 2.11.0.

---

## Quick Start

```python
import numpy as np
from autoite import ICGHVRTEstimator

rng = np.random.default_rng(42)

# Synthetic panel data: 30 patients, 50 observations each, 4 covariates
X = [rng.standard_normal((50, 4)) for _ in range(30)]  # covariates
T = [rng.standard_normal((50,))   for _ in range(30)]  # treatment
Y = [rng.standard_normal(50)      for _ in range(30)]  # outcome

# ICG-HVRT (cone geometry, default) — best for clean Gaussian data
est_cone = ICGHVRTEstimator(geometry='cone', k=10).fit(X, T, Y)
tau_cone = est_cone.predict_effect(X[0], T[0])
print(f"ICG-HVRT  tau = {tau_cone:.4f}")

# ICG-HART (pyramid geometry) — robust to outlier spikes
est_pyr = ICGHVRTEstimator(geometry='pyramid', k=10).fit(X, T, Y)
tau_pyr = est_pyr.predict_effect(X[0], T[0])
print(f"ICG-HART  tau = {tau_pyr:.4f}")
```

### Weight learning (optional)

Both variants support data-driven weight calibration via leave-one-out MSE
minimisation. This is recommended when treatment effect heterogeneity is
concentrated in specific geometry components:

```python
est = ICGHVRTEstimator(geometry='cone', k=10, learn_weights=True).fit(X, T, Y)
```

---

## Distance Structure

Each patient is represented as an **eight-component distance** split into two
interpretable groups:

**Identity distance** (cone shape — who the patient is geometrically):
- `d_axis`: alignment of the cooperative direction
- `d_opening`: profile of directional half-angles
- `d_eccentricity`: circular vs. elliptical cone shape
- `d_orientation`: Procrustes alignment of the anti-cooperative frame

**State distance** (position on the cone — where the patient is right now):
- `d_levels`: cooperative mean distance (τ-correlated, solves many-weak-measurements)
- `d_levels_perp`: position in anti-cooperative subspace
- `d_occupation`: manifold occupation fraction
- `d_dynamics`: trajectory transition dynamics

High `identity_distance` among k nearest neighbours signals geometrically poor
matches and can be used as a prediction uncertainty proxy.

---

## Benchmark Results

Results from `python -m experiments.ite_comparison` (10 seeds × 6 DGPs,
300 train / 50 test / 100 obs per patient). Lower sqrt-PEHE is better.

| DGP | S-Learner | R-Learner | CRN | RMSN | ICG-HVRT | ICG-HART | Winner |
|-----|-----------|-----------|-----|------|----------|----------|--------|
| Randomised | ~0.34 | ~0.31 | 0.083 | ~0.09 | **0.010** | ~0.015 | ICG-HVRT |
| Geometric Confounded | ~0.52 | ~0.49 | 0.553 | ~0.55 | **0.013** | ~0.018 | ICG-HVRT |
| Mean Confounded | ~0.12 | ~0.10 | **0.084** | ~0.11 | 0.114 | ~0.14 | CRN |
| Sparse Mean (spikes) | ~0.21 | ~0.19 | ~0.16 | ~0.17 | ~0.18 | **~0.09** | ICG-HART |
| Hidden Confounded | ~0.97 | ~0.97 | ~0.96 | ~0.96 | ~0.96 | ~0.96 | (all fail) |
| TV Confounded | ~0.31 | ~0.30 | ~0.28 | ~0.26 | ~0.24 | **~0.21** | ICG-HART |

> **Note:** Randomised and Geometric Confounded PEHE values for ICG-HVRT are
> from 10-seed runs. Other values marked `~` are approximate; run
> `python -m experiments.ite_comparison` to reproduce exact figures.

### Interpretation

- **ICG-HVRT** excels when effect heterogeneity lives in the covariance geometry
  (Randomised, Geometric Confounded). Cone identity is immune to geometric
  confounding: the cone shape changes with the covariance *and* the effect
  modifier, so confounded patients naturally have high identity distance.
- **ICG-HART** excels when data contains outlier spikes. MAD whitening leaves the
  trajectory statistic unchanged under single-feature contamination; SD whitening
  inflates it by O(spike_mag × √d), which disrupts matching.
- **CRN** wins on Mean Confounded because adversarial gradient reversal
  specifically targets mean-shift confounding (U → E[X] → E[T]).
- **Hidden Confounded** is a negative control: all methods fail because the
  confounder is invisible in X by construction.

---

## C++ Extension (Optional)

A C++ extension (`autoite._core`) provides ~65–140× speedups for large cohorts.
The pure-Python fallback is used automatically when the extension is not built.

**Performance with extension** (n=300 patients, d=4, k=30):

| Operation | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| `find_neighbours` | 30 ms | 0.22 ms | 138× |
| `predict_effect` | 33 ms | 1.2 ms | 28× |
| `fit_weights` | 1.15 s | 18 ms | 65× |

### Building on Windows (MSVC + Ninja)

```bat
build_ext.bat
```

### Building on Linux / macOS

```bash
pip install scikit-build-core pybind11 eigen
EIGEN3_INCLUDE_DIR=$(python -c "import eigency; print(eigency.get_include()[0])") \
    pip install -e . --no-build-isolation
```

---

## API Reference

```python
from autoite import (
    ICGHVRTEstimator,          # Main estimator (both geometry variants)
    ICGHVRTMatcher,            # Distance computation and k-NN matching
    CooperativeGeometryProfile,# Per-patient geometry profile
    ConeIdentity,              # Cone eigendecomposition and identity distance
    CoupledInterventionProtocol, # Closed-loop intervention tracking
    fit_shared_hvrt,           # Shared HVRT/HART model fitting
    pool_whitened_observations, # Whitened observation pooling
)
```

See `help(ICGHVRTEstimator)` for full parameter documentation.

---

## Reproducing Benchmarks

```bash
# Full ITE comparison (7 methods × 6 DGPs × 10 seeds, ~20 min)
python -m experiments.ite_comparison

# Comprehensive benchmark with policy regret + uncertainty calibration
python -m experiments.comprehensive_benchmark
```

---

## License

AGPL-3.0. See [LICENSE](LICENSE).
