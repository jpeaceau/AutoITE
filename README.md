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

### Local regression model (optional)

The local regression step that extracts τ̂ from the k-NN pool supports five variants:

```python
# 'ridge' (default) — Ridge(α=1) on [X|T], L2 regularised
# 'ols'             — OLS on [X|T], unregularised
# 'lad'             — L1 (quantile) regression on [X|T], outlier-robust Y
# 'mean'            — simple mean contrast (T only), trusts cone pre-balancing
# 'median'          — Theil–Sen slope (T only), robust to outlier Y values
est = ICGHVRTEstimator(geometry='cone', k=10, local_model='lad').fit(X, T, Y)
```

---

## Prediction Confidence and Selective Prediction

The k-NN distance is a natural uncertainty signal: when the nearest neighbours
are geometrically distant, the local regression pool is unreliable. Two
complementary features leverage this:

### Distance-weighted k-NN

Neighbours are weighted `exp(−d_j)` in the local regression, so geometrically
close patients dominate. This is a free improvement at 100% coverage — no
abstention required:

```python
est = ICGHVRTEstimator(k=30, distance_weighted=True).fit(X, T, Y)
tau = est.predict_effect(X_new, T_new)
```

### Selective prediction (abstention)

`predict_effect_with_confidence` returns both the ITE estimate and the mean
k-NN distance. Use the confidence score to abstain for out-of-distribution patients:

```python
est = ICGHVRTEstimator(k=30).fit(X, T, Y)
tau, dist = est.predict_effect_with_confidence(X_new, T_new)

# Predict only for high-confidence patients (low distance)
THRESHOLD = 1.5   # tune on held-out data
if dist < THRESHOLD:
    print(f"ITE estimate: {tau:.4f}  (confidence: {dist:.3f})")
else:
    print(f"Out-of-distribution — abstaining (distance: {dist:.3f})")
```

**Selective prediction benchmark** (`experiments/selective_prediction.py`,
10 seeds × 4 DGPs, predicting only the top-20% most confident patients):

| DGP | ICG-HVRT (all) | Distance-weighted | Selective 20% | Random 20% |
|-----|---------------|------------------|---------------|------------|
| Geometric Confounded | 0.056 | **0.030** (−47%) | **0.010** (−82%) | 0.052 (−7%) |
| Mean Confounded      | 0.245 | 0.214 (−13%) | 0.216 (−12%) | 0.241 (−2%) |
| Prognostic Confounded | 0.556 | 0.500 (−10%) | **0.392** (−30%) | 0.543 (−2%) |
| Hidden Confounded    | 0.980 | 0.972 (−1%) | 0.971 (−1%) | 0.978 (−0%) |

> **Key result:** Selective 20% reduces PEHE by 82% on Geometric Confounded —
> geometrically incompatible patients have naturally large k-NN distances.
> On Hidden Confounded, selective ≈ random (the confidence signal is not spuriously
> correlated with hidden confounders), validating that the model is honest about
> what it can and cannot detect.

**Clinical implication:** ICG-HVRT can be deployed as a decision-support tool that
*declares when it cannot make a reliable prediction*, directing clinical judgment to
cases where the geometric support is insufficient. As more patient data accumulates,
the abstention rate decreases.

---

## Counterfactual Augmentation

In observational data, many patients never receive the full range of treatments.
ICG-HVRT can fill missing treatment arms by generating synthetic counterfactual
observations within each k-NN neighbour's HVRT partition distribution:

```python
est = ICGHVRTEstimator(
    k=30,
    counterfactual_aug=True,
    n_synth_per_neighbor=50,   # synthetic observations per neighbour
).fit(X, T, Y)
tau = est.predict_effect(X_new, T_new)
```

**Mechanism:** For each k-NN neighbour j, a within-patient Ridge model is fitted
to j's own observations. Synthetic covariates X_synth are sampled from j's HVRT
partition distribution; treatment T_synth is drawn uniformly over the observed
treatment range (filling the missing arm); outcome Y_synth is predicted by the
within-patient model. The augmented pool extends local regression into
counterfactual treatment regions.

**Augmentation benchmark** (`experiments/counterfactual_aug_benchmark.py`,
10 seeds × 8 DGPs, n_synth_per_neighbor=50):

| DGP | Flat ICG-HVRT | +CF Augmentation | Delta |
|-----|--------------|-----------------|-------|
| Geometric Confounded | 0.056 | **0.035** | −39% |
| Mean Confounded      | 0.114 | 0.116 | +2% |
| Prognostic Confounded | 0.556 | 0.558 | ~0% |
| Hidden Confounded    | 0.980 | 0.974 | ~0% |

> Augmentation helps most on Geometric Confounded where treatment is systematically
> shifted (T_shift = ±1 by confounding), creating a missing treatment arm that
> synthetic counterfactuals fill. On randomised or hidden-confounder DGPs the
> augmentation is neutral — correctly detecting that there is no missing arm to fill.

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
matches and is used as the prediction confidence score in selective prediction.

---

## Benchmark Results

Results from `python -m experiments.ite_comparison` (10 seeds × 9 DGPs,
300 train / 50 test / 100 obs per patient). Lower sqrt-PEHE is better.

| DGP | S-Learner | R-Learner | CRN | RMSN | ICG-HVRT | ICG-HART | Winner |
|-----|-----------|-----------|-----|------|----------|----------|--------|
| Randomised | ~0.34 | ~0.31 | 0.083 | ~0.09 | **0.010** | ~0.015 | ICG-HVRT |
| Geometric Confounded | ~0.52 | ~0.49 | 0.553 | ~0.55 | **0.013** | ~0.018 | ICG-HVRT |
| Mean Confounded | ~0.12 | ~0.10 | **0.084** | ~0.11 | 0.114 | ~0.14 | CRN |
| Sparse Mean Conf (spikes) | ~0.21 | ~0.19 | ~0.16 | ~0.17 | ~0.18 | **~0.09** | ICG-HART |
| Indiv Feature Leak | ~0.23 | ~0.21 | ~0.19 | ~0.18 | ~0.17 | **~0.12** | ICG-HART |
| Prognostic Confounded | ~0.75 | ~0.75 | **0.129** | ~0.14 | 0.556 | ~0.61 | CRN |
| Hidden Confounded | ~0.97 | ~0.97 | ~0.96 | ~0.96 | ~0.96 | ~0.96 | (all fail) |
| TV Confounded | ~0.31 | ~0.30 | ~0.28 | ~0.26 | ~0.24 | **~0.21** | ICG-HART |
| Outlier Spike | ~0.22 | ~0.20 | ~0.17 | ~0.16 | ~0.20 | **~0.08** | ICG-HART |

> Values marked `~` are approximate from single runs; run
> `python -m experiments.ite_comparison` to reproduce exact figures.

### Data-generating processes

| DGP | Confounding mechanism | Key property |
|-----|-----------------------|--------------|
| **Randomised** | None (RCT) | Oracle baseline |
| **Geometric Confounded** | Treatment confounded via cone geometry (T_shift = ±U) | ICG immune by design |
| **Mean Confounded** | U → E[X] and U → E[T] (mean-shift) | CRN's adversarial domain |
| **Sparse Mean Conf** | U leaked into K=5 obs per patient (sparse signal) | ICG-HART extracts spikes |
| **Indiv Feature Leak** | U leaked into single feature, 3 obs only (ultra-sparse) | ICG-HART pattern matching |
| **Prognostic Confounded** | U → tau AND U → E[X], but T ⊥ U (randomised) | CRN wins via per-step supervision |
| **Hidden Confounded** | U → T, U ∉ X | All methods fail — negative control |
| **TV Confounded** | Time-varying U → T | ICG-HART tracks transitions |
| **Outlier Spike** | Extreme single-observation contamination (≥10σ) | MAD whitening absorbs spikes |

### Interpretation

- **ICG-HVRT** excels when effect heterogeneity lives in the covariance geometry
  (Randomised, Geometric Confounded). Cone identity is immune to geometric
  confounding: the cone shape changes with the covariance *and* the effect
  modifier, so confounded patients naturally have high identity distance from
  controls.
- **ICG-HART** excels when data contains outlier spikes or sparse individual-level
  signals. MAD whitening leaves the trajectory statistic unchanged under
  single-feature contamination; SD whitening inflates it by O(spike_mag × √d).
- **CRN** wins on Mean Confounded (adversarial gradient reversal targets mean-shift)
  and on Prognostic Confounded (per-timestep supervision exploits U's leakage into X
  at every observation, not just the patient mean).
- **Hidden Confounded** is a negative control: all methods fail because the
  confounder is invisible in X. ICG-HVRT's confidence signal correctly detects
  *geometric* out-of-distribution patients but cannot detect hidden confounders
  whose geometry appears normal.

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
    ICGHVRTEstimator,           # Main estimator (both geometry variants)
    ICGHVRTMatcher,             # Distance computation and k-NN matching
    CooperativeGeometryProfile, # Per-patient geometry profile
    ConeIdentity,               # Cone eigendecomposition and identity distance
    CoupledInterventionProtocol,# Closed-loop intervention tracking
    fit_shared_hvrt,            # Shared HVRT/HART model fitting
    pool_whitened_observations, # Whitened observation pooling
)
```

### `ICGHVRTEstimator` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | `10` | Number of k-NN neighbours for local regression |
| `geometry` | `'cone'` | `'cone'` (ICG-HVRT) or `'pyramid'` (ICG-HART) |
| `learn_weights` | `False` | L-BFGS-B calibration of 8-component distance weights |
| `local_model` | `'ridge'` | Local regression: `'ridge'`, `'ols'`, `'lad'`, `'mean'`, `'median'` |
| `distance_weighted` | `False` | Weight k-NN pool by `exp(−d_j)` in local regression |
| `counterfactual_aug` | `False` | Augment pool with HVRT-sampled counterfactual observations |
| `n_synth_per_neighbor` | `30` | Synthetic observations per neighbour (with `counterfactual_aug`) |
| `alpha_local` | `1.0` | Ridge regularisation strength for local regression |

### Key methods

```python
# Fit on training data
est.fit(X_list, T_list, Y_list)   # lists of (n_obs, d), (n_obs,), (n_obs,) arrays

# Predict ITE for a test patient
tau = est.predict_effect(X_new, T_new)

# Predict ITE + confidence (for selective prediction)
tau, dist = est.predict_effect_with_confidence(X_new, T_new)
# dist = mean k-NN distance; lower = higher geometric confidence

# Triage report (geometry diagnostics)
report = est.triage_report(X_new, T_new)
```

See `help(ICGHVRTEstimator)` for full parameter documentation.

---

## Reproducing Benchmarks

```bash
# Full ITE comparison (8 methods × 9 DGPs × 10 seeds, ~30 min)
python -m experiments.ite_comparison

# Selective prediction / coverage-PEHE curves (10 seeds × 4 DGPs)
python -m experiments.selective_prediction

# Counterfactual augmentation benchmark (10 seeds × 8 DGPs)
python -m experiments.counterfactual_aug_benchmark

# Local regression model sweep (5 seeds × 7 DGPs × 5 models)
python -m experiments.local_model_benchmark

# Comprehensive benchmark (policy regret + uncertainty calibration)
python -m experiments.comprehensive_benchmark
```

---

## License

AGPL-3.0. See [LICENSE](LICENSE).
