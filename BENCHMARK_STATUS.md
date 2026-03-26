# AutoITE Benchmark Status

## Current Issue
The C++ extension (`autoite._core`) doesn't compile with the current
build backend (setuptools). The `compute_distances` function is required
by the matcher but has no Python fallback.

## To Fix
Either:
1. Switch pyproject.toml build backend to scikit-build-core (like GeoXGB)
2. Add Python fallback paths in matcher.py for when _core is unavailable

## Benchmark Plan
1. Fix build → run baseline with hvrt 2.11.0 (pip) → record PEHE scores
2. Replace hvrt dependency with GeoXGBPriv's improved HVRT (via SynKonosCPP)
3. Re-run same benchmarks → compare PEHE scores
4. The improved HVRT has: min_gain, discretization fixes, novelty enforcement

## DGPs to Test
- Randomised (baseline, no confounding)
- Geometric confound (cooperation-dependent ITE)
- Mean confound (location-based ITE)
- Hidden confound (negative control — should be unlearnable)
- Time-varying confound (adversarial for ICG-HVRT)
- Prognostic confound (biomarker-based)
