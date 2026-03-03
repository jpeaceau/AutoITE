# ICG-HVRT v0.2.0 Update Specification

## Amendment to ICG-HVRT v0.1.0

**Author:** Jake Peace
**Date:** March 2026
**Amends:** icg_hvrt_specification.md (v0.1.0)
**Nature of Change:** Structural revision — replaces five-component distance with Identity-State decomposition

---

## 1. Problem Statement

Version 0.1.0 compares patients through a five-component distance (Levels, Direction, Shape, Occupation, Dynamics). The Direction component captures the cone axis via the cooperative direction w_i = Σ_i^{−1/2} **1**, and the Shape component captures manifold curvature via the matrix pencil or Log-Euclidean proxy. A single scalar opening angle θ_i = arccos(1/√(w_i^T Σ_i w_i)) summarises the cone's width.

This is incomplete. Each patient's cooperative cone is not circular — it is an elliptical cone with direction-dependent opening angles, patient-specific eccentricity, and patient-specific orientation of the narrow and wide directions. The v0.1.0 distance collapses d−1 directional half-angles into a single scalar, losing the cone's cross-sectional shape entirely.

Two patients can have aligned axes and the same average opening angle but very different eccentricities. One has a circular cone (features resist anti-cooperative deviation equally in all directions). The other has a highly eccentric cone (features deviate anti-cooperatively easily in one specific direction but not others). These patients live in qualitatively different cooperative geometries, but v0.1.0 calls them similar.

---

## 2. The Eigendecomposition of C_i

The cooperative geometry operator C_i = Σ_i^{−1/2}(**11**^T − I)Σ_i^{−1/2} has signature (1, d−1) by Sylvester's law of inertia. Its eigendecomposition:

    C_i = λ_i⁺ · v_i⁺(v_i⁺)^T + Σ_{k=1}^{d-1} λ_{ik}⁻ · v_{ik}⁻(v_{ik}⁻)^T

yields:

- **λ_i⁺ > 0**, eigenvector **v_i⁺** — the cone axis (cooperative direction, normalised). This is what v0.1.0 captured as w_i/‖w_i‖.
- **λ_{i1}⁻, ..., λ_{i,d-1}⁻ < 0**, eigenvectors **V_i⁻ = [v_{i1}⁻, ..., v_{i,d-1}⁻]** — the anti-cooperative frame. In whitened space these are all equal (λ⁻ = −1, circular cone). In original space they are generally unequal (elliptical cone). v0.1.0 did not extract these.

The full cone geometry follows from this eigendecomposition:

- **Directional half-angles:** θ_{ik} = arctan(√(λ_i⁺ / |λ_{ik}⁻|)) for k = 1, ..., d−1
- **Eccentricity:** ecc_i = max_k(θ_{ik}) / min_k(θ_{ik})
- **Opening profile:** the sorted vector (θ_{i1}, ..., θ_{i,d-1})

---

## 3. Structural Change: Identity-State Decomposition

**Replace** the five-component distance with a two-stage decomposition:

### Stage 1: Cone Identity (Compatibility Gate)

The shape of the patient's cone family. Determines whether two patients' cooperative geometries are even comparable.

**Components (replaces v0.1.0's Direction + Shape):**

| Component | Formula | What v0.1.0 Had |
|-----------|---------|-----------------|
| Axis alignment | d_axis = arccos(\|v_i⁺ · v_j⁺\|) | Had (as d_w) |
| Opening profile | d_opening = ‖sort(θ_i) − sort(θ_j)‖₂ | Missing — collapsed to single scalar |
| Eccentricity | d_ecc = \|log(ecc_i) − log(ecc_j)\| | Missing entirely |
| Orientation | d_orient = Procrustes(V_i⁻, V_j⁻) after axis alignment | Missing entirely |

**Combined identity distance:**

    d_identity = β₁·d_axis + β₂·d_opening + β₃·d_ecc + β₄·d_orient

Recommended weights: β₁ = 3.0, β₂ = 2.0, β₃ = 1.0, β₄ = 1.5 (normalised by empirical standard deviations).

**Identity gate (replaces v0.1.0's single direction gate):**

    gate_pass(i, j) = (d_axis < π/4) AND (d_opening < π/6) AND (d_ecc < log(2))

The orientation component d_orient contributes to the distance but not the gate — orientation differences are clinically meaningful but not disqualifying.

### Stage 2: Cone State (Similarity Distance)

Where on the compatible cone the patient lives and how they move. Applied only within identity-gate survivors.

**Components (retains v0.1.0's Levels + Occupation + Dynamics, unchanged):**

    d_state = γ₁·d_μ + γ₂·d_occ + γ₃·d_dyn

Recommended weights: γ₁ = 1.0, γ₂ = 2.0, γ₃ = 1.5.

---

## 4. Revised Matching Algorithm

**Replace** v0.1.0's flat five-component matching with:

```
STAGE 1 — IDENTITY GATE:
  For each training patient j:
    Compute d_identity(i, j)
    Apply gate: d_axis < π/4 AND d_opening < π/6 AND d_ecc < log(2)
  Compatible_i = {j : gate passed}
  If |Compatible_i| < k_min → TRIAGE

STAGE 2 — STATE MATCH (within Compatible_i only):
  Compute d_state(i, j) for j ∈ Compatible_i
  Select k nearest neighbours by d_state
  Fit local model, estimate τ̂_i
```

**Optimised cascade** (replaces v0.1.0's single-level cascade):

1. Pre-filter on d_axis: O(n·d). Cheapest, most discriminative.
2. Filter on d_opening and d_ecc: O(|axis-compatible|·d). Still cheap.
3. Compute d_orient for shape-compatible survivors only: O(|shape-compatible|·d³). Expensive Procrustes, but on a reduced set.
4. State match within identity-compatible: O(|identity-compatible|·(d + K²)).

---

## 5. New Stress Tests

Add two stress tests targeting the previously missing cone invariants:

**Test 6: Eccentricity Gate.** τ = 3·I(ecc_i < 1.5) − 1. Patients with near-circular cones benefit; highly eccentric cones are harmed. This is the critical test — v0.1.0 fails because it has no eccentricity measure.

**Test 7: Orientation Gate.** τ = 2·I(|v_{i1}⁻ · e₁₂| > 0.7). Treatment works when the dominant anti-cooperative direction aligns with a specific feature pair. v0.1.0 fails because it has no orientation measure.

**Expected results update:**

| Test | v0.1.0 | v0.2.0 |
|------|--------|--------|
| Eccentricity Gate | Fails | Passes |
| Orientation Gate | Fails | Passes |
| All other tests | Passes | Passes (no regression) |

---

## 6. Coupled Intervention Protocol Update

**Replace** v0.1.0's cooperative-direction-only target with a full cone identity target:

**v0.1.0 target:** Rotate w_pre toward w_target. Single angular check.

**v0.2.0 target:** Transform full cone identity toward target. Per-component verification:

- Axis: Has v_post⁺ rotated toward v_target⁺?
- Opening: Has θ_post converged toward θ_target?
- Eccentricity: Has ecc_post moved toward ecc_target?
- Orientation: Has V_post⁻ aligned with V_target⁻?

**Gate decision logic:**
- All components improved AND within thresholds → Proceed to Stage 2
- Some improved, others unchanged → Continue Stage 1, adjust protocol
- Axis or eccentricity worsened → Halt, clinical review

This catches the case v0.1.0 misses: a patient whose axis rotated correctly but whose eccentricity increased. Direction is right but cone shape is wrong — not ready for Stage 2.

**Clinical interpretation of cone identity components:**

| Component | Clinician Language |
|-----------|-------------------|
| Axis rotation | "Biomarkers now cooperating along the target pattern" |
| Opening change | "Cooperative threshold has widened/narrowed" |
| Eccentricity change | "Cooperative structure is more/less uniform across biomarker combinations" |
| Orientation change | "Dominant cooperative pattern has shifted between feature groups" |

---

## 7. API Changes

### New class: ConeIdentity

```python
from icg_hvrt import ConeIdentity

identity = ConeIdentity.from_covariance(sigma_i)

identity.axis                    # v⁺ (d,)
identity.positive_eigenvalue     # λ⁺ (scalar)
identity.negative_eigenvalues    # (d-1,) sorted by magnitude
identity.anti_cooperative_frame  # V⁻ (d, d-1)
identity.opening_profile         # (d-1,) sorted half-angles
identity.eccentricity            # scalar

d = ConeIdentity.distance(id_i, id_j)
# Returns: .axis, .opening, .eccentricity, .orientation, .total

passed, margin = ConeIdentity.gate_check(id_i, id_j)
```

### Modified class: ICGHVRTMatcher

```python
# v0.1.0 (DEPRECATED):
matcher = ICGHVRTMatcher(
    alpha_levels=1.0, alpha_direction=2.0, alpha_shape=1.0,
    alpha_occupation=1.5, alpha_dynamics=1.0,
    direction_gate=np.pi/4,
)

# v0.2.0 (REPLACEMENT):
matcher = ICGHVRTMatcher(
    # Identity weights
    beta_axis=3.0, beta_opening=2.0,
    beta_eccentricity=1.0, beta_orientation=1.5,
    # Identity gate thresholds
    gate_axis=np.pi/4, gate_opening=np.pi/6,
    gate_eccentricity=np.log(2),
    # State weights (unchanged)
    gamma_levels=1.0, gamma_occupation=2.0, gamma_dynamics=1.5,
)
```

### Modified class: CoupledInterventionProtocol

```python
# v0.1.0 (DEPRECATED):
protocol = CoupledInterventionProtocol(
    target_direction=w_target,
    rotation_threshold=np.pi/6,
)

# v0.2.0 (REPLACEMENT):
protocol = CoupledInterventionProtocol(
    target_identity=ConeIdentity(...),  # Full cone target
    axis_threshold=np.pi/6,
    opening_threshold=np.pi/8,
    eccentricity_threshold=np.log(1.5),
)
```

---

## 8. Computational Impact

The eigendecomposition of C_i is O(d³) per patient — same order as v0.1.0's Σ_i^{−1/2} computation. The Procrustes distance for d_orient is O(d³) per pair, but is only computed for patients surviving the cheaper axis/opening/eccentricity filters. Net computational overhead relative to v0.1.0 is modest and concentrated in the orientation component, which is deferred to the last pre-filter stage.

---

## 9. Summary of Changes

| Aspect | v0.1.0 | v0.2.0 |
|--------|--------|--------|
| Patient representation | Point on manifold + manifold shape | Cone of manifolds with full identity |
| Distance structure | Five flat components | Two-stage: Identity gate → State match |
| Cone axis | Captured (d_w) | Captured (d_axis, same) |
| Cone opening | Single scalar average | Full d−1 directional profile |
| Cone eccentricity | Missing | Captured (d_ecc) |
| Cone orientation | Missing | Captured (d_orient via Procrustes) |
| Compatibility gate | Axis-only | Axis + opening + eccentricity |
| Coupled intervention target | Direction only | Full cone identity |
| Closed-loop verification | Single angular check | Per-component transformation tracking |
| Eccentricity Gate stress test | Fails | Passes |
| Orientation Gate stress test | Fails | Passes |