"""Three-World Problem: Clean, Confounded, Collider DAG Stress Test.

Tests ICG's ability to distinguish causal structure using Riemannian geometry
and distance-based blending of effect recovery vs null detection models.
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autoite.riemannian import PatientData, MatrixBuilder, RiemannianGeometry
from autoite.jit import IntrinsicJIT

np.random.seed(2025)


@dataclass
class ThreeWorldGenerator:
    """Generate synthetic data for Clean, Confounded, and Collider regimes."""
    n_patients_per_regime: int = 500
    n_observations: int = 100
    noise_std: float = 0.3
    _patient_counter: int = 0

    def _make_patient(self, obs, regime, regime_name, true_effect):
        """Create PatientData with proper field names."""
        self._patient_counter += 1
        T, Y = obs[:, 2], obs[:, 3]
        return PatientData(
            patient_id=self._patient_counter,
            regime=regime,
            regime_name=regime_name,
            observations=obs,
            true_treatment_effect=true_effect,
            mean_treatment=float(np.mean(T)),
            mean_outcome=float(np.mean(Y)),
            observed_correlation=float(np.corrcoef(T, Y)[0, 1])
        )

    def _generate_clean(self) -> PatientData:
        """Regime A: X -> T -> Y (true effect = 1.0)"""
        X1 = np.random.normal(0, 1, self.n_observations)
        X2 = np.random.normal(0, 1, self.n_observations)
        T = 0.5 * X1 + np.random.normal(0, 0.5, self.n_observations)
        Y = 1.0 * T + 0.3 * X1 + np.random.normal(0, self.noise_std, self.n_observations)
        obs = np.column_stack([X1, X2, T, Y])
        return self._make_patient(obs, 'A', 'Clean', 1.0)

    def _generate_confounded(self) -> PatientData:
        """Regime B: U -> T, U -> Y (true effect = 0.0, spurious correlation)"""
        U = np.random.normal(0, 1, self.n_observations)
        X1 = np.random.normal(0, 1, self.n_observations)
        X2 = U + np.random.normal(0, 0.3, self.n_observations)
        T = 0.8 * U + np.random.normal(0, 0.5, self.n_observations)
        Y = 0.0 * T + 0.8 * U + np.random.normal(0, self.noise_std, self.n_observations)
        obs = np.column_stack([X1, X2, T, Y])
        return self._make_patient(obs, 'B', 'Confounded', 0.0)

    def _generate_collider(self) -> PatientData:
        """Regime C: T -> C <- Y (true effect = 0.0, conditioning creates spurious)"""
        X1 = np.random.normal(0, 1, self.n_observations)
        X2 = np.random.normal(0, 1, self.n_observations)
        T = np.random.normal(0, 1, self.n_observations)
        Y_true = np.random.normal(0, 1, self.n_observations)
        C = T + Y_true + np.random.normal(0, 0.3, self.n_observations)
        selection = C > np.median(C)
        T_sel, Y_sel = T[selection], Y_true[selection]
        n_sel = len(T_sel)
        if n_sel < self.n_observations:
            idx = np.random.choice(n_sel, self.n_observations, replace=True)
            T_sel, Y_sel = T_sel[idx], Y_sel[idx]
        X1 = np.random.normal(0, 1, self.n_observations)
        X2 = np.random.normal(0, 1, self.n_observations)
        obs = np.column_stack([X1, X2, T_sel, Y_sel])
        return self._make_patient(obs, 'C', 'Collider', 0.0)

    def generate_dataset(self) -> list:
        self._patient_counter = 0
        patients = []
        for _ in range(self.n_patients_per_regime):
            patients.append(self._generate_clean())
            patients.append(self._generate_confounded())
            patients.append(self._generate_collider())
        return patients


def run_three_world_test():
    """Main test: Distance-based blending of JIT-Aug and X-Learner."""
    print("=" * 80)
    print("THREE-WORLD PROBLEM: Distance-Blended ICG Test")
    print("=" * 80)

    # Generate data
    generator = ThreeWorldGenerator(n_patients_per_regime=500, n_observations=100, noise_std=0.3)
    all_patients = generator.generate_dataset()

    # Build matrices and compute geometry
    matrix_builder = MatrixBuilder(regularization_lambda=1e-3)
    all_matrices = matrix_builder.build_matrices_for_patients(all_patients)
    frechet_mean = RiemannianGeometry.compute_frechet_mean(all_matrices)
    distances = RiemannianGeometry.compute_global_alignment_scores(all_matrices, frechet_mean)

    # Train/test split
    indices = np.arange(len(all_patients))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=[p.regime for p in all_patients]
    )

    train_patients = [all_patients[i] for i in train_idx]
    test_patients = [all_patients[i] for i in test_idx]
    train_matrices = all_matrices[train_idx]
    test_matrices = all_matrices[test_idx]
    train_distances = distances[train_idx]
    test_distances = distances[test_idx]

    # Prepare JIT data
    X_train = [p.observations[:, :2] for p in train_patients]
    T_train = [p.observations[:, 2:3] for p in train_patients]
    Y_train = [p.observations[:, 3] for p in train_patients]

    # Train JIT-Aug (effect recovery)
    jit_aug = IntrinsicJIT(k=50)
    jit_aug.fit(X_train, T_train, Y_train)

    # Distance statistics
    dist_mean, dist_std = np.mean(train_distances), np.std(train_distances)
    dist_p25 = np.percentile(train_distances, 25)

    # Local X-Learner
    def local_xlearner(patient):
        X_c = np.array([p.observations[:, :2].mean(0) for p in train_patients])
        T_c = np.array([p.observations[:, 2].mean() for p in train_patients])
        Y_c = np.array([p.observations[:, 3].mean() for p in train_patients])
        t_med = np.median(T_c)
        treated = T_c >= t_med
        if treated.sum() < 3 or (~treated).sum() < 3:
            return 0.0
        mu_t = Ridge(alpha=1.0).fit(X_c[treated], Y_c[treated])
        mu_c = Ridge(alpha=1.0).fit(X_c[~treated], Y_c[~treated])
        tau_t = Y_c[treated] - mu_c.predict(X_c[treated])
        tau_c = mu_t.predict(X_c[~treated]) - Y_c[~treated]
        cate_t = Ridge(alpha=1.0).fit(X_c[treated], tau_t)
        cate_c = Ridge(alpha=1.0).fit(X_c[~treated], tau_c)
        X_q = patient.observations[:, :2].mean(0).reshape(1, -1)
        return 0.5 * (cate_t.predict(X_q)[0] + cate_c.predict(X_q)[0])

    # Evaluate
    results = {r: {'jit_aug': [], 'xl': [], 'blend': []} for r in ['A', 'B', 'C']}
    true_effects = {'A': 1.0, 'B': 0.0, 'C': 0.0}

    for i, patient in enumerate(test_patients):
        X_test = patient.observations[:, :2]
        T_test = patient.observations[:, 2:3]
        d = test_distances[i]

        pred_aug = jit_aug.predict_effect(X_test, T_test)
        pred_xl = local_xlearner(patient)

        # P25 threshold blend
        pred_blend = pred_aug if d < dist_p25 else pred_xl

        results[patient.regime]['jit_aug'].append(pred_aug)
        results[patient.regime]['xl'].append(pred_xl)
        results[patient.regime]['blend'].append(pred_blend)

    # Print results
    print("\n[Distance by Regime]")
    for regime in ['A', 'B', 'C']:
        mask = [test_patients[i].regime == regime for i in range(len(test_patients))]
        regime_dists = test_distances[mask]
        print(f"  Regime {regime}: {np.mean(regime_dists):.3f} +/- {np.std(regime_dists):.3f}")

    print("\n[Results: MAE +/- Std]")
    print(f"  {'Model':<20} {'Regime A (T=1)':>16} {'Regime B (T=0)':>16} {'Regime C (T=0)':>16} {'Avg MAE':>10}")
    print(f"  {'-'*80}")

    for model in ['jit_aug', 'xl', 'blend']:
        name = {'jit_aug': 'JIT-Aug', 'xl': 'X-Learner', 'blend': 'Blend (P25)'}[model]
        row = f"  {name:<20}"
        total = 0
        for regime in ['A', 'B', 'C']:
            preds = np.array(results[regime][model])
            mae = np.mean(np.abs(preds - true_effects[regime]))
            std = np.std(preds)
            row += f" {mae:>6.3f} +/- {std:<5.3f}"
            total += mae
        row += f" {total/3:>10.4f}"
        print(row)

    # Summary
    print("\n[Key Findings]")
    for regime in ['A', 'B', 'C']:
        best = min(['jit_aug', 'xl', 'blend'],
                   key=lambda m: np.mean(np.abs(np.array(results[regime][m]) - true_effects[regime])))
        name = {'jit_aug': 'JIT-Aug', 'xl': 'X-Learner', 'blend': 'Blend (P25)'}[best]
        print(f"  Regime {regime}: Best = {name}")

    return results


if __name__ == "__main__":
    run_three_world_test()
