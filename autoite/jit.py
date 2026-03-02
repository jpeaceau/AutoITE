import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from scipy.linalg import logm, norm

class IntrinsicJIT:
    def __init__(self, k=30, geometry='augmented', local_model='ridge'):
        self.k = k
        self.geometry = geometry       # 'pure' or 'augmented'
        self.local_model_type = local_model # 'ridge' or 'lasso'
        self.history_state = []        # Stores (Mean, Cov, LogCov) tuples
        self.history_obs = []
        self.history_y = []

    def fit(self, X_list, T_list, Y_list):
        for X, T, Y in zip(X_list, T_list, Y_list):
            obs = np.column_stack([X, T])

            # Compute State: (Mean, Covariance)
            mu = np.mean(obs, axis=0)
            cov = np.cov(obs, rowvar=False)
            
            # Pre-calculate logm for efficiency
            # Regularize
            c_reg = cov + 1e-6 * np.eye(cov.shape[0])
            log_cov = logm(c_reg)

            self.history_state.append((mu, cov, log_cov))
            self.history_obs.append(obs)
            self.history_y.append(Y)

    def predict_effect(self, X_new, T_new):
        # 1. Query State
        obs_new = np.column_stack([X_new, T_new])
        mu_new = np.mean(obs_new, axis=0)
        cov_new = np.cov(obs_new, rowvar=False)
        
        c_new_reg = cov_new + 1e-6 * np.eye(cov_new.shape[0])
        log_cov_new = logm(c_new_reg)
        
        # 2. Scan History
        dists = []
        for h in self.history_state:
            mu_h, cov_h, log_cov_h = h
            
            # Structural Distance
            dist_struct = norm(log_cov_new - log_cov_h)
            
            if self.geometry == 'pure':
                dists.append(dist_struct)
                continue
                
            # Level Distance
            dist_level = np.linalg.norm(mu_new - mu_h)
            dists.append(dist_struct + dist_level)

        # 3. Select Cohort
        cohort_idx = np.argsort(dists)[:self.k]

        # 4. Local Inference
        X_train = np.vstack([self.history_obs[i] for i in cohort_idx])
        Y_train = np.hstack([self.history_y[i] for i in cohort_idx])

        if self.local_model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif self.local_model_type == 'lasso':
            model = Lasso(alpha=0.01) # Mild sparsity
        else:
            model = LinearRegression()

        model.fit(X_train, Y_train)
        return model.coef_[-1]
