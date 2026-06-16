import numpy as np

class KalmanFilter:
    def __init__(self, mu0: np.array, cov0: np.ndarray, u0: np.array, z0: np.array):
        self.mu = mu0
        self.cov = cov0
        self.u = u0
        self.z = z0

    def predict_1_step(self, A: np.ndarray, B: np.ndarray, R: np.ndarray, xt: np.array) -> tuple:
        w = np.random.multivariate_normal(np.zeros(len(self.mu)), R)
        xt = A @ xt + B @ self.u + w
        mu_pred = A @ self.mu + B @ self.u
        cov_pred = A @ self.cov @ A.T + R

        return (mu_pred, cov_pred, xt)

    def calculate_gain(self, C: np.ndarray, Q: np.ndarray):
        K = self.cov @ C.T @ np.linalg.inv(C @ self.cov @ C.T + Q)
        return K

    def correct_1_step(self, C: np.ndarray, Q: np.ndarray, z: np.array) -> tuple:
        K = self.calculate_gain(C, Q)
        mu_cor= self.mu + K @ (z - C @ self.mu)
        cov_cor = (np.eye(z.shape[0]) - K @ C) @ self.cov

        return (mu_cor, cov_cor)

    def update_system(self, At, Bt, Ct, fs, n, model: str) -> tuple:
        # system matrices: At, Bt, Ct can be dependence on time in dynamic systems
        if model == "uniform_accel":
            At[0][1] = n / fs
            Bt[0] = (n / fs) ** 2 / 2
            Bt[1] = n / fs
        
        return At, Bt, Ct

    def update_measurement(self, Ct, Q, xt):
        v = np.random.multivariate_normal(np.zeros(len(self.mu)), Q)
        zt = Ct @ xt + v

        return zt

    def simulate(self, A, B, C, Q, R, xt, time_sec, fs, model: str, is_dynamic: bool):
        n_steps = int(time_sec * fs)
        
        states = []
        covariances = []
        measurements = []
        true_states = []
        predictions = []
        pred_covs = []

        self.mu = self.mu.copy()
        self.cov = self.cov.copy()

        for i in range(n_steps):
            predictions.append(self.mu.copy())
            pred_covs.append(self.cov.copy())
            
            mu_i_pred, cov_i_pred, xt = self.predict_1_step(A, B, R, xt)

            self.mu = mu_i_pred
            self.cov = cov_i_pred

            zt = self.update_measurement(C, Q, xt)
            measurements.append(zt)
            mu_i_cor, cov_i_cor = self.correct_1_step(C, Q, zt)
           
            states.append(mu_i_cor.copy())
            covariances.append(cov_i_cor.copy())
            true_states.append(xt.copy())
            
            if is_dynamic:
                A, B, C = self.update_system(A, B, C, fs, i, model)

        return {"states": states, "covariances": covariances, 
                "measurements": measurements, "true_states": true_states,
                "predictions": predictions, "pred_covs": pred_covs}