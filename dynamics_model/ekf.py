import numpy as np
from dynamics_model import *

class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        
        # YOUR IMPLEMENTATION HERE
        G = env.G(self.mu, u)
        V = env.V(self.mu, u)
        M = env.noise_from_motion(u, self.alphas)

        #prediction
        mu_next = env.forward(self.mu, u)
        Phi = env.observe(mu_next, marker_id)
        sigma_new = G @ self.sigma @ G.transpose() + V @ M @ V.transpose()

        #correction
        H = env.H(mu_next, marker_id)
        R = self.beta[0, 0]
        K = np.array([sigma_new @ H.transpose() / (H @ sigma_new @ H.transpose() + R)]).reshape((-1, 1))

        self.mu = mu_next + K * minimized_angle(z - Phi).reshape((-1, 1))
        self.sigma = (np.eye(3) - K @ H) @ sigma_new

        return self.mu, self.sigma
