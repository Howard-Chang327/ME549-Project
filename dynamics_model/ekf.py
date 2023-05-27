import numpy as np
from dynamics_model import *

def EKF(dym,action,mu,state,sigma,dt):
    
    #get linearized A and B at each point
    A = dym.A(mu, action, dt)
    B = dym.B(mu, action, dt)
    H = dym.H()
    
    m, n = H.shape
    
    Q = np.eye(n) * 0.01
    R = np.eye(m) * 0.01
    
    v = np.sqrt(Q) @ np.random.normal(0,1,n) # gaussian noise
    w = np.sqrt(R) @ np.random.normal(0,1,m) # gaussian noise
    
    true_traj = state + v # true trajectory = state + noise             
    
    Xhat_pred = A @ mu + B @ action # predicted state
    P_pred = A @ sigma @ A.T + Q # predicted covariance
    Y = H @ true_traj + w  # measurement = true trajectory + noise 
    
    S = H @ sigma @ H.T + R
    S_inv = np.linalg.inv(S)   
    K = P_pred @ H.T @ S_inv # Kalman gain
    
    X_next = Xhat_pred + K @ (Y - H @ Xhat_pred)
    P_next = P_pred - K @ S @ K.T
    
    return X_next, P_next, Y, H @ true_traj



