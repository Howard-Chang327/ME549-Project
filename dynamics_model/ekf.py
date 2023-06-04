import numpy as np
from dynamics_model import *

def EKF(dym,action,mu,state,sigma,dt):
    """EKF for every time step.

    Args:
        dym: dynamics model
        action: control input
        mu: x hat (state estimate)
        state: ideal state
        sigma: covariance matrix of x hat
        dt: time step

    Returns:
        X_next: next state estimate
        P_next: covariance matrix of next state estimate
        Y: measurement
        H @ true_traj: true trajectory
    """
    
    # get linearized A and B at each point
    A = dym.A(mu, action, dt)
    B = dym.B(mu, action, dt)
    H = dym.H()
    
    m, n = H.shape
    
    # covariance matrix of noise
    Q = np.eye(n) * 0.01
    R = np.eye(m) * 0.01
    
    # gaussian noise
    v = np.sqrt(Q) @ np.random.normal(0,1,n) 
    w = np.sqrt(R) @ np.random.normal(0,0.5,m)
    
    # true trajectory = ideal state + noise 
    true_traj = state + v   
    
    # measurement = true trajectory + noise           
    Y = H @ true_traj + w  
     
    # time update
    Xhat_pred = A @ mu + B @ action # predicted state
    P_pred = A @ sigma @ A.T + Q # predicted covariance
    
   
    
    S = H @ sigma @ H.T + R
    S_inv = np.linalg.inv(S)   
    K = P_pred @ H.T @ S_inv # Kalman gain
    
    # information update
    X_next = Xhat_pred + K @ (Y - H @ Xhat_pred)
    P_next = P_pred - K @ S @ K.T
    
    return X_next, P_next, Y, H @ true_traj



