import numpy as np
from dynamics_model import *

def EKF(dym,action,mu,state,sigma,dt):
    '''
    1. if add v and w in the dynamics, it will crash.
    2. how do we deal with the C term? it's not introduced in the KF framework.
    3. when setting covariance too small, it will generate problems
    '''


    # v = np.sqrt(Q)@np.random.randn(n)
    # w = np.sqrt(R)@np.random.randn(m)

    # X = np.zeros((N,n))  #updated by linearized dynamics
    # Y = np.zeros((N,m))
    
    #get linearized A and B at each point
    A = dym.A(mu, action, dt)
    B = dym.B(mu, action, dt)
    H = dym.H()
    # Q = np.diag(np.random.rand(6)) * 0.001
    # R = np.diag(np.random.rand(2)) * 0.001
    # w = np.random.rand(2) * 0.001 #size 2 * 1
    Q = np.eye(6) * 0.01
    R = np.eye(2) * 0.01
    
    v = np.random.multivariate_normal(mean=np.zeros(6), cov=Q)
    # w = np.random.multivariate_normal(mean=np.zeros(2), cov=R)
    w = np.sqrt(R) @ np.random.normal(0,0.5,2)
                 
    
    Xhat_pred = A @ mu + B @ action #predicted state
    P_pred = A @ sigma @ A.T + Q #predicted covariance
    Y = H @ state + w  #measurement
    
    S = H @ sigma @ H.T + R
    S_inv = np.linalg.inv(S)   
    K = P_pred @ H.T @ S_inv # Kalman gain
    
    X_next = Xhat_pred + K @ (Y - H @ Xhat_pred)
    P_next = P_pred - K @ S @ K.T
    
    return X_next, P_next, Y



