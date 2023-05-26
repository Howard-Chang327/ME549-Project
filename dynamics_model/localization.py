from dynamics_model import BasePlanarQuadrotor
import matplotlib.pyplot as plt
import numpy as np
from ekf import *

if __name__ == '__main__':
    
    dynamics = BasePlanarQuadrotor()
    
    # initial control sequence
    initial_control = dynamics.control_sequence()
    x0 = np.array([0,0,5,0,0,0]) # initial state
    
    N = len(initial_control) # number of time steps
    
    k_dt = 0.17069465800865732
    
    # nonlinear dynamics
    state = np.zeros((N+1,6))
    state[0] = x0
    for i in range(len(initial_control)):
        state[i+1] = (dynamics.discrete_step(x0,initial_control[i],k_dt))
        x0 = state[i+1]
        
    # print(state[0])
    # print(len(state))
    # plt.plot(state[:,0],state[:,2])

    #########################################
    # ekf
    
    x0 = np.array([0,0,5,0,0,0])
    sigma_0 = np.eye(6) * 0.01
    path = [x0]
    measurement = []
    
    for i in range(19):
        x_next, sigma_next, Y = EKF(dynamics, initial_control[i], x0, sigma_0, k_dt)
        x0 = x_next
        sigma_0 = sigma_next
        path.append(x_next)
        measurement.append(Y)
        # print("for debug simga next:", i, sigma_next)
        # print("for debug x_next:", i, x_next[0], x_next[2] ,Y)
    
    x0 = np.array([0,0,5,0,0,0])
    path_jacob = [x0]
    
    for i in range(19):
        # x_dot = dynamics.A_con(x0, initial_control[i]) @ x0 + dynamics.B_con(x0, initial_control[i]) @ initial_control[i]
        # x_new_jacob = x0 + k_dt * x_dot
        x_new_jacob = dynamics.A(x0, initial_control[i], k_dt) @ x0 + dynamics.B(x0, initial_control[i], k_dt) @ initial_control[i]
        path_jacob.append(x_new_jacob)
        x0 = x_new_jacob
        # print("for debug x_new_jacob:", i, x_new_jacob[0], x_new_jacob[2])
        
    measurement = np.array(measurement)
    path_jacob = np.array(path_jacob)    
    path = np.array(path)
    plt.scatter(measurement[:,0],measurement[:,1], color='green', label='measurement')
    plt.plot(path[:,0],path[:,2], color='red', label='ekf')
    plt.plot(path_jacob[:,0],path_jacob[:,2], color='blue', label='jacob')
    plt.legend()
    plt.show()    
    

    
    
    
        
    
    
    