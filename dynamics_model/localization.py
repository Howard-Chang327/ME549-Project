from dynamics_model import BasePlanarQuadrotor
import matplotlib.pyplot as plt
import numpy as np
from ekf import *

if __name__ == '__main__':
    
    dynamics = BasePlanarQuadrotor()
    
    # x0 = np.array([0,0,5,0,0,0])
    # sigma_0 = np.eye(6) * 0.01
    # path_ideal = [x0]
    # path_ekf = [x0]
    # measurement = []
    # N = 20
    # dt = 0.17069465800865732
    
    # for i in range(19):
    #     # action = dynamics.control_generate()
    #     action = dynamics.control_sequence()[i]
    #     print("action: ", action)
    #     # ideal path
    #     init_state_ideal = x0
    #     ideal_next = (dynamics.discrete_step(init_state_ideal,action,dt))
    #     path_ideal.append(ideal_next)
    #     init_state_ideal = ideal_next
        
    #     # ekf path
    #     init_state_ekf = x0
    #     ekf_next, sigma_next, Y = EKF(dynamics, action, init_state_ekf, sigma_0, dt)
    #     init_state_ekf = ekf_next
    #     sigma_0 = sigma_next
    #     path_ekf.append(ekf_next)
    #     measurement.append(Y)

    # path_ideal = np.array(path_ideal)
    # path_ekf = np.array(path_ekf)
    # measurement = np.array(measurement)
    
    # plt.plot(path_ideal[:,0],path_ideal[:,2], color='red', label='ekf')
    # plt.plot(path_ekf[:,0],path_ekf[:,2], color='blue', label='jacob')
    # plt.scatter(measurement[:,0],measurement[:,1], color='green', label='measurement')
    # plt.legend()
    # plt.show()    
    
    
    #########################
    
    x0 = np.array([0,0,5,0,0,0])
    x0_ekf = np.array([0,0,5,0,0,0])
    sigma_0 = np.eye(6) * 0.01
    path = [x0]
    path_ekf = [x0_ekf]
    measurement = []
    N = 20
    dt = 0.1

    for i in range(N):
        action = dynamics.control_generate()
        x_next = dynamics.discrete_step(path[i],action,dt)
        path.append(x_next)
        x0 = x_next
        print("x next: ", x_next)
        
        # ekf 
        ekf_next, sigma_next, Y = EKF(dynamics, action, x0_ekf, x_next, sigma_0, dt)
        path_ekf.append(ekf_next)
        x0_ekf = ekf_next
        sigma_0 = sigma_next
        measurement.append(Y)
         
    path = np.array(path)
    path_ekf = np.array(path_ekf)
    measurement = np.array(measurement)
    
    plt.plot(path[:,0],path[:,2], color='red', label='ideal')
    plt.plot(path_ekf[:,0],path_ekf[:,2], color='blue', label='ekf')
    plt.scatter(measurement[:,0],measurement[:,1], color='green', label='measurement')
    plt.legend()
    plt.show()
    
    
    
    
    
        
    
    
    