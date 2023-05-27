from dynamics_model import * 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from ekf import *

if __name__ == '__main__':
    
    dynamics = BasePlanarQuadrotor()

    # initial state
    x0 = np.array([0,0,0,0,0,0])
    x0_ekf = np.array([0,0,0,0,0,0])
    sigma_0 = np.eye(6) * 0.01
    ideal_path = [x0]
    path_ekf = [x0_ekf]
    measurement = []
    true_path = []
    N = 20
    dt = 0.12

    for i in range(N):  
        
        # ideal path
        action = dynamics.control_generate()
        x_next = dynamics.discrete_step(x0,action,dt)
        ideal_path.append(x_next)
        x0 = x_next
        
        # ekf 
        ekf_next, sigma_next, Y, true_traj = EKF(dynamics, action, x0_ekf, x_next, sigma_0, dt)
        path_ekf.append(ekf_next)
        x0_ekf = ekf_next
        sigma_0 = sigma_next
        measurement.append(Y)
        true_path.append(true_traj)
         
    ideal_path = np.array(ideal_path)
    path_ekf = np.array(path_ekf)
    measurement = np.array(measurement)
    true_path = np.array(true_path)
    
    # plt.plot(path[:,0],path[:,2], color='yellow', label='ideal')
    # # plt.plot(path_ekf[:,0],path_ekf[:,2], color='m', label='ekf')
    # plt.scatter(measurement[:,0],measurement[:,1], color='pink', label='measurement')
    # plt.plot(true_path[:,0],true_path[:,1], color='c', label='true')
    # plt.legend()
    
    fig, ax = plt.subplots()
    
    ax.plot(ideal_path[:,0],ideal_path[:,2], color='yellow', label='ideal')
    ax.scatter(measurement[:,0],measurement[:,1], color='pink', label='measurement')
    ax.plot(true_path[:,0],true_path[:,1], color='c', label='true')
    ax.legend()
    
    line_ekf, = ax.plot([], [], 'x', color='m')  # Node moving along EKF path
    line_ekf_path, = ax.plot([], [], '-', color='m')  # Path of the moving node

    # Initialization function for the animation
    def init():
        ax.set_xlim(np.min(path_ekf[:,0]), np.max(path_ekf[:,0]))
        ax.set_ylim(np.min(path_ekf[:,2]), np.max(path_ekf[:,2]))
        return line_ekf, line_ekf_path,

    # Update function for the animation
    def update(frame):
        line_ekf.set_data(path_ekf[frame, 0], path_ekf[frame, 2])
        line_ekf_path.set_data(path_ekf[:frame+1, 0], path_ekf[:frame+1, 2])
        return line_ekf, line_ekf_path,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(N), init_func=init, blit=True)

    # Show the animation
    plt.show()
        
    
    
    