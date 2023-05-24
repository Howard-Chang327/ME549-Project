from dynamics_model import BasePlanarQuadrotor
import matplotlib.pyplot as plt
import numpy as np






if __name__ == '__main__':
    
    dynamics = BasePlanarQuadrotor()
    
    # initial control sequence
    initial_control = dynamics.control_sequence()
    
    x0 = np.array([0,0,5,0,0,0]) # initial state
    k_dt = 0.17069465800865732
    dt = 0.1
    
    # nonlinear dynamics
    state = np.zeros((len(initial_control)+1,6))
    state[0] = x0
    for i in range(len(initial_control)):
        state[i+1] = (dynamics.discrete_step(x0,initial_control[i],k_dt))
        x0 = state[i+1]
        
    print(state)
    print(len(state))
    plt.plot(state[:,0],state[:,2])
    plt.show()

    # linearized dynamics
    linear_state = np.zeros((len(initial_control)+1,6))
    linear_state[0] = x0
    for i in range(len(initial_control)):
        linear_state[i+1] = dynamics.A(state[i+1], initial_control[i]) @ linear_state[i] + dynamics.B(state[i+1], initial_control[i]) @ initial_control[i]
    
    print(linear_state)
    print(len(linear_state))
    plt.plot(linear_state[:,0],linear_state[:,2])
    plt.show()

    #########################################
#     print(dynamics.m)
#     step = dynamics.discrete_step(state, control, dt)
#     print(step)
    
#     planar_quad = BasePlanarQuadrotor()

#     N = 500
#     dt = 0.1

#     np.random.seed(1)
#     control_sequence = np.random.normal(20,0.1,(N,2))
#     x_initial = np.array([0,0,5,0,0,0])


# def plot_traj(N,dt,x0,control_sequence):
#     states = np.zeros((N+1,6))
#     states[0] = x0
#     for i in range(N):
#         next_state = planar_quad.discrete_step(states[i],control_sequence[i],dt)
#         # print(planar_quad.ode(states[i],control_sequence[i]))
#         states[i+1] = next_state
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     # ax.set_xlim([-100,100])
#     # ax.set_ylim([-100,100])

#     ax.set_title('trajectory')
#     ax.plot(states[:,0],states[:,2])

#     plt.show()

#     t_slider = IntSlider(min=1, max=N, step=1, value=1, description='t')

#     # Define the interaction function
#     def interact_plot_traj(time_steps):
#         plot_traj(time_steps, dt, x_initial, control_sequence)

#     # Create the interactive plot
#     interact(interact_plot_traj,time_steps = t_slider)
        
        
    
    
