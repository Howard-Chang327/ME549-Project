"""
Dynamics model for planar quadcoptor
"""

import numpy as np
# import jax
# import jax.numpy as jnp

class BasePlanarQuadrotor:

    def __init__(self):
        # Dynamics constants
        # yapf: disable
        self.x_dim = 6         # state dimension (see dynamics below)
        self.u_dim = 2         # control dimension (see dynamics below)
        self.g = 9.807         # gravity (m / s**2)
        self.m = 2.5           # mass (kg)
        self.l = 1.0           # half-length (m)
        self.Iyy = 1.0         # moment of inertia about the out-of-plane axis (kg * m**2)
        self.Cd_v = 0.25       # translational drag coefficient
        self.Cd_phi = 0.02255  # rotational drag coefficient
        # yapf: enable

        # Control constraints
        self.max_thrust_per_prop = 0.75 * self.m * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0  # at least until variable-pitch quadrotors become mainstream :D

    def ode(self, state, control):
        """Continuous-time dynamics of a planar quadrotor expressed as an ODE."""
        x, v_x, y, v_y, phi, omega = state
        T_1, T_2 = control
        return np.array([
            v_x,
            (-(T_1 + T_2) * np.sin(phi) - self.Cd_v * v_x) / self.m,
            v_y,
            ((T_1 + T_2) * np.cos(phi) - self.Cd_v * v_y) / self.m - self.g,
            omega,
            ((T_2 - T_1) * self.l - self.Cd_phi * omega) / self.Iyy,
        ])

    def discrete_step(self, state, control, dt):
        """Discrete-time dynamics (Euler-integrated) of a planar quadrotor."""
        
        return state + dt * self.ode(state, control)
    
    def A_con(self, state, control):
        """Jacobian matrix for the continuous-time dynamics."""
        x, v_x, y, v_y, phi, omega = state
        T_1, T_2 = control
        return np.array([
            [0, 1, 0, 0, 0, 0],
            [0, -self.Cd_v / self.m, 0, 0, -(T_1 + T_2) * np.cos(phi) / self.m, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, -self.Cd_v / self.m, -(T_1 + T_2) * np.sin(phi) / self.m, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, -self.Cd_phi / self.Iyy]
        ])

    def B_con(self, state, control):
        """Jacobian matrix for the continuous-time control."""
        x, v_x, y, v_y, phi, omega = state
        T_1, T_2 = control
        return np.array([
            [0, 0],
            [-np.sin(phi) / self.m, -np.sin(phi) / self.m],
            [0, 0],
            [np.cos(phi) / self.m, np.cos(phi) / self.m],
            [0, 0],
            [-self.l / self.Iyy, self.l / self.Iyy]
        ])
        
    def A(self, state, control, dt):
        """Jacobian matrix for the discrete-time dynamics."""
        x, v_x, y, v_y, phi, omega = state
        T_1, T_2 = control
        return np.array([
            [1, dt, 0, 0, 0, 0], 
            [0, -self.Cd_v*dt/self.m + 1, 0, 0, dt*(-T_1 - T_2)*np.cos(phi)/self.m, 0], 
            [0, 0, 1, dt, 0, 0], 
            [0, 0, 0, -self.Cd_v*dt/self.m + 1, -dt*(T_1 + T_2)*np.sin(phi)/self.m, 0], 
            [0, 0, 0, 0, 1, dt], 
            [0, 0, 0, 0, 0, -self.Cd_phi*dt/self.Iyy + 1]
        ]) # type: ignore
        
    def B(self, state, control, dt):
        """Jacobian matrix for the discrete-time control."""
        x, v_x, y, v_y, phi, omega = state
        T_1, T_2 = control
        return np.array([
            [0, 0], 
            [-dt*np.sin(phi)/self.m, -dt*np.sin(phi)/self.m], 
            [0, 0], 
            [dt*np.cos(phi)/self.m, dt*np.cos(phi)/self.m], 
            [0, 0], 
            [-dt*self.l/self.Iyy, dt*self.l/self.Iyy]
        ])
        
    def H(self):
        """Jacobian matrix for the measurement model."""
        return np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
    def control_sequence(self):
        initial_path = np.array([
            [12.94349588, 12.97691164], 
            [12.92116354, 12.94490242],
            [12.89679699, 12.91398908],
            [12.87103051, 12.88351406],
            [12.84465181, 12.85266965],
            [12.81850192, 12.82059408],
            [12.79339224, 12.78645827],
            [12.77001642, 12.74955271],
            [12.74885155, 12.70938729],
            [12.73004465, 12.66580248],
            [12.71329318, 12.61908956],
            [12.69773434, 12.57009592],
            [12.68186434, 12.52029691],
            [12.66352302, 12.47180147],
            [12.63997233, 12.427266  ],
            [12.60810796, 12.38968941],
            [12.56481901, 12.36207806],
            [12.50749652, 12.34698935],
            [12.43464627, 12.34598717],
            [12.34652004, 12.35906655],
            [12.24561336, 12.38417606],
            [12.13688087, 12.41688802],
            [12.02756034, 12.4503671 ],
            [11.92652716, 12.47570427],
            [11.84325934, 12.4827875 ],
            [11.78650149, 12.46172228],
            [11.76270047, 12.40455777],
            [11.77460418, 12.30737214],
            [11.81998633, 12.17222878],
            [11.8907987 , 12.00857708],
            [11.97267514, 11.83319242],
            [12.08548669, 11.71521207],
            [12.22180289, 11.67918489],
            [12.32154746, 11.68221583],
            [12.38189322, 11.72436274],
            [12.40662854, 11.80053552],
            [12.40438442, 11.90243045],
            [12.3863155 , 12.02027938],
            [12.3638027 , 12.14432001],
            [12.34661088, 12.26597365],
            [12.34176122, 12.37869427],
            [12.35316205, 12.47843004],
            [12.38187299, 12.56365004],
            [12.42678805, 12.63496466],
            [12.48550532, 12.69444487],
            [12.55519262, 12.74479798],
            [12.63332968, 12.78857542],
            [12.71824494, 12.82754857],
            [12.80940457, 12.86235206],
            [12.9073966 , 12.8924723 ]])
    
        return initial_path
    
