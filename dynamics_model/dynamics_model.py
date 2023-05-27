import numpy as np

class BasePlanarQuadrotor:
    """
    Dynamics model for planar quadcoptor.
    """

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
        """Jacobian matrix for the continuous-time control input."""
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
        """Jacobian matrix for the discrete-time control input."""
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
            [1, 0, 0, 0, 0, 0], # x
            [0, 0, 1, 0, 0, 0], # y          
            [0 ,0, 0 ,0, 1, 0]  # phi
        ])
    
    def control_generate(self):
        """Generate a random control input T_1, T_2."""
        return np.random.normal(20,1,2)

    
