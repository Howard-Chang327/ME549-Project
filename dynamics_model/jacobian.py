import sympy as sp
from sympy import *

# Define the variables
x, v_x, y, v_y, phi, omega, T_1, T_2, m, Cd_v, Cd_phi, Iyy, g, l = sp.symbols('x v_x y v_y phi omega T_1 T_2 m Cd_v Cd_phi Iyy g l')

# Define the functions
f1 = v_x
f2 = (-(T_1 + T_2) * sp.sin(phi) - Cd_v * v_x) / m
f3 = v_y
f4 = ((T_1 + T_2) * sp.cos(phi) - Cd_v * v_y) / m - g
f5 = omega
f6 = ((T_2 - T_1) * l - Cd_phi * omega) / Iyy

# Create a matrix for the functions
F = sp.Matrix([f1, f2, f3, f4, f5, f6])

# Create a matrix for the variables
V_dynamics = sp.Matrix([x, v_x, y, v_y, phi, omega])
V_control = sp.Matrix([T_1, T_2])

# Turn the continuous-time dynamics into discrete-time dynamics

# Define the time step
dt = sp.symbols('dt')

# Define the next state variables
x_next, v_x_next, y_next, v_y_next, phi_next, omega_next = sp.symbols('x_next v_x_next y_next v_y_next phi_next omega_next')

# Create a matrix for the next state variables
V_next = sp.Matrix([x_next, v_x_next, y_next, v_y_next, phi_next, omega_next])

# Compute the next state using the Euler method
V_next = V_dynamics + dt * F

# Print the next state
# print(V_next)

f1d = dt*v_x + x
f2d = dt*(-Cd_v*v_x + (-T_1 - T_2)*sp.sin(phi))/m + v_x
f3d = dt*v_y + y
f4d = dt*(-g + (-Cd_v*v_y + (T_1 + T_2)*sp.cos(phi))/m) + v_y
f5d = dt*omega + phi
f6d = dt*(-Cd_phi*omega + (T_2 - T_1)*l)/Iyy + omega

F_d = sp.Matrix([f1d, f2d, f3d, f4d, f5d, f6d])
# Compute the Jacobian
J_d = F_d.jacobian(V_dynamics)
J_c = F_d.jacobian(V_control)
# # Print the Jacobian
print(J_c)

