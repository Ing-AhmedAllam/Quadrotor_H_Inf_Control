import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import rigidbody_visualize



cur_dir = os.getcwd()
file_path = os.path.join(cur_dir, "quadrotor_data.npz")
data = np.load(file_path)
# Extract data from the loaded file
time_arr = data['time']
bisection_x_norm_arr = data['bisection_x_norm']
# bisection_x_norm_arr = bisection_x_norm_arr[:,0,0]
bisection_time_arr = data['bisection_time']
v_arr = data['v']
r_arr = data['r']
euler_arr = data['euler']
pos_arr = data['position']
angular_velocity_arr = data['angular_velocity']
tau_arr = data['tau']
d_arr = data['disturbance']
gamma_arr = data['gamma']
gamma_lb_arr = data['gamma_lb']
np.savez(file_path, time=time_arr, bisection_x_norm=bisection_x_norm_arr,
         bisection_time=bisection_time_arr, v=v_arr, r=r_arr, euler=euler_arr,
         position=pos_arr, angular_velocity=angular_velocity_arr, tau=tau_arr,
         disturbance=d_arr, gamma=gamma_arr, gamma_lb=gamma_lb_arr)

ITERATION_TIMES = 20000 # Number of iterations for the simulation: 20 seconds at 1000Hz
dt = 0.001

xd = np.zeros((ITERATION_TIMES, 3))
vd = np.zeros((ITERATION_TIMES, 3))


# Circular Trajectory Parameters
radius = 1.5        
circum_rate = 0.25
climb_rate = -0.05
yaw_rate = 0.05
for i in tqdm(range(ITERATION_TIMES)):
    #Planning
    # Pose
    pos_x = radius * np.cos(circum_rate * dt * np.pi * i)
    pos_y = radius * np.sin(circum_rate * dt * np.pi * i)
    pos_z = climb_rate * dt * i
    if pos_z <= -1.0:
        pos_z = -1.0

    xd[i] = np.array([pos_x, pos_y, pos_z])
    # dyn.orientation = h_inf_controller.get_orientation()
    
    # Velocity
    vd[i] = np.array([radius * -np.sin(circum_rate * dt * np.pi * i),
                      radius * np.cos(circum_rate * dt * np.pi * i),
                                     climb_rate])

# Attitude (Euler Angles)
plt.figure('attitude (euler angles)')
plt.subplot(3, 1, 1)
plt.plot(time_arr, euler_arr[:, 0], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('roll [deg]')

plt.subplot(3, 1, 2)
plt.plot(time_arr, euler_arr[:, 1], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('pitch [deg]')

plt.subplot(3, 1, 3)
plt.plot(time_arr, euler_arr[:, 2], linewidth=1)
plt.plot([0, ITERATION_TIMES * dt], [0, 0], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('yaw [deg]')
plt.legend(['yaw', 'setpoint'])

# Angular Velocity
plt.figure('Angular velocity')
plt.subplot(3, 1, 1)
plt.plot(time_arr, angular_velocity_arr[:, 0], linewidth=1)
plt.plot([0, ITERATION_TIMES * dt], [0, 0], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('x [deg/s]')
plt.legend(['p', 'setpoint'])

plt.subplot(3, 1, 2)
plt.plot(time_arr, angular_velocity_arr[:, 1], linewidth=1)
plt.plot([0, ITERATION_TIMES * dt], [0, 0], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('y [deg/s]')
plt.legend(['q', 'setpoint'])

plt.subplot(3, 1, 3)
plt.plot(time_arr, angular_velocity_arr[:, 2], linewidth=1)
plt.plot([0, ITERATION_TIMES * dt], [0, 0], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('z [deg/s]')
plt.legend(['r', 'setpoint'])

# Velocity (NED Frame)
plt.figure('velocity (NED frame)')
plt.subplot(3, 1, 1)
plt.plot(time_arr, v_arr[:, 0], linewidth=1)
plt.plot(time_arr, vd[:, 0], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('v_x [m/s]')
plt.legend(['x velocity', 'setpoint'])

plt.subplot(3, 1, 2)
plt.plot(time_arr, v_arr[:, 1], linewidth=1)
plt.plot(time_arr, vd[:, 1], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('v_y [m/s]')
plt.legend(['y velocity', 'setpoint'])

plt.subplot(3, 1, 3)
plt.plot(time_arr, -v_arr[:, 2], linewidth=1)
plt.plot(time_arr, -vd[:, 2], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('v_z [m/s]')
plt.legend(['z velocity', 'setpoint'])

# Position (NED Frame)
plt.figure('position (NED frame)')
plt.subplot(3, 1, 1)
plt.plot(time_arr, pos_arr[:, 0], linewidth=1)
plt.plot(time_arr, xd[:, 0], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('x [m]')
plt.legend(['x position', 'setpoint'])

plt.subplot(3, 1, 2)
plt.plot(time_arr, pos_arr[:, 1], linewidth=1)
plt.plot(time_arr, xd[:, 1], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('y [m]')
plt.legend(['y position', 'setpoint'])

plt.subplot(3, 1, 3)
plt.plot(time_arr, -pos_arr[:, 2], linewidth=1)
plt.plot(time_arr, -xd[:, 2], linewidth=1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('z [m]')
plt.legend(['z position', 'setpoint'])

# Time cost of CARE solver
plt.figure('time cost of the H-infinity control synthesizer')
plt.title('time cost')
plt.plot(time_arr, bisection_time_arr)
plt.xlabel('time [s]')
plt.ylabel('cost [s]')
plt.legend(['bisection algorithm'])
plt.grid(True)

# Precision of CARE solver
plt.figure('precision of the H-infinity control synthesizer')
plt.title('precision (norm of CARE)')
plt.plot(time_arr, bisection_x_norm_arr)
plt.xlabel('time [s]')
plt.ylabel('CARE residual')
plt.legend(['SDA'])
plt.grid(True)

# Disturbance
plt.figure('disturbances')
plt.subplot(3, 2, 1)
plt.plot(time_arr, d_arr[:, 0])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel(r'$f_{w,x}$')

plt.subplot(3, 2, 3)
plt.plot(time_arr, d_arr[:, 1])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel(r'$f_{w,y}$')

plt.subplot(3, 2, 5)
plt.plot(time_arr, d_arr[:, 2])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel(r'$f_{w,z}$')

plt.subplot(3, 2, 2)
plt.plot(time_arr, d_arr[:, 3])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel(r'$\tau_{w,x}$')

plt.subplot(3, 2, 4)
plt.plot(time_arr, d_arr[:, 4])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel(r'$\tau_{w,y}$')

plt.subplot(3, 2, 6)
plt.plot(time_arr, d_arr[:, 5])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel(r'$-\tau_{w,z}$')

# Gamma evolution
plt.figure('Optimal H-infinity control gamma')
plt.title('optimal r')
plt.plot(time_arr, gamma_lb_arr, label='r_lb')
plt.plot(time_arr, gamma_arr, label='r_x')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('r')
plt.legend()

plt.show()

# Equivalent to: disp("Press any key to leave");
input("Press Enter to leave...")

rigidbody_visualize([5, 5, 5], pos_arr, r_arr, ITERATION_TIMES, dt)