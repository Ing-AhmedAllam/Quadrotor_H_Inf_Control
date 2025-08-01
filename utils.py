from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
import os

class Dataholder():
    def __init__(self, iterations, dt):
        self.iterations = iterations
        self.dt = dt

        # Initialize arrays to hold simulation data
        self.time_array = np.arange(0, iterations * self.dt, self.dt)
        self.bisection_x_norm_array = np.zeros(iterations)
        self.bisection_time_array = np.zeros(iterations)

        self.v_array = np.zeros((iterations, 3))
        self.r_array = np.zeros((iterations, 3, 3))
        self.euler_array = np.zeros((iterations, 3))
        self.position_array = np.zeros((iterations, 3))
        self.angular_velocity_array = np.zeros((iterations, 3))
        self.tau_array = np.zeros((iterations, 3))
        self.disturbance_array = np.zeros((iterations, 6))
        self.gamma_array = np.zeros(iterations)
        self.gamma_lb_array = np.zeros(iterations)
        self.xd = np.zeros((iterations, 3))  # Desired position
        self.vd = np.zeros((iterations, 3))  # Desired velocity
    
    def save(self, filename):
        """
        Save the data to a file.
        :param filename: Name of the file to save the data
        """
        cur_dir = os.getcwd()
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        file_path = os.path.join(cur_dir, filename)
        np.savez(file_path, time=self.time_array, bisection_x_norm=self.bisection_x_norm_array,
                 bisection_time=self.bisection_time_array, v=self.v_array, r=self.r_array,
                 euler=self.euler_array, position=self.position_array,
                 angular_velocity=self.angular_velocity_array, tau=self.tau_array,
                 disturbance=self.disturbance_array, gamma=self.gamma_array,
                 gamma_lb=self.gamma_lb_array, xd=self.xd, vd=self.vd)
    
    def plot(self):
        """
        Placeholder for plotting method.
        This method can be implemented to visualize the stored data.
        """
        # Attitude (Euler Angles)
        plt.figure('attitude (euler angles)')
        plt.subplot(3, 1, 1)
        plt.plot(self.time_array, self.euler_array[:, 0], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('roll [deg]')

        plt.subplot(3, 1, 2)
        plt.plot(self.time_array, self.euler_array[:, 1], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('pitch [deg]')

        plt.subplot(3, 1, 3)
        plt.plot(self.time_array, self.euler_array[:, 2], linewidth=1)
        plt.plot([0, self.iterations * self.dt], [0, 0], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('yaw [deg]')
        plt.legend(['yaw', 'setpoint'])

        # Angular Velocity
        plt.figure('Angular velocity')
        plt.subplot(3, 1, 1)
        plt.plot(self.time_array, self.angular_velocity_array[:, 0], linewidth=1)
        plt.plot([0, self.iterations * self.dt], [0, 0], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('x [deg/s]')
        plt.legend(['p', 'setpoint'])

        plt.subplot(3, 1, 2)
        plt.plot(self.time_array, self.angular_velocity_array[:, 1], linewidth=1)
        plt.plot([0, self.iterations * self.dt], [0, 0], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('y [deg/s]')
        plt.legend(['q', 'setpoint'])

        plt.subplot(3, 1, 3)
        plt.plot(self.time_array, self.angular_velocity_array[:, 2], linewidth=1)
        plt.plot([0, self.iterations * self.dt], [0, 0], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('z [deg/s]')
        plt.legend(['r', 'setpoint'])

        # Velocity (NED Frame)
        plt.figure('velocity (NED frame)')
        plt.subplot(3, 1, 1)
        plt.plot(self.time_array, self.v_array[:, 0], linewidth=1)
        plt.plot(self.time_array, self.vd[:, 0], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('v_x [m/s]')
        plt.legend(['x velocity', 'setpoint'])

        plt.subplot(3, 1, 2)
        plt.plot(self.time_array, self.v_array[:, 1], linewidth=1)
        plt.plot(self.time_array, self.vd[:, 1], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('v_y [m/s]')
        plt.legend(['y velocity', 'setpoint'])

        plt.subplot(3, 1, 3)
        plt.plot(self.time_array, -self.v_array[:, 2], linewidth=1)
        plt.plot(self.time_array, -self.vd[:, 2], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('v_z [m/s]')
        plt.legend(['z velocity', 'setpoint'])

        # Position (NED Frame)
        plt.figure('position (NED frame)')
        plt.subplot(3, 1, 1)
        plt.plot(self.time_array, self.position_array[:, 0], linewidth=1)
        plt.plot(self.time_array, self.xd[:, 0], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('x [m]')
        plt.legend(['x position', 'setpoint'])

        plt.subplot(3, 1, 2)
        plt.plot(self.time_array, self.position_array[:, 1], linewidth=1)
        plt.plot(self.time_array, self.xd[:, 1], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('y [m]')
        plt.legend(['y position', 'setpoint'])

        plt.subplot(3, 1, 3)
        plt.plot(self.time_array, -self.position_array[:, 2], linewidth=1)
        plt.plot(self.time_array, -self.xd[:, 2], linewidth=1)
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('z [m]')
        plt.legend(['z position', 'setpoint'])

        # Time cost of CARE solver
        plt.figure('time cost of the H-infinity control synthesizer')
        plt.title('time cost')
        plt.plot(self.time_array, self.bisection_time_array)
        plt.xlabel('time [s]')
        plt.ylabel('cost [s]')
        plt.legend(['bisection algorithm'])
        plt.grid(True)

        # Precision of CARE solver
        plt.figure('precision of the H-infinity control synthesizer')
        plt.title('precision (norm of CARE)')
        plt.plot(self.time_array, self.bisection_x_norm_array)
        plt.xlabel('time [s]')
        plt.ylabel('CARE residual')
        plt.legend(['SDA'])
        plt.grid(True)
        
        # Disturbance
        plt.figure('disturbances')
        plt.subplot(3, 2, 1)
        plt.plot(self.time_array, self.d_array[:, 0])
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel(r'$f_{w,x}$')

        plt.subplot(3, 2, 3)
        plt.plot(self.time_array, self.d_array[:, 1])
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel(r'$f_{w,y}$')

        plt.subplot(3, 2, 5)
        plt.plot(self.time_array, self.d_array[:, 2])
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel(r'$f_{w,z}$')

        plt.subplot(3, 2, 2)
        plt.plot(self.time_array, self.d_array[:, 3])
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel(r'$\tau_{w,x}$')

        plt.subplot(3, 2, 4)
        plt.plot(self.time_array, self.d_array[:, 4])
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel(r'$\tau_{w,y}$')

        plt.subplot(3, 2, 6)
        plt.plot(self.time_array, self.d_array[:, 5])
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel(r'$-\tau_{w,z}$')

        # Gamma evolution
        plt.figure('Optimal H-infinity control gamma')
        plt.title('optimal r')
        plt.plot(self.time_array, self.gamma_lb_array, label='r_lb')
        plt.plot(self.time_array, self.gamma_array, label='r_x')
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('r')
        plt.legend()

        plt.show()

        # Equivalent to: disp("Press any key to leave");
        input("Press Enter to leave...")

def cylinder_transform(px, py, pz, pos, R):
    ring1_old = np.stack([px[0, :], py[0, :], pz[0, :]], axis=1)
    ring2_old = np.stack([px[1, :], py[1, :], pz[1, :]], axis=1)
    
    ring1_rot = ring1_old @ R.T
    ring2_rot = ring2_old @ R.T
    
    ret_px = np.vstack([ring1_rot[:, 0], ring2_rot[:, 0]]) + pos[0]
    ret_py = np.vstack([ring1_rot[:, 1], ring2_rot[:, 1]]) + pos[1]
    ret_pz = np.vstack([ring1_rot[:, 2], ring2_rot[:, 2]]) - pos[2]  # NED z flip
    
    return ret_px, ret_py, ret_pz

def rigidbody_visualize(plot_size, rigidbody_pos, rigidbody_R, iterate_times, sleep_time):
    skip_cnt = max(1, round(0.05 / sleep_time))

    # Define quadrotor shapes
    theta = np.linspace(0, 2 * np.pi, 30)
    x1 = np.outer(np.ones(2), 0.2 * np.cos(theta))
    y1 = np.outer(np.ones(2), 0.2 * np.sin(theta))
    z1 = np.array([[0], [1]]) * np.ones_like(theta)

    x2 = np.outer(np.ones(2), 0.15 * np.cos(theta))
    y2 = np.outer(np.ones(2), 0.15 * np.sin(theta))
    z2 = np.outer([0, 1], np.ones_like(theta))

    # Initial geometry
    i11, i12, i13 = z1, y1, x1
    i21, i22, i23 = -z1, x1, y1
    i31, i32, i33 = y1, z1, x1
    i41, i42, i43 = y1, -z1, x1
    i51, i52, i53 = x2 + 1, y2, z2
    i61, i62, i63 = x2 - 1, y2, z2
    i71, i72, i73 = x2, y2 + 1, z2
    i81, i82, i83 = x2, y2 - 1, z2

    print("Start timing the elapsed time of rigidbody visualization:")
    start_time = time.time()

    fig = plt.figure('simulation visualization (NED)')
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, iterate_times, skip_cnt):
        ax.clear()
        sx, sy, sz = plot_size
        ax.set_xlim(-sx, sx)
        ax.set_ylim(-sy, sy)
        ax.set_zlim(-sz, sz)

        pos = rigidbody_pos[i, :]
        R = rigidbody_R[i, :, :]

        parts = [
            (i11, i12, i13, 'red'),
            (i21, i22, i23, 'blue'),
            (i31, i32, i33, 'yellow'),
            (i41, i42, i43, 'green'),
            (i51, i52, i53, 'red'),
            (i61, i62, i63, 'blue'),
            (i71, i72, i73, 'yellow'),
            (i81, i82, i83, 'green'),
        ]

        for px, py, pz, color in parts:
            tx, ty, tz = cylinder_transform(px, py, pz, pos, R)
            ax.plot_surface(tx, ty, tz, color=color, rstride=1, cstride=1, linewidth=0, antialiased=False)

        ax.view_init(elev=30, azim=135)
        ax.grid(True)
        plt.pause(sleep_time)

    elapsed = time.time() - start_time
    print(f"Visualization finished in {elapsed:.2f} seconds")

