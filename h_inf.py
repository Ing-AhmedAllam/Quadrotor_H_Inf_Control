import numpy as np
from time import time
from se3_math import se3_math

from h_inf_syn import *

class H_INF():
    def __init__(self, dynamics):
        self.dynamics = dynamics
        Ix = dynamics.inertia[0, 0]
        self.Ix = Ix
        Iy = dynamics.inertia[1, 1]
        self.Iy = Iy
        Iz = dynamics.inertia[2, 2]
        self.Iz = Iz
        m = dynamics.mass
        self.m = m
        
        # Math object for mathematical operations
        self.h_inf_math = se3_math()
        
        # Setpoints
        self.xd = np.zeros(3)  # Desired position
        self.vd = np.zeros(3)  # Desired velocity
        self.yaw_d = 0.0  # Desired yaw angle

        B1 = [
            [0,    0,   0,   0,   0,   0],
            [0,    0,   0,   0,   0,   0],
            [0,    0,   0,   0,   0,   0],
            [0,    0,   0, 1/Ix, 0,   0],
            [0,    0,   0,   0, 1/Iy, 0],
            [0,    0,   0,   0,   0, 1/Iz],
            [1/m,  0,   0,   0,   0,   0],
            [0,  1/m,   0,   0,   0,   0],
            [0,    0, 1/m,   0,   0,   0],
            [0,    0,   0,   0,   0,   0],
            [0,    0,   0,   0,   0,   0],
            [0,    0,   0,   0,   0,   0]
        ]

        self.B1 = np.array(B1)

        B2 = [
            [0,    0,   0,   0],
            [0,    0,   0,   0],
            [0,    0,   0,   0],
            [0,  1/Ix,  0,   0],
            [0,    0, 1/Iy,  0],
            [0,    0,   0, 1/Iz],
            [0,    0,   0,   0],
            [0,    0,   0,   0],
            [-1/m, 0,   0,   0],
            [0,    0,   0,   0],
            [0,    0,   0,   0],
            [0,    0,   0,   0]
        ]

        self.B2 = np.array(B2)
        
        c1 = np.zeros((14, 12))
        c1[0, 2] = 125;   #yaw
        c1[1, 3] = 10;    #roll rate
        c1[2, 4] = 10;    #pitch rate
        c1[3, 5] = 25;    #yaw rate
        c1[4, 6] = 50;    #vx
        c1[5, 7] = 50;    #vy
        c1[6, 8] = 100;   #vz
        c1[7, 9] = 200;   #x
        c1[8, 10] = 200;  #y
        c1[9, 11] = 160;  #z
        self.C1 = c1

        D12 = np.zeros((14, 4))
        D12[10, 0] = 1  # f
        D12[11, 1] = 1  # tau_x
        D12[12, 2] = 1  # tau_y
        D12[13, 3] = 1  # tau_z
        self.D12 = D12

        self.C2 = np.eye(12)
        self.D21 = 0.0
        
        self.B2t = self.B2.T
    
    def control(self, dyn, dataholder, i):
        """
        Control method to compute the control inputs based on the current dynamics.
        :param dyn: Current dynamics of the quadrotor
        :return: Control inputs (force and torque)
        """
        # Compute control inputs based on desired states and current dynamics
        euler_angles = self.h_inf_math.dcm2euler(dyn.R)
        v_b = dyn.R @ dyn.velocity  # Body velocity
        
        # State vector
        x = np.array([euler_angles[0], euler_angles[1], euler_angles[2],
                      dyn.angular_velocity[0], dyn.angular_velocity[1], dyn.angular_velocity[2],
                      v_b[0], v_b[1], v_b[2],
                      dyn.position[0], dyn.position[1], dyn.position[2]])

        # Desired state vector
        x0 = np.array([0, 0, 0, 0, 0, 0,
                       self.vd[0], self.vd[1], self.vd[2],
                       self.xd[0], self.xd[1], self.xd[2]])

        tic = time()
        # Compute control inputs using H-infinity synthesis
        gamma, gamma_lb, X, ric_residual = h_inf_syn(dyn.A, self.B1, self.B2, self.C1, 0)
        toc = time()
        bisec_time = toc - tic
        
        # Calculate the feedback control input
        C0_hat = -self.B2t @ X
        u_fb = C0_hat @ (x - x0)
        
        # Calculate the feedforward control input
        gravity_ff = np.array([[0], [0], [dyn.mass * dyn.gravity]]).T @ (dyn.R @ np.array([[0], [0], [1]]))
        gravity_ff = gravity_ff[0, 0]
        u_ff = np.array([gravity_ff, 0, 0, 0])
        
        #Complete Control Input
        u = (u_fb + u_ff)  # Reshape to (4, 1) for force and torques

        dyn.f = dyn.R @ np.array([0, 0, u[0]])  # Force in body frame
        dyn.tau = np.array([u[1], u[2], u[3]])  # Torque in body frame

        #store data for analysis
        dataholder.time_array[i] = i * dyn.dt
        dataholder.bisection_time_array[i] = bisec_time
        dataholder.bisection_x_norm_array[i] = ric_residual
        dataholder.gamma_array[i] = gamma
        dataholder.gamma_lb_array[i] = gamma_lb
        dataholder.v_array[i] = dyn.velocity
        dataholder.r_array[i] = dyn.R
        dataholder.euler_array[i] = np.rad2deg(euler_angles)
        dataholder.position_array[i] = dyn.position
        dataholder.angular_velocity_array[i] = dyn.angular_velocity
        dataholder.tau_array[i] = dyn.tau
        dataholder.disturbance_array[i] = dyn.d
        dataholder.xd[i] = self.xd
        dataholder.vd[i] = self.vd

