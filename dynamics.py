from se3_math import se3_math
import numpy as np
from scipy.linalg import expm

dyn_math = se3_math()

class dynamics():
    def __init__(self,dt, gravity,
                 mass, inertia,
                 position, velocity,acceleration,
                 orientation, angular_velocity, angular_acceleration,
                 force, torque):
       assert type(dt) == float, "dt must be a float"
       assert type(gravity) == float, "gravity must be a float"
       assert type(mass) == float, "mass must be a float"
       assert type(inertia) == np.ndarray, "inertia must be a numpy array"
       assert inertia.shape == (3, 3), "inertia must be a 3x3 numpy array"
       assert type(position) == np.ndarray, "position must be a numpy array"
       assert position.shape == (3,), "position must be a 3-element numpy array"
       assert type(velocity) == np.ndarray, "velocity must be a numpy array"
       assert velocity.shape == (3,), "velocity must be a 3-element numpy array"
       assert type(acceleration) == np.ndarray, "acceleration must be a numpy array"
       assert acceleration.shape == (3,), "acceleration must be a 3-element numpy array"
       assert type(orientation) == np.ndarray, "orientation must be a numpy array"
       assert orientation.shape == (4,), "orientation must be a 4-element numpy array"
       assert type(angular_velocity) == np.ndarray, "angular_velocity must be a numpy array"
       assert angular_velocity.shape == (3,), "angular_velocity must be a 3-element numpy array"
       assert type(angular_acceleration) == np.ndarray, "angular_acceleration must be a numpy array"
       assert angular_acceleration.shape == (3,), "angular_acceleration must be a 3-element numpy array"
       assert type(force) == np.ndarray, "force must be a numpy array"
       assert force.shape == (3,), "force must be a 3-element numpy array"
       assert type(torque) == np.ndarray, "torque must be a numpy array"
       assert torque.shape == (3,), "torque must be a 3-element numpy array"

       self.dt = dt  # Time step in seconds

       self.gravity = gravity  # Gravity constant in m/s^2
       self.mass = mass      # Mass of the quadrotor in kg
       self.inertia = inertia  # Inertia matrix diagonal elements in kg*m^2
       print(f"Gravity: {self.gravity} m/s^2, Mass: {self.mass} kg")
       print(f"Inertia Matrix ({self.inertia.shape}): {self.inertia}")

       self.drag_coefficient = 0.01  # Drag coefficient for the quadrotor

       self.position = position  # Initial position
       self.velocity = velocity  # Initial velocity
       self.acceleration = acceleration  # Initial acceleration
       self.orientation = orientation  # Initial orientation as a quaternion [qx, qy, qz, qw]
       self.angular_velocity = angular_velocity  # Initial angular velocity
       self.angular_acceleration = angular_acceleration  # Initial angular acceleration

       self.R = dyn_math.quaternion2dcm(self.orientation)  # Initial Direction Cosine Matrix
       self.R_det = np.linalg.det(self.R)  # Determinant of the DCM, should be close to 1 for a valid rotation matrix

       self.d = np.array([0.0]*6)
       self.sigma_f_w = 0.5 # distribution of the force disturbance
       self.sigma_tau_w = 0.1 # distribution of the torque disturbance
       self.tau_c = 3.2 # correlation time for the wind disturbance
       
       self.prv_angle = 0.0

       self.f = force  # Force applied to the quadrotor
       self.tau = torque  # Torque applied to the quadrotor
       
       self.A = np.zeros((12, 12))  # State matrix for the dynamics

    def integrator(self, f_now, f_dot):
       """
       Integrate the force and acceleration over time.
       :param f_now: Current force vector
       :param f_dot: Change in force vector
       :param dt: Time step
       :return: Updated force vector
       """
       assert f_now.shape == f_dot.shape, "[W] value and rate of change should be of the same shape"
       f = f_now + f_dot * self.dt
       return f
    
    def update_A(self):
        # Update A matrix
        euler_angles = dyn_math.dcm2euler(self.R)
        v_b = self.R @ self.velocity  # Body velocity
        
        p = self.angular_velocity[0]
        q = self.angular_velocity[1]
        r = self.angular_velocity[2]
        
        u = v_b[0]
        v = v_b[1]
        w = v_b[2]

        s_phi = np.sin(euler_angles[0])
        c_phi = np.cos(euler_angles[0])
        s_theta = np.sin(euler_angles[1])
        c_theta = np.cos(euler_angles[1])
        s_psi = np.sin(euler_angles[2])
        c_psi = np.cos(euler_angles[2])
        t_theta = np.tan(euler_angles[1])
        sec_theta = 1.0 / np.cos(euler_angles[1])
        Ix = self.inertia[0, 0]
        Iy = self.inertia[1, 1]
        Iz = self.inertia[2, 2]
    
        a1 = [ -r*s_phi*t_theta + q*c_phi*t_theta,
               r*(c_phi*sec_theta**2 + q*s_phi*sec_theta**2),
               0, 1, s_phi*t_theta, c_phi*t_theta, 0, 0, 0, 0, 0, 0]
        a2 = [(-q*s_phi - r*c_phi), 0, 0, 0, c_phi, -s_phi, 0, 0, 0, 0, 0, 0]
        a3 = [ -r*s_phi/c_theta + q*c_phi/c_theta,
               r*c_phi*sec_theta*t_theta + q*s_phi*sec_theta*t_theta,
               0, 0, s_phi/c_theta, c_phi/c_theta, 0, 0, 0, 0, 0, 0]
        a4 = [0, 0, 0, 0, (Iy-Iz)/Ix*r, (Iy-Iz)/Ix*q, 0, 0, 0, 0, 0, 0]
        a5 = [0, 0, 0, (Iz-Ix)/Iy*r, 0, (Iz-Ix)/Iy*p, 0, 0, 0, 0, 0, 0]
        a6 = [0, 0, 0, (Ix-Iy)/Iz*q, (Ix-Iy)/Iz*p, 0, 0, 0, 0, 0, 0, 0]
        a7 = [0, -self.gravity*c_theta, 0, 0, -w, v, 0, r, -q, 0, 0, 0]
        a8 = [self.gravity*c_phi*c_theta, -self.gravity*s_phi*s_theta, 0, w, 0, -u, -r, 0, p, 0, 0, 0]
        a9 = [-self.gravity*c_theta*s_phi, -self.gravity*s_theta*c_phi, 0, -v, u, 0, q, -p, 0, 0, 0, 0]
        a10 = [w*(c_phi*s_psi - s_phi*c_psi*s_theta) + v*(s_phi*s_psi + c_psi*c_phi*s_theta),
               w*(c_phi*c_psi*c_theta) + v*(c_psi*s_phi*c_theta) - u*(c_psi*s_theta),
               w*(s_phi*c_psi - c_phi*s_psi*s_theta) - v*(c_phi*c_psi - c_phi*c_psi*s_theta) + u*(c_theta*c_psi),
               0, 0, 0, c_psi*c_theta, (-c_phi*s_psi + c_psi*s_phi*s_theta), (s_phi*s_psi + c_phi*c_psi*s_theta), 0, 0, 0]
        a11 = [v*(-s_phi*c_psi + c_phi*s_psi*s_theta) - w*(c_psi*c_phi + s_phi*s_psi*s_theta),
               v*(s_phi*s_psi*c_theta) + w*(c_phi*s_psi*c_theta) - u*(s_theta*s_psi),
               v*(-c_phi*s_psi + s_phi*c_psi*s_theta) + w*(s_psi*s_phi + c_phi*c_psi*s_theta) + u*(c_theta*c_psi),
               0, 0, 0, c_theta*s_psi, (c_phi*c_psi + s_phi*s_psi*s_theta), (-c_psi*s_phi + c_phi*s_psi*s_theta), 0, 0, 0]
        a12 = [ -w*s_phi*c_theta + v*c_theta*c_phi,
                -w*c_phi*s_theta - u*c_theta - v*s_theta*s_phi,
                0, 0, 0, 0, -s_theta, c_theta*s_phi, c_phi*c_theta, 0, 0, 0]
        
        A = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12])
        self.A = A
        
    def update(self):
       inv_corr_time = -1.0 / self.tau_c
       A_d = np.eye(6) * inv_corr_time  # Diagonal matrix for disturbance correlation
       noise = np.random.normal(0, 1, (6))  # Generate noise for disturbances
       noise[:3] = self.sigma_f_w * noise[:3]  # Scale force disturbances
       noise[3:] = self.sigma_tau_w * noise[3:] # Scale torque disturbances

       self.d = noise
       self.angular_velocity = self.integrator(self.angular_velocity, self.angular_acceleration)
       self.velocity = self.integrator(self.velocity, self.acceleration)

       #calculate rotation matrix by intergrating DCM differential equation
       angular_dt = self.angular_velocity * self.dt
       I = np.eye(3)
       #  print("I shape: ", I.shape)
       # dR = expm(dyn_math.hat_map_3d(angular_dt))
       dR = dyn_math.hat_map_3d(angular_dt) + I

       self.R = self.R @ dR # Rotate the DCM
       self.R = dyn_math.orthonormalize_dcm(self.R) # Ensure orthonormality
       assert np.isclose(np.linalg.det(self.R), 1.0), "[W] DCM must have a determinant close to 1."

       self.prv_angle = dyn_math.get_prv_angle(self.R)  # Get the principal rotation vector angle

       # Calculate position and orientation updates
       self.position = self.integrator(self.position, self.velocity)
       e3 = np.array([0.0, 0.0, 1.0])  # Unit vector in the z-direction
       mv_dot = (self.mass * self.gravity * e3) - self.f + self.d[:3]  # Total force including disturbances
       self.acceleration = mv_dot / self.mass

       jw = self.inertia @ self.angular_velocity  # Angular momentum
       # Compute the cross product of angular velocity (3,) with each column of jw (3,)
       assert jw.shape == (3,), f"[W] jw must be a 3-element numpy array instead of {jw.shape} "
       wjw = np.cross(self.angular_velocity, jw)
       assert wjw.shape == (3,), f"[W] wjw must be a 3-element numpy array instead of {wjw.shape} "
       wjw = wjw

       self.angular_acceleration = np.linalg.inv(self.inertia) @ ((self.tau - wjw - self.d[3:]))  # Update angular acceleration

       self.update_A()
