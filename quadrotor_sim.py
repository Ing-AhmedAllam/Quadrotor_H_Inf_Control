from tqdm import tqdm
import numpy as np

from se3_math import se3_math
from dynamics import dynamics
from h_inf import H_INF
from utils import Dataholder, rigidbody_visualize


math = se3_math()
print("Simulation Parameters:   ")
iterations = 20000 # Number of iterations for the simulation: 20 seconds at 1000Hz
dt = 0.001 # Time step in seconds
print(f"Iterations: {iterations}, dt: {dt} seconds")
print("Initializing Dynamics and H_INF Controller...")
inertia = np.diag([0.01466, 0.01466, 0.02848])

# Initialize dynamics with parameters
dyn = dynamics(dt= dt, gravity=9.81,
               mass=1.0, inertia=inertia,
               position=np.zeros(3), velocity=np.zeros(3), acceleration=np.zeros(3),
               orientation=np.array([0, 0, 0, 1]), angular_velocity=np.zeros(3), angular_acceleration=np.zeros(3),
               force=np.zeros(3), torque=np.zeros(3))

# Set initial attitude with roll, pitch, yaw
init_attitude = np.array([0.0, 0.0, 0.0])  # Roll, Pitch, Yaw in radians
dyn.R = math.euler2dcm(init_attitude[0], init_attitude[1], init_attitude[2])

h_inf_controller = H_INF(dyn)

dataholder = Dataholder(iterations, dt)

# Circular Trajectory Parameters
radius = 1.5        
circum_rate = 0.25
climb_rate = -0.05
yaw_rate = 0.05

for i in tqdm(range(iterations)):
    #Planning
    #Yaw
    if i == 0:
        h_inf_controller.yaw_d = 0.0  # Reset desired yaw angle at the start
    else:
        h_inf_controller.yaw_d = h_inf_controller.yaw_d + yaw_rate*dyn.dt *2*np.pi
        if h_inf_controller.yaw_d > np.pi:
            h_inf_controller.yaw_d -= 2 * np.pi

    # Pose
    pos_x = radius * np.cos(circum_rate * dyn.dt  * np.pi * i)
    pos_y = radius * np.sin(circum_rate * dyn.dt  * np.pi * i)
    pos_z = climb_rate * dyn.dt * i
    if pos_z <= -1.0:
        pos_z = -1.0

    h_inf_controller.xd = np.array([pos_x, pos_y, pos_z])
    # dyn.orientation = h_inf_controller.get_orientation()
    
    # Velocity
    h_inf_controller.vd = np.array([radius * -np.sin(circum_rate * dyn.dt * np.pi * i),
                                     radius * np.cos(circum_rate * dyn.dt * np.pi * i),
                                     climb_rate])

    dyn.position = h_inf_controller.xd
    dyn.velocity = h_inf_controller.vd
    
    # Control
    dyn.update()
    h_inf_controller.control(dyn, dataholder, i)
    
# Store data
dataholder.save(f"quadrotor_data.npz")
# Plot results
dataholder.plot()
# Visualize the quadrotor trajectory
rigidbody_visualize([5, 5, 5], dyn.position, dyn.R, iterations, dt)
    
