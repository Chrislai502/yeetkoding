import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle  # Import Circle for drawing car boundaries

# Define the initial positions, velocities, and radii
P1 = np.array([0.0, 0.0])    # Initial position of Car 1
V1 = np.array([1.0, 0.5])    # Velocity of Car 1
r1 = 1.5                     # Radius of Car 1

P2 = np.array([5.0, 2.0])    # Initial position of Car 2
V2 = np.array([-0.5, -1.0])  # Velocity of Car 2
r2 = 1.5                   # Radius of Car 2

# Calculate the initial relative position and velocity
D0 = P1 - P2
V = V1 - V2

# Calculate coefficients of the quadratic equation
a = np.dot(V, V)
b = 2 * np.dot(D0, V)
c = np.dot(D0, D0) - (r1 + r2)**2

# Function to calculate the collision time
def calculate_collision_time(a, b, c):

    if np.isclose(a, 0):
        if np.isclose(b, 0):
            if c <= 0:
                return 0.0  # Cars are already touching
            else:
                return float('-inf')  # No collision
        else:
            t = -c / b
            return t if t >= 0 else float('-inf')
    else:
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return float('-inf')  # No real roots, no collision
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        # Choose the smallest non-negative time
        times = [t for t in [t1, t2] if t >= 0]
        return min(times) if times else float('-inf')

# Calculate the collision time
t_collision = calculate_collision_time(a, b, c)
if t_collision == float('-inf'):
    print("The cars will not collide.")
else:
    print(f"The cars will collide at t = {t_collision:.2f} units of time.")

# Function to get position at time t
def position_at_time(P, V, t):
    return P + V * t

# Set up the visualization
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 4)

# Plot initial positions
car1_plot, = ax.plot([], [], 'bo', markersize=10, label='Car 1')
car2_plot, = ax.plot([], [], 'ro', markersize=10, label='Car 2')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
collision_point_plot, = ax.plot([], [], 'gx', markersize=12, label='Collision Point')

# Number lines for time
time_line, = ax.plot([], [], 'k-', linewidth=1)

# Add circles around the cars
car1_circle = Circle((0, 0), r1, color='blue', fill=False)  # Car 1 circle
car2_circle = Circle((0, 0), r2, color='red', fill=False)   # Car 2 circle
ax.add_patch(car1_circle)
ax.add_patch(car2_circle)

# Initialize the animation
def init():
    car1_plot.set_data([], [])
    car2_plot.set_data([], [])
    collision_point_plot.set_data([], [])
    time_text.set_text('')
    time_line.set_data([], [])
    car1_circle.center = (P1[0], P1[1])
    car2_circle.center = (P2[0], P2[1])
    return car1_plot, car2_plot, collision_point_plot, time_text, time_line, car1_circle, car2_circle

# Animation function
def animate(t):
    # Update positions
    P1_t = position_at_time(P1, V1, t)
    P2_t = position_at_time(P2, V2, t)
    car1_plot.set_data(P1_t[0], P1_t[1])
    car2_plot.set_data(P2_t[0], P2_t[1])
    time_text.set_text(f'Time = {t:.2f}')
    
    # Update circle positions
    car1_circle.center = P1_t
    car2_circle.center = P2_t
    
    # Draw number line (time axis)
    time_line.set_data([t, t], [-1, 4])

    # Mark collision point if collision has occurred
    if t_collision != float('-inf') and t >= t_collision:
        collision_point = position_at_time(P1, V1, t_collision)
        collision_point_plot.set_data(collision_point[0], collision_point[1])
    
    return car1_plot, car2_plot, collision_point_plot, time_text, time_line, car1_circle, car2_circle

# Time parameters for the animation
t_max = t_collision + 1 if t_collision != float('-inf') else 10
num_frames = 200
t_values = np.linspace(0, t_max, num_frames)

# Create the animation
anim = FuncAnimation(fig, animate, frames=t_values, init_func=init,
                     interval=50, blit=True)

# Add legend
ax.legend(loc='upper right')

plt.show()
