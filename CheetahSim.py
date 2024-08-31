import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
size = 200  # Domain size
Du, Dv = 0.16, 0.08  # Diffusion coefficients for u and v
F, k = 0.035, 0.065  # Feed rate and kill rate
dt = 1.0  # Time step
steps = 10000  # Number of iterations

# Initialize the domain with small random perturbations
u = np.ones((size, size))
v = np.zeros((size, size))
r = 20  # Radius of initial pattern perturbation

# Add initial perturbations
np.random.seed(0)
u[size//2-r:size//2+r, size//2-r:size//2+r] = 0.50 + 0.1 * np.random.rand(2 * r, 2 * r)
v[size//2-r:size//2+r, size//2-r:size//2+r] = 0.25 + 0.1 * np.random.rand(2 * r, 2 * r)

def laplacian(Z):
    """Compute the Laplacian of matrix Z."""
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)

def update():
    """Perform one update step of the reaction-diffusion simulation."""
    global u, v

    Lu = laplacian(u)
    Lv = laplacian(v)
    
    uvv = u * v * v
    u += (Du * Lu - uvv + F * (1 - u)) * dt
    v += (Dv * Lv + uvv - (F + k) * v) * dt

def run_simulation():
    """Run the reaction-diffusion simulation and display the result."""
    fig, ax = plt.subplots()
    img = ax.imshow(u, cmap='copper', interpolation='bilinear')

    def animate(frame):
        for _ in range(100):  # Update multiple times per frame for faster animation
            update()
        img.set_data(u)
        return [img]

    anim = FuncAnimation(fig, animate, frames=steps//100, interval=1, blit=True)
    plt.show()

run_simulation()
