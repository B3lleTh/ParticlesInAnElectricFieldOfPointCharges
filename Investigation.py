import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# charges: (x, y, value)
charges = [
    (-1.0, 0.0, 5.0),
    (1.0, 0.0, -5.0),
    (0.0, 1.5, 3.0)
]

# grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# potential function
def potential(X, Y):
    V = np.zeros_like(X)
    for (cx, cy, q) in charges:
        r = np.sqrt((X - cx)**2 + (Y - cy)**2) + 1e-5  # avoid division by zero
        V += q / r
    return V

# electric field
def electric_field(X, Y):
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    for (cx, cy, q) in charges:
        dx = X - cx
        dy = Y - cy
        r = np.sqrt(dx**2 + dy**2) + 1e-5
        Ex += q * dx / r**3
        Ey += q * dy / r**3
    return Ex, Ey

# compute field and potential
V = potential(X, Y)
Ex, Ey = electric_field(X, Y)

# particles
n_particles = 20
pos = np.random.uniform(-2, 2, (n_particles, 2))
vel = np.zeros((n_particles, 2))

dt = 0.05

# trails for visualization
trails = [[pos[i].copy()] for i in range(n_particles)]

# plot setup
fig, ax = plt.subplots(figsize=(6,6))

# background (potential)
ax.contourf(X, Y, V, levels=50, cmap='coolwarm', alpha=0.6)

# field arrows
ax.quiver(X[::5, ::5], Y[::5, ::5], Ex[::5, ::5], Ey[::5, ::5], color='black')

# field lines
ax.streamplot(X, Y, Ex, Ey, color='k', density=1, linewidth=0.5)

# draw charges
for (cx, cy, q) in charges:
    if q > 0:
        ax.scatter(cx, cy, color='red', s=100)
    else:
        ax.scatter(cx, cy, color='blue', s=100)

particles_plot, = ax.plot([], [], 'go')

# trail lines
trail_lines = [ax.plot([], [], lw=1)[0] for _ in range(n_particles)]

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Particles in Electric Field")

# animation update
def update(frame):
    global pos, vel

    Ex_p = np.zeros(n_particles)
    Ey_p = np.zeros(n_particles)

    # field at particle positions
    for (cx, cy, q) in charges:
        dx = pos[:,0] - cx
        dy = pos[:,1] - cy
        r = np.sqrt(dx**2 + dy**2) + 1e-5
        Ex_p += q * dx / r**3
        Ey_p += q * dy / r**3

    # simple motion (q/m = 1)
    vel[:,0] += Ex_p * dt
    vel[:,1] += Ey_p * dt

    pos += vel * dt

    # update trails
    for i in range(n_particles):
        trails[i].append(pos[i].copy())
        if len(trails[i]) > 30:
            trails[i].pop(0)

    # update particle positions
    particles_plot.set_data(pos[:,0], pos[:,1])

    # update trails
    for i, line in enumerate(trail_lines):
        trail = np.array(trails[i])
        line.set_data(trail[:,0], trail[:,1])

    return [particles_plot] + trail_lines

# run animation
ani = FuncAnimation(fig, update, frames=200, interval=50)

plt.show()