# Here we implement a controller driven by neural network architecture to select per time step the input frequencies of a standing wave

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import sys
import math

# Define maximum clusters
MAX_KX = 10
MAX_KY = 10
MAX_CLUSTERS = MAX_KX * MAX_KY  # Adjust as needed

# Environment Setup
@ti.data_oriented
class AcousticEnv():
    def __init__(self, particles: int):
        """ti.init(arch=ti.cpu) #initialization arch ti.cpu/ti.gpu"""

ti.init(arch=ti.cpu)  # initialization arch ti.cpu/ti.gpu
PI = 3.1415926
res = 512
# global control
paused = ti.field(ti.i32, ())  # a scalar i32

N = 61  # number of particles
particle_m_max = 5.0  # mass
nest_size = 0.6  # nest size
particle_radius_max = 10.0 / float(res)  # particle radius for rendering
init_vel = 100  # initial velocity

h = 1e-5  # time step
substepping = 3  # the number of sub-iterations within a time step

# declare fields (pos, vel, force of the particles)
pos = ti.Vector.field(2, ti.f32, N)
pos_1 = ti.Vector.field(2, ti.f32, 1)
vel = ti.Vector.field(2, ti.f32, N)
vel_p1 = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)

particle_radius = ti.field(ti.f32, N)
particle_m = ti.field(ti.f32, N)

energy = ti.field(ti.f32, shape=2)  # [1] current energy [0] initial energy
particle_color = ti.Vector.field(3, ti.f32, N)

# Acoustic properties
po = 10e6  # acoustic pressure

ax = np.array([1.0])
ay = np.array([1.0])
kx = np.array([3], dtype=int)
ky = np.array([1], dtype=int)

# convert arrays into Taichi fields
ax_field = ti.field(dtype=ti.f32, shape=ax.shape)
ay_field = ti.field(dtype=ti.f32, shape=ay.shape)
kx_field = ti.field(dtype=ti.i32, shape=kx.shape)
ky_field = ti.field(dtype=ti.i32, shape=ky.shape)

ax_field.from_numpy(ax)
ay_field.from_numpy(ay)
kx_field.from_numpy(kx)
ky_field.from_numpy(ky)

# initialize with ones
num_waves_x = ti.field(dtype=ti.i32, shape=())
num_waves_y = ti.field(dtype=ti.i32, shape=())
num_waves_x[None] = len(ax)
num_waves_y[None] = len(ay)

# drag properties
drag = 3e6

@ti.kernel
def initialize():  # initialize pos, vel, force of each particle
    for i in range(N):
        theta = ti.random() * 2 * PI  # theta = (0, 2 pi)
        r = (ti.sqrt(ti.random()) * 0.7 + 0.3) * nest_size  # r = (0.3 1)*nest_size
        pos[i] = ti.Vector([ti.random(), ti.random()])
        offset = init_vel * r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        vel[i] = [-offset.y, offset.x]  # vel direction is perpendicular to its offset

        # particle_radius[i] = max(0.4, ti.random()) * particle_radius_max
        particle_radius[i] = 0.5 * particle_radius_max
        particle_m[i] = (particle_radius[i] / particle_radius_max)**2 * particle_m_max

        energy[0] += 0.5 * particle_m[i] * (vel[i][0]**2 + vel[i][1]**2)
        energy[1] += 0.5 * particle_m[i] * (vel[i][0]**2 + vel[i][1]**2)

        particle_color[i][0] = 1 - particle_m[i] / particle_m_max
        particle_color[i][1] = 1 - particle_m[i] / particle_m_max
        particle_color[i][2] = 1 - particle_m[i] / particle_m_max

# Spawning the cluster
cluster_x = 1/6
cluster_y = 1/2
cluster_radius = particle_radius_max

@ti.kernel
def initialize_cluster():
    for i in range(N):
        # Hexagonal lattice
        theta = 0.0
        pos[0] = curr = ti.Vector([cluster_x, cluster_y])
        index = layer = 1
        while index < N:
            curr += cluster_radius * ti.Vector([ti.cos(theta), ti.sin(theta)])
            for j in range(2, 8):
                angle = j * PI / 3 + theta
                for _ in range(layer):
                    curr += cluster_radius * ti.Vector([ti.cos(angle), ti.sin(angle)])
                    pos[index] = curr
                    index += 1
                    if index >= N:
                        break
                if index >= N:
                    break
            layer += 1

        theta = ti.random() * 2 * PI
        r = (ti.sqrt(ti.random()) * 0.7 + 0.3) * nest_size
        offset = init_vel * r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        vel[i] = [-offset.y, offset.x]  # vel direction is perpendicular to its offset

        particle_radius[i] = 0.5 * particle_radius_max
        particle_m[i] = (particle_radius[i] / particle_radius_max)**2 * particle_m_max

        energy[0] += 0.5 * particle_m[i] * (vel[i][0]**2 + vel[i][1]**2)
        energy[1] += 0.5 * particle_m[i] * (vel[i][0]**2 + vel[i][1]**2)

        particle_color[i][0] = 0.3 + 0.7 * ti.random()
        particle_color[i][1] = 0.3 + 0.7 * ti.random()
        particle_color[i][2] = 0.3 + 0.7 * ti.random()

@ti.kernel
def compute_force():
    # clear force
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])

    # compute acoustic force
    for i in range(N):
        f_x = 0.0
        f_y = 0.0

        for wave in range(num_waves_x[None]):
            f_x += particle_radius[i]**3 * 1000000 * ax_field[wave] * ti.sin(2 * PI * pos[i][0] * kx_field[wave])

        for wave in range(num_waves_y[None]):
            f_y += particle_radius[i]**3 * 1000000 * ay_field[wave] * ti.sin(2 * PI * pos[i][1] * ky_field[wave])

        # Compute total force for this position
        f_vector = ti.Vector([f_x, f_y]) * po
        force[i] += f_vector

    # force due to drag
    for i in range(N):
        f = -drag * particle_radius[i] * vel[i]
        force[i] += f

@ti.kernel
def collision_update():
    for i in range(N - 1):
        for j in range(i + 1, N):
            diff = pos[j] - pos[i]
            r = diff.norm(1e-4)  # norm of Vector diff and minimum value is 1e-5 (clamp to 1e-5)
            a = particle_radius[i] + particle_radius[j]
            scale = a / r * 2**(1/6)
            this_force = -max(scale**12 - 2 * scale**6, 0) * particle_m[i] * particle_m[j] * diff * 1e8
            force[i] += this_force
            force[j] -= this_force

@ti.kernel
def update():  # update the position and velocity of each particle
    dt = h / substepping
    for i in range(N):
        vel[i] += dt * force[i] / particle_m[i]
        pos[i] += dt * vel[i]
        # collision detection at edges, flip the velocity
        if pos[i][0] < 0.0 + particle_radius[i] or pos[i][0] > 1.0 - particle_radius[i]:
            vel[i][0] *= -1
        if pos[i][1] < 0.0 + particle_radius[i] or pos[i][1] > 1.0 - particle_radius[i]:
            vel[i][1] *= -1

@ti.kernel
def compute_energy():
    energy[1] = 0.0
    for i in range(N):
        energy[1] += 0.5 * particle_m[i] * (vel[i][0]**2 + vel[i][1]**2)

neighbors = ti.field(ti.i32, N)
particle_node_assignments = ti.field(ti.i32, N)

# Clustering Logic
@ti.kernel
def calculate_neighbors_and_init_clusters(d: float):
    for i in range(N):
        neighbors[i] = 0  # Reset neighbors count
        particle_node_assignments[i] = -1  # Reset cluster assignments (no cluster yet)

    for i in range(N):
        for j in range(i + 1, N):
            r = (pos[j] - pos[i]).norm()
            if r < particle_radius[i] + particle_radius[j] + 2e-3:  # Detect touching particles
                neighbors[i] += 1
                neighbors[j] += 1

@ti.kernel
def assign_clusters(d: float):
    cluster_id[None] = 0
    for i in range(N):
        if neighbors[i] >= 2 and particle_node_assignments[i] == -1:  # Step 1: Start a new cluster
            particle_node_assignments[i] = cluster_id[None]
            expand_cluster(cluster_id[None], i, d)  # Step 2: Expand the cluster
            cluster_id[None] += 1  # Increment cluster ID for the next cluster

# Define stack size
MAX_STACK_SIZE = 1000  # Set this large enough to hold the maximum number of particles you expect in a cluster

# Define the stack using Taichi fields
stack = ti.field(dtype=ti.i32, shape=MAX_STACK_SIZE)  # Stack to hold particle indices
stack_pointer = ti.field(dtype=ti.i32, shape=())  # Stack pointer

@ti.func
def push(val: int):
    stack[stack_pointer[None]] = val
    stack_pointer[None] += 1

@ti.func
def pop() -> int:
    stack_pointer[None] -= 1
    return stack[stack_pointer[None]]

@ti.func
def stack_empty() -> bool:
    return stack_pointer[None] == 0

@ti.func
def expand_cluster(cluster_id: int, particle_id: int, d: float):
    stack_pointer[None] = 0  # Initialize stack pointer
    push(particle_id)  # Push the initial particle onto the stack
    particle_node_assignments[particle_id] = cluster_id  # Assign starting particle to the cluster

    while not stack_empty():  # Loop while there are particles in the stack
        current_particle = pop()  # Take the top particle from the stack

        # Recursively add touching particles to the cluster
        for j in range(N):
            if particle_node_assignments[j] == -1:  # Unassigned particle
                r = (pos[j] - pos[current_particle]).norm()
                if r < particle_radius[current_particle] + particle_radius[j] + 2e-3:
                    particle_node_assignments[j] = cluster_id  # Add to the cluster
                    push(j)  # Add this particle to the stack for further exploration

        # Add particles within distance 'd'
        for j in range(N):
            if particle_node_assignments[j] == -1:  # Unassigned particle
                r = (pos[j] - pos[current_particle]).norm()
                if r < d:  # Check if within distance d
                    particle_node_assignments[j] = cluster_id
                    push(j)  # Add this particle to the stack for further exploration

def run_clustering(d: float):
    calculate_neighbors_and_init_clusters(d)
    assign_clusters(d)

# Create a list to store the average number of neighbors over time
average_neighbors_over_time = []

# Create lists to store the number of particles with 0, 1, 2, 3, 4, 5, and 6 neighbors
neighbor_count_over_time = {i: [] for i in range(7)}

# Create a Taichi field to store the number of particles with 0, 1, 2, 3, 4, 5, and 6 neighbors
neighbor_counts_field = ti.field(dtype=ti.i32, shape=(7,))

@ti.kernel
def calc_neighbors():  # number of neighbors of each particle
    for i in range(N):
        neighbors[i] = 0

    for i in range(N):
        for j in range(i + 1, N):
            r = (pos[j] - pos[i]).norm()
            if r < particle_radius[i] + particle_radius[j] + 2e-3:
                neighbors[i] += 1
                neighbors[j] += 1

    # Clear the neighbor counts field
    for i in range(7):
        neighbor_counts_field[i] = 0

    # Count how many particles have 0, 1, 2, ..., 6 neighbors
    for i in range(N):
        if neighbors[i] <= 6:  # Cap the number of neighbors at 6 for this analysis
            neighbor_counts_field[neighbors[i]] += 1

# Function to calculate and store the average number of neighbors
def update_average_neighbors():
    avg_neighbors = np.mean(neighbors.to_numpy())
    average_neighbors_over_time.append(avg_neighbors)

# After the kernel is run, transfer the data from the Taichi field to a Python list
def update_neighbor_count_over_time():
    counts = neighbor_counts_field.to_numpy()  # Convert the Taichi field to a NumPy array
    for i in range(7):
        neighbor_count_over_time[i].append(counts[i])  # Append the counts to the Python list

def plot_average_neighbors_over_time():
    plt.figure(figsize=(10, 6))
    plt.plot(average_neighbors_over_time, label="Average Neighbors Over Time", color="blue", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Average Number of Neighbors")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"average_neighbors_{c}.png")
    plt.show()

def plot_neighbor_distribution_over_time():
    plt.figure(figsize=(10, 6))

    # Plot each neighbor count over time
    for i in range(7):
        plt.plot(neighbor_count_over_time[i], label=f"Particles with {i} neighbors", linewidth=2)

    plt.xlabel("Time Step")
    plt.ylabel("Number of Particles")
    plt.title("Neighbor Distribution Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"neighbor_distribution_{c}.png")
    plt.show()

def plot():
    import matplotlib.pyplot as plt

    positions = np.linspace(0, 1, 500)

    # Compute wave values and superpositions
    def compute_waves(amplitudes, wavenumbers, positions):
        num_waves = len(amplitudes)
        wave_values = np.zeros((num_waves, len(positions)))
        for i in range(num_waves):
            wave_values[i, :] = amplitudes[i] * np.sin(2 * np.pi * positions * wavenumbers[i])
        superposition = np.sum(wave_values, axis=0)
        return wave_values, superposition

    # Compute for x and y
    wave_values_x, superposition_x = compute_waves(ax, kx, positions)
    wave_values_y, superposition_y = compute_waves(ay, ky, positions)

    # Plotting
    def plot_waves_and_superposition(wave_values, superposition, positions, title):
        plt.figure(figsize=(10, 6))
        for i in range(wave_values.shape[0]):
            plt.plot(positions, wave_values[i, :], label=f"Wave {i+1}")
        plt.plot(positions, superposition, label="Superposition", color="black", linewidth=2, linestyle="--")
        plt.title(title)
        plt.xlabel("Position")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

    # Plot for x
    plot_waves_and_superposition(wave_values_x, superposition_x, positions, "Waves and Superposition in X direction")

    # Plot for y
    plot_waves_and_superposition(wave_values_y, superposition_y, positions, "Waves and Superposition in Y direction")

    # Create a 2D grid of positions
    X, Y = np.meshgrid(positions, positions)

    # Calculate the force magnitude at each point in the grid
    # Assume the superposition in x affects the x component of the force and y affects the y component
    Force_x = np.outer(np.ones_like(positions), superposition_x)
    Force_y = np.outer(superposition_y, np.ones_like(positions))
    Force_magnitude = np.sqrt(Force_x**2 + Force_y**2)

    # Plotting the 2D colormap of force magnitudes
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Force_magnitude, shading="auto", cmap="ocean_r")
    plt.colorbar(label="Force Magnitude")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.title("2D Colormap of Force Magnitude from Superpositions")
    plt.show()

c = 0
def plot_radial_distribution():
    import matplotlib.pyplot as plt
    global c

    positions = pos.to_numpy()

    rs = np.linspace(0, np.sqrt(2), 500)  # Updated the range to cover the entire domain
    gs = np.zeros_like(rs)
    mid = np.mean(positions, axis=0)  # Compute the center of mass of the particles

    plt.figure(figsize=(10, 6))
    for j in range(N):
        distance = np.linalg.norm(positions[j] - mid)
        for i, r in enumerate(rs):
            contribution = np.sqrt(max(0., 1. - ((distance - r) ** 2) / (particle_radius[j] ** 2)))
            gs[i] += contribution

    plt.plot(rs, gs, label="Radial Distribution Function", color="black", linewidth=2, linestyle="--")
    plt.xlabel("Distance from Center")
    plt.ylabel("Radial Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"radial_distribution_{c}.png")
    plt.show()
    c += 1

if len(sys.argv) > 1 and sys.argv[1] == "plot":
    plot()
    exit()

# start the simulation
d = 1.5 * particle_radius_max  # Set the distance threshold for clustering

gui = ti.GUI("Bulk Acoustic 2D Simulation", (res, res))  # create a window of resolution 512*512

initialize_cluster()

xchanging = True
message = "Changing number of nodes along X-axis"

# Declare cluster_id as a Taichi field
cluster_id = ti.field(dtype=ti.i32, shape=())

# Initialize variables for plotting clusters
colors = [
    0xFF0000,  # Red
    0x00FF00,  # Green
    0x0000FF,  # Blue
    0xFFFF00,  # Yellow
    0xFF00FF,  # Magenta
    0x00FFFF,  # Cyan
    0xFFFFFF,  # White
    0xFFA500,  # Orange
    0x800080,  # Purple
    0x00FF7F,  # Spring Green
    0x808000,  # Olive
    0x008080,  # Teal
    0x000000,  # Black
]

def adjust_brightness(color, factor):
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF

    r = int(min(max(r * factor, 0), 255))
    g = int(min(max(g * factor, 0), 255))
    b = int(min(max(b * factor, 0), 255))

    return (r << 16) + (g << 8) + b

while gui.running:  # update frames, interval is time step h

    for e in gui.get_events(ti.GUI.PRESS):  # event processing
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == "r":
            initialize_cluster()
        elif e.key == ti.GUI.SPACE:
            paused[None] = not paused[None]
        elif e.key in "123456789":
            val = int(e.key)
            if xchanging:
                kx_field[0] = kx[0] = val
            else:
                ky_field[0] = ky[0] = val
        elif e.key == "x":
            xchanging = True
            message = "Changing number of nodes along X-axis"
        elif e.key == "y":
            xchanging = False
            message = "Changing number of nodes along Y-axis"
        elif e.key == "g":
            plot_radial_distribution()
            plot_average_neighbors_over_time()  # Plot the average number of neighbors
            plot_neighbor_distribution_over_time()  # Plot the neighbor distribution over time

    if not paused[None]:
        for i in range(substepping):  # run substepping times for each time step
            compute_force()
            collision_update()
            update()
            vel_p1.copy_from(vel)
            vel.copy_from(vel_p1)
            compute_energy()
            calc_neighbors()
            run_clustering(d)  # Run the new clustering algorithm
            update_average_neighbors()
            update_neighbor_count_over_time()  # Update the neighbor count data for plotting
            # print("Current energy = {}, Initial energy = {}, Ratio = {}".format(energy[1],energy[0],energy[1]/energy[0]))

        total_clusters = cluster_id[None]

        # Convert Taichi field to NumPy array
        assignments = particle_node_assignments.to_numpy()

        # Create a mask for unassigned particles
        unassigned_mask = assignments == -1

        # For unassigned particles, set their cluster ID to 'total_clusters' (the next integer)
        assignments_unassigned = assignments.copy()
        assignments_unassigned[unassigned_mask] = total_clusters

        # Now, total number of clusters including the unassigned cluster is 'total_clusters + 1'
        total_clusters_with_unassigned = total_clusters + 1

        # Count the number of particles in each cluster
        counts = np.bincount(assignments_unassigned.astype(np.int32), minlength=total_clusters_with_unassigned)

    gui.clear(0x112F41)  # Hex code of the color: 0x000000 = black, 0xffffff = white

    # Re-add the lines indicating pressure minima
    lines_start = np.zeros((kx[0] + ky[0], 2))
    lines_end = np.ones((kx[0] + ky[0], 2))
    lines_start[:kx[0], 0] = lines_end[:kx[0], 0] = np.linspace(.5/kx[0], 1-.5/kx[0], kx[0])
    lines_start[kx[0]:, 1] = lines_end[kx[0]:, 1] = np.linspace(.5/ky[0], 1-.5/ky[0], ky[0])

    gui.lines(lines_start, lines_end, color=0x808080, radius=1.0)

    # Get the neighbors array
    neighbors_np = neighbors.to_numpy()
    assignments = particle_node_assignments.to_numpy()
    num_colors = len(colors)

    for i in range(N):
        pos_1[0] = pos[i]

        cluster_id_i = assignments[i]
        num_neighbors = neighbors_np[i]
        h = min(num_neighbors / 6, 1)
        h = max(h, 0)
        brightness_factor = h  # Assuming h ranges from 0 to 1

        if cluster_id_i == -1:
            base_color = 0x808080  # Gray for unassigned particles
        else:
            base_color = colors[cluster_id_i % num_colors]

        color = adjust_brightness(base_color, brightness_factor)

        gui.circles(pos_1.to_numpy(), \
            color=color, \
            radius=particle_radius[i] * float(res))

    # Define starting position and spacing
    start_x = 0.05  # Position near the left edge
    start_y = 0.95  # Start near the top
    spacing = 0.03   # Vertical spacing between lines

    # Iterate over each cluster and display its count
    for cluster_idx in range(total_clusters):
        cluster_text = f"Cluster {cluster_idx + 1}: {counts[cluster_idx]} particles"
        y_pos = start_y - cluster_idx * spacing
        # Ensure that text doesn't go below the bottom of the screen
        if y_pos < 0.05:
            break
        gui.text(cluster_text, pos=(start_x, y_pos), color=0xFFFFFF, font_size=16)

    # Display the unassigned particles
    cluster_text = f"Unassigned: {counts[total_clusters]} particles"
    y_pos = start_y - (total_clusters) * spacing
    if y_pos >= 0.05:
        gui.text(cluster_text, pos=(start_x, y_pos), color=0xFFFFFF, font_size=16)

    # Add numbering labels to clusters
    pos_np = pos.to_numpy()
    for cluster_idx in range(total_clusters):
        indices = np.where(assignments == cluster_idx)[0]
        if len(indices) > 0:
            cluster_positions = pos_np[indices]
            centroid = np.mean(cluster_positions, axis=0)
            # Ensure the label stays within bounds
            centroid = np.clip(centroid, 0.05, 0.95)
            gui.text(
                str(cluster_idx + 1),
                pos=(centroid[0], centroid[1]),
                color=0xFFFFFF,
                font_size=20,
            )

    gui.text(message, pos=(0.05, 0.99), color=0xFFFFFF, font_size=20)
    # pos=(x, y) coordinates range from (0,0) bottom-left to (1,1) top-right

    gui.text(f"Number of clusters: {total_clusters}", pos=(0.05, 0.05), color=0xFFFFFF, font_size=20)

    gui.fps_limit = 30
    gui.show()
