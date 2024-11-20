# scp /Users/xanderbackus/Acoustic_UROP/PyGame_Acoustics_2D_cluster_manipulation.py xabackus@f12.csail.mit.edu:~/
# ssh xabackus@f12.csail.mit.edu
# python3 PyGame_Acoustics_2D_cluster_manipulation.py
# If you want the script to continue running after you disconnect, use nohup:
# nohup python3 PyGame_Acoustics_2D_cluster_manipulation.py &

import numpy as np
import pygame
import sys
import math
import matplotlib.pyplot as plt
import cv2  # For video writing
import json  # For saving/loading key press data
import os   # For file path operations
import scipy.optimize  # For linear sum assignment

# Set the mode: 'write', 'read', or 'simulation'
# Uncomment the desired mode
# mode = 'write'       # Record key presses
# mode = 'read'        # Replay simulation based on recorded key presses
mode = 'simulation'    # ML-based control to optimize action timing
learning_rate = 1

render_bool = False     # Boolean variable; controls whether simulation renders while running gradient descent

# Constants and Initialization
PI = 3.1415926
res = 512  # Simulation area size (512x512 pixels)
dashboard_width = 400  # Width for the dashboard area

# Define dashboard heights
text_dashboard_height = 300  # Height for displaying text information
average_graph_height = 200   # Height for the average neighbors graph
cumulative_graph_height = 200  # Height for the cumulative neighbor proportions graph

# Calculate window dimensions
window_width = res + dashboard_width
window_height = max(res, text_dashboard_height + average_graph_height + cumulative_graph_height)

# Pygame Initialization
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Bulk Acoustic 2D Simulation with Dashboards")
clock = pygame.time.Clock()

# Video recording setup
record_video = True  # Set to True to record video
video_filename = 'simulation_video.mp4'
video_writer = None
video_fps = 30  # Frames per second

if record_video:
    frame_width = window_width
    frame_height = window_height
    # Use 'mp4v' codec for better compatibility
    video_writer = cv2.VideoWriter(
        video_filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_fps,
        (frame_width, frame_height)
    )

# Simulation Parameters
paused = False
running = True  # Flag to control the main loop
N = 12  # Number of particles (changed to 12)
particle_m_max = 5.0  # Maximum mass of particles
nest_size = 0.6  # Nest size
particle_radius_max = 10.0 / float(res)  # Maximum particle radius (~0.0195)
init_vel = 100  # Initial velocity

h = 1e-5  # Time step
substepping = 3  # Number of sub-iterations within a time step

# Set the distance threshold for clustering
d = 1.5 * particle_radius_max  # Global variable 'd'

# Fields for particle properties
pos = np.zeros((N, 2), dtype=np.float32)  # Positions
vel = np.zeros((N, 2), dtype=np.float32)  # Velocities
force = np.zeros((N, 2), dtype=np.float32)  # Forces

particle_radius = np.zeros(N, dtype=np.float32)  # Radii
particle_m = np.zeros(N, dtype=np.float32)  # Masses

energy = np.zeros(2, dtype=np.float32)  # [1] current energy, [0] initial energy
particle_color = np.zeros((N, 3), dtype=np.float32)  # Colors (not used in rendering)

# Acoustic properties
po = 10e7  # Acoustic pressure

ax = np.array([1.0])  # Amplitude in x-direction
ay = np.array([1.0])  # Amplitude in y-direction
kx = np.array([1], dtype=int)  # Wave number in x-direction (initially 1)
ky = np.array([1], dtype=int)  # Wave number in y-direction

ax_field = ax
ay_field = ay
kx_field = kx
ky_field = ky

num_waves_x = len(ax)
num_waves_y = len(ay)

# Drag properties
drag = 3e6

# Clustering variables
neighbors = np.zeros(N, dtype=np.int32)  # Number of neighbors for each particle
particle_node_assignments = np.full(N, -1, dtype=np.int32)  # Cluster assignments
cluster_id = 0  # Current cluster ID

# For plotting
average_neighbors_over_time = []  # Track average neighbors over time
neighbor_count_over_time = {i: [] for i in range(7)}  # Track counts of neighbor numbers over time
neighbor_counts_field = np.zeros(7, dtype=np.int32)  # Field to store neighbor counts
weighted_avg_density_over_time = []  # List to store weighted average cluster density over time
weighted_avg_aspect_ratio_over_time = []  # List to store weighted average aspect ratio over time

# Initialize key press tracking variables
key_press_data_indices = []  # Stores data indices when number keys are pressed
time_step = 0  # Tracks the current time step

key_press_data = []  # List of tuples: (time_step, key_name)

if mode == 'read':
    # Load key_press_data from file
    with open('key_press_data.txt', 'r') as f:
        key_press_data = json.load(f)
    key_press_index = 0
else:
    key_press_data = []

# Set random seed for reproducibility
np.random.seed(0)

# Initialize particles randomly
def initialize():
    global energy
    energy[0] = 0.0
    energy[1] = 0.0
    for i in range(N):
        pos[i] = np.random.rand(2)
        vel[i] = np.zeros(2)
        particle_radius[i] = 0.5 * particle_radius_max
        particle_m[i] = (particle_radius[i] / particle_radius_max) ** 2 * particle_m_max
        particle_color[i] = 0.3 + 0.7 * np.random.rand(3)

# Compute forces acting on particles
def compute_force():
    # Clear forces
    force[:, :] = 0.0

    # Compute acoustic force
    for i in range(N):
        f_x = 0.0
        f_y = 0.0

        for wave in range(num_waves_x):
            f_x += particle_radius[i] ** 3 * 1e6 * ax_field[wave] * np.sin(2 * PI * pos[i][0] * kx_field[wave])

        for wave in range(num_waves_y):
            f_y += particle_radius[i] ** 3 * 1e6 * ay_field[wave] * np.sin(2 * PI * pos[i][1] * ky_field[wave])

        f_vector = np.array([f_x, f_y]) * po
        force[i] += f_vector

    # Force due to drag
    for i in range(N):
        f = -drag * particle_radius[i] * vel[i]
        force[i] += f

    # Compute collision forces and add to force
    compute_collision_force()

# Compute collision forces between particles
def compute_collision_force():
    global force
    # Initialize collision force
    collision_force = np.zeros_like(force)
    EPSILON = 1e-4
    DAMPING = 0.1
    K = 1e11  # Stiffness constant

    # Calculate pairwise differences and distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # Shape (N, N, 2)
    r = np.linalg.norm(diff, axis=2) + EPSILON  # Shape (N, N)

    # Calculate overlaps
    overlap = particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :] - r  # Shape (N, N)
    overlap = np.maximum(overlap, 0)  # Only positive overlaps (compressions)

    # Normal force magnitude (Hertzian contact model)
    normal_force_magnitude = K * overlap ** (3 / 2)  # Shape (N, N)

    # Compute unit vectors along diff
    normal_vector = diff / r[..., np.newaxis]  # Shape (N, N, 2)

    # Compute relative velocities
    relative_velocity = vel[:, np.newaxis, :] - vel[np.newaxis, :, :]  # Shape (N, N, 2)

    # Compute relative velocity along the normal direction
    vel_along_normal = np.sum(relative_velocity * normal_vector, axis=2)  # Shape (N, N)

    # Damping force magnitude
    damping_force_magnitude = -DAMPING * vel_along_normal  # Shape (N, N)

    # Total force magnitude
    total_force_magnitude = normal_force_magnitude + damping_force_magnitude  # Shape (N, N)

    # Total force vector
    total_force = total_force_magnitude[..., np.newaxis] * normal_vector  # Shape (N, N, 2)

    # Compute collision force on each particle
    collision_force = np.sum(total_force, axis=1) - np.sum(total_force, axis=0)  # Shape (N, 2)

    # Add collision force to total force
    force += collision_force

# Update positions and velocities
def update():
    dt = h / substepping
    for i in range(N):
        vel[i] += dt * force[i] / particle_m[i]
        pos[i] += dt * vel[i]
        # Collision detection at edges
        if pos[i][0] < 0.0 + particle_radius[i]:
            pos[i][0] = 0.0 + particle_radius[i]
            vel[i][0] *= -1
        if pos[i][0] > 1.0 - particle_radius[i]:
            pos[i][0] = 1.0 - particle_radius[i]
            vel[i][0] *= -1
        if pos[i][1] < 0.0 + particle_radius[i]:
            pos[i][1] = 0.0 + particle_radius[i]
            vel[i][1] *= -1
        if pos[i][1] > 1.0 - particle_radius[i]:
            pos[i][1] = 1.0 - particle_radius[i]
            vel[i][1] *= -1

# Compute total kinetic energy
def compute_energy():
    energy[1] = 0.0
    for i in range(N):
        energy[1] += 0.5 * particle_m[i] * np.sum(vel[i] ** 2)

# Clustering logic
def calculate_neighbors_and_init_clusters(d):
    global neighbors, particle_node_assignments
    neighbors[:] = 0
    particle_node_assignments[:] = -1
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(pos[j] - pos[i])
            if r < particle_radius[i] + particle_radius[j] + 2e-3:
                neighbors[i] += 1
                neighbors[j] += 1

def expand_cluster(cluster_id_value, particle_id, d):
    stack = []
    stack.append(particle_id)
    particle_node_assignments[particle_id] = cluster_id_value

    while stack:
        current_particle = stack.pop()
        for j in range(N):
            if particle_node_assignments[j] == -1:
                r = np.linalg.norm(pos[j] - pos[current_particle])
                if r < particle_radius[current_particle] + particle_radius[j] + 2e-3:
                    particle_node_assignments[j] = cluster_id_value
                    stack.append(j)
                elif r < d:
                    particle_node_assignments[j] = cluster_id_value
                    stack.append(j)

def assign_clusters(d):
    global cluster_id
    cluster_id = 0
    for i in range(N):
        if neighbors[i] >= 2 and particle_node_assignments[i] == -1:
            particle_node_assignments[i] = cluster_id
            expand_cluster(cluster_id, i, d)
            cluster_id += 1

def run_clustering(d):
    calculate_neighbors_and_init_clusters(d)
    assign_clusters(d)

def calc_neighbors():
    global neighbors, neighbor_counts_field
    neighbors[:] = 0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(pos[j] - pos[i])
            if r < particle_radius[i] + particle_radius[j] + 2e-3:
                neighbors[i] += 1
                neighbors[j] += 1

    neighbor_counts_field[:] = 0
    for i in range(N):
        if neighbors[i] <= 6:
            neighbor_counts_field[neighbors[i]] += 1

def update_average_neighbors():
    avg_neighbors = np.mean(neighbors)
    average_neighbors_over_time.append(avg_neighbors)

def update_neighbor_count_over_time():
    counts = neighbor_counts_field.copy()
    for i in range(7):
        neighbor_count_over_time[i].append(counts[i])

def calculate_weighted_avg_density():
    global weighted_avg_density_over_time
    total_weighted_density = 0.0
    total_particles = 0
    for cluster_idx in range(cluster_id):
        indices = np.where(particle_node_assignments == cluster_idx)[0]
        num_particles_in_cluster = len(indices)
        if num_particles_in_cluster > 1:
            # For each particle, compute distance to its nearest neighbor in the cluster
            distances = []
            for i in indices:
                # Get positions of other particles in the same cluster
                other_indices = indices[indices != i]
                if len(other_indices) == 0:
                    continue  # Only one particle left, skip
                dists = np.linalg.norm(pos[other_indices] - pos[i], axis=1)
                min_dist = np.min(dists)
                distances.append(min_dist)
            if len(distances) > 0:
                avg_distance = np.mean(distances)
                # Weight by number of particles in the cluster
                total_weighted_density += avg_distance * num_particles_in_cluster
                total_particles += num_particles_in_cluster
    if total_particles > 0:
        weighted_avg_density = total_weighted_density / total_particles
    else:
        weighted_avg_density = 0.0  # No clusters, density is zero
    weighted_avg_density_over_time.append(weighted_avg_density)

def calculate_aspect_ratios():
    global weighted_avg_aspect_ratio_over_time
    total_weighted_aspect_ratio = 0.0
    total_particles_in_clusters = 0
    cluster_aspect_ratios = {}

    for cluster_idx in range(cluster_id):
        indices = np.where(particle_node_assignments == cluster_idx)[0]
        num_particles_in_cluster = len(indices)
        if num_particles_in_cluster > 1:
            # Get positions of particles in the cluster
            cluster_positions = pos[indices]
            # Compute covariance matrix
            covariance_matrix = np.cov(cluster_positions.T)
            # Compute eigenvalues
            eigenvalues, _ = np.linalg.eig(covariance_matrix)
            # Sort eigenvalues
            sorted_eigenvalues = np.sort(eigenvalues)
            # Avoid division by zero
            if sorted_eigenvalues[0] == 0:
                aspect_ratio = np.inf  # Set to infinity if division by zero
            else:
                aspect_ratio = np.sqrt(sorted_eigenvalues[-1] / sorted_eigenvalues[0])
            cluster_aspect_ratios[cluster_idx] = aspect_ratio
            # Weighted sum
            total_weighted_aspect_ratio += aspect_ratio * num_particles_in_cluster
            total_particles_in_clusters += num_particles_in_cluster
        else:
            # For clusters with only one particle, aspect ratio is 1
            cluster_aspect_ratios[cluster_idx] = 1.0
            total_weighted_aspect_ratio += 1.0 * num_particles_in_cluster
            total_particles_in_clusters += num_particles_in_cluster

    if total_particles_in_clusters > 0:
        weighted_avg_aspect_ratio = total_weighted_aspect_ratio / total_particles_in_clusters
    else:
        weighted_avg_aspect_ratio = 0.0

    weighted_avg_aspect_ratio_over_time.append(weighted_avg_aspect_ratio)
    return cluster_aspect_ratios

# Initialize the simulation
initialize()
xchanging = True  # Flag to indicate whether changing kx or ky
message = "Changing number of nodes along X-axis"

# Colors for clusters
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
    """Adjust the brightness of a color."""
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF

    r = int(min(max(r * factor, 0), 255))
    g = int(min(max(g * factor, 0), 255))
    b = int(min(max(b * factor, 0), 255))

    return (r, g, b)

# Font for text
pygame.font.init()
font = pygame.font.SysFont('Arial', 16)

# Define dashboard areas
text_area_rect = pygame.Rect(res, 0, dashboard_width, text_dashboard_height)
average_graph_rect = pygame.Rect(res, text_dashboard_height, dashboard_width, average_graph_height)
cumulative_graph_rect = pygame.Rect(res, text_dashboard_height + average_graph_height, dashboard_width, cumulative_graph_height)

# Colors for neighbor counts in cumulative graph
neighbor_colors = [
    (255, 0, 0),     # Red for 0 neighbors
    (255, 165, 0),   # Orange for 1 neighbor
    (255, 255, 0),   # Yellow for 2 neighbors
    (0, 255, 0),     # Green for 3 neighbors
    (0, 127, 255),   # Light blue for 4 neighbors
    (0, 0, 255),     # Blue for 5 neighbors
    (139, 0, 255),   # Purple for 6 neighbors
]

neighbor_labels = ["0", "1", "2", "3", "4", "5", "6"]

def process_key_press(key):
    """Process key presses to control simulation parameters."""
    global xchanging, message, kx_field, ky_field, kx, ky, paused, running, key_press_data_indices, record_video, video_writer, key_press_data
    if key == pygame.K_ESCAPE:
        running = False
    elif key == pygame.K_r:
        initialize()
    elif key == pygame.K_SPACE:
        paused = not paused
    elif key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
        val = int(pygame.key.name(key))
        if xchanging:
            kx_field[0] = kx[0] = val
        else:
            ky_field[0] = ky[0] = val
        # Record the current time_step when a number key is pressed
        key_press_data_indices.append(time_step)
        # Record key press data
        key_name = pygame.key.name(key)
        key_press_data.append((time_step, key_name))
    elif key == pygame.K_x:
        xchanging = True
        message = "Changing number of nodes along X-axis"
        # Record key press data
        key_press_data.append((time_step, 'x'))
    elif key == pygame.K_y:
        xchanging = False
        message = "Changing number of nodes along Y-axis"
        # Record key press data
        key_press_data.append((time_step, 'y'))
    elif key == pygame.K_g:
        # Plot graphs using Matplotlib
        plot_average_neighbors_over_time()
        plot_neighbor_distribution_over_time()
        plot_weighted_avg_density_over_time()
        plot_weighted_avg_aspect_ratio_over_time()
        # Release the video writer
        if record_video and video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"Video saved as {video_filename}")
        if mode == 'write':
            # Save key_press_data to file
            with open('key_press_data.txt', 'w') as f:
                json.dump(key_press_data, f)
            print("Key press data saved to key_press_data.txt")
        elif mode == 'read':
            # In 'read' mode, when 'g' is pressed, we stop the simulation
            running = False
        elif mode == 'simulation':
            # In 'simulation' mode, when 'g' is pressed, we stop the simulation
            running = False

# Function to render the simulation
def render_simulation(cluster_aspect_ratios):
    global assignments, cluster_id, pos
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    # Clear the screen
    window.fill((17, 47, 65))  # Dark background

    # Draw pressure minima lines
    lines = []
    for x_line in np.linspace(.5 / kx[0], 1 - .5 / kx[0], kx[0]):
        start_pos = (int(x_line * res), 0)
        end_pos = (int(x_line * res), res)
        lines.append((start_pos, end_pos))

    for y_line in np.linspace(.5 / ky[0], 1 - .5 / ky[0], ky[0]):
        start_pos = (0, int(y_line * res))
        end_pos = (res, int(y_line * res))
        lines.append((start_pos, end_pos))

    for line in lines:
        pygame.draw.line(window, (128, 128, 128), line[0], line[1], 1)  # Gray lines

    # Draw particles
    num_colors = len(colors)
    assignments = particle_node_assignments.copy()

    unassigned_mask = assignments == -1

    assignments_unassigned = assignments.copy()
    assignments_unassigned[unassigned_mask] = cluster_id

    total_clusters_with_unassigned = cluster_id + 1

    counts = np.bincount(assignments_unassigned.astype(np.int32), minlength=total_clusters_with_unassigned)

    for i in range(N):
        cluster_id_i = assignments[i]
        num_neighbors = neighbors[i]
        brightness_factor = min(max(num_neighbors / 6, 0), 1)

        if cluster_id_i == -1:
            base_color = 0x808080  # Gray for unassigned particles
        else:
            base_color = colors[cluster_id_i % num_colors]

        color = adjust_brightness(base_color, brightness_factor)

        # Ensure positions are within [0,1]
        pos_clipped = np.clip(pos[i], 0.0, 1.0)
        screen_pos = (int(pos_clipped[0] * res), int(pos_clipped[1] * res))
        radius = int(max(particle_radius[i] * res, 2))  # Minimum radius of 2 pixels for visibility
        pygame.draw.circle(window, color, screen_pos, radius)

    # Draw Text Dashboard
    pygame.draw.rect(window, (30, 30, 30), text_area_rect)  # Background for text area

    # Display cluster information
    start_x = text_area_rect.left + 10
    start_y = text_area_rect.top + 10  # Starting y position for text
    spacing = 20  # Spacing between lines

    # Render cluster information
    y_pos = start_y
    for cluster_idx in range(cluster_id):
        aspect_ratio = cluster_aspect_ratios.get(cluster_idx, 1.0)
        cluster_text = f"Cluster {cluster_idx + 1}: {counts[cluster_idx]} particles; aspect ratio = {aspect_ratio:.2f}"
        if y_pos + spacing > text_area_rect.bottom:
            break
        text_surface = font.render(cluster_text, True, (255, 255, 255))
        window.blit(text_surface, (start_x, y_pos))
        y_pos += spacing

    # Display unassigned particles
    if y_pos + spacing <= text_area_rect.bottom:
        cluster_text = f"Unassigned: {counts[cluster_id]} particles"
        text_surface = font.render(cluster_text, True, (255, 255, 255))
        window.blit(text_surface, (start_x, y_pos))
        y_pos += spacing

    # Display messages
    if y_pos + spacing <= text_area_rect.bottom:
        message_surface = font.render(message, True, (255, 255, 255))
        window.blit(message_surface, (start_x, y_pos))
        y_pos += spacing

    # Display average neighbors
    if y_pos + spacing <= text_area_rect.bottom:
        avg_neighbors = average_neighbors_over_time[-1] if average_neighbors_over_time else 0
        avg_neighbors_surface = font.render(f"Average Neighbors: {avg_neighbors:.2f}", True, (255, 255, 255))
        window.blit(avg_neighbors_surface, (start_x, y_pos))
        y_pos += spacing

    # Display number of clusters
    if y_pos + spacing <= text_area_rect.bottom:
        num_clusters_surface = font.render(f"Number of Clusters: {cluster_id}", True, (255, 255, 255))
        window.blit(num_clusters_surface, (start_x, y_pos))
        y_pos += spacing

    # Display weighted average cluster density
    if y_pos + spacing <= text_area_rect.bottom:
        weighted_avg_density = weighted_avg_density_over_time[-1] if weighted_avg_density_over_time else 0
        density_surface = font.render(f"Weighted Avg Cluster Density: {weighted_avg_density:.4f}", True, (255, 255, 255))
        window.blit(density_surface, (start_x, y_pos))
        y_pos += spacing

    # Display weighted average aspect ratio
    if y_pos + spacing <= text_area_rect.bottom:
        weighted_avg_aspect_ratio = weighted_avg_aspect_ratio_over_time[-1] if weighted_avg_aspect_ratio_over_time else 0
        aspect_ratio_surface = font.render(f"Weighted Avg Aspect Ratio: {weighted_avg_aspect_ratio:.2f}", True, (255, 255, 255))
        window.blit(aspect_ratio_surface, (start_x, y_pos))
        y_pos += spacing

    pygame.display.flip()

    # Capture the screen and write to video
    if record_video:
        # Capture the screen
        frame = pygame.surfarray.array3d(window)
        # Convert from (width, height, 3) to (height, width, 3)
        frame = np.transpose(frame, (1, 0, 2))
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Write frame to video
        video_writer.write(frame)

####################################################################################
# ML code for 'simulation' mode begins here

if mode == 'simulation':
    def generate_initial_guesses(actions, initial_time=50, initial_step=10, time_gap=51):
        initial_guesses = []
        def helper(current_guess, idx):
            if idx == len(actions):
                initial_guesses.append(current_guess.copy())
                return
            if idx == 0:
                # For the first action, i ranges from initial_time to 100, step initial_step
                for i in range(initial_time, 100, initial_step):
                    current_guess.append(i)
                    helper(current_guess, idx+1)
                    current_guess.pop()
            else:
                # For subsequent actions, ranges from previous value + 1 to previous value + time_gap
                prev_value = current_guess[-1]
                for i in range(prev_value + 1, prev_value + time_gap):
                    current_guess.append(i)
                    helper(current_guess, idx+1)
                    current_guess.pop()
        helper([], 0)
        return initial_guesses

    def optimize_action(num_groups_input, group_proportions_input, actions_input, wait_time_input, initial_guesses_input=None):
        global num_groups, group_proportions, actions, wait_time, initial_guesses
        num_groups = num_groups_input
        group_proportions = group_proportions_input
        actions = actions_input
        wait_time = wait_time_input
        if initial_guesses_input is None:
            initial_guesses = generate_initial_guesses(actions)
        else:
            initial_guesses = initial_guesses_input

        # Function to compute the reward based on the current particle positions
        def compute_reward(total_steps):
            # Run clustering to identify clusters
            d_local = d  # Use same distance threshold
            run_clustering(d_local)
            # Get the number of clusters detected
            num_detected_clusters = cluster_id
            # Get counts of particles in each cluster
            cluster_counts = []
            for cid in range(num_detected_clusters):
                indices = np.where(particle_node_assignments == cid)[0]
                count = len(indices)
                cluster_counts.append(count)
            # Compute the proportion of particles in each cluster
            cluster_proportions = [count / N for count in cluster_counts]
            # Build the cost matrix
            cost_matrix = np.zeros((num_detected_clusters, num_groups))
            for i in range(num_detected_clusters):
                for j in range(num_groups):
                    cost_matrix[i][j] = abs(cluster_proportions[i] - group_proportions[j])
            # Solve the assignment problem
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_ind, col_ind].sum()
            # Compute reward
            reward = (-1 * total_steps) - 100000 * total_cost
            return reward, total_cost

        # Function to run the simulation with given action times and actions
        def run_simulation_with_actions(action_times, actions, render=render_bool):
            global pos, vel, kx_field, ky_field, time_step, paused, running, key_press_data, cluster_id, particle_node_assignments
            # Reset simulation state
            initialize()
            kx_field[0] = kx[0] = 1  # Start with kx = 1
            time_step = 0
            total_steps = 0
            action_taken_flags = [False] * len(action_times)
            action_times = [int(t) for t in action_times]  # Ensure action times are integers
            max_action_time = max(action_times)
            last_action_time = max_action_time
            sim_running = True
            while sim_running:
                if not paused:
                    for _ in range(substepping):
                        compute_force()
                        update()
                        compute_energy()
                        calc_neighbors()
                        run_clustering(d)
                        update_average_neighbors()
                        update_neighbor_count_over_time()
                        calculate_weighted_avg_density()
                        cluster_aspect_ratios = calculate_aspect_ratios()

                    time_step += 1
                    total_steps += 1

                    # Apply actions if it's time and action hasn't been taken yet
                    for idx, action_time in enumerate(action_times):
                        if not action_taken_flags[idx] and time_step >= action_time:
                            kx_field[0] = kx[0] = actions[idx]  # Change kx to the action value
                            action_taken_flags[idx] = True
                            if idx == len(action_times) - 1:
                                last_action_time = time_step  # Record the time of last action

                    # If all actions have been taken, check if a certain number of steps have passed since last action
                    if all(action_taken_flags) and time_step >= last_action_time + wait_time:
                        # Compute reward
                        reward, total_cost = compute_reward(total_steps)
                        print(f"Actions taken at steps {action_times}: Reward = {reward}")
                        return reward, total_cost

                    # If total_steps exceeds some limit, stop simulation (prevent infinite loop)
                    if total_steps > 10000:
                        reward, total_cost = compute_reward(total_steps)
                        print(f"Simulation exceeded maximum steps. Reward = {reward}")
                        return reward, total_cost

                    # Render the simulation if required
                    if render:
                        render_simulation(cluster_aspect_ratios)
                        # Process events
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                sim_running = False
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    sim_running = False
                        clock.tick(60)  # Limit to 60 FPS

        # Gradient descent optimization function
        best_reward = None
        best_action_times = None

        print("Evaluating initial guesses...")
        for guess in initial_guesses:
            # guess is a list of action times
            action_times = guess
            reward, total_cost = run_simulation_with_actions(action_times, actions, render=render_bool)
            print(f"Action times {action_times}: Reward = {reward}, Total Cost = {total_cost:.4f}")
            if best_reward is None or reward > best_reward:
                best_reward = reward
                best_action_times = action_times.copy()

        # Use the best initial guess for gradient descent
        action_times = best_action_times.copy()
        previous_reward = best_reward

        print("\nStarting gradient descent optimization...")
        while True:
            improved = False
            for i in range(len(action_times)):
                current_time = action_times[i]
                # Try moving backward
                new_time_backward = max(1, current_time - learning_rate)
                new_action_times = action_times.copy()
                new_action_times[i] = new_time_backward
                reward_backward, total_cost_backward = run_simulation_with_actions(new_action_times, actions, render=render_bool)
                # Try moving forward
                new_time_forward = current_time + learning_rate
                new_action_times_forward = action_times.copy()
                new_action_times_forward[i] = new_time_forward
                reward_forward, total_cost_forward = run_simulation_with_actions(new_action_times_forward, actions, render=render_bool)
                # Decide which direction to move
                if reward_backward > previous_reward:
                    action_times[i] = new_time_backward
                    previous_reward = reward_backward
                    improved = True
                    print(f"Action time at index {i} updated to {new_time_backward} (earlier)")
                elif reward_forward > previous_reward:
                    action_times[i] = new_time_forward
                    previous_reward = reward_forward
                    improved = True
                    print(f"Action time at index {i} updated to {new_time_forward} (later)")
            if not improved:
                # No further improvement
                break

        print("\nOptimal action times found.")
        print(f"Final action times {action_times}: Reward = {previous_reward}")

        # Run the simulation one more time with rendering
        reward, total_cost = run_simulation_with_actions(action_times, actions, render=True)
        print(f"Final total cost: {total_cost:.4f}")

        # Output the key presses similar to 'write' mode
        key_press_data.clear()
        # Record key presses for 'x' and number corresponding to action at the action times
        for action_time, action in zip(action_times, actions):
            # Record 'x' key press
            key_press_data.append((action_time, 'x'))
            # Record number key press corresponding to 'action'
            key_name = str(action)
            key_press_data.append((action_time + 1, key_name))

        # Save key_press_data to file
        with open('key_press_data.txt', 'w') as f:
            json.dump(key_press_data, f)
        print("Optimized key press data saved to key_press_data.txt")
        print(f"Optimal action times: {action_times}")

    # Call the optimize_action function with desired parameters
    num_groups = 3  # Number of clusters to split the particles into
    group_proportions = [1/3, 1/3, 1/3]  # Desired proportions for each group, must sum to 1
    actions = [2, 3]  # Actions: changes to the number of nodes in x-direction
    wait_time = 50

    optimize_action(num_groups, group_proportions, actions, wait_time)

# Plotting functions
def plot_average_neighbors_over_time():
    plt.figure(figsize=(10, 6))
    plt.plot(average_neighbors_over_time)
    plt.title('Average Neighbors Over Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Average Neighbors')
    # Plot vertical lines at key press indices adjusted for substepping
    if key_press_data_indices:
        for idx in key_press_data_indices:
            plt.axvline(x=idx * substepping, color='red', linestyle='--')
    plt.show()

def plot_neighbor_distribution_over_time():
    plt.figure(figsize=(10, 6))
    for i in range(7):
        plt.plot(neighbor_count_over_time[i], label=f'{i} neighbors')
    plt.title('Neighbor Distribution Over Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Count')
    plt.legend()
    # Plot vertical lines at key press indices adjusted for substepping
    if key_press_data_indices:
        for idx in key_press_data_indices:
            plt.axvline(x=idx * substepping, color='red', linestyle='--')
    plt.show()

def plot_weighted_avg_density_over_time():
    plt.figure(figsize=(10, 6))
    plt.plot(weighted_avg_density_over_time)
    plt.title('Weighted Average Cluster Density Over Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Weighted Avg Density')
    # Plot vertical lines at key press indices adjusted for substepping
    if key_press_data_indices:
        for idx in key_press_data_indices:
            plt.axvline(x=idx * substepping, color='red', linestyle='--')
    plt.show()

def plot_weighted_avg_aspect_ratio_over_time():
    plt.figure(figsize=(10, 6))
    plt.plot(weighted_avg_aspect_ratio_over_time)
    plt.title('Weighted Average Aspect Ratio Over Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Weighted Avg Aspect Ratio')
    # Plot vertical lines at key press indices adjusted for substepping
    if key_press_data_indices:
        for idx in key_press_data_indices:
            plt.axvline(x=idx * substepping, color='red', linestyle='--')
    plt.show()

# End of ML code
####################################################################################

# Main simulation loop
if mode != 'simulation':
    running = True
    while running:
        if mode == 'write':
            # In 'write' mode, capture key presses
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key_name = pygame.key.name(event.key)
                    key_press_data.append((time_step, key_name))
                    process_key_press(event.key)
        elif mode == 'read':
            # In 'read' mode, replay recorded key presses
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            # Process virtual key presses from key_press_data
            while key_press_index < len(key_press_data) and key_press_data[key_press_index][0] == time_step:
                key_name = key_press_data[key_press_index][1]
                key = pygame.key.key_code(key_name)
                process_key_press(key)
                if key_name == 'g':
                    running = False
                key_press_index += 1

        if not paused:
            for _ in range(substepping):
                compute_force()
                update()
                compute_energy()
                calc_neighbors()
                run_clustering(d)
                update_average_neighbors()
                update_neighbor_count_over_time()
                calculate_weighted_avg_density()
                cluster_aspect_ratios = calculate_aspect_ratios()

            time_step += 1  # Increment the time step after each simulation step

            total_clusters = cluster_id

            assignments = particle_node_assignments.copy()

            unassigned_mask = assignments == -1

            assignments_unassigned = assignments.copy()
            assignments_unassigned[unassigned_mask] = total_clusters

            total_clusters_with_unassigned = total_clusters + 1

            counts = np.bincount(assignments_unassigned.astype(np.int32), minlength=total_clusters_with_unassigned)

        # Clear the screen
        window.fill((17, 47, 65))  # Dark background

        # Render the simulation
        render_simulation(cluster_aspect_ratios)

        clock.tick(30)  # Limit to 30 FPS

    # When the simulation ends, ensure the video writer is released
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print(f"Video saved as {video_filename}")

    pygame.quit()
    sys.exit()
