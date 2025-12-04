# randomex.py (reactive robot with SLAM data output)

import numpy as np
import matplotlib.pyplot as plt

STEP_SIZE = 10
NUM_STEPS = 5000
SENSOR_RADIUS = 5     # 50×50 patch around robot


def sense_local_patch(grid, x, y):
    """Returns the 50×50 sensory patch centered on the robot."""
    x = int(x)
    y = int(y)
    x0 = max(0, x - SENSOR_RADIUS)
    x1 = min(grid.shape[1], x + SENSOR_RADIUS)
    y0 = max(0, y - SENSOR_RADIUS)
    y1 = min(grid.shape[0], y + SENSOR_RADIUS)

    patch = grid[y0:y1, x0:x1]
    return patch


def sense_local_obstacles(grid, x, y):
    """Returns True if any obstacle is inside the sensory patch."""
    patch = sense_local_patch(grid, x, y)
    return np.any(patch == 1)


def random_direction():
    theta = np.random.rand() * 2 * np.pi
    return np.cos(theta), np.sin(theta), theta


def simulate_robot(grid):
    # Start in free space
    while True:
        x = np.random.randint(0, grid.shape[1])
        y = np.random.randint(0, grid.shape[0])
        if grid[y, x] == 0:
            break

    # Initial direction
    dx_dir, dy_dir, theta = random_direction()

    # SLAM data log
    slam_data = []

    path_x = [x]
    path_y = [y]

    for _ in range(NUM_STEPS):

        # Proposed new location
        nx = x + dx_dir * STEP_SIZE
        ny = y + dy_dir * STEP_SIZE

        # Out of bounds → pick new direction
        if nx < 0 or nx >= grid.shape[1] or ny < 0 or ny >= grid.shape[0]:
            dx_dir, dy_dir, theta = random_direction()
            continue

        # If obstacles detected in future patch → turn randomly
        if sense_local_obstacles(grid, nx, ny):
            dx_dir, dy_dir, theta = random_direction()
            continue

        # Compute SLAM motion model values
        dx = (dx_dir * STEP_SIZE)
        dy = (dy_dir * STEP_SIZE)
        v = STEP_SIZE  # constant speed per timestep

        # Apply movement
        x = nx
        y = ny

        # Get sensory patch
        sensor_patch = sense_local_patch(grid, x, y)

        # Log SLAM data for this timestep
        slam_data.append({
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "theta": theta,
            "v": v,
            "sensor": sensor_patch.copy()   # 50×50 patch
        })

        path_x.append(x)
        path_y.append(y)

    return path_x, path_y, slam_data


def main():
    grid = np.load("map.npy")

    path_x, path_y, slam_data = simulate_robot(grid)

    # ---- Visualization ----
    plt.imshow(grid, cmap="gray_r")
    plt.plot(path_x, path_y, linewidth=0.7)
    plt.scatter([path_x[0]], [path_y[0]], c="green", s=40, label="Start")
    plt.scatter([path_x[-1]], [path_y[-1]], c="red", s=40, label="End")
    plt.legend()
    plt.title("Reactive Robot Exploration + SLAM Data Output")
    plt.show()

    # Print example of SLAM data
    print("\nExample SLAM timestep:")
    print({k: (v if not isinstance(v, np.ndarray) else "50x50 patch")
           for k, v in slam_data[0].items()})

    return slam_data


if __name__ == "__main__":
    slam_output = main()
