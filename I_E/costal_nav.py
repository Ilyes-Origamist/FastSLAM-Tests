import numpy as np
import matplotlib.pyplot as plt

# --- NAVIGATION PARAMETERS ---
STEP_SIZE = 3
NUM_STEPS = 5000

# The robot's small sensing radius (still used for collision checking)
SENSOR_RADIUS = 10    

# Ray probe length (your chosen value)
PROBE_DIST = 13        # <<<<<<<<< KEY SETTING

TURN_ANGLE = np.deg2rad(6)  # smooth turning for coastal navigation


# -------------------------------------------------------------------------
# RAYCAST SENSOR FUNCTIONS
# -------------------------------------------------------------------------
def ray_probe(grid, x, y, dx, dy, max_dist):
    """
    Casts a ray from (x,y) in direction (dx,dy)
    up to max_dist pixels.
    Returns the distance until obstacle hit, 
    or max_dist if no obstacle.
    """
    for d in range(1, max_dist + 1):
        px = int(x + dx * d)
        py = int(y + dy * d)
        if px < 0 or px >= grid.shape[1] or py < 0 or py >= grid.shape[0]:
            return d
        if grid[py, px] == 1:
            return d
    return max_dist


def get_wall_distances(grid, x, y, dx, dy):
    """
    Returns distances to walls in FRONT, LEFT, RIGHT directions using ray probes.
    """

    # Forward probe
    d_fwd = ray_probe(grid, x, y, dx, dy, PROBE_DIST)

    # Left probe (rotate 90° left)
    left_dx = -dy
    left_dy = dx
    d_left = ray_probe(grid, x, y, left_dx, left_dy, PROBE_DIST)

    # Right probe (rotate 90° right)
    right_dx = dy
    right_dy = -dx
    d_right = ray_probe(grid, x, y, right_dx, right_dy, PROBE_DIST)

    return d_fwd, d_left, d_right


def random_direction():
    theta = np.random.rand() * 2 * np.pi
    return np.cos(theta), np.sin(theta), theta


def turn_left(dx, dy, theta):
    theta += TURN_ANGLE
    return np.cos(theta), np.sin(theta), theta


def turn_right(dx, dy, theta):
    theta -= TURN_ANGLE
    return np.cos(theta), np.sin(theta), theta


# -------------------------------------------------------------------------
# MAIN NAVIGATION LOGIC
# -------------------------------------------------------------------------
def simulate_robot(grid):

    # Place robot in a free cell
    while True:
        x = np.random.randint(0, grid.shape[1])
        y = np.random.randint(0, grid.shape[0])
        if grid[y, x] == 0:
            break

    dx, dy, theta = random_direction()
    mode = "EXPLORE"

    path_x = [x]
    path_y = [y]

    for step in range(NUM_STEPS):

        # --- RAYCAST WALL DISTANCES ---
        d_fwd, d_left, d_right = get_wall_distances(grid, x, y, dx, dy)

        # Trigger coastal mode if ANY wall is reasonably close
        if mode == "EXPLORE" and min(d_left, d_right, d_fwd) < PROBE_DIST * 0.8:
            mode = "COAST"

        # Switch back to explore mode if no walls nearby
        if mode == "COAST" and min(d_left, d_right, d_fwd) > PROBE_DIST * 0.95:
            mode = "EXPLORE"
            dx, dy, theta = random_direction()

        # ======================================================
        # EXPLORE MODE (random wandering + collision avoidance)
        # ======================================================
        if mode == "EXPLORE":
            nx = x + dx * STEP_SIZE
            ny = y + dy * STEP_SIZE

            # Out of bounds → randomize direction
            if nx < 0 or nx >= grid.shape[1] or ny < 0 or ny >= grid.shape[0]:
                dx, dy, theta = random_direction()
                continue

            # If forward ray detects obstacle too close → turn randomly
            if d_fwd < SENSOR_RADIUS + 2:
                dx, dy, theta = random_direction()
                continue

            x, y = nx, ny

        # ======================================================
        # COAST MODE (REAL WALL-FOLLOWING)
        # ======================================================
        elif mode == "COAST":

            # --- Forward wall too close → steer left ---
            if d_fwd < 6:
                dx, dy, theta = turn_left(dx, dy, theta)

            # --- Wall on left but clear on right → drift right
            elif d_left < d_right:
                dx, dy, theta = turn_right(dx, dy, theta)

            # --- Wall on right but clear on left → drift left
            elif d_right < d_left:
                dx, dy, theta = turn_left(dx, dy, theta)

            # Otherwise continue straight

            # Attempt movement
            nx = x + dx * STEP_SIZE
            ny = y + dy * STEP_SIZE

            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                x, y = nx, ny
            else:
                # if out of bounds, turn away and continue
                dx, dy, theta = turn_left(dx, dy, theta)

        # Record path
        path_x.append(x)
        path_y.append(y)

    return path_x, path_y


# -------------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------------
def main():
    grid = np.load("map.npy")

    path_x, path_y = simulate_robot(grid)

    plt.figure(figsize=(7, 7))
    plt.imshow(grid, cmap="gray_r")
    plt.plot(path_x, path_y, linewidth=0.7, color="blue")
    plt.scatter([path_x[0]], [path_y[0]], c="green", s=40, label="Start")
    plt.scatter([path_x[-1]], [path_y[-1]], c="red", s=40, label="End")
    plt.legend()
    plt.title("Ray-Probe Coastal Navigation Robot")
    plt.show()


if __name__ == "__main__":
    main()
