# generate_map.py
import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 500
NUM_OBSTACLES = 40
OBSTACLE_MIN = 10
OBSTACLE_MAX = 40

def add_square_obstacle(grid):
    """Places a random square obstacle into the grid."""
    size = np.random.randint(OBSTACLE_MIN, OBSTACLE_MAX)
    x = np.random.randint(0, GRID_SIZE - size)
    y = np.random.randint(0, GRID_SIZE - size)
    grid[y:y+size, x:x+size] = 1  # mark cells as occupied
    return grid

def generate_map():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for _ in range(NUM_OBSTACLES):
        grid = add_square_obstacle(grid)

    np.save("map.npy", grid)
    print("Saved map.npy")

    plt.imshow(grid, cmap="gray_r")
    plt.title("Generated Map")
    plt.show()

if __name__ == "__main__":
    generate_map()
