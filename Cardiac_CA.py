import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d

# --- Parameters ---
GRID_WIDTH = 1000
GRID_HEIGHT = 1000
TIME_STEPS = 100
REFRACTORY_PERIOD = 4  
EXCITATION_THRESHOLD = 1 


# --- States ---
RESTING = 0
EXCITED = 1


# --- Initialization ---
grid = np.full((GRID_HEIGHT, GRID_WIDTH), RESTING, dtype=int)

# Initial stimulus
# stim_size = 3
# center_y, center_x = GRID_HEIGHT // 2, GRID_WIDTH // 2
# grid[center_y - stim_size//2 : center_y + stim_size//2 + 1,
#      center_x - stim_size//2 : center_x + stim_size//2 + 1] = EXCITED

if GRID_HEIGHT > 0: 
     grid[GRID_HEIGHT - 1, :] = EXCITED

# --- Moore Neighborhood Kernel ---
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# kernel = np.array([[1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1],
#                     [1, 1, 0, 1, 1],
#                     [1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1]])



# --- Update Function ---
def update_grid(current_grid):
    new_grid = current_grid.copy() 

    is_excited_map = (current_grid == EXCITED).astype(int)
    excited_neighbor_count = convolve2d(is_excited_map, kernel, mode='same', boundary='fill', fillvalue=0)

    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            current_state = current_grid[r, c]
            neighbors_excited = excited_neighbor_count[r, c]

            #RULE 1: If the cell is in the resting state and has at least one excited neighbor, it becomes excited.
            if current_state == RESTING:
                if neighbors_excited >= EXCITATION_THRESHOLD:
                    new_grid[r, c] = EXCITED
            # RULE 2: If the cell is excited, it becomes refractory.
            elif current_state == EXCITED:
                new_grid[r, c] = 2 
            # RULE 3: If the cell is in the refractory period, it becomes resting after the refractory period.
            elif current_state >= 2: 
                if current_state < REFRACTORY_PERIOD + 1:
                    new_grid[r, c] += 1 
                else: 
                    new_grid[r, c] = RESTING 

    return new_grid

# --- Visualization ---
colors = ['white', 'red'] + ['blue'] * REFRACTORY_PERIOD
cmap = ListedColormap(colors)
bounds = list(range(REFRACTORY_PERIOD + 3))
norm = plt.Normalize(vmin=0, vmax=REFRACTORY_PERIOD + 1)

fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
ax.set_title("Cardiac Propagation Automata")
plt.xticks([])
plt.yticks([])

# --- Animation Function ---
def animate(frame_num):
    global grid
    grid = update_grid(grid)
    img.set_array(grid)
    ax.set_title("Cardiac Propagation")
    return [img]

# --- Run Simulation ---
ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=10, blit=True)
plt.tight_layout()
plt.show()

