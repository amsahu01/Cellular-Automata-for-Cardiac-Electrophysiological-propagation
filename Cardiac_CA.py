"""
Cellular Automaton for Cardiac Electrophysiological Propagation.

This script simulates a 2D cellular automaton model representing
the propagation of an electrical wave in cardiac tissue. The model
includes resting, excited, and multi-stage refractory states.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d
import time # For potential performance checks, not used by default in this version

# --- Parameters ---
GRID_WIDTH = 1000
GRID_HEIGHT = 1000
TIME_STEPS = 100
REFRACTORY_PERIOD = 10
EXCITATION_THRESHOLD = 1

# --- States ---
RESTING = 0
EXCITED = 1
# Refractory states will be integers from 2 up to REFRACTORY_PERIOD + 1

# --- Moore Neighborhood Kernel ---
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# Alternative larger kernel (commented out)
# kernel = np.array([[1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 0, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1]])

def initialize_grid(grid_height, grid_width, stim_type="center", stim_size=3):
    """
    Initializes the grid with all cells in the RESTING state and applies an initial stimulus.

    Args:
        grid_height (int): The height of the grid.
        grid_width (int): The width of the grid.
        stim_type (str, optional): Type of stimulus. "center" for a central square,
                                   "bottom_row" for the entire bottom row,
                                   "bottom_left_corner" for a square at the bottom-left.
                                   Defaults to "center".
        stim_size (int, optional): Size of the square stimulus if type is "center"
                                   or "bottom_left_corner". Defaults to 3.

    Returns:
        np.ndarray: The initialized grid.

    Raises:
        ValueError: If grid dimensions are not positive or stimulus parameters are invalid.
    """
    if not (grid_height > 0 and grid_width > 0):
        raise ValueError("Grid height and width must be positive integers.")

    grid = np.full((grid_height, grid_width), RESTING, dtype=int)

    if stim_type == "center":
        if not (stim_size > 0 and stim_size <= grid_height and stim_size <= grid_width):
            raise ValueError("Stimulus size is invalid or too large for the grid.")
        center_y, center_x = grid_height // 2, grid_width // 2
        grid[center_y - stim_size//2 : center_y + stim_size//2 + (stim_size % 2), 
             center_x - stim_size//2 : center_x + stim_size//2 + (stim_size % 2)] = EXCITED
    elif stim_type == "bottom_row":
        grid[grid_height - 1, :] = EXCITED
    elif stim_type == "bottom_left_corner": 
        if not (stim_size > 0 and stim_size <= grid_height and stim_size <= grid_width):
            raise ValueError("Stimulus size for 'bottom_left_corner' is invalid or too large for the grid.")
        grid[grid_height - stim_size : grid_height, 0 : stim_size] = EXCITED
    else:
        print(f"Warning: Unknown stimulus type '{stim_type}'. Grid initialized without active stimulus cells.")

    return grid
    
def update_grid_vectorized(current_grid, refractory_period_val, excitation_threshold_val):
    """
    Updates the cellular automaton grid for one time step using vectorized operations.

    Applies rules for resting, excited, and refractory states based
    on the current state of each cell and its excited neighbors.

    Args:
        current_grid (np.ndarray): The 2D NumPy array representing the
                                   current state of the grid.
        refractory_period_val (int): The duration of the refractory period.
        excitation_threshold_val (int): The minimum number of excited neighbors
                                       to excite a resting cell.

    Returns:
        np.ndarray: The 2D NumPy array representing the grid state
                    at the next time step.
    """
    new_grid = current_grid.copy()

    # 1. Identify cells that are currently EXCITED (state 1) for convolution
    is_excited_map = (current_grid == EXCITED).astype(int)

    # 2. Count excited neighbors for all cells using convolution
    excited_neighbor_count = convolve2d(is_excited_map, kernel, mode='same', boundary='fill', fillvalue=0)

    # --- Apply rules using Boolean masks (vectorized) ---

    # RULE 1: Resting cells that meet the excitation threshold become excited
    resting_cells_mask = (current_grid == RESTING)
    can_be_excited_mask = (excited_neighbor_count >= excitation_threshold_val)
    to_excite_mask = resting_cells_mask & can_be_excited_mask
    new_grid[to_excite_mask] = EXCITED

    # RULE 2: Excited cells become refractory (start at state 2)
    # We must ensure we are only affecting cells that were EXCITED in the *current_grid*
    # and not those that just became excited (to_excite_mask)
    excited_in_current_grid_mask = (current_grid == EXCITED)
    new_grid[excited_in_current_grid_mask] = 2  

    # RULE 3: Refractory cells progress or become resting
    # Cells currently in any refractory state (>= 2)
    currently_refractory_mask = (current_grid >= 2)

    # Identify refractory cells that need to increment their state
    progressing_refractory_mask = currently_refractory_mask & (current_grid < refractory_period_val + 1)
    new_grid[progressing_refractory_mask] = current_grid[progressing_refractory_mask] + 1

    # Identify refractory cells that have completed their refractory period and become resting
    completed_refractory_mask = currently_refractory_mask & (current_grid == refractory_period_val + 1)
    new_grid[completed_refractory_mask] = RESTING

    return new_grid


def create_animation_elements(initial_grid, refractory_period_val):
    """
    Sets up the Matplotlib figure, axes, and initial image for the animation.

    Args:
        initial_grid (np.ndarray): The starting grid state.
        refractory_period_val (int): The duration of the refractory period
                                    (used for colormap setup).

    Returns:
        tuple: (fig, ax, img) Matplotlib figure, axes, and image objects.
    """
    colors = ['white', 'red'] + ['blue'] * refractory_period_val
    cmap = ListedColormap(colors)
    max_state_value = refractory_period_val + 1
    bounds = list(range(max_state_value + 2)) 
    norm = plt.Normalize(vmin=RESTING, vmax=max_state_value)

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(initial_grid, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title("Cardiac Propagation Automata")
    plt.xticks([])
    plt.yticks([])
    return fig, ax, img

def animate_step(frame_num, grid_ref, img_ref, ax_ref, refractory_p, excitation_t):
    """
    Function called by FuncAnimation for each frame to update the animation.

    Args:
        frame_num (int): The current frame number (passed by FuncAnimation).
        grid_ref (list): A list containing the grid (used as a mutable reference).
        img_ref (matplotlib.image.AxesImage): The image object to update.
        ax_ref (matplotlib.axes.Axes): The axes object for setting the title.
        refractory_p (int): The refractory period value.
        excitation_t (int): The excitation threshold value.


    Returns:
        list: A list containing the updated image artist (required by FuncAnimation).
    """
    #print(f"Time step: {frame_num}") 
    grid_ref[0] = update_grid_vectorized(grid_ref[0], refractory_p, excitation_t)
    img_ref.set_array(grid_ref[0])
    ax_ref.set_title("Cardiac Propagation Automata")
    return [img_ref]

def main():
    """
    Main function to set up and run the cellular automaton simulation and animation.
    """
    # --- Parameter Assertions (Basic Error Handling) ---
    try:
        assert REFRACTORY_PERIOD >= 0, "REFRACTORY_PERIOD cannot be negative."
        assert EXCITATION_THRESHOLD > 0, "EXCITATION_THRESHOLD must be positive."
        
    except AssertionError as e:
        print(f"Parameter Error: {e}")
        return 
    # Initialize grid
    # To use bottom row stimulus: current_grid = initialize_grid(GRID_HEIGHT, GRID_WIDTH, stim_type="bottom_row")
    current_grid = initialize_grid(GRID_HEIGHT, GRID_WIDTH, stim_type="center", stim_size=15) 
    grid_container = [current_grid] # Use a list to pass grid by reference to animate_step

    # --- Visualization and Animation Setup ---
    fig, ax, img = create_animation_elements(grid_container[0], REFRACTORY_PERIOD)

    print(f"Starting animation with Excitation Threshold: {EXCITATION_THRESHOLD}, Refractory Period: {REFRACTORY_PERIOD}...")
    print(f"Grid size: {GRID_WIDTH}x{GRID_HEIGHT}. Total time steps: {TIME_STEPS}.")
    
    # Define the animation function
    animation_function = lambda frame: animate_step(frame, grid_container, img, ax, REFRACTORY_PERIOD, EXCITATION_THRESHOLD)

    ani = FuncAnimation(fig, animation_function,
                        frames=TIME_STEPS, interval=10, blit=True)

    plt.tight_layout()
    plt.show()

    print("Simulation finished.")

if __name__ == "__main__":
    main()
