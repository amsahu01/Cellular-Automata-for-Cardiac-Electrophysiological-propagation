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

# --- Import from our analysis module ---
from ca_analysis import SimulationDataCollector


# --- States ---
RESTING = 0
EXCITED = 1
# Refractory states will be integers from 2 up to REFRACTORY_PERIOD + 1

# --- Moore Neighborhood Kernel ---
# kernel = np.array([[1, 1, 1],
#                    [2, 0, 2],
#                    [1, 1, 1]])

kernel = np.array([[1, 1, 0.8, 1, 1],
                   [1, 1, 0.8, 1, 1],
                   [1.5, 1.5, 0, 1.5, 1.5],
                   [1, 1, 0.8, 1, 1],
                   [1, 1, 0.8, 1, 1]], dtype=float)



def initialize_grid(grid_height, grid_width, stim_type="center", stim_size=3):
    """
    Initializes the grid with all cells in the RESTING state and applies an initial stimulus.

    Args:
        grid_height (int): The height of the grid.
        grid_width (int): The width of the grid.
        stim_type (str, optional): Type of stimulus. "center" for a central square,
                                   "bottom_row" for the entire bottom row,
                                   "bottom_left_corner" for a square at the bottom-left,
                                   "bottom_center" for a square at the center of the bottom edge.
                                   Defaults to "center".
        stim_size (int, optional): Size of the square stimulus if type is "center",
                                   "bottom_left_corner", or "bottom_center". Defaults to 3.

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
    elif stim_type == "bottom_center":
        if not (stim_size > 0 and stim_size <= grid_height and stim_size <= grid_width):
            raise ValueError("Stimulus size for 'bottom_center' is invalid or too large for the grid.")
        bottom_y = grid_height - stim_size
        center_x = grid_width // 2
        grid[bottom_y : grid_height, 
             center_x - stim_size//2 : center_x + stim_size//2 + (stim_size % 2)] = EXCITED
    else:
        print(f"Warning: Unknown stimulus type '{stim_type}'. Grid initialized without active stimulus cells.")

    return grid
  
def update_grid(current_grid, refractory_period_val, excitation_threshold_val):
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

def animate_step(frame_num, grid_ref, img_ref, ax_ref,
                  excitation_t, refractory_p, data_collector_obj,
                  fig_to_save_from, screenshot_time_steps=None): # Modified args
    """
    Updates the simulation for a single time step and refreshes the animation display.

    This function is called by `matplotlib.animation.FuncAnimation` for each frame
    of the animation. It performs the following actions:
    1. Retrieves the current state of the simulation grid.
    2. Calls `update_grid` to compute the grid state for the next time step.
    3. Updates the reference to the grid with the new state.
    4. Updates the data displayed by the `matplotlib.image.AxesImage` object.
    5. Updates the title of the plot to show the current time step and parameters.
    6. If a `SimulationDataCollector` object is provided, it calls its `record_step`
       method to store data about the current state.
    7. If `screenshot_time_steps` are specified and the current frame number
       (plus one, as frames are 0-indexed) is in this list, it saves a
       screenshot of the current animation frame.

    Args:
        frame_num (int): The current frame number (time step), provided by
                         `FuncAnimation`. This is 0-indexed.
        grid_ref (list): A list containing a single element: the 2D NumPy array
                         representing the current state of the simulation grid.
                         This is passed as a list to allow modification within
                         the function (pass-by-reference behavior for the grid).
        img_ref (matplotlib.image.AxesImage): The `AxesImage` object (from `imshow`)
                                              that displays the grid. This function
                                              will update its data.
        ax_ref (matplotlib.axes.Axes): The Matplotlib `Axes` object on which the
                                       image is displayed. Used to update the title.
        excitation_t (int): The excitation threshold for cells.
        refractory_p (int): The refractory period duration for cells.
        data_collector_obj (SimulationDataCollector or None): An instance of the
            `SimulationDataCollector` class used to record data at each step.
            If `None`, data collection is skipped.
        fig_to_save_from (matplotlib.figure.Figure): The Matplotlib `Figure` object
            from which screenshots will be saved.
        screenshot_time_steps (list of int, optional): A list of 1-indexed time
            steps at which to save a screenshot of the animation.
            Defaults to `None`.

    Returns:
        list: A list containing the updated `AxesImage` object (`[img_ref]`).
              This is required by `FuncAnimation` for blitting.
    """
    current_grid = grid_ref[0]
    new_grid_state = update_grid(current_grid, refractory_p, excitation_t)
    grid_ref[0] = new_grid_state

    img_ref.set_array(new_grid_state)
    ax_ref.set_title(f"Cardiac Propagation - Time: {frame_num + 1} (ET:{excitation_t} RP:{refractory_p})")

    # --- Trigger Data Collection via the collector object ---
    if data_collector_obj:
        data_collector_obj.record_step(frame_num, new_grid_state)

    # --- Save Screenshot at specific time steps ---
    if screenshot_time_steps and (frame_num + 1) in screenshot_time_steps:
        screenshot_filename = f"simulation_step_{frame_num + 1}.png"
        fig_to_save_from.savefig(screenshot_filename)
        print(f"Saved screenshot: {screenshot_filename}")

    return [img_ref]


def main():
    """
    Main function to set up and run the cellular automaton simulation.

    This function orchestrates the entire simulation process:
    1.  Defines simulation parameters: grid dimensions, number of time steps,
        refractory period, excitation threshold, stimulus type, and stimulus size.
    2.  Defines time steps at which to save screenshots of the simulation.
    3.  Collects simulation parameters into a dictionary for later use (e.g., plotting).
    4.  Performs basic assertions to ensure parameter validity.
    5.  Initializes the simulation grid using `initialize_grid`.
    6.  Initializes a `SimulationDataCollector` to store data at each step.
    7.  Sets up the Matplotlib animation elements (figure, axes, image) using
        `create_animation_elements`.
    8.  Creates a `FuncAnimation` object to run the simulation, updating the grid
        and display at each time step via the `animate_step` function.
    9.  Displays the animation.
    10. After the animation completes, it calls the data collector's `plot_results`
        method to generate and display quantitative analysis plots.

    The simulation models the propagation of an electrical wave in cardiac tissue,
    visualizing cell states (resting, excited, refractory) over time.
    """
    grid_width = 300
    grid_height = 300
    time_steps = 150
    refractory_period = 4
    excitation_threshold = 3
    stimulus_type = "center"  # Options: "bottom_row", "center", "bottom_left_corner"
    stimulus_size = 5
   


     # --- Define at which time steps you want screenshots (1-indexed) ---
    SCREENSHOT_AT_STEPS = [1, 40, 80] # Example: save at these time steps

    simulation_params = {
        "ET": excitation_threshold, "RP": refractory_period,
        "stim_type": stimulus_type, "stim_size": stimulus_size,
        "grid_W": grid_width, "grid_H": grid_height
    }
    # --- Parameter Assertions (Basic Error Handling) ---
    try:
        assert grid_width > 0 and grid_height > 0
        assert refractory_period >= 0 and excitation_threshold > 0 and time_steps > 0
    except AssertionError as e:
        print(f"Parameter Error: One or more basic assertions failed for parameters. {e}")
        return

    print(f"Starting simulation with: Size={grid_width}x{grid_height}, Time Steps={time_steps}") 

    try:
        current_grid = initialize_grid(grid_height, grid_width,
                                       stim_type=stimulus_type, stim_size=stimulus_size)
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return

    if SCREENSHOT_AT_STEPS:
        print(f"Screenshots will be saved at time steps: {SCREENSHOT_AT_STEPS}")

    grid_list_container = [current_grid]
    # --- Initialize Data Collector ---
    data_collector = SimulationDataCollector(grid_height, grid_width)

    fig_anim, ax_anim, img_anim = create_animation_elements(grid_list_container[0], refractory_period)

    anim_func = lambda fn: animate_step(fn, grid_list_container, img_anim, ax_anim,
                                         excitation_threshold, refractory_period,
                                         data_collector, fig_anim, SCREENSHOT_AT_STEPS) 

    ani = FuncAnimation(fig_anim, anim_func, frames=time_steps, interval=10, blit=True, repeat=False)
    fig_anim.tight_layout()
    plt.show()

    print("Animation finished. Plotting quantitative results...")

    # --- Plot Quantitative Results using the collector ---
    data_collector.plot_results(params=simulation_params)

    print("Simulation completely finished.")

if __name__ == "__main__":
    main()
