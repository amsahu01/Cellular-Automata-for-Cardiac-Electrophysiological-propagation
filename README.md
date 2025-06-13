# Cellular Automaton for Cardiac Electrophysiological Propagation

This Python script simulates a 2D cellular automaton model representing the propagation of an electrical wave, often used as a simplified model for cardiac tissue excitability. The model features cells with resting, excited, and multi-stage refractory states, and has been optimized for performance and code structure.

## Features

*   **2D Grid:** Simulates propagation on a configurable 2D grid.
*   **Cell States:**
    *   **Resting (0):** Cell is excitable.
    *   **Excited (1):** Cell is active and can excite its neighbors.
    *   **Refractory (2 to N):** Cell is in a recovery phase and cannot be re-excited. The duration (`REFRACTORY_PERIOD`) is configurable.
*   **Moore Neighborhood:** Each cell considers its 8 immediate neighbors (horizontal, vertical, and diagonal) for excitation counting.
*   **Configurable Parameters:**
    *   Grid dimensions (`GRID_WIDTH`, `GRID_HEIGHT`)
    *   Number of simulation time steps (`TIME_STEPS`)
    *   Duration of the refractory period (`REFRACTORY_PERIOD`)
    *   Excitation threshold (`EXCITATION_THRESHOLD`): Minimum number of excited neighbors required to excite a resting cell.
*   **Versatile Initial Stimulus (via `initialize_grid` function):**
    *   `"center"`: A square stimulus of configurable size at the grid's center.
    *   `"bottom_row"`: Excites the entire bottom row of the grid.
    *   `"bottom_left_corner"`: Excites a square stimulus of configurable size at the bottom-left corner of the grid.
*   **Vectorized Update Logic (`update_grid`):** The core cell state update rules are implemented using NumPy vectorization (boolean array indexing and array operations) for significantly improved performance on large grids compared to explicit Python loops.
*   **Structured Code:**
    *   Follows standard Python practices, including a `main()` function and `if __name__ == "__main__":` guard for better modularity and reusability.
    *   Dedicated functions for grid initialization, state updates, and visualization setup.
    *   Comprehensive docstrings for functions and the module, aiding understanding and potential automated documentation generation.
*   **Real-time Visualization:** Uses Matplotlib to animate the wave propagation.
    *   White: Resting cells
    *   Red: Excited cells
    *   Blue: Refractory cells
*   **Basic Error Handling:** Includes initial checks for parameter validity and stimulus setup.

## Requirements

To run this script, you need:

1.  **Python 3.x** (developed and tested with Python 3.7+)
2.  **NumPy:** For numerical operations and array manipulation.
    ```bash
    pip install numpy
    ```
3.  **Matplotlib:** For plotting and animation.
    ```bash
    pip install matplotlib
    ```
4.  **SciPy:** For the `convolve2d` function used in efficient neighbor counting.
    ```bash
    pip install scipy
    ```

You can install all of them at once using:
```bash
pip install numpy matplotlib scipy
