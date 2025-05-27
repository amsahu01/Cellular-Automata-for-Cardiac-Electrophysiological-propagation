# Cellular-Automata-for-Cardiac-Electrophysiological-propagation

This Python script simulates a 2D cellular automaton model representing the propagation of an electrical wave, often used as a simplified model for cardiac tissue excitability. The model features cells with resting, excited, and multi-stage refractory states.

## Features

*   **2D Grid:** Simulates propagation on a configurable 2D grid.
*   **Cell States:**
    *   **Resting (0):** Cell is excitable.
    *   **Excited (1):** Cell is active and can excite its neighbors.
    *   **Refractory (2 to N):** Cell is in a recovery phase and cannot be re-excited. The duration is configurable.
*   **Moore Neighborhood:** Each cell considers its 8 immediate neighbors (horizontal, vertical, and diagonal).
*   **Configurable Parameters:**
    *   Grid dimensions (`GRID_WIDTH`, `GRID_HEIGHT`)
    *   Number of simulation time steps (`TIME_STEPS`)
    *   Duration of the refractory period (`REFRACTORY_PERIOD`)
    *   Excitation threshold (`EXCITATION_THRESHOLD`): Minimum number of excited neighbors required to excite a resting cell.
*   **Initial Stimulus:**
    *   Can be a central square of excited cells.
    *   Can be the entire bottom row of excited cells.
*   **Vectorized Update Logic:** The core cell state update rules are implemented using NumPy vectorization for improved performance on large grids.
*   **Visualization:** Uses Matplotlib to animate the wave propagation in real-time.
    *   White: Resting cells
    *   Red: Excited cells
    *   Blue: Refractory cells

## Requirements

To run this script, you need:

1.  **Python 3.x**
2.  **NumPy:** For numerical operations and array manipulation.
    ```bash
    pip install numpy
    ```
3.  **Matplotlib:** For plotting and animation.
    ```bash
    pip install matplotlib
    ```
4.  **SciPy:** For the `convolve2d` function used in neighbor counting.
    ```bash
    pip install scipy
    ```

You can install all of them at once using:
```bash
pip install numpy matplotlib scipy
