# ca_analysis.py

import numpy as np
import matplotlib.pyplot as plt

class SimulationDataCollector:
    """
    A class to collect and store data during a cellular automaton simulation.
    """
    def __init__(self, grid_height, grid_width):
        self.time_points = []
        self.excited_counts = []
        self.total_activated = []
        self.activated_map = np.full((grid_height, grid_width), False, dtype=bool)

    def record_step(self, frame_num, current_grid_state):
        """
        Records data for the current simulation step.

        Args:
            frame_num (int): The current frame number (0-indexed).
            current_grid_state (np.ndarray): The state of the grid after the update.
        """
        self.time_points.append(frame_num + 1) 

        num_excited = np.sum(current_grid_state == 1) 
        self.excited_counts.append(num_excited)

        # Update activated map: mark newly excited cells or any cell currently excited
        self.activated_map[current_grid_state == 1] = True 
        current_total_activated = np.sum(self.activated_map)
        self.total_activated.append(current_total_activated)

    def plot_results(self, params=None):
        """
        Plots the collected quantitative data after the simulation.

        Args:
            params (dict, optional): Dictionary of simulation parameters for plot titles.
                                     Defaults to None for generic titles.
        """
        if not self.time_points:
            print("No data collected to plot.")
            return

        if params is None: 
            params = {"ET": "N/A", "RP": "N/A", "stim_type": "N/A", "stim_size": "N/A"}


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot 1: Number of Excited Cells vs. Time
        ax1.plot(self.time_points, self.excited_counts, marker='.', linestyle='-', color='r')
        ax1.set_ylabel("Number of Excited Cells")
        ax1.set_title(f"Excited Cells Over Time\n"
                      f"(ET:{params.get('ET', 'N/A')}, RP:{params.get('RP', 'N/A')}, "
                      f"Stim:{params.get('stim_type', 'N/A')}/{params.get('stim_size', 'N/A')})")
        ax1.grid(True)

        # Plot 2: Total Activated Area vs. Time
        ax2.plot(self.time_points, self.total_activated, marker='.', linestyle='-', color='b')
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Total Cells Activated")
        ax2.set_title(f"Cumulative Activated Area Over Time")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

