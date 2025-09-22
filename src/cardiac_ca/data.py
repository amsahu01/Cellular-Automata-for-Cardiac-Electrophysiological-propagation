from __future__ import annotations
from typing import List
import numpy as np
from .constants import EXCITED_STAGES


class SimulationDataCollector:
    """Collect simple activity summaries during the run.

    Tracks counts per frame and a cumulative activation map. This is a minimal
    helper to generate quick plots (e.g. excited area vs. time) after the
    simulation.

    Parameters
    ----------
    grid_height : int
        Grid height.
    grid_width : int
        Grid width.
    """

    def __init__(self, grid_height: int, grid_width: int):
        self.time_points: List[int] = []
        self.excited_counts: List[int] = []
        self.total_activated: List[int] = []
        self.activated_map = np.full(
            (grid_height, grid_width), False, dtype=bool
        )

    def record_step(self, frame_num: int, current_grid_state: np.ndarray) -> None:
        """Record one timepoint of activity statistics.

        Parameters
        ----------
        frame_num : int
            Zero-based frame index.
        current_grid_state : np.ndarray
            2‑D integer grid of states at the current frame.
        """
        self.time_points.append(frame_num + 1)
        num_excited = np.sum(
            (current_grid_state >= 1)
            & (current_grid_state <= EXCITED_STAGES)
        )
        self.excited_counts.append(int(num_excited))
        self.activated_map[
            (current_grid_state >= 1)
            & (current_grid_state <= EXCITED_STAGES)
        ] = True
        self.total_activated.append(int(np.sum(self.activated_map)))