from __future__ import annotations
import numpy as np
from .constants import EXCITED


class LATTracker:
    """Track Local Activation Time (LAT).

    LAT is defined as the first time a cell enters the first excited substate
    (state == 1). Call :meth:`update_with_grid` once per simulation step, then
    convert stored steps to milliseconds with :meth:`lat_ms`.

    Parameters
    ----------
    height : int
        Grid height.
    width : int
        Grid width.
    """

    def __init__(self, height: int, width: int):
        self.lat_steps = np.full((height, width), np.inf, dtype=float)
        # Start at -1 so the first update stamps step 0.
        self.current_step = -1

    def update_with_grid(self, grid: np.ndarray) -> None:
        """Record LAT hits for the current frame.

        Parameters
        ----------
        grid : np.ndarray
            2‑D integer grid of states. Cells with ``grid == 1`` are treated
            as newly excited for LAT purposes.
        """
        self.current_step += 1
        newly_excited = grid == EXCITED
        mask = newly_excited & np.isinf(self.lat_steps)
        self.lat_steps[mask] = self.current_step

    def lat_ms(self, dt_ms: float) -> np.ndarray:
        """Return the LAT map in milliseconds.

        Cells that never activate are returned as ``np.nan``.

        Parameters
        ----------
        dt_ms : float
            Milliseconds per simulation step.

        Returns
        -------
        np.ndarray
            LAT map (float32/float64), with ``np.nan`` for never-activated
            cells.
        """
        lat = self.lat_steps.copy()
        lat[np.isinf(lat)] = np.nan
        return lat * float(dt_ms)
