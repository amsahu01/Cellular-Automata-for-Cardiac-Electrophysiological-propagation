# /Users/amansahu/Downloads/Cellular-Automata/NEW_CA/src/cardiac_ca/apd.py
from __future__ import annotations
import numpy as np

class APDTracker:
    """APD_effective = time from first entry into state 1 (EXCITED)
    until first return to RESTING (0). Tracks many cells/episodes.

    Call `update(grid)` once per time step; at the end read `apd_list_ms()`.
    """
    def __init__(self, height: int, width: int, dt_ms: float):
        self.dt_ms = float(dt_ms)
        self._active = np.full((height, width), False, dtype=bool)
        self._start = np.full((height, width), -1, dtype=int)
        self._durations_ms: list[float] = []
        self.step = -1

    def update(self, grid: np.ndarray) -> None:
        self.step += 1
        # Start of APD window: newly excited (state == 1) and not already active
        starts = (grid == 1) & (~self._active)
        self._active[starts] = True
        self._start[starts] = self.step

        # End of APD window: return to rest (state == 0) from active
        finishes = (grid == 0) & self._active
        if np.any(finishes):
            d_steps = self.step - self._start[finishes]
            self._durations_ms.extend((d_steps * self.dt_ms).tolist())
            self._active[finishes] = False
            self._start[finishes] = -1

    def apd_list_ms(self) -> np.ndarray:
        return np.array(self._durations_ms, dtype=float)
