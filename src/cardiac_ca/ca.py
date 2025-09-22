from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.signal import convolve2d
from .constants import RESTING, EXCITED
from .data import SimulationDataCollector
from .lat import LATTracker
from .apd import APDTracker 


def update_grid_hybrid(
    current_grid: np.ndarray,
    V: np.ndarray,
    refractory_period_val: int,
    theta: float,
    alpha: float,
    k_exc: int,
    conv_kernel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance the cellular automaton by one step.

    This hybrid rule uses an anisotropic Gaussian kernel to compute a local
    input field from cells in excited substates, integrates that input into a
    leaky accumulator ``V`` for resting cells, and applies a threshold to
    determine new excitations. Excited states progress deterministically into
    refractory, then back to rest.

    Parameters
    ----------
    current_grid : np.ndarray
        Integer state grid at time ``t``.
    V : np.ndarray
        Floating accumulator (same shape) holding the input "voltage" at ``t``.
    refractory_period_val : int
        Refractory length in steps.
    theta : float
        Threshold on ``V`` for triggering excitation.
    alpha : float
        Leak factor for ``V`` (close to 1.0 means slower decay).
    k_exc : int
        Number of excited substates (1..k_exc).
    conv_kernel : np.ndarray
        Convolution kernel for neighbor influence.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(new_grid, V_next)`` at time ``t+1``.
    """
    return CardiacCA._update_grid_hybrid(
        current_grid, V, refractory_period_val, theta, alpha, k_exc, conv_kernel
    )


class CardiacCA:
    """Encapsulates the cellular automaton state and update logic."""

    def __init__(
        self,
        grid: np.ndarray,
        V: Optional[np.ndarray],
        kernel: np.ndarray,
        refractory_period: int,
        theta: float,
        alpha: float,
        k_exc: int,
        lat_tracker: Optional[LATTracker] = None,
        data_collector: Optional[SimulationDataCollector] = None,
        apd_tracker: Optional[APDTracker] = None, 
    ) -> None:
        self.grid = grid
        self.V = np.zeros_like(grid, dtype=np.float32) if V is None else V
        self.kernel = kernel
        self.refractory_period = refractory_period
        self.theta = theta
        self.alpha = alpha
        self.k_exc = k_exc
        self.lat_tracker = lat_tracker
        self.data_collector = data_collector
        self.apd_tracker = apd_tracker

    def step(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one simulation step and update trackers/collectors."""
        new_grid, new_V = self._update_grid_hybrid(
            self.grid,
            self.V,
            self.refractory_period,
            self.theta,
            self.alpha,
            self.k_exc,
            self.kernel,
        )
        self.grid, self.V = new_grid, new_V
        if self.data_collector is not None:
            self.data_collector.record_step(frame_idx, new_grid)
        if self.lat_tracker is not None:
            self.lat_tracker.update_with_grid(new_grid)
        if self.apd_tracker is not None:                   
            self.apd_tracker.update(new_grid)
        return new_grid, new_V

    @staticmethod
    def _update_grid_hybrid(
        current_grid: np.ndarray,
        V: np.ndarray,
        refractory_period_val: int,
        theta: float,
        alpha: float,
        k_exc: int,
        conv_kernel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        new_grid = current_grid.copy()
        V_next = V.copy()

        excited_mask = (current_grid >= 1) & (current_grid <= k_exc)
        I = convolve2d(
            excited_mask.astype(np.float32),
            conv_kernel,
            mode="same",
            boundary="symm",  # reflective slab
        )

        resting = current_grid == RESTING
        V_next[resting] = alpha * V[resting] + (1.0 - alpha) * I[resting]

        to_excite = resting & (V_next >= theta)
        new_grid[to_excite] = EXCITED
        V_next[to_excite] = 0.0

        # Progress excited substates 1..K
        for s in range(1, k_exc):
            mask = current_grid == s
            new_grid[mask] = s + 1

        # Last excited -> start refractory
        mask_last_exc = current_grid == k_exc
        refractory_start = k_exc + 1
        new_grid[mask_last_exc] = refractory_start
        refractory_last = refractory_start + refractory_period_val - 1
        refractory = current_grid >= refractory_start

        progressing = refractory & (current_grid < refractory_last)
        new_grid[progressing] = current_grid[progressing] + 1

        finished = refractory & (current_grid == refractory_last)
        new_grid[finished] = RESTING
        V_next[finished] = 0.0

        return new_grid, V_next