from __future__ import annotations
import numpy as np
from .constants import RESTING, EXCITED


def initialize_grid(
    grid_height: int,
    grid_width: int,
    stim_type: str = "bottom_cap",
    stim_size: int = 27,
    stim_depth: int = 3,
) -> np.ndarray:
    """Return an initial grid with a chosen stimulus pattern.

    Parameters
    ----------
    grid_height : int
        Grid height.
    grid_width : int
        Grid width.
    stim_type : str, optional
        One of ``{"center", "bottom_center", "bottom_cap"}``. Default is
        ``"bottom_cap"``.
    stim_size : int, optional
        Stimulus width/diameter depending on pattern. Default is 27.
    stim_depth : int, optional
        Vertical depth for the ``"bottom_center"`` stimulus. Default is 3.

    Returns
    -------
    np.ndarray
        Integer grid of shape ``(grid_height, grid_width)`` with initial
        excited cells encoded as ``EXCITED``.
    """
    if not (grid_height > 0 and grid_width > 0):
        raise ValueError("Grid height and width must be positive integers.")
    grid = np.full((grid_height, grid_width), RESTING, dtype=np.int32)

    if stim_type == "center":
        cy, cx = grid_height // 2, grid_width // 2
        y0 = max(0, cy - stim_size // 2)
        y1 = min(grid_height, cy + stim_size // 2 + (stim_size % 2))
        x0 = max(0, cx - stim_size // 2)
        x1 = min(grid_width, cx + stim_size // 2 + (stim_size % 2))
        grid[y0:y1, x0:x1] = EXCITED
    elif stim_type == "bottom_center":
        cx = grid_width // 2
        y0 = max(0, grid_height - stim_depth)
        x0 = max(0, cx - stim_size // 2)
        x1 = min(grid_width, cx + stim_size // 2 + (stim_size % 2))
        grid[y0:grid_height, x0:x1] = EXCITED
    elif stim_type == "bottom_cap":
        radius = max(6, stim_size // 2)
        cy = grid_height - 2
        cx = grid_width // 2
        y, x = np.ogrid[:grid_height, :grid_width]
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        mask &= (y >= cy - radius)
        grid[mask] = EXCITED
    else:
        print(f"Warning: Unknown stimulus '{stim_type}'. No active cells.")
    return grid