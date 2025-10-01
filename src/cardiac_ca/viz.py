from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from .constants import RESTING, IMSHOW_INTERPOLATION


def create_animation_elements(
    initial_grid,
    refractory_period_val: int,
    k_exc: int,
):
    """Create figure, axis, and image handle for animation frames.

    Parameters
    ----------
    initial_grid : np.ndarray
        Initial integer state grid used for the first frame.
    refractory_period_val : int
        Refractory length in steps (controls colormap extent).
    k_exc : int
        Number of excited substates (controls colormap extent).

    Returns
    -------
    tuple[plt.Figure, plt.Axes, plt.AxesImage]
        Figure, axis, and image handle for blitting updates.
    """
    colors = ["white"] + ["red"] * k_exc + ["blue"] * refractory_period_val
    cmap = ListedColormap(colors)
    max_state_value = k_exc + refractory_period_val
    norm = Normalize(vmin=RESTING, vmax=max_state_value)
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(
        initial_grid,
        cmap=cmap,
        norm=norm,
        interpolation=IMSHOW_INTERPOLATION,
    )
    ax.set_title("Cardiac Propagation Simulation", fontsize=16)
    plt.xticks([])
    plt.yticks([])
    return fig, ax, img