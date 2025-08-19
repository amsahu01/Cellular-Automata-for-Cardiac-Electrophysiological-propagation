"""Cardiac propagation cellular automaton.

This module simulates a 2‑D cardiac excitation wave using a hybrid rule:
integer states for excitation/refractory progression and a floating‑point
"voltage" accumulator (``V``) that integrates neighborhood input via an
anisotropic Gaussian kernel. The kernel controls the preferred conduction
axis and anisotropy, yielding smooth wavefronts.

The module provides:

* A self-contained simulation with animation and screenshots
* Local Activation Time (LAT) measurement
* Conduction velocity (CV) estimators from LAT
* Data collection for basic activity summaries

"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, Normalize
from scipy.signal import convolve2d

# ----------------------------- States ---------------------------------------

#: Integer state for resting cells.
RESTING = 0
#: First excited substate (1..K are excited).
EXCITED = 1

# ------------------- Physical & numerical resolution ------------------------

#: Spatial resolution in millimetres per node.
DX_MM = 0.10
#: Temporal resolution in milliseconds per step.
DT_MS = 0.125
#: Refractory duration expressed in CA steps.
REFRACTORY_STEPS = 100
#: Square grid size (height == width).
GRID_SIZE = 401

# ------------------- Anisotropic kernel parameters --------------------------

#: Anisotropy ratio (sigma_long / sigma_trans) \(\CV ratio\).
ANISO_RATIO = 1.5
#: Transverse sigma in cells.
SIGMA_TRANS = 4.0
#: Longitudinal sigma in cells.
SIGMA_LONG = ANISO_RATIO * SIGMA_TRANS
#: Fiber angle in degrees (0 == +x direction).
FIBER_ANGLE_DEG = 90.0
#: Odd kernel size; larger gives smoother fronts but higher cost.
KERNEL_SIZE = 15
#: Post-normalization gain applied to the kernel.
KERNEL_GAIN = 3

# ------------------- Activation parameters ---------------------------

#: Excitation threshold on V.
THETA = 0.085
#: Leak (memory) factor for V.
ALPHA = 0.94
#: Number of excited substates (thickness of excited rim).
EXCITED_STAGES = 3

# ------------------- Visualization -----------------------------------------

#: Matplotlib ``imshow`` interpolation.
IMSHOW_INTERPOLATION = "bilinear" 

# ------------------- LAT/CV utilities --------------------------------


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


def cv_between_points(
    lat_steps: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    dx_mm: float,
    dt_ms: float,
) -> float:
    """Estimate CV (mm/ms) from two points and the LAT step map.

    Parameters
    ----------
    lat_steps : np.ndarray
        LAT stored in steps (not milliseconds). Typically from
        ``LATTracker.lat_steps``.
    p1, p2 : tuple[int, int]
        Points as ``(y, x)`` indices.
    dx_mm : float
        Millimetres per pixel.
    dt_ms : float
        Milliseconds per step.

    Returns
    -------
    float
        Estimated conduction velocity in mm/ms, or ``np.nan`` when the
        estimate is not possible (e.g. missing activations or zero time
        difference).
    """
    y1, x1 = p1
    y2, x2 = p2
    t1 = lat_steps[y1, x1]
    t2 = lat_steps[y2, x2]
    if not np.isfinite(t1) or not np.isfinite(t2) or t1 == t2:
        return float("nan")
    dist_mm = np.hypot((y2 - y1), (x2 - x1)) * dx_mm
    time_ms = abs(t2 - t1) * dt_ms
    return float(dist_mm / time_ms)


def sample_line_points(
    p1: Tuple[int, int], p2: Tuple[int, int], n: int = 100
) -> np.ndarray:
    """Return unique grid points sampling the line segment ``p1 -> p2``.

    Parameters
    ----------
    p1, p2 : tuple[int, int]
        Endpoints as ``(y, x)``.
    n : int, optional
        Number of samples along the segment (before de-duplication).

    Returns
    -------
    np.ndarray
        Unique integer coordinates ``[[y, x], ...]`` along the segment.
    """
    y1, x1 = p1
    y2, x2 = p2
    ys = np.linspace(y1, y2, n)
    xs = np.linspace(x1, x2, n)
    pts = np.vstack([np.round(ys).astype(int), np.round(xs).astype(int)]).T
    pts = np.unique(pts, axis=0)
    return pts


def cv_from_lat_profile(
    lat_steps: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    dx_mm: float,
    dt_ms: float,
    n_samples: int = 100,
) -> float:
    """Estimate CV along a transect using a robust slope fit.

    The method samples points along the segment ``p1 -> p2`` and performs a
    least-squares fit of distance vs. time difference (relative to the first
    valid sample). The slope yields CV in mm/ms.

    Parameters
    ----------
    lat_steps : np.ndarray
        LAT stored in steps (not milliseconds).
    p1, p2 : tuple[int, int]
        Endpoints as ``(y, x)``.
    dx_mm : float
        Millimetres per pixel.
    dt_ms : float
        Milliseconds per step.
    n_samples : int, optional
        Number of sample points (pre de-duplication). Default is 100.

    Returns
    -------
    float
        Estimated CV in mm/ms, or ``np.nan`` if a reliable estimate cannot be
        obtained (e.g. too few samples or zero slope).
    """
    pts = sample_line_points(p1, p2, n_samples)
    t0 = None
    y0 = x0 = None
    times: List[float] = []
    dists: List[float] = []
    for y, x in pts:
        t = lat_steps[y, x]
        if np.isfinite(t):
            if t0 is None:
                t0 = t
                y0, x0 = y, x
            d_mm = np.hypot(y - y0, x - x0) * dx_mm
            times.append((t - t0) * dt_ms)
            dists.append(d_mm)
    if len(times) < 5:
        return float("nan")
    t = np.array(times)
    d = np.array(dists)
    denom = float(np.dot(t, t))
    if denom == 0.0:
        return float("nan")
    v = float(np.dot(t, d) / denom)
    return v


# ------------------- Data Collector ---------------------------


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


# ------------------- Kernel construction ------------------------------------


def anisotropic_gaussian_kernel(
    size: int = 21,
    sigma_long: float = 3.5,
    sigma_trans: float = 1.8,
    angle_deg: float = 0.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Create a normalized anisotropic Gaussian kernel.

    The kernel is oriented by ``angle_deg`` and normalized to sum to 1 before
    applying the ``gain`` factor. The center is zeroed to avoid self-coupling.

    Parameters
    ----------
    size : int, optional
        Odd kernel size (width == height). Default is 21.
    sigma_long : float, optional
        Longitudinal standard deviation (pixels). Default is 3.5.
    sigma_trans : float, optional
        Transverse standard deviation (pixels). Default is 1.8.
    angle_deg : float, optional
        Rotation angle in degrees (0 == +x). Default is 0.
    gain : float, optional
        Post-normalization gain. Default is 1.0.

    Returns
    -------
    np.ndarray
        2‑D float32 kernel of shape ``(size, size)``.
    """
    assert size % 2 == 1, "Kernel size must be odd."
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    phi = np.deg2rad(angle_deg)
    c, s = np.cos(phi), np.sin(phi)
    xp = c * x + s * y
    yp = -s * x + c * y
    g = np.exp(
        -((xp ** 2) / (2.0 * sigma_long ** 2)
          + (yp ** 2) / (2.0 * sigma_trans ** 2))
    ).astype(np.float32)
    # No self-coupling
    g[r, r] = 0.0
    ssum = g.sum()
    if ssum > 0:
        g /= ssum
    g *= gain
    return g


kernel = anisotropic_gaussian_kernel(
    size=KERNEL_SIZE,
    sigma_long=SIGMA_LONG,
    sigma_trans=SIGMA_TRANS,
    angle_deg=FIBER_ANGLE_DEG,
    gain=KERNEL_GAIN,
)


# ------------------- Grid initialization ------------------------------------


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


# ------------------- Hybrid update rule -------------------------------------


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


# ------------------- Animation helpers --------------------------------------


def create_animation_elements(
    initial_grid: np.ndarray,
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
    ax.set_title("Cardiac Propagation Automata (Hybrid Anisotropic)")
    plt.xticks([])
    plt.yticks([])
    return fig, ax, img


# ------------------- Main ----------------------------------------------------


def main() -> None:
    """Run the simulation, animation, LAT/CV summary, and plots.

    The function builds the kernel and initial grid, runs the animation loop
    for a fixed number of steps, records online LAT and activity summaries, and
    finally plots an LAT heatmap and prints a few CV estimates. Optional frame
    screenshots can be saved during the run.
    """
    grid_width = GRID_SIZE
    grid_height = GRID_SIZE
    time_steps = 400
    refractory_period = REFRACTORY_STEPS

    stimulus_type = "bottom_cap"
    stimulus_size = 27
    stim_depth = 3

    screenshot_steps = [1, 80, 200]

    print(
        f"Starting simulation: Size={grid_width}x{grid_height}, "
        f"Steps={time_steps}"
    )

    current_grid = initialize_grid(
        grid_height,
        grid_width,
        stim_type=stimulus_type,
        stim_size=stimulus_size,
        stim_depth=stim_depth,
    )
    V = np.zeros_like(current_grid, dtype=np.float32)

    # Trackers/collectors
    lat_tracker = LATTracker(grid_height, grid_width)
    data_collector = SimulationDataCollector(grid_height, grid_width)

    # Visualization setup
    fig_anim, ax_anim, img_anim = create_animation_elements(
        current_grid, REFRACTORY_STEPS, EXCITED_STAGES
    )

    # Use containers to allow in-place updates captured by closure
    grid_list_container = [current_grid]
    V_container = [V]

    def anim_func(frame_idx: int):
        current_grid = grid_list_container[0]
        V = V_container[0]
        new_grid, new_V = update_grid_hybrid(
            current_grid,
            V,
            refractory_period,
            THETA,
            ALPHA,
            EXCITED_STAGES,
            kernel,
        )

        grid_list_container[0] = new_grid
        V_container[0] = new_V

        img_anim.set_array(new_grid)
        rho = SIGMA_LONG / SIGMA_TRANS
        ax_anim.set_title(
            "t={:d}  theta={:.3f} alpha={:.3f}  ".format(frame_idx + 1, THETA, ALPHA)
            + "rho={:.2f} phi={}°  ".format(rho, int(FIBER_ANGLE_DEG))
            + "RP={}  K={}".format(refractory_period, EXCITED_STAGES)
        )

        # Record metrics
        data_collector.record_step(frame_idx, new_grid)
        lat_tracker.update_with_grid(new_grid)

        if (frame_idx + 1) in screenshot_steps:
            fig_anim.savefig(
                f"simulation_step_{frame_idx + 1}.png", dpi=150
            )

        return [img_anim]

    ani = FuncAnimation(
        fig_anim,
        anim_func,
        frames=time_steps,
        interval=10,
        blit=True,
        repeat=False,
    )
    fig_anim.tight_layout()
    plt.show()

    # ------------------- LAT/CV summary and plots ---------------------------

    LAT_steps = lat_tracker.lat_steps

    # Coverage summary
    finite_mask = np.isfinite(LAT_steps)
    coverage = float(finite_mask.mean() * 100.0)
    if np.any(finite_mask):
        lat_min_ms = float(np.nanmin(LAT_steps[finite_mask]) * DT_MS)
        lat_max_ms = float(np.nanmax(LAT_steps[finite_mask]) * DT_MS)
    else:
        lat_min_ms = float("nan")
        lat_max_ms = float("nan")

    print(f"LAT finite coverage: {coverage:.1f}% of grid")
    print(f"LAT min/max (ms): {lat_min_ms:.1f} / {lat_max_ms:.1f}")

    # # Point-to-point vertical (bottom -> mid)
    # p_bottom_mid = (grid_height - 30, grid_width // 2)
    # p_upper_mid = (grid_height // 2 - 100, grid_width // 2)
    # cv_vert = cv_between_points(
    #     LAT_steps, p_bottom_mid, p_upper_mid, DX_MM, DT_MS
    # )

    # # Along fiber: 
    # p1 = (grid_height - 60, grid_width // 2 - 60)
    # p2 = (grid_height - 220, grid_width // 2 + 60)
    # cv_fiber = cv_from_lat_profile(
    #     LAT_steps, p1, p2, DX_MM, DT_MS, n_samples=140
    # )

    # # Across fiber: approximately horizontal segment
    # q1 = (grid_height - 120, grid_width // 2 - 120)
    # q2 = (grid_height - 120, grid_width // 2 + 120)
    # cv_cross = cv_from_lat_profile(
    #     LAT_steps, q1, q2, DX_MM, DT_MS, n_samples=140
    # )

    # print(f"CV(bottom->mid) ~ {cv_vert:.3f} mm/ms")
    # print(f"CV along fiber   ~ {cv_fiber:.3f} mm/ms")
    # print(f"CV across fiber  ~ {cv_cross:.3f} mm/ms")

    # LAT map plot
    lat_ms_map = np.where(finite_mask, LAT_steps * DT_MS, np.nan)
    plt.figure(figsize=(6.6, 6.6))
    plt.imshow(lat_ms_map, cmap="turbo")
    plt.title("Local Activation Time (ms)")
    cbar = plt.colorbar()
    cbar.set_label("ms")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("lat_map.png", dpi=150)
    plt.show()

    print("Done. Saved: lat_map.png and optional step screenshots.")


if __name__ == "__main__":
    main()
