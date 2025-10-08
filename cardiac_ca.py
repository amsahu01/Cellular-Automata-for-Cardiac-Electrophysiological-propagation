"""Cardiac Electrophysiological Propagation (Cellular Automata)"""

from __future__ import annotations

import math
from dataclasses import dataclass
import time

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d, fftconvolve


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------


@dataclass
class Config:
    # Grid and time
    height: int = 401
    width: int = 401
    dt_ms: float = 0.1
    t_max_ms: float = 2000.0  # safety cap

    # State durations (APD = T_excited + T_refractory)
    T_excited_ms: float = 1.0
    T_refractory_ms: float = 100.0

    # Kernel parameters
    kernel_size: int = 15
    sigma_trans: float = 4.0
    aniso_ratio: float = 1  # sigma_long / sigma_trans
    angle_deg: float = 0.0
    gain: float = 3.0

    # Boundary handling 
    boundary_mode: str = "reflective"

    # Activation threshold (uniform)
    threshold: float = 0.25

    # Stimulus placement
    stimulus_type: str = "bottom_cap"
    stimulus_size: int = 12

    # Visualization
    animate: bool = True
    steps_per_frame: int = 1
    fps: int = 30
    cmap_name: str = "state"
    show_apd_at_end: bool = True


# -------------------------------------------------------------
# Core simulation helpers
# -------------------------------------------------------------


RESTING = 0
EXCITED = 1
REFRACTORY = 2


def anisotropic_gaussian_kernel(
    size: int,
    sigma_long: float,
    sigma_trans: float,
    angle_deg: float,
    gain: float,
) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    phi = math.radians(angle_deg)
    c, s = math.cos(phi), math.sin(phi)
    xp = c * x + s * y
    yp = -s * x + c * y
    g = np.exp(-((xp**2) / (2.0 * sigma_long**2) + (yp**2) / (2.0 * sigma_trans**2))).astype(np.float32)
    g[r, r] = 0.0
    total = g.sum()
    if total > 0.0:
        g /= total
    g *= gain
    return g


# -------------------------------------------------------------
# Simulation loop
# -------------------------------------------------------------


@dataclass
class SimState:
    state: np.ndarray
    t_activation: np.ndarray
    t_deactivation: np.ndarray
    threshold: np.ndarray
    activated_mask: np.ndarray
    time_ms: float = 0.0
    step: int = 0


def initialise_state(cfg: Config) -> tuple[SimState, np.ndarray]:
    grid_shape = (cfg.height, cfg.width)
    state = np.zeros(grid_shape, dtype=np.uint8)
    t_activation = np.full(grid_shape, np.nan, dtype=np.float64)
    t_deactivation = np.full(grid_shape, np.nan, dtype=np.float64)
    threshold = np.full(grid_shape, cfg.threshold, dtype=np.float64)
    activated_mask = np.zeros(grid_shape, dtype=bool)

    stim_type = cfg.stimulus_type.lower()
    stim_size = max(1, int(cfg.stimulus_size))
    y, x = np.ogrid[: cfg.height, : cfg.width]

    if stim_type == "bottom_cap":
        radius = max(6, stim_size // 2)
        cy = max(0, cfg.height - 2)
        cx = cfg.width // 2
        stim_mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius**2
        stim_mask &= y >= (cy - radius)
    else:
        raise ValueError(f"Unsupported stimulus type '{cfg.stimulus_type}'")

    state[stim_mask] = EXCITED
    t_activation[stim_mask] = 0.0
    apd = cfg.T_excited_ms + cfg.T_refractory_ms
    t_deactivation[stim_mask] = apd
    activated_mask |= stim_mask

    sigma_long = cfg.aniso_ratio * cfg.sigma_trans  # derive longitudinal width from ratio
    kernel = anisotropic_gaussian_kernel(
        size=cfg.kernel_size,
        sigma_long=sigma_long,
        sigma_trans=cfg.sigma_trans,
        angle_deg=cfg.angle_deg,
        gain=cfg.gain,
    )
    return SimState(state, t_activation, t_deactivation, threshold, activated_mask), kernel


def update_rules(cfg: Config, sim: SimState, kernel: np.ndarray) -> None:
    excited_mask = (sim.state == EXCITED).astype(np.float64)
    boundary = cfg.boundary_mode.lower()
    if boundary in {"reflective", "mirror", "symm"}:
        input_field = convolve2d(
            excited_mask,
            kernel,
            mode="same",
            boundary="symm",
        )
    else:
        # Default to zero-padding (Dirichlet) via FFT-based convolution.
        input_field = fftconvolve(excited_mask, kernel, mode="same")


    activate_mask = (sim.state == RESTING) & (input_field >= sim.threshold)

    time_next = sim.time_ms + cfg.dt_ms

    excite_to_refrac = (sim.state == EXCITED) & (
        time_next >= sim.t_activation + cfg.T_excited_ms - 1e-9
    )
    refrac_to_rest = (sim.state == REFRACTORY) & (
        time_next >= sim.t_deactivation - 1e-9
    )

    sim.state[excite_to_refrac] = REFRACTORY
    sim.state[refrac_to_rest] = RESTING

    sim.state[activate_mask] = EXCITED
    sim.t_activation[activate_mask] = time_next
    sim.t_deactivation[activate_mask] = (
        time_next + cfg.T_excited_ms + cfg.T_refractory_ms
    )
    sim.activated_mask |= activate_mask

    sim.time_ms = time_next
    sim.step += 1


# -------------------------------------------------------------
# Visualization
# -------------------------------------------------------------


def state_colormap(name: str) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    if name.lower() == "state":
        colors = [
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        cmap = mcolors.ListedColormap(colors, name="state_map")
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    else:
        cmap = plt.get_cmap(name)
        norm = mcolors.Normalize(vmin=RESTING, vmax=REFRACTORY)
    return cmap, norm


def run_simulation(cfg: Config) -> SimState:
    sim, kernel = initialise_state(cfg)
    if cfg.animate:
        animate_simulation(cfg, sim, kernel)
    else:
        while not is_complete(cfg, sim):
            update_rules(cfg, sim, kernel)
    return sim


def animate_simulation(cfg: Config, sim: SimState, kernel: np.ndarray) -> None:
    cmap, norm = state_colormap(cfg.cmap_name)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Cardiac CA Simulation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    image = ax.imshow(sim.state, cmap=cmap, norm=norm, origin="upper", interpolation="bilinear")
    hud = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
    )
    hud.set_text(format_status(sim, wall_time_s=0.0))

    fps = max(1, cfg.fps)
    steps_per_frame = max(1, cfg.steps_per_frame)
    start_wall = time.perf_counter()

    def update(_frame):
        for _ in range(steps_per_frame):
            if is_complete(cfg, sim):
                break
            update_rules(cfg, sim, kernel)
        image.set_data(sim.state)
        wall_elapsed = time.perf_counter() - start_wall
        hud.set_text(format_status(sim, wall_elapsed))
        if is_complete(cfg, sim):
            anim.event_source.stop()
        return image, hud

    anim = animation.FuncAnimation(
        fig,
        update,
        interval=1000 / fps,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()

    if cfg.show_apd_at_end:
        show_activation_maps(sim)


def format_status(sim: SimState, wall_time_s: float) -> str:
    num_excited = int(np.count_nonzero(sim.state == EXCITED))
    num_refrac = int(np.count_nonzero(sim.state == REFRACTORY))
    return (
        f"t = {sim.time_ms:7.1f} ms\n"
        f"wall_t = {wall_time_s:6.2f} s\n"
        f"excited = {num_excited:6d}\n"
        f"refractory = {num_refrac:6d}"
    )


def show_activation_maps(sim: SimState) -> None:
    activation = sim.t_activation
    deactivation = sim.t_deactivation
    apd = deactivation - activation

    datasets = [
        (activation, "Activation Time (ms)"),
        (deactivation, "Deactivation Time (ms)"),
        (apd, "APD (ms)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (data, title) in zip(axes, datasets):
        im = ax.imshow(data, origin="upper", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Termination, metrics, and entry point
# -------------------------------------------------------------


def is_complete(cfg: Config, sim: SimState) -> bool:
    if sim.step == 0:
        return False
    if cfg.t_max_ms is not None and sim.time_ms >= cfg.t_max_ms:
        return True
    return np.all(sim.state == RESTING)


def compute_metrics(sim: SimState) -> dict[str, float]:
    apd = sim.t_deactivation - sim.t_activation
    valid = apd[~np.isnan(apd)]
    coverage = float(np.count_nonzero(~np.isnan(apd)) / apd.size)
    metrics = {
        "t_end_ms": float(sim.time_ms),
        "num_steps": int(sim.step),
        "coverage": coverage,
    }
    if valid.size:
        metrics.update(
            {
                "apd_mean_ms": float(np.mean(valid)),
                "apd_median_ms": float(np.median(valid)),
                "apd_min_ms": float(np.min(valid)),
                "apd_max_ms": float(np.max(valid)),
                "activation_spread_ms": float(
                    np.max(sim.t_deactivation[~np.isnan(sim.t_deactivation)]) - np.min(sim.t_activation[~np.isnan(sim.t_activation)])
                ),
            }
        )
    else:
        metrics.update(
            {
                "apd_mean_ms": float("nan"),
                "apd_median_ms": float("nan"),
                "apd_min_ms": float("nan"),
                "apd_max_ms": float("nan"),
                "activation_spread_ms": float("nan"),
            }
        )
    return metrics


def main() -> None:
    cfg = Config()
    sim = run_simulation(cfg)
    metrics = compute_metrics(sim)
    print(f"Simulation finished in {metrics['num_steps']} steps, t_end = {metrics['t_end_ms']:.2f} ms")
    print(f"Coverage: {metrics['coverage']:.3f}")
    print(f"APD mean: {metrics['apd_mean_ms']:.2f} ms")


if __name__ == "__main__":
    main()
