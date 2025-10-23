"""Cardiac Electrophysiological Propagation (Cellular Automata)"""
# python cardiac_ca_copy.py --params optimised_kernel_thresholds.npz
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import fftconvolve

from viz_metrics import animate_simulation, compute_metrics, show_activation_maps


EPS = 1e-12


def generate_gaussian_threshold_matrix(
    height: int,
    width: int,
    sigma_y: float | None = None,
    sigma_x: float | None = None,
    sigma_y_frac: float = 0.25,
    sigma_x_frac: float = 0.25,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    invert: bool = False,
    angle_degrees: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Return a bottom-centered 2D Gaussian threshold matrix.

    Coordinates follow row-major image conventions: rows increase downward, columns increase rightward, and
    the origin is the top-left corner. The Gaussian peak is fixed at (row=H-1, col=W//2) so it emanates from the
    bottom center of the grid. Rotation pivots the ellipse around that peak.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive integers.")
    if max_threshold < min_threshold:
        raise ValueError("max_threshold must be greater than or equal to min_threshold.")

    sigma_y_val = float(sigma_y) if sigma_y is not None else float(height) * float(sigma_y_frac)
    sigma_x_val = float(sigma_x) if sigma_x is not None else float(width) * float(sigma_x_frac)
    if sigma_y_val <= 0.0 or sigma_x_val <= 0.0:
        raise ValueError("Gaussian standard deviations must be positive.")

    mu_row = height - 1
    mu_col = width // 2
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")

    y0 = yy - mu_row
    x0 = xx - mu_col
    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xp = cos_t * x0 + sin_t * y0
    yp = -sin_t * x0 + cos_t * y0

    exponent = -0.5 * ((xp / sigma_x_val) ** 2 + (yp / sigma_y_val) ** 2)
    gaussian = np.exp(exponent)

    if normalize:
        g_min = float(gaussian.min())
        g_max = float(gaussian.max())
        gaussian = (gaussian - g_min) / max(g_max - g_min, EPS)

    mapped = 1.0 - gaussian if invert else gaussian
    thresholds = min_threshold + (max_threshold - min_threshold) * mapped
    thresholds = np.clip(thresholds, min_threshold, max_threshold)

    if thresholds.shape != (height, width):
        raise AssertionError("Threshold matrix has incorrect shape.")
    if not np.all(np.isfinite(thresholds)):
        raise AssertionError("Threshold matrix contains non-finite values.")
    if thresholds.min() < min_threshold - 1e-6 or thresholds.max() > max_threshold + 1e-6:
        raise AssertionError("Threshold matrix values exceed configured bounds.")

    return thresholds.astype(np.float32, copy=False)


# -------------------------------------------------------------
# Configurations
# -------------------------------------------------------------


@dataclass
class Config:
    # Grid and time
    height: int = 401
    width: int = 401
    dt_ms: float = 1.0
    max_steps: int = 10000  # safety cap

    # Phase durations (number of simulation steps)
    excited_steps: int = 3
    refractory_steps: int = 1000

    # Kernel parameters
    kernel_size: int = 15
    sigma_trans: float = 2.0  # transverse spread of the Gaussian kernel
    aniso_ratio: float = 2.5  # sigma_long / sigma_trans
    angle_deg: float = 90.0
    gain: float = 1.0
    custom_kernel: np.ndarray | None = None

    # Activation threshold configuration
    threshold_strategy: str = "gaussian"  # "uniform" or "gaussian"
    threshold: float = 0.12  # used when threshold_strategy == "uniform"
    threshold_min: float = 0.05
    threshold_max: float = 0.2
    threshold_sigma_y: float | None = None
    threshold_sigma_x: float | None = None
    threshold_sigma_y_frac: float = 0.018
    threshold_sigma_x_frac: float = 0.018
    threshold_invert: bool = False
    threshold_angle_deg: float = 0.0
    threshold_normalize: bool = True

    # Stimulus placement
    stimulus_type: str = "bottom_cap"
    stimulus_size: int = 6

    # Visualization
    animate: bool = True
    steps_per_frame: int = 1
    fps: int = 30
    cmap_name: str = "state"
    show_apd_at_end: bool = False


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


def _build_threshold_matrix(cfg: Config) -> np.ndarray:
    """Construct the threshold matrix according to the configured strategy."""
    grid_shape = (cfg.height, cfg.width)
    strategy = cfg.threshold_strategy.lower()
    if strategy == "uniform":
        matrix = np.full(grid_shape, cfg.threshold, dtype=np.float32)
    elif strategy == "gaussian":
        matrix = generate_gaussian_threshold_matrix(
            height=cfg.height,
            width=cfg.width,
            sigma_y=cfg.threshold_sigma_y,
            sigma_x=cfg.threshold_sigma_x,
            sigma_y_frac=cfg.threshold_sigma_y_frac,
            sigma_x_frac=cfg.threshold_sigma_x_frac,
            min_threshold=cfg.threshold_min,
            max_threshold=cfg.threshold_max,
            invert=cfg.threshold_invert,
            angle_degrees=cfg.threshold_angle_deg,
            normalize=cfg.threshold_normalize,
        )
    else:
        raise ValueError(f"Unsupported threshold strategy '{cfg.threshold_strategy}'")

    if matrix.shape != grid_shape:
        raise AssertionError("Threshold matrix shape mismatch.")
    if not np.all(np.isfinite(matrix)):
        raise AssertionError("Threshold matrix contains non-finite values.")
    if strategy == "gaussian":
        if matrix.min() < cfg.threshold_min - 1e-6 or matrix.max() > cfg.threshold_max + 1e-6:
            raise AssertionError("Gaussian thresholds exceed configured bounds.")

    return matrix.astype(np.float32, copy=False)


# -------------------------------------------------------------
# Simulation loop
# -------------------------------------------------------------


@dataclass
class SimState:
    state: np.ndarray
    activation_step: np.ndarray
    deactivation_step: np.ndarray
    threshold: np.ndarray
    activated_mask: np.ndarray
    time_ms: float = 0.0
    step: int = 0
    activation_counts: list[int] = field(default_factory=list)


def initialise_state(cfg: Config) -> tuple[SimState, np.ndarray]:
    grid_shape = (cfg.height, cfg.width)
    state = np.zeros(grid_shape, dtype=np.uint8)
    activation_step = np.full(grid_shape, -1, dtype=np.int32)
    deactivation_step = np.full(grid_shape, -1, dtype=np.int32)
    threshold = _build_threshold_matrix(cfg)
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
    activation_step[stim_mask] = 1
    apd_steps = cfg.excited_steps + cfg.refractory_steps
    deactivation_step[stim_mask] = apd_steps + 1
    activated_mask |= stim_mask

    if cfg.custom_kernel is not None:
        kernel = np.asarray(cfg.custom_kernel, dtype=np.float32)
        if kernel.shape != (cfg.kernel_size, cfg.kernel_size):
            raise ValueError(
                f"Custom kernel must have shape ({cfg.kernel_size}, {cfg.kernel_size}), "
                f"received {kernel.shape}."
            )
        if not np.all(np.isfinite(kernel)):
            raise ValueError("Custom kernel contains non-finite values.")
    else:
        sigma_long = cfg.aniso_ratio * cfg.sigma_trans  # derive longitudinal width from ratio
        kernel = anisotropic_gaussian_kernel(
            size=cfg.kernel_size,
            sigma_long=sigma_long,
            sigma_trans=cfg.sigma_trans,
            angle_deg=cfg.angle_deg,
            gain=cfg.gain,
        )
    return SimState(state, activation_step, deactivation_step, threshold, activated_mask), kernel


def simulate_step(cfg: Config, sim: SimState, kernel: np.ndarray) -> None:
    excited_mask = (sim.state == EXCITED).astype(np.float64)
    input_field = fftconvolve(excited_mask, kernel, mode="same")

    activate_mask = (sim.state == RESTING) & (input_field >= sim.threshold)
    num_newly_excited = int(np.count_nonzero(activate_mask))

    step_next = sim.step + 1

    excited_steps = cfg.excited_steps
    refrac_steps = cfg.refractory_steps
    apd_steps = excited_steps + refrac_steps

    excite_to_refrac = (sim.state == EXCITED) & (
        sim.activation_step >= 0
    ) & (step_next >= sim.activation_step + excited_steps)

    refrac_to_rest = (sim.state == REFRACTORY) & (
        sim.deactivation_step >= 0
    ) & (step_next >= sim.deactivation_step)

    sim.state[excite_to_refrac] = REFRACTORY
    sim.state[refrac_to_rest] = RESTING

    sim.state[activate_mask] = EXCITED
    sim.activation_step[activate_mask] = step_next
    sim.deactivation_step[activate_mask] = step_next + apd_steps
    sim.activation_counts.append(num_newly_excited)

    sim.activated_mask |= activate_mask

    sim.step = step_next
    sim.time_ms = sim.step * cfg.dt_ms


def run_simulation(cfg: Config) -> SimState:
    sim, kernel = initialise_state(cfg)
    if cfg.animate:
        animate_simulation(cfg, sim, kernel, simulate_step, is_complete)
    else:
        while not is_complete(cfg, sim):
            simulate_step(cfg, sim, kernel)
    return sim


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------


def is_complete(cfg: Config, sim: SimState) -> bool:
    if sim.step == 0:
        return False
    if cfg.max_steps is not None and sim.step >= cfg.max_steps:
        return True
    return np.all(sim.state == RESTING)


def _apply_saved_parameters(cfg: Config, params_path: Path) -> Config:
    params_path = params_path.expanduser()
    if not params_path.is_file():
        raise FileNotFoundError(f"Parameter file '{params_path}' does not exist.")

    with np.load(params_path) as data:
        if "kernel" not in data:
            raise KeyError("Parameter file is missing 'kernel'.")
        kernel = np.asarray(data["kernel"], dtype=np.float32)
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel array must be a square 2D matrix.")
        size = int(kernel.shape[0])
        cfg.kernel_size = size
        cfg.custom_kernel = kernel
        cfg.threshold_strategy = "gaussian"

        for attr in ("threshold_min", "threshold_max", "threshold_sigma_y_frac", "threshold_sigma_x_frac"):
            if attr in data:
                setattr(cfg, attr, float(data[attr]))
        if "threshold_sigma_y" in data:
            cfg.threshold_sigma_y = float(data["threshold_sigma_y"])
        if "threshold_sigma_x" in data:
            cfg.threshold_sigma_x = float(data["threshold_sigma_x"])
        if "gain" in data:
            cfg.gain = float(data["gain"])

    return cfg


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cardiac CA simulation.")
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Path to an optimised parameter file (.npz) to configure the simulation.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = Config()
    if args.params is not None:
        cfg = _apply_saved_parameters(cfg, args.params)
    start = time.time()
    sim = run_simulation(cfg)
    end = time.time()
    print(f"Total simulation time: {end - start:.4f} seconds")
    metrics = compute_metrics(cfg, sim)
    if cfg.show_apd_at_end and not cfg.animate:
        show_activation_maps(cfg, sim)
    min = sim.activation_step.min()
    max = sim.activation_step.max()
    print(f"Activation step range: min={min}, max={max}")
    
    print(f"Simulation finished in {metrics['num_steps']} steps")
    print(f"Coverage: {metrics['coverage']:.3f}")
    print(f"APD steps: {metrics['apd_steps']}")


if __name__ == "__main__":
    main()
