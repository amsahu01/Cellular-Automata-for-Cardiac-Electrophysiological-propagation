"""Cardiac Electrophysiological Propagation (Cellular Automata)"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import time

import numpy as np
from scipy.signal import fftconvolve

from viz_metrics import animate_simulation, compute_metrics, show_activation_maps


# -------------------------------------------------------------
# Configurations
# -------------------------------------------------------------


@dataclass
class Config:
    # Grid and time
    height: int = 401
    width: int = 401
    dt_ms: float = 0.1
    max_steps: int = 2000  # safety cap

    # Phase durations (number of simulation steps)
    excited_steps: int = 3
    refractory_steps: int = 1000

    # Kernel parameters
    kernel_size: int = 15
    sigma_trans: float = 4.0
    aniso_ratio: float = 1  # sigma_long / sigma_trans
    angle_deg: float = 0.0
    gain: float = 3.0

    # Activation threshold (uniform)
    threshold: float = 0.25

    # Stimulus placement
    stimulus_type: str = "bottom_cap"
    stimulus_size: int = 12

    # Visualization
    animate: bool = False
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
    threshold = np.full(grid_shape, cfg.threshold, dtype=np.float32)
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
    activation_step[stim_mask] = 0
    apd_steps = cfg.excited_steps + cfg.refractory_steps
    deactivation_step[stim_mask] = apd_steps
    activated_mask |= stim_mask

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


def main() -> None:
    cfg = Config()
    start = time.time()
    sim = run_simulation(cfg)
    end = time.time()
    print(f"Total simulation time: {end - start:.4f} seconds")
    metrics = compute_metrics(cfg, sim)
    if cfg.show_apd_at_end and not cfg.animate:
        show_activation_maps(cfg, sim)
    # print(f"state activation {sim.activation_step}")
    # print(f"state deactivation {sim.deactivation_step}")
    print(f"Simulation finished in {metrics['num_steps']} steps")
    print(f"Coverage: {metrics['coverage']:.3f}")
    print(f"APD steps: {metrics['apd_steps']:}")
    # for step_index, count in enumerate(sim.activation_counts, start=1):
    #     print(f"Step {step_index}: {count} cells transitioned 0->1")
    iter = 1
    print(f"cells transitioned 0->1 at step {iter}: {sim.activation_counts[iter-1]}")


if __name__ == "__main__":
    main()
