from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from .constants import (
    GRID_SIZE, REFRACTORY_STEPS,
    KERNEL_SIZE, SIGMA_TRANS, ANISO_RATIO, FIBER_ANGLE_DEG, KERNEL_GAIN,
    THETA, ALPHA, EXCITED_STAGES,
)
from .kernel import anisotropic_gaussian_kernel


@dataclass
class SimulationConfig:
    grid_size: int = GRID_SIZE
    time_steps: int = 300
    refractory_steps: int = REFRACTORY_STEPS

    # Stimulus
    stimulus_type: str = "bottom_cap"
    stimulus_size: int = 27
    stim_depth: int = 3

    # Kernel
    kernel_size: int = KERNEL_SIZE
    sigma_trans: float = SIGMA_TRANS
    aniso_ratio: float = ANISO_RATIO
    fiber_angle_deg: float = FIBER_ANGLE_DEG
    kernel_gain: float = KERNEL_GAIN

    # Activation
    theta: float = THETA
    alpha: float = ALPHA
    excited_stages: int = EXCITED_STAGES

    # IO/visual
    screenshot_steps: Tuple[int, ...] = (1, 80, 200)

    @property
    def sigma_long(self) -> float:
        return self.aniso_ratio * self.sigma_trans

    def build_kernel(self):
        return anisotropic_gaussian_kernel(
            size=self.kernel_size,
            sigma_long=self.sigma_long,
            sigma_trans=self.sigma_trans,
            angle_deg=self.fiber_angle_deg,
            gain=self.kernel_gain,
        )