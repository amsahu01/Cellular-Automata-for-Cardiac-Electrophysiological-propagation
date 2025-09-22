from __future__ import annotations
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .constants import DT_MS
from .apd import APDTracker  
from .grid import initialize_grid
from .lat import LATTracker
from .data import SimulationDataCollector
from .ca import CardiacCA
from .viz import create_animation_elements
from .config import SimulationConfig


class SimulationApp:
    """High-level orchestrator for running the CA, animation, and summaries."""

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.kernel = self.cfg.build_kernel()
        gsize = self.cfg.grid_size
        self.grid = initialize_grid(
            gsize, gsize,
            stim_type=self.cfg.stimulus_type,
            stim_size=self.cfg.stimulus_size,
            stim_depth=self.cfg.stim_depth,
        )
        self.V = np.zeros_like(self.grid, dtype=np.float32)
        self.lat_tracker = LATTracker(gsize, gsize)
        self.apd_tracker = APDTracker(gsize, gsize, dt_ms=DT_MS)
        self.data = SimulationDataCollector(gsize, gsize)
        self.ca = CardiacCA(
            grid=self.grid,
            V=self.V,
            kernel=self.kernel,
            refractory_period=self.cfg.refractory_steps,
            theta=self.cfg.theta,
            alpha=self.cfg.alpha,
            k_exc=self.cfg.excited_stages,
            lat_tracker=self.lat_tracker,
            apd_tracker=self.apd_tracker,
            data_collector=self.data,
        )

    def _title(self, frame_idx: int) -> str:
        rho = self.cfg.sigma_long / self.cfg.sigma_trans
        return (
            "t={:d}  theta={:.3f} alpha={:.3f}  ".format(frame_idx + 1, self.cfg.theta, self.cfg.alpha)
            + "rho={:.2f} phi={}°  ".format(rho, int(self.cfg.fiber_angle_deg))
            + "RP={}  K={}".format(self.cfg.refractory_steps, self.cfg.excited_stages)
        )

    def run(self) -> None:
        gsize = self.cfg.grid_size
        fig_anim, ax_anim, img_anim = create_animation_elements(
            self.ca.grid, self.cfg.refractory_steps, self.cfg.excited_stages
        )

        def anim_func(frame_idx: int):
            new_grid, _ = self.ca.step(frame_idx)
            img_anim.set_array(new_grid)
            ax_anim.set_title(self._title(frame_idx))
            if (frame_idx + 1) in self.cfg.screenshot_steps:
                fig_anim.savefig(
                    f"simulation_step_{frame_idx + 1}.png", dpi=150
                )
            return [img_anim]

        ani = FuncAnimation(
            fig_anim,
            anim_func,
            frames=self.cfg.time_steps,
            interval=10,
            blit=True,
            repeat=False,
        )
        fig_anim.tight_layout()
        print(
            f"Starting simulation: Size={gsize}x{gsize}, Steps={self.cfg.time_steps}"
        )
        plt.show()

        # ------------------- LAT/CV summary and plots ---------------------------
        LAT_steps = self.lat_tracker.lat_steps

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

        # --- APD summary (effective: excited + refractory) ---
        apds = self.apd_tracker.apd_list_ms()
        if apds.size:
            apd_mean = float(np.mean(apds))
            apd_std  = float(np.std(apds))
            print(f"CA APD_effective: mean={apd_mean:.1f} ms, std={apd_std:.1f} ms, n={apds.size}")
            # Optional: quick histogram
            # import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4))
            plt.hist(apds, bins=40)
            plt.xlabel("APD_effective (ms)")
            plt.ylabel("Count")
            plt.title("CA APD_effective distribution")
            plt.tight_layout()
            plt.savefig("apd_hist.png", dpi=150)
            plt.show()
        else:
            print("CA APD_effective: none observed (try increasing time_steps).")

        print("Done. Saved: lat_map.png and optional step screenshots.")