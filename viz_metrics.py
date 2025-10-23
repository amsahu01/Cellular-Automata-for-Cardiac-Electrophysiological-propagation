from __future__ import annotations

import time
from typing import Callable, Tuple, TYPE_CHECKING

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from cardiac_ca import Config, SimState


def state_colormap(name: str) -> Tuple[mcolors.Colormap, mcolors.Normalize]:
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
        norm = mcolors.Normalize(vmin=0, vmax=2)
    return cmap, norm


def _steps_to_ms(array: np.ndarray, dt_ms: float) -> np.ndarray:
    arr = array.astype(np.float64, copy=False)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = arr >= 0
    if np.any(mask):
        out[mask] = arr[mask] * dt_ms
    return out


def format_status(sim: "SimState", wall_time_s: float) -> str:
    num_excited = int(np.count_nonzero(sim.state == 1))
    num_refrac = int(np.count_nonzero(sim.state == 2))
    return (
        f"t = {sim.time_ms:7.1f} ms\n"
        f"step = {sim.step:6d}\n"
        f"wall_t = {wall_time_s:6.2f} s\n"
        f"excited = {num_excited:6d}\n"
        f"refractory = {num_refrac:6d}"
    )


def animate_simulation(
    cfg: "Config",
    sim: "SimState",
    kernel: np.ndarray,
    step_fn: Callable[["Config", "SimState", np.ndarray], None],
    stop_fn: Callable[["Config", "SimState"], bool],
) -> None:
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
            if stop_fn(cfg, sim):
                break
            step_fn(cfg, sim, kernel)
        image.set_data(sim.state)
        wall_elapsed = time.perf_counter() - start_wall
        hud.set_text(format_status(sim, wall_elapsed))
        if stop_fn(cfg, sim):
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
        show_activation_maps(cfg, sim)


def show_activation_maps(cfg: "Config", sim: "SimState") -> None:
    activation_ms = _steps_to_ms(sim.activation_step, cfg.dt_ms)
    deactivation_ms = _steps_to_ms(sim.deactivation_step, cfg.dt_ms)
    apd_ms = deactivation_ms - activation_ms

    datasets = [
        (activation_ms, "Activation Time (ms)"),
        (deactivation_ms, "Deactivation Time (ms)"),
        (apd_ms, "APD (ms)"),
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


def compute_metrics(cfg: "Config", sim: "SimState") -> dict[str, float]:
    activation = sim.activation_step.astype(np.int64, copy=False)
    deactivation = sim.deactivation_step.astype(np.int64, copy=False)
    valid = (activation >= 0) & (deactivation >= 0)

    total_cells = activation.size
    coverage = float(np.count_nonzero(valid) / total_cells) if total_cells else 0.0

    metrics: dict[str, float] = {
        "t_end_ms": float(sim.step * cfg.dt_ms),
        "num_steps": int(sim.step),
        "coverage": coverage,
    }

    if np.any(valid):
        apd_steps = (deactivation - activation)[valid].astype(np.float64, copy=False)
        apd_ms = apd_steps * cfg.dt_ms
        metrics.update(
            {
                "apd_mean_ms": float(np.mean(apd_ms)),
                "apd_steps": int(np.mean(apd_steps)),
                "apd_median_ms": float(np.median(apd_ms)),
                "apd_min_ms": float(np.min(apd_ms)),
                "apd_max_ms": float(np.max(apd_ms)),
                "activation_spread_ms": float(
                    (deactivation[valid].max() - activation[valid].min()) * cfg.dt_ms
                ),
            }
        )
    else:
        metrics.update(
            {
                "apd_mean_ms": float("nan"),
                "apd_steps": float("nan"),
                "apd_median_ms": float("nan"),
                "apd_min_ms": float("nan"),
                "apd_max_ms": float("nan"),
                "activation_spread_ms": float("nan"),
            }
        )
    return metrics
