"""
This script uses the Nelder–Mead simplex algorithm to minimise the squared relative error
between the CA LAT map and a reference monodomain LAT map, by adjusting three parameters:
- sigma_trans (transverse spread of the Gaussian kernel)
- aniso_ratio (ratio of longitudinal to transverse spread)
- threshold (activation threshold)      
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
import time

# Project imports
from cardiac_ca import Config, run_simulation


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Path to the reference monodomain LAT data
LAT_PATH = Path("single_CL_LAT.dat")

# Simulation settings for the CA model.
MAX_STEPS = 10000         # allow enough steps for the wave to cover the tissue
DT_OVERRIDE: float | None = None  # set to a float (ms) to override Config.dt_ms

# Nelder–Mead needs an initial simplex: 4 vertices because we optimise 3 params.
# Each row lists [sigma_trans, aniso_ratio, threshold]. All must be positive.
INITIAL_SIMPLEX = np.array(
    [
        [2.019187, 2.408187, 0.098790],  
        [2.0, 2.5, 0.085],  
        [3.0, 3.00, 0.12],  
        [4.00, 2.0, 0.09], 

    ],
    dtype=np.float64,
)

# Objective options.
EPS_MS = 0.0           # denominator floor (ms) in the relative error
PRINT_EACH_EVAL = True    # log every objective evaluation for transparency
PRINT_BEST_PER_ITER = False  # log which evaluation becomes the best vertex each iteration
MAX_EVALS = 2000          # hard limit on function evaluations
TOL_X = 1e-5             # Nelder–Mead parameter tolerance
TOL_F = 1e-3              # Nelder–Mead objective tolerance (in squared-rel units)
PENALTY_LAT_MS = 1000     # LAT assigned to cells that never activate
PRINT_CA_LAT_ARRAY = True  # show full CA LAT grid (can be large)
PRINT_MONO_LAT_ARRAY = True  # show full monodomain LAT grid (can be large)


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def load_monodomain_lat(path: Path, shape: tuple[int, int]) -> NDArray[np.float64]:
    """Read the monodomain LAT file and reshape it to the CA grid."""
    data = np.loadtxt(path, dtype=np.float64)
    expected_size = shape[0] * shape[1]
    if data.size != expected_size:
        raise ValueError(f"LAT file has {data.size} entries, expected {expected_size} for grid {shape}.")
    lat_ms = np.flipud(data.reshape(shape))
    if np.any(lat_ms <= 0.0):
        raise ValueError("Monodomain LAT contains non-positive values; relative error would blow up.")
    print(
        f"[info] Monodomain LAT grid: shape={lat_ms.shape}, "
        f"min={lat_ms.min():.3f} ms, max={lat_ms.max():.3f} ms"
    )
    return lat_ms


def ensure_positive_simplex(simplex: NDArray[np.float64]) -> None:
    """Validate that every simplex vertex respects the positivity constraints."""
    if simplex.shape != (4, 3):
        raise ValueError("INITIAL_SIMPLEX must be a 4x3 array (4 vertices for 3 parameters).")
    if np.any(simplex <= 0.0):
        raise ValueError("All simplex entries must be strictly > 0 (constraint requested by the user).")


def to_raw(params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map positive physical parameters into an unconstrained (log) space."""
    return np.log(params)


def to_physical(raw: NDArray[np.float64]) -> tuple[float, float, float]:
    """Map unconstrained optimiser variables back into physical space (> 0)."""
    sigma_trans = float(np.exp(raw[0]))
    aniso_ratio = float(np.exp(raw[1]))
    threshold = float(np.exp(raw[2]))
    return sigma_trans, aniso_ratio, threshold


def activation_lat_in_ms(steps: NDArray[np.int32], dt_ms: float) -> NDArray[np.float64]:
    """Convert activation steps to milliseconds, using PENALTY_LAT_MS for inactive cells."""
    lat_ms = steps.astype(np.float64) * dt_ms
    lat_ms[steps < 0] = PENALTY_LAT_MS
    return lat_ms


def squared_relative_error(ca_ms: NDArray[np.float64], mono_ms: NDArray[np.float64], eps_ms: float) -> float:
    """Compute the summed squared relative error between two LAT grids."""
    if ca_ms.shape != mono_ms.shape:
        raise ValueError(f"Grid mismatch: CA grid {ca_ms.shape} vs monodomain grid {mono_ms.shape}.")
    denom = np.maximum(mono_ms, eps_ms)
    residual = (ca_ms - mono_ms) / denom
    return float(np.sum(residual * residual))


def build_objective(
    cfg: Config,
    mono_lat_ms: NDArray[np.float64],
    eps_ms: float,
    eval_log: list[dict[str, Any]],
    verbose: bool,
):
    """Create the objective function that Nelder–Mead will minimise."""
    report_state = {"grid": False, "array": False, "count": 0}

    def objective(raw_params: NDArray[np.float64]) -> float:
        nonlocal report_state

        report_state["count"] += 1

        sigma_trans, aniso_ratio, threshold = to_physical(raw_params)

        # Build a config instance with the proposed parameters (constraints handled via exp).
        run_cfg = replace(
            cfg,
            sigma_trans=sigma_trans,
            aniso_ratio=aniso_ratio,
            threshold=threshold,
            animate=False,
            show_apd_at_end=False,
        )

        # Run the CA simulation and collect activation times.
        sim = run_simulation(run_cfg)
        ca_lat_ms = activation_lat_in_ms(sim.activation_step, run_cfg.dt_ms)

        # Log the CA grid statistics once so we can confirm alignment.
        if verbose and not report_state["grid"]:
            print(
                f"[info] CA LAT grid: shape={ca_lat_ms.shape}, "
                f"min={ca_lat_ms.min():.3f} ms, max={ca_lat_ms.max():.3f} ms"
            )
            report_state["grid"] = True
        if PRINT_CA_LAT_ARRAY and not report_state["array"]:
            np.set_printoptions(precision=3, suppress=True)
            print("[info] CA LAT grid values:")
            print(ca_lat_ms)
            report_state["array"] = True

        # Evaluate the mismatch.
        error_value = squared_relative_error(ca_lat_ms, mono_lat_ms, eps_ms)
        if verbose:
            print(
                f"[eval {report_state['count']:04d}] sigma_trans={sigma_trans:7.6f}, "
                f"aniso_ratio={aniso_ratio:7.6f}, "
                f"threshold={threshold:7.6f} --> J={error_value:.6f}"
            )
        eval_log.append(
            {
                "index": report_state["count"],
                "raw": np.array(raw_params, copy=True),
                "sigma_trans": sigma_trans,
                "aniso_ratio": aniso_ratio,
                "threshold": threshold,
                "objective": error_value,
            }
        )
        return error_value

    return objective


# =============================================================================
# 3. OPTIMISATION WORKFLOW
# =============================================================================

def main() -> None:
    # Step 3.1: prepare the baseline CA configuration.
    cfg = Config()
    cfg = replace(cfg, max_steps=MAX_STEPS, animate=False, show_apd_at_end=False)
    if DT_OVERRIDE is not None:
        cfg = replace(cfg, dt_ms=DT_OVERRIDE)

    # Step 3.2: load the monodomain LAT map and print basic stats.
    mono_lat_ms = load_monodomain_lat(LAT_PATH, (cfg.height, cfg.width))
    if PRINT_MONO_LAT_ARRAY:
        np.set_printoptions(precision=4, suppress=True)
        print("[info] Monodomain LAT grid values:")
        print(mono_lat_ms)


    # Step 3.3: validate and transform the simplex into raw optimiser space.
    ensure_positive_simplex(INITIAL_SIMPLEX)
    simplex_raw = np.vstack([to_raw(row) for row in INITIAL_SIMPLEX])
    print("[info] Initial simplex (physical parameters):")
    for idx, row in enumerate(INITIAL_SIMPLEX):
        print(f"       vertex {idx}: sigma_trans={row[0]:.4f}, aniso_ratio={row[1]:.4f}, threshold={row[2]:.4f}")

    # Step 3.4: build the objective and run Nelder–Mead.
    evaluation_log: list[dict[str, Any]] = []
    objective = build_objective(cfg, mono_lat_ms, EPS_MS, evaluation_log, PRINT_EACH_EVAL)

    iteration_state = {"count": 0}

    def nm_callback(xk: NDArray[np.float64]) -> None:
        iteration_state["count"] += 1
        match = next(
            (entry for entry in reversed(evaluation_log) if np.allclose(entry["raw"], xk, atol=1e-6, rtol=0.0)),
            None,
        )
        if PRINT_BEST_PER_ITER:
            if match is not None:
                print(
                    f"[iter {iteration_state['count']:04d}] best vertex accepted from eval {match['index']:04d}: "
                    f"sigma_trans={match['sigma_trans']:.6f}, aniso_ratio={match['aniso_ratio']:.6f}, "
                    f"threshold={match['threshold']:.6f}, J={match['objective']:.6f}"
                )
            else:
                sigma_trans, aniso_ratio, threshold = to_physical(xk)
                print(
                    f"[iter {iteration_state['count']:04d}] best vertex (not in log): "
                    f"sigma_trans={sigma_trans:.6f}, aniso_ratio={aniso_ratio:.6f}, threshold={threshold:.6f}"
                )

    star = time.time()
    result = minimize(
        objective,
        simplex_raw[0],
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex_raw,
            "maxfev": MAX_EVALS,
            "xatol": TOL_X,
            "fatol": TOL_F,
            "return_all": True,
        },
        callback=nm_callback,
    )
    end = time.time()
    print(f"\n[info] Optimisation time: {end - star:.2f} seconds")

    # Step 3.5: report the outcome in physical units.
    final_sigma, final_aniso, final_threshold = to_physical(result.x)
    print("\n[summary] Optimisation finished")
    print(f"          success        : {result.success}")
    print(f"          message        : {result.message}")
    print(f"          evaluations    : {result.nfev}")
    print(f"          best sigma_trans = {final_sigma:.6f}")
    print(f"          best aniso_ratio = {final_aniso:.6f}")
    print(f"          best threshold   = {final_threshold:.6f}")
    print(f"          final objective  = {result.fun:.3f}")


if __name__ == "__main__":
    main()
