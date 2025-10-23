# python optimize_kernel_threshold.py --verbose  --initial-simplex-scale 0.2 --max-iter 100 --tol 1e-3 --save optimised_kernel_thresholds.npz
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from cardiac_ca_copy import Config, anisotropic_gaussian_kernel, run_simulation, EPS


# Clamp range for the minimum Gaussian activation threshold (used in unpack_threshold_params).
THRESH_MIN_BOUNDS = (0.001, 0.99)
# Clamp range for the maximum Gaussian activation threshold (used in unpack_threshold_params).
THRESH_MAX_BOUNDS = (0.001, 0.99)
# Allowed fractional spreads for the Gaussian threshold along y/x axes.
SIGMA_FRAC_BOUNDS = (1e-4, 1.0)
# Penalty for each reference cell the CA fails to activate when scoring LAT error.
PENALTY_MISSING = 1e4
# Penalty added if the simulation produces no valid LAT field or throws.
PENALTY_FAILURE = 1e6
# Default convolution kernel size used for optimisation and simulation runs.
DEFAULT_KERNEL_SIZE = 15


# Load mono-domain LAT grid from disk and validate its size.
def load_reference_lat(path: Path, shape: tuple[int, int]) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    expected = shape[0] * shape[1]
    if data.size != expected:
        raise ValueError(f"Expected {expected} LAT entries, found {data.size} in '{path}'.")
    lat = data.reshape(shape)
    if not np.all(np.isfinite(lat)):
        raise ValueError("Reference LAT grid contains non-finite values.")
    return lat


# Convert the flattened optimisation vector into a normalised convolution kernel.
def build_kernel_from_vector(vector: Sequence[float], size: int = DEFAULT_KERNEL_SIZE) -> np.ndarray:
    arr = np.abs(np.asarray(vector, dtype=np.float64).reshape(size, size))
    centre = size // 2
    arr[centre, centre] = 0.0  # disable self-feedback
    total = arr.sum()
    if total <= EPS:
        fallback = np.ones((size, size), dtype=np.float64)
        fallback[centre, centre] = 0.0
        arr = fallback
        total = arr.sum()
    arr /= total
    return arr.astype(np.float32, copy=False)


# Clamp a scalar into inclusive bounds.
def clamp(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(np.clip(value, lower, upper))


# Map four raw threshold parameters into constrained values.
def unpack_threshold_params(params: Sequence[float]) -> tuple[float, float, float, float]:
    if len(params) != 4:
        raise ValueError("Threshold parameter vector must contain exactly four values.")
    t_min_raw, t_max_raw, sig_y_raw, sig_x_raw = map(float, params)
    t_min = clamp(t_min_raw, THRESH_MIN_BOUNDS)
    t_max = clamp(t_max_raw, THRESH_MAX_BOUNDS)
    if t_max <= t_min:
        t_max = t_min + 1e-3
    sigma_y_frac = clamp(sig_y_raw, SIGMA_FRAC_BOUNDS)
    sigma_x_frac = clamp(sig_x_raw, SIGMA_FRAC_BOUNDS)
    return t_min, t_max, sigma_y_frac, sigma_x_frac


# Convert activation step indices to LAT values in milliseconds.
def activation_steps_to_lat(sim_steps: np.ndarray, dt_ms: float) -> np.ndarray:
    lat = np.full(sim_steps.shape, np.nan, dtype=np.float64)
    valid = sim_steps >= 0
    if np.any(valid):
        lat[valid] = sim_steps[valid].astype(np.float64) * float(dt_ms)
    return lat


# Compute summed squared relative LAT error with penalties for coverage gaps.
def summed_squared_relative_error(model: np.ndarray, reference: np.ndarray) -> float:
    if model.shape != reference.shape:
        raise ValueError("Model and reference LAT grids must share the same shape.")

    ref_valid = np.isfinite(reference) & (reference > 0.0)
    model_valid = np.isfinite(model)

    aligned = ref_valid & model_valid
    if not np.any(aligned):
        return PENALTY_FAILURE

    rel = (model[aligned] - reference[aligned]) / np.maximum(reference[aligned], EPS)
    error = float(np.sum(rel**2))

    missing = ref_valid & ~model_valid
    if np.any(missing):
        error += PENALTY_MISSING * float(np.count_nonzero(missing))

    return error


class ObjectiveEvaluator:
    # Construct evaluator wrapper that tracks calls and holds shared state.
    def __init__(
        self,
        base_cfg: Config,
        target_lat: np.ndarray,
        verbose: bool = False,
    ) -> None:
        self.base_cfg = base_cfg
        self.target_lat = target_lat
        self.verbose = verbose
        self.calls = 0

    # Evaluate the objective for a given optimisation parameter vector.
    def __call__(self, params: np.ndarray) -> float:
        self.calls += 1
        try:
            k = self.base_cfg.kernel_size ** 2
            kernel = build_kernel_from_vector(params[:k], self.base_cfg.kernel_size)
            thresholds = unpack_threshold_params(params[k:])
            cfg = replace(
                self.base_cfg,
                custom_kernel=kernel,
                threshold_strategy="gaussian",
                threshold_min=thresholds[0],
                threshold_max=thresholds[1],
                threshold_sigma_y_frac=thresholds[2],
                threshold_sigma_x_frac=thresholds[3],
            )

            sim = run_simulation(cfg)
            lat = activation_steps_to_lat(sim.activation_step, cfg.dt_ms)
            # np.set_printoptions(precision=4, suppress=True)
            # print(lat)
            error = summed_squared_relative_error(lat, self.target_lat)

            coverage = float(np.count_nonzero(np.isfinite(lat))) / lat.size
            if coverage <= 0.0:
                error += PENALTY_FAILURE

        except Exception as exc:  
            error = PENALTY_FAILURE
            if self.verbose:
                print(f"[ObjectiveEvaluator] Evaluation {self.calls} failed: {exc}")

        if self.verbose:
            print(f"[ObjectiveEvaluator] Call {self.calls:03d} -> error={error:.6f}")
        return error


# Build the starting optimisation vector from the current configuration.
def initial_parameter_vector(cfg: Config) -> np.ndarray:
    sigma_long = cfg.aniso_ratio * cfg.sigma_trans
    kernel = anisotropic_gaussian_kernel(
        size=cfg.kernel_size,
        sigma_long=sigma_long,
        sigma_trans=cfg.sigma_trans,
        angle_deg=cfg.angle_deg,
        gain=cfg.gain,
    )
    threshold_params = np.array(
        [
            cfg.threshold_min,
            cfg.threshold_max,
            cfg.threshold_sigma_y_frac,
            cfg.threshold_sigma_x_frac,
        ],
        dtype=np.float64,
    )
    return np.concatenate([kernel.astype(np.float64).ravel(), threshold_params])


# Run Nelder–Mead with standard tolerances (optionally custom simplex) and return best result.
def optimise_parameters(
    evaluator: ObjectiveEvaluator,
    initial: np.ndarray,
    max_iter: int,
    tol: float,
    simplex_scale: float | None = None,
) -> tuple[np.ndarray, float, bool, str]:
    options = {"maxiter": max_iter, "xatol": tol, "fatol": tol, "maxfev": max_iter * 4}
    if simplex_scale is not None and simplex_scale > 0.0:
        base = initial.astype(np.float64, copy=True)
        n = base.size
        step = simplex_scale * np.maximum(np.abs(base), 1e-3)
        simplex = np.vstack([base, base + np.eye(n) * step])
        options["initial_simplex"] = simplex
    result = minimize(evaluator, initial, method="Nelder-Mead", options=options)
    return result.x, float(result.fun), bool(result.success), str(result.message)


# Parse CLI arguments controlling optimisation inputs and runtime options.
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise CA kernel weights and threshold parameters via Nelder-Mead."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("single_CL_LAT.dat"),
        help="Path to the mono-domain LAT reference file (default: single_CL_LAT.dat).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=60,
        help="Maximum Nelder-Mead iterations."
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Termination tolerance applied to simplex size and function value.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print objective progress for each evaluation.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("optimised_kernel_thresholds.npz"),
        help="Optional path to save the optimised kernel and thresholds (.npz format).",
    )
    parser.add_argument(
        "--initial-simplex-scale",
        type=float,
        default=None,
        help=(
            "Relative perturbation scale for constructing the Nelder-Mead initial simplex. "
            "Example: 0.1 perturbs each parameter by 10%% of its magnitude."
        ),
    )
    return parser.parse_args(argv)


# Entry point for command-line execution of the optimiser.
def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    base_cfg = Config()
    base_cfg = replace(
        base_cfg,
        animate=False,
        max_steps=base_cfg.max_steps or 10000,
        kernel_size=DEFAULT_KERNEL_SIZE,
    )

    target_lat = load_reference_lat(args.target, (base_cfg.height, base_cfg.width))
    target_lat = np.flipud(target_lat)
    # np.set_printoptions(precision=4, suppress=True)
    # print("[info] Monodomain LAT grid values:")
    # print(target_lat)

    initial = initial_parameter_vector(base_cfg)
    evaluator = ObjectiveEvaluator(base_cfg, target_lat, verbose=args.verbose)

    best_params, best_value, success, message = optimise_parameters(
        evaluator,
        initial,
        max_iter=args.max_iter,
        tol=args.tol,
        simplex_scale=args.initial_simplex_scale,
    )

    k = base_cfg.kernel_size ** 2
    best_kernel = build_kernel_from_vector(best_params[:k], base_cfg.kernel_size)
    best_thresholds = unpack_threshold_params(best_params[k:])

    print("Nelder-Mead optimisation finished.")
    print(f"  Success: {success}")
    print(f"  Message: {message}")
    print(f"  Best objective: {best_value:.6f}")
    print("  Threshold parameters:")
    print(f"    min={best_thresholds[0]:.6f}")
    print(f"    max={best_thresholds[1]:.6f}")
    print(f"    sigma_y_frac={best_thresholds[2]:.6f}")
    print(f"    sigma_x_frac={best_thresholds[3]:.6f}")
    print("  Kernel weights (normalised, centre excluded):")
    np.set_printoptions(precision=6, suppress=True)
    print(best_kernel)

    if args.save is not None:
        np.savez_compressed(
            args.save,
            kernel=best_kernel,
            threshold_min=best_thresholds[0],
            threshold_max=best_thresholds[1],
            threshold_sigma_y_frac=best_thresholds[2],
            threshold_sigma_x_frac=best_thresholds[3],
            objective=best_value,
            success=success,
            message=message,
        )
        print(f"Saved optimised parameters to {args.save}.")


if __name__ == "__main__":
    main()
