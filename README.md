# Cellular Automata LAT Optimisation Workflow

This README documents the two standalone scripts that live alongside the legacy project:

- `cardiac_ca_copy.py` – runs the 2-D cardiac cellular automaton (CA) and reports LAT/APD metrics.  
- `optimize_kernel_threshold.py` – performs Nelder–Mead optimisation of the CA kernel weights and Gaussian threshold map so that the simulated LAT field matches a mono-domain reference.

These scripts work together to fit the CA parameters to the reference data in `single_CL_LAT.dat`, then re-run the CA with the optimised settings.

---


## Optimising LAT (``optimize_kernel_threshold.py``)

Run the optimiser from the repository root:

```bash
python optimize_kernel_threshold.py --verbose \
    --initial-simplex-scale 0.1 \
    --max-iter 300 \
    --tol 1e-3 \
    --save optimised_kernel_thresholds.npz
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--target PATH` | `single_CL_LAT.dat` | Mono-domain reference LAT grid (text file). Must reshape to the CA grid size. |
| `--max-iter N` | `60` | Nelder–Mead iteration limit. Increase (e.g., 300–500) for higher dimensional kernels. |
| `--tol VALUE` | `1e-3` | Termination tolerances applied to simplex size (`xatol`) and function value (`fatol`). |
| `--verbose` | off | Print the objective value for every evaluation. Helpful to inspect convergence. |
| `--save PATH` | `optimised_kernel_thresholds.npz` | Persist the best kernel/thresholds, the final objective, and success flag. |
| `--initial-simplex-scale VALUE` | `None` | If provided, perturbs each parameter by `VALUE × max(|x0|, 1e-3)` when constructing the initial simplex. Use values in `[0.05, 0.2]` to explore more aggressively. |

### What the optimiser does

1. Builds an initial 15×15 anisotropic Gaussian kernel (centre weight forced to zero) and concatenates it with four Gaussian-threshold scalars: `threshold_min`, `threshold_max`, `threshold_sigma_y_frac`, `threshold_sigma_x_frac`.
2. For each parameter vector proposed by Nelder–Mead it:
   - Reshapes the vector back into a normalised kernel.
   - Clamps threshold scalars inside configured bounds.
   - Runs `cardiac_ca_copy.Config` with the custom kernel.
   - Computes the summed squared relative error between simulated and reference LAT on all cells where both are valid. (Penalties for missing coverage can be configured via the constants near the top of the file.)
3. On exit, prints the best parameters and saves them if `--save` is supplied.

The saved `.npz` contains:

```
kernel                     # 2-D array (kernel_size x kernel_size)
threshold_min              # float
threshold_max              # float
threshold_sigma_y_frac     # float
threshold_sigma_x_frac     # float
objective                  # final objective value
success / message          # SciPy solver status
```

---

## Running the CA with optimised parameters (``cardiac_ca_copy.py``)

Run the simulation with either the default configuration or an exported `.npz`:

```bash
# Default configuration
python cardiac_ca_copy.py

# Use optimised kernel/thresholds
python cardiac_ca_copy.py --params optimised_kernel_thresholds.npz
```

### What happens during a run

1. The script constructs a `Config` dataclass with grid size `401×401`, stimulus placement on the bottom cap, and CA timings.
2. If `--params` is supplied, `_apply_saved_parameters` injects the stored kernel and threshold values, and updates `Config.kernel_size` to match the saved kernel.
3. The CA runs until it returns to all-resting or hits `max_steps`.
4. Summary metrics are printed:
   - Wall-clock runtime.
   - Simulation step count and coverage (fraction of cells that activated).
   - Activation/deactivation step range and mean APD steps.
5. If `Config.show_apd_at_end` is set, LAT/APD heat maps are rendered via `viz_metrics.py`.

You can edit the defaults in the `Config` dataclass (kernel size, refractory steps, stimulus geometry) or override them inside your own script before calling `run_simulation`.

---


