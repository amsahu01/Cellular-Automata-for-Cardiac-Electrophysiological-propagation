# Cardiac Cellular Automata 

This repository hosts two main entry points for simulating and tuning a 2‑D cellular automaton (CA) model of cardiac electrophysiology:

- `cardiac_ca.py` runs the forward simulation, visualises propagation, and reports basic activation metrics.
- `LAT_optimization.py` fits the CA parameters to a reference local-activation-time (LAT) map using Nelder–Mead.

Both scripts share the same `Config` dataclass, so changes you make to the CA settings flow naturally between standalone simulations and optimisation runs.

---

## Requirements

- Python 3.10+ (the project relies on dataclass features and newer typing syntax).
- Python packages: `numpy`, `scipy`, `matplotlib` (for the helpers in `viz_metrics.py`). Use `pip install numpy scipy matplotlib`.
- A reference LAT grid file (default: `single_CL_LAT.dat`) for stimulation masks and optimisation targets.

---

## Running the Cellular Automaton (`cardiac_ca.py`)

1. **Review configuration**  
   The `Config` dataclass near the top of the file controls every aspect of the simulation:
   - **Grid & timing**: `height`, `width`, `dt_ms`, `max_steps`.
   - **Action potential durations**: `excited_steps`, `refractory_steps`.
   - **Kernel parameters**: `kernel_size`, `sigma_trans`, `aniso_ratio`, `angle_deg`, `gain`.
   - **Uniform threshold**: `threshold`.
   - **Stimulus**: `stimulus_type` (`"bottom_cap"` or `"lat_mask"`), `stimulus_size`, `stimulus_data_path`, `stimulus_value`.
   - **Visualisation**: toggles for animation (`animate`, `steps_per_frame`, `fps`, `cmap_name`) and end-of-run APD plots.

2. **Provide stimulus data (if needed)**  
   The default `stimulus_type="lat_mask"` expects `single_CL_LAT.dat` in the repository root. Each value indicates whether the corresponding cell is part of the initial activation mask (`stimulus_value` determines the selected entries). Set `stimulus_type="bottom_cap"` to bypass the file requirement.

3. **Run the simulation**
   ```bash
   python cardiac_ca.py
   ```
   The script initialises the state, repeatedly calls `simulate_step`, and optionally animates the wavefront using helpers in `viz_metrics.py`. At the end it prints:
   - Wall-clock runtime.
   - Activation-step range (`SimState.activation_step.min/max`).
   - Summary metrics from `viz_metrics.compute_metrics` (steps taken, coverage, APD).

4. **Interpret the output**  
   - `SimState` captures the evolving grid state, activation/deactivation step maps, and the per-cell threshold matrix.  
   - `animate_simulation` and `show_activation_maps` help validate propagation visually (useful when tweaking kernel anisotropy or thresholds).

---

## Optimising LAT Fit (`LAT_optimization.py`)

`LAT_optimization.py` calibrates three CA parameters—`sigma_trans`, `aniso_ratio`, and `threshold`—to minimise the squared relative error between the simulated LAT surface and a monodomain reference.

1. **Key constants (top of the file)**
   - `LAT_PATH`: path to the monodomain LAT file (default `single_CL_LAT.dat`).
   - `MAX_STEPS`, `DT_OVERRIDE`: override simulation-length/temporal resolution if desired.
   - `INITIAL_SIMPLEX`: four vertices (3 parameters + 1) defining the starting simplex in physical units; all values must stay positive.
   - Objective/solver controls: tolerances, max evaluations, denominators (`EPS_MS`), penalties for never-activated cells (`PENALTY_LAT_MS`), and verbosity switches (`PRINT_*` flags).

2. **Workflow overview**
   - `load_monodomain_lat` reshapes and validates the reference LAT grid.
   - `ensure_positive_simplex`, `to_raw`, and `to_physical` enforce >0 constraints by mapping parameters into log space for SciPy's unconstrained `minimize`.
   - `build_objective` runs `run_simulation` with each candidate parameter set, converts activation steps to ms via `activation_lat_in_ms`, and accumulates an evaluation log for diagnostics.
   - `scipy.optimize.minimize(..., method="Nelder-Mead")` iterates until tolerances or `MAX_EVALS` are hit, optionally printing every evaluation and/or best vertex per iteration.

3. **Run optimisation**
   ```bash
   python LAT_optimization.py
   ```
   Expect console output describing:
   - Reference and CA grid statistics (min/max LAT in ms).
   - Every objective evaluation (parameter triple and squared relative error) if `PRINT_EACH_EVAL` is `True`.
   - Final success status, best-fit parameters, and achieved objective value.

4. **Using the fitted parameters**  
   Copy the reported `sigma_trans`, `aniso_ratio`, and `threshold` back into `cardiac_ca.Config` (or another config instance) to rerun the CA with the optimised settings. Because both scripts import the same `Config`, you can also modify `LAT_optimization.py` to seed the optimiser with custom defaults (e.g., a different grid size or stimulus) before running `minimize`.

---
