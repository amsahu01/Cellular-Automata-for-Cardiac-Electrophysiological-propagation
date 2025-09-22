# Cardiac Electrophysiological Propagation (Cellular Automata)

A Python implementation of a 2-D **cardiac excitation** model using a **hybrid cellular automaton** (CA): integer states for *rest/excited/refractory* plus a floating “voltage-like” accumulator driven by an **anisotropic Gaussian kernel**. The repo includes tools to compute **LAT** (Local Activation Time), **APD\_effective** (excited + refractory), basic visualization, and a **Nelder–Mead** optimizer for parameter fitting.

---

## Features

* **Hybrid CA update rule** with anisotropic neighbor drive (convolutional kernel).
* **Stimuli** presets (center, bottom\_center, bottom\_cap).
* **Metrics**

  * **LAT** map and coverage.
  * **APD\_effective** tracker (time in excited + refractory).
* **Visualization**

  * Animation frames (optional snapshots).
  * Heatmaps (LAT, APD histogram).
* **Optimization**

  * Fast 1-D Nelder–Mead demo fitting **refractory time** to a target APD.


---

## Project structure

```
/src
  /cardiac_ca
    __init__.py
    apd.py           # APDTracker: measures APD_effective (state 1 → back to 0)
    app.py           # SimulationApp orchestrator (run loop, plots, summaries)
    ca.py            # CardiacCA + update rule (convolution + state machine)
    config.py        # SimulationConfig (parameters, kernel builder)
    constants.py     # Defaults: DX, DT_MS, EXCITED_STAGES, etc.
    data.py          # Simple per-frame activity stats
    grid.py          # Stimulus / initial condition builders
    kernel.py        # Anisotropic Gaussian kernel (+ factory)
    lat.py           # LATTracker
    viz.py           # Colormap, imshow helpers
  run_sim.py         # Run a full CA simulation with plots
  optimize_apd_nm.py # Fast Nelder–Mead (1-D): fits refractory time to APD
  lat_map.png        # Generated LAT heatmap (example)
  apd_hist.png       # Generated APD histogram (example)
  simulation_step_*.png  # Optional snapshots
```

---

## Installation

> Python 3.9+ recommended.

```bash
# (optional) create a clean environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install -U numpy scipy matplotlib
```

---

## How the model works 

* Each cell has an **integer state**:

  * `0` RESTING,
  * `1..k_exc` EXCITED (k\_exc = `EXCITED_STAGES`),
  * `k_exc+1 .. k_exc+REFRACTORY_STEPS` REFRACTORY, then back to `0`.
* A **voltage-like accumulator** `V` integrates neighbor input:

  * Neighbors are cells in **excited** states; their mask is convolved with an **anisotropic Gaussian kernel** (fiber angle & anisotropy).
  * Resting cells update `V_next = α·V + (1−α)·I`; if `V_next ≥ θ`, they fire (`state=1`) and reset `V` locally.
* **APD\_effective** (your definition) is:

  ```
  APD_effective = (EXCITED_STAGES + REFRACTORY_STEPS) * DT_MS
  ```

  measured per-cell by `APDTracker` as *time from first entering state 1 to first return to 0*.

---

## Run a simulation

```bash
python src/run_sim.py
```


Artifacts:

* `lat_map.png` – LAT heatmap (ms).
* `apd_hist.png` – Histogram of APD\_effective (ms).
* `simulation_step_*.png` – Optional snapshots at configured frames.

**Where to change parameters**: `src/cardiac_ca/constants.py` and `src/cardiac_ca/config.py`.

Key defaults (you can edit):

```python
# constants.py
DT_MS = 0.1                 # ms per step
EXCITED_STAGES = 3          # number of excited substates
REFRACTORY_STEPS = 200      # refractory length in steps
ANISO_RATIO = 1.5           # sigma_long / sigma_trans
SIGMA_TRANS = 4.0
FIBER_ANGLE_DEG = 90.0
KERNEL_GAIN = 3

# config.py (simulation run settings)
time_steps = 300            # total simulation steps
grid_size = 401
stimulus_type = "bottom_cap"
```

> **Tip**: Ensure `time_steps` is large enough for most cells to return to rest at least once, i.e. `time_steps ≥ (EXCITED_STAGES + REFRACTORY_STEPS) + max(LAT_steps)`.

---

## APD & LAT definitions

* **LAT** (Local Activation Time): first time a cell enters `state == 1`.

  * Tracked by `LATTracker`: `lat_steps` → `ms` via `DT_MS`.
* **APD\_effective**: first entry into `state == 1` → first return to `state == 0`.

  * Tracked by `APDTracker` (per-cell episodes; you can inspect the distribution).

---

## Fast Nelder–Mead demo (fit APD)

The provided `src/optimize_apd_nm.py` is a **1-D** fast prototype that fits **refractory time (ms)** so the CA’s **mean APD** matches a target.
Run:

```bash
python src/optimize_apd_nm.py
```

What it does:

* Uses a tiny grid (`81×81`) and short runs for quick function evaluations.
* Optimizes **T\_ref\_ms** (converted inside the simulator to integer steps).
* Objective:

  $$
  J = \left(\frac{\overline{APD}_{CA} - APD_{target}}{APD_{target}}\right)^2
  $$



---


This project draws on concepts from cardiac electrophysiology (APD, LAT, CV) and classic derivative-free optimization (Nelder–Mead).
