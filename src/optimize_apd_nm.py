from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

# Pull constants and building blocks from your package
from cardiac_ca import (
    DT_MS, EXCITED_STAGES, KERNEL_SIZE,
    THETA, ALPHA, SIGMA_TRANS, ANISO_RATIO, KERNEL_GAIN, FIBER_ANGLE_DEG,
    anisotropic_gaussian_kernel, initialize_grid, CardiacCA, APDTracker
)

# --- Target APD in ms ---
APD_TARGET_MS = 272.0

def run_short_sim_apd_fast(T_ref_ms: float) -> float:
    """Headless CA run: tiny grid + short time so each objective is ~fast."""
    # SMALL GRID for speed
    H = W = 81
    dt_ms = DT_MS

    # map ms -> steps
    refractory_steps = max(1, int(round(T_ref_ms / dt_ms)))
    k_exc = EXCITED_STAGES

    # kernel built from fixed parameters for now
    sigma_long = ANISO_RATIO * SIGMA_TRANS
    kernel = anisotropic_gaussian_kernel(
        size=KERNEL_SIZE,
        sigma_long=sigma_long,
        sigma_trans=SIGMA_TRANS,
        angle_deg=FIBER_ANGLE_DEG,
        gain=KERNEL_GAIN,
    )

    # init
    g = initialize_grid(H, W, stim_type="bottom_cap", stim_size=21, stim_depth=3)
    V = np.zeros_like(g, dtype=np.float32)
    tracker = APDTracker(H, W, dt_ms=dt_ms)
    ca = CardiacCA(
        grid=g, V=V, kernel=kernel,
        refractory_period=refractory_steps,
        theta=THETA, alpha=ALPHA, k_exc=k_exc,
        lat_tracker=None, data_collector=None, apd_tracker=tracker
    )

    # SHORT RUN: ~2 cycles (or min 800 steps)
    per_cycle_steps = k_exc + refractory_steps
    steps = int(max(2 * per_cycle_steps, 800))
    for t in range(steps):
        ca.step(t)

    apds = tracker.apd_list_ms()
    # require some completions to be reliable but keep it low for speed
    if apds.size < 50:
        return np.nan
    return float(np.mean(apds))

# 1D objective function
def objective_1d(x):
    (T_ref_ms,) = np.atleast_1d(x).astype(float)
    # soft bounds to keep the search sane
    if T_ref_ms < 20 or T_ref_ms > 600:
        return 1e4 + (T_ref_ms - 260.0)**2
    apd_mean = run_short_sim_apd_fast(T_ref_ms)
    if not np.isfinite(apd_mean):
        return 1e5
    rel_err = (apd_mean - APD_TARGET_MS) / max(1e-6, APD_TARGET_MS)
    return float(rel_err**2)

# initial guess + simplex for 1D (shape must be (2,1))
x0 = np.array([260.0], dtype=float)
initial_simplex = np.vstack([x0, x0 + 30.0])

evals = {"n": 0}
def cb(xk):
    evals["n"] += 1
    print(f"[iter {evals['n']:02d}] T_ref_ms={xk[0]:.2f}, J={objective_1d(xk):.5f}")

res = minimize(
    objective_1d, x0, method="Nelder-Mead",
    options=dict(maxiter=60, fatol=4e-3, xatol=4e-3, adaptive=True, initial_simplex=initial_simplex),
    callback=cb
)

print("\nSuccess:", res.success)
print("Message:", res.message)
print("Final J:", res.fun)
best_T_ref_ms = float(res.x[0])
best_refractory_steps = int(round(best_T_ref_ms / DT_MS))
print("Best T_ref_ms:", best_T_ref_ms)
print("Best refractory_steps:", best_refractory_steps)
