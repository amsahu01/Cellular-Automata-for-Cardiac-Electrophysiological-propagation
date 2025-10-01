"""
APD-only optimization with variable EXCITED_STAGES (``K``) and REFRACTORY_STEPS (``R``).

This script enumerates a small discrete set of excited substates ``K`` and, for each
choice, performs a 1‑D **Nelder-Mead** search over the *refractory time* ``T_ref_ms``
(then mapped to integer steps ``R = round(T_ref_ms / DT_ms)``) to minimize an
**APD-only** objective. The simulator uses a small headless run for speed.

Objective Function
----------------------

Let :math:`A_*` be the target APD (ms) and let :math:`\overline{APD}(K,R)` denote the
mean APD measured from completed episodes in a run with parameters ``(K, R)``.
We minimize the squared relative error

.. math::

    J(K,R) = \left( \frac{\overline{APD}(K,R) - A_*}{A_*} \right)^2.

**Identifiability note.** In this CA model, :math:`APD_{eff} \approx (K+R)\,DT_{ms}`,
so several ``(K, R)`` pairs can achieve similar error. We therefore enumerate ``K`` and
solve a 1‑D problem in ``T_ref_ms`` for each ``K``.

"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

# Pull constants and building blocks from your package
from cardiac_ca import (
    DT_MS, EXCITED_STAGES, KERNEL_SIZE,
    THETA, ALPHA, SIGMA_TRANS, ANISO_RATIO, KERNEL_GAIN, FIBER_ANGLE_DEG, GRID_SIZE,
    anisotropic_gaussian_kernel, initialize_grid, CardiacCA, APDTracker
)

# --- Target APD in ms ---
APD_TARGET_MS = 272.0


def run_short_sim_apd_fast(T_ref_ms: float, k_exc: int) -> float:
    """
    Run a small, headless CA simulation and return the **mean APD** (ms).

    The simulator uses an ``81x81`` grid, a bottom-cap stimulus, and an anisotropic
    Gaussian kernel with the current default parameters. The run length is chosen to be
    long enough to complete ~1-2 activation cycles for most cells.

    :param T_ref_ms: Refractory time **in milliseconds** to be mapped to integer steps
        as ``R = max(1, round(T_ref_ms / DT_MS))``.
    :type T_ref_ms: float
    :param k_exc: Number of **EXCITED_STAGES** (``K``) used by the state machine.
    :type k_exc: int
    :returns: Mean effective APD across completed episodes (ms). If fewer than
        ``50`` APD episodes complete, returns ``numpy.nan`` so the optimizer can
        penalize this configuration.
    :rtype: float
    :notes: The total number of steps is
        ``steps = max(2 * (k_exc + R), 2 * H)``, with ``H=W=81``. Boundaries are
        reflective; the kernel center weight is zero to avoid self-coupling.
    """
    # SMALL GRID for speed
    H = W = 81
    dt_ms = DT_MS

    # map ms -> steps
    refractory_steps = max(1, int(round(T_ref_ms / dt_ms)))

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
        theta=THETA, alpha=ALPHA, k_exc=int(k_exc),
        lat_tracker=None, data_collector=None, apd_tracker=tracker
    )

    # SHORT RUN: ~2 cycles (or min ~2*H steps)
    per_cycle_steps = int(k_exc) + refractory_steps
    steps = int(max(2 * per_cycle_steps, 2 * H))
    for t in range(steps):
        ca.step(t)

    apds = tracker.apd_list_ms()
    if apds.size < 50:
        return np.nan
    return float(np.mean(apds))


# Sweep excited-stage counts to explore the (near‑)degenerate APD-only fit.
K_RANGE = [1, 2, 3, 4, 5]
results = []
for K in K_RANGE:
    def objective_R_only(x: np.ndarray) -> float:
        """
        APD-only objective *for a fixed* ``K`` while optimizing ``T_ref_ms``.

        :param x: 1-D array with the current guess for ``T_ref_ms`` (ms). Only the first
            element is used; additional elements are ignored.
        :type x: numpy.ndarray
        :returns: Squared relative APD error
            ``J = ((APD_mean(T_ref_ms, K) - APD_TARGET_MS) / APD_TARGET_MS)**2``.
            Returns a large penalty if ``T_ref_ms`` is outside a loose box
            ``[20, 600]`` ms or if too few APD episodes complete.
        :rtype: float
        :notes: This function intentionally keeps the simulator deterministic and uses
            a small grid for speed. The *best* ``T_ref_ms`` found here should be
            re-validated on the full-size grid.
        """
        (T_ref_ms,) = np.atleast_1d(x).astype(float)
        if T_ref_ms < 20 or T_ref_ms > 600:
            return 1e4 + (T_ref_ms - 260.0) ** 2
        apd_mean = run_short_sim_apd_fast(T_ref_ms, k_exc=K)
        if not np.isfinite(apd_mean):
            return 1e5
        rel_err = (apd_mean - APD_TARGET_MS) / max(1e-6, APD_TARGET_MS)
        return float(rel_err ** 2)

    # Initial simplex around a reasonable starting point
    x0 = np.array([260.0], dtype=float)
    initial_simplex = np.vstack([x0, x0 + 30.0])

    res = minimize(
        objective_R_only,
        x0,
        method="Nelder-Mead",
        options=dict(maxiter=60, fatol=4e-3, xatol=4e-3, adaptive=True, initial_simplex=initial_simplex),
    )
    best_T_ref = float(res.x[0])
    best_R = int(round(best_T_ref / DT_MS))
    results.append(dict(K=K, J=res.fun, T_ref_ms=best_T_ref, R_steps=best_R))
    print(f"K={K}: Best T_ref_ms={best_T_ref:.2f}, R_steps={best_R}, J={res.fun:.5f}")
