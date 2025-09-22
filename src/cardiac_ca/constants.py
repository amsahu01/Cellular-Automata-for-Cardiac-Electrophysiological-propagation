# ----------------------------- States ---------------------------------------

#: Integer state for resting cells.
RESTING = 0
#: First excited substate (1..K are excited).
EXCITED = 1

# ------------------- Physical & numerical resolution ------------------------

#: Spatial resolution in millimetres per node.
DX_MM = 0.10
#: Temporal resolution in milliseconds per step.
DT_MS = 0.1
#: Refractory duration expressed in CA steps.
REFRACTORY_STEPS = 200
#: Square grid size (height == width).
GRID_SIZE = 401

# ------------------- Anisotropic kernel parameters --------------------------

#: Anisotropy ratio (sigma_long / sigma_trans) \(\CV ratio\).
ANISO_RATIO = 1.5
#: Transverse sigma in cells.
SIGMA_TRANS = 4.0
#: Longitudinal sigma in cells.
SIGMA_LONG = ANISO_RATIO * SIGMA_TRANS
#: Fiber angle in degrees (0 == +x direction).
FIBER_ANGLE_DEG = 90.0
#: Odd kernel size; larger gives smoother fronts but higher cost.
KERNEL_SIZE = 15
#: Post-normalization gain applied to the kernel.
KERNEL_GAIN = 3

# ------------------- Activation parameters ---------------------------

#: Excitation threshold on V.
THETA = 0.085
#: Leak (memory) factor for V.
ALPHA = 0.94
#: Number of excited substates (thickness of excited rim).
EXCITED_STAGES = 3

# ------------------- Visualization -----------------------------------------

#: Matplotlib ``imshow`` interpolation.
IMSHOW_INTERPOLATION = "bilinear"

