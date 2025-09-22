from __future__ import annotations
import numpy as np


def anisotropic_gaussian_kernel(
    size: int = 21,
    sigma_long: float = 3.5,
    sigma_trans: float = 1.8,
    angle_deg: float = 0.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Create a normalized anisotropic Gaussian kernel.

    The kernel is oriented by ``angle_deg`` and normalized to sum to 1 before
    applying the ``gain`` factor. The center is zeroed to avoid self-coupling.

    Parameters
    ----------
    size : int, optional
        Odd kernel size (width == height). Default is 21.
    sigma_long : float, optional
        Longitudinal standard deviation (pixels). Default is 3.5.
    sigma_trans : float, optional
        Transverse standard deviation (pixels). Default is 1.8.
    angle_deg : float, optional
        Rotation angle in degrees (0 == +x). Default is 0.
    gain : float, optional
        Post-normalization gain. Default is 1.0.

    Returns
    -------
    np.ndarray
        2‑D float32 kernel of shape ``(size, size)``.
    """
    return KernelFactory.create(size, sigma_long, sigma_trans, angle_deg, gain)


class KernelFactory:
    """Factory for creating anisotropic Gaussian kernels."""

    @staticmethod
    def create(
        size: int = 21,
        sigma_long: float = 3.5,
        sigma_trans: float = 1.8,
        angle_deg: float = 0.0,
        gain: float = 1.0,
    ) -> np.ndarray:
        assert size % 2 == 1, "Kernel size must be odd."
        r = size // 2
        y, x = np.mgrid[-r : r + 1, -r : r + 1]
        phi = np.deg2rad(angle_deg)
        c, s = np.cos(phi), np.sin(phi)
        xp = c * x + s * y
        yp = -s * x + c * y
        g = np.exp(
            -((xp ** 2) / (2.0 * sigma_long ** 2)
              + (yp ** 2) / (2.0 * sigma_trans ** 2))
        ).astype(np.float32)
        # No self-coupling
        g[r, r] = 0.0
        ssum = g.sum()
        if ssum > 0:
            g /= ssum
        g *= gain
        return g