"""King PSF built from per-species (sigma, gamma) look-up tables."""

from itertools import product

import numpy as np
import torch
import xarray as xr


def _bracket(coord: np.ndarray, value: float) -> list[tuple[int, float]]:
    """Return the two grid indices around *value* and their linear weights."""
    i = int(np.clip(np.searchsorted(coord, value) - 1, 0, len(coord) - 2))
    # Clamped to the LUT edges: no extrapolation of the kernels.
    t = float(np.clip((value - coord[i]) / (coord[i + 1] - coord[i]), 0.0, 1.0))
    return [(i, 1.0 - t), (i + 1, t)]


def mean_king_psf(
    luts: dict[str, xr.Dataset],
    species: dict[str, float],
    aot: float,
    rh: float,
    wl: float,
    n: int,
    res: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the species-weighted, trilinearly interpolated King kernel.

    The kernel is not linear in ``(σ, γ)``, so the parameters are never
    interpolated: normalised King kernels ``(1 + r² / (2σ²γ))^{-γ}`` are
    evaluated at the 8 LUT nodes around ``(aot, rh, wl)`` and summed with
    weight ``fraction x trilinear weight``. Kernels are accumulated in
    place, so peak memory is three ``n x n`` buffers whatever the number
    of species.

    Parameters
    ----------
    luts : dict[str, xr.Dataset]
        Species name to its LUT, with variables ``sigma`` [km] and
        ``gamma`` over dims ``(aot, rh, wl)``.
    species : dict[str, float]
        Species name to fraction of the aerosol optical thickness.
        Fractions must sum to 1 for the kernel to be normalised.
    aot : float
        Aerosol optical thickness at 550 nm.
    rh : float
        Relative humidity [%].
    wl : float
        Wavelength [nm].
    n : int
        Kernel size in pixels, odd.
    res : float
        Pixel size [km].
    device : torch.device or str, optional
        Device the kernel is allocated on.

    Returns
    -------
    torch.Tensor
        Kernel of shape ``(n, n)``, summing to 1.
    """
    t = torch.linspace(-(n // 2) * res, (n // 2) * res, n, device=device)
    r2 = t[None, :] ** 2 + t[:, None] ** 2
    acc = torch.zeros_like(r2)
    tmp = torch.empty_like(r2)
    for name, fraction in species.items():
        lut = luts[name]
        corners = [
            _bracket(lut[dim].values, value)
            for dim, value in (("aot", aot), ("rh", rh), ("wl", wl))
        ]
        for (ia, wa), (ih, wh), (iw, ww) in product(*corners):
            weight = fraction * wa * wh * ww
            if weight == 0.0:
                continue
            params = lut.isel(aot=ia, rh=ih, wl=iw)
            sigma, gamma = float(params.sigma), float(params.gamma)
            torch.div(r2, 2.0 * sigma**2 * gamma, out=tmp)
            tmp.add_(1.0).pow_(-gamma)
            acc.add_(tmp, alpha=weight / tmp.sum().item())
    return acc
