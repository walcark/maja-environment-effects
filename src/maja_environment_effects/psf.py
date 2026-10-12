"""King PSF read from the per-species (sigma, gamma) look-up tables.

The kernel is not linear in its parameters, so the kernels are interpolated
rather than the parameters, and the species are weighted by their share of
the aerosol optical thickness.  See :func:`mean_king_psf`.
"""

from itertools import product

import numpy as np
import torch
import xarray as xr

#: Axes the tables tabulate but do not vary.  Read at their only node.
FIXED_AXES = ("sza", "vza", "href")

#: Axes the kernel is interpolated over.
SWEPT_AXES = ("aot", "rh", "h")


def _bracket(coord: np.ndarray, value: float) -> list[tuple[int, float]]:
    """Return the two grid indices around *value* and their linear weights."""
    if len(coord) == 1:
        return [(0, 1.0)]
    i = int(np.clip(np.searchsorted(coord, value) - 1, 0, len(coord) - 2))
    # Clamped to the LUT edges: no extrapolation of the kernels.
    t = float(np.clip((value - coord[i]) / (coord[i + 1] - coord[i]), 0.0, 1.0))
    return [(i, 1.0 - t), (i + 1, t)]


def _at_band(lut: xr.Dataset, wl: float, href: float) -> xr.Dataset:
    """Return the table reduced to one band and to the fixed axes.

    The band is selected rather than interpolated: the tables hold one node
    per band of the sensor, so a Sentinel-2 wavelength falls on a node.  A
    sensor whose bands differ lands on the nearest one.
    """
    sel = {ax: lut[ax].values[0] for ax in FIXED_AXES if ax != "href"}
    sel["href"] = href
    return lut.sel(band=wl, method="nearest").sel(sel, method="nearest")


def mean_king_psf(
    luts: dict[str, xr.Dataset],
    species: dict[str, float],
    aot: float,
    rh: float,
    wl: float,
    n: int,
    res: float,
    h: float = 0.0,
    href: float = 2.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the species-weighted, trilinearly interpolated King kernel.

    The kernel is not linear in ``(σ, γ)``, so the parameters are never
    interpolated: normalised King kernels ``(1 + r² / (2σ²γ))^{-γ}`` are
    evaluated at the 8 LUT nodes around ``(aot, rh, h)`` and summed with
    weight ``fraction x trilinear weight``. Kernels are accumulated in
    place, so peak memory is three ``n x n`` buffers whatever the number
    of species.

    Parameters
    ----------
    luts : dict[str, xr.Dataset]
        Species name to its LUT, with variables ``sigma`` [km] and
        ``gamma`` over dims ``(band, sza, vza, h, href, rh, aot)``.
    species : dict[str, float]
        Species name to fraction of the aerosol optical thickness.
        Fractions must sum to 1 for the kernel to be normalised.
    aot : float
        Aerosol optical thickness at 550 nm.
    rh : float
        Relative humidity [%].
    wl : float
        Wavelength [nm], read at the nearest band node of the table.
    n : int
        Kernel size in pixels, odd.
    res : float
        Pixel size [km].
    h : float, optional
        Ground elevation [km].
    href : float, optional
        Aerosol scale height [km]; 2 km is the typical MAJA case.
    device : torch.device or str, optional
        Device the kernel is allocated on.

    Returns
    -------
    torch.Tensor
        Kernel of shape ``(n, n)``, summing to 1.

    Notes
    -----
    Of the seven axes the deposited tables carry, three hold a single node,
    since this correction does not make them vary: a solar zenith angle of
    40 degrees, a nadir view, and an aerosol scale height of 2 km.  The
    spectral band is read at its own node rather than interpolated, so what
    is interpolated is ``(aot, rh, h)``.
    """
    t = torch.linspace(-(n // 2) * res, (n // 2) * res, n, device=device)
    r2 = t[None, :] ** 2 + t[:, None] ** 2
    acc = torch.zeros_like(r2)
    tmp = torch.empty_like(r2)
    for name, fraction in species.items():
        lut = _at_band(luts[name], wl, href)
        corners = [
            _bracket(lut[axis].values, value)
            for axis, value in zip(SWEPT_AXES, (aot, rh, h))
        ]
        for (ia, wa), (ih, wh), (ie, we) in product(*corners):
            weight = fraction * wa * wh * we
            if weight == 0.0:
                continue
            params = lut.isel(aot=ia, rh=ih, h=ie)
            sigma, gamma = float(params.sigma), float(params.gamma)
            torch.div(r2, 2.0 * sigma**2 * gamma, out=tmp)
            tmp.add_(1.0).pow_(-gamma)
            acc.add_(tmp, alpha=weight / tmp.sum().item())
    return acc
