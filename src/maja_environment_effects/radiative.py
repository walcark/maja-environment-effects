"""Radiative terms of a scene, interpolated from the look-up table.

MAJA reads them per pixel at the coarse resolution, from the aerosol optical
thickness and the elevation of that pixel, so the interpolation is vectorised
over the maps the product carries rather than evaluated at a tile average.

The table holds one entry per aerosol species. Transmittances do not add
across species, so the mixture is the fraction-weighted mean of the terms,
which is first order in the optical thickness. It is the same approximation
as reading a single pre-mixed aerosol model, and what it saves is one SMART-G
run per date instead of one per species.
"""

import xarray as xr

#: Terms read at the solar zenith angle rather than the viewing one.
_DOWNWARD = {"tdir_down": "tdir", "tdif_down": "tdif"}

#: Terms read at the viewing zenith angle.
_UPWARD = {"tdir_up": "tdir", "tdif_up": "tdif"}


def radiative_terms(
    lut: xr.Dataset,
    species: dict[str, float],
    wl: float,
    aot: xr.DataArray,
    h: xr.DataArray,
    sza: float,
    vza: float,
) -> dict[str, xr.DataArray]:
    """Return the six radiative terms of one band, over the dims of *aot*.

    Parameters
    ----------
    lut : xr.Dataset
        Table as written by ``scripts/make_radiative_lut.py``.
    species : dict[str, float]
        Species name to fraction of the aerosol optical thickness.
    wl : float
        Wavelength [nm], read at the nearest band node of the table.
    aot, h : xr.DataArray
        Maps of the aerosol optical thickness at 550 nm and of the ground
        elevation [km], over the same dims.
    sza, vza : float
        Solar and viewing zenith angles [degree].

    Returns
    -------
    dict[str, xr.DataArray]
        ``rho_atm``, ``tdir_down``, ``tdif_down``, ``tdir_up``, ``tdif_up``
        and ``sph_alb``, each over the dims of *aot*.
    """
    fractions = xr.DataArray(
        list(species.values()), coords={"species": list(species)}, dims="species"
    )
    # One band, and the mixture: the species axis is reduced here, so the
    # interpolation below runs once rather than once per species.
    band = (lut.sel(band=wl, method="nearest") * fractions).sum("species")

    out: dict[str, xr.DataArray] = {}
    for name, var in _DOWNWARD.items():
        out[name] = band[var].interp(aot=aot, h=h, theta=sza)
    for name, var in _UPWARD.items():
        out[name] = band[var].interp(aot=aot, h=h, theta=vza)
    out["sph_alb"] = band["sph_alb"].interp(aot=aot, h=h)
    out["rho_atm"] = band["rho_atm"].interp(aot=aot, h=h, sza=sza, vza=vza)
    return {k: v.drop_vars([c for c in v.coords if c not in aot.dims])
            for k, v in out.items()}
