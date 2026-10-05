"""5S atmospheric correction formulas used by MAJA, with environment effects."""

import torch
import xarray as xr

from .convolution import fft_convolve_2d
from .psf import mean_king_psf


def correct_scene(
    scene: xr.Dataset,
    luts: dict[str, xr.Dataset],
    device: torch.device | str = "cpu",
) -> xr.Dataset:
    """Run ``rho_toa -> rho_unif -> rho_env -> rho_s`` on every band.

    The PSF used is the King Kernel of the scene's atmosphere, whose AOT is
    averaged on the full image.

    Parameters
    ----------
    scene : xr.Dataset
        Scene as returned by :func:`maja_environment_effects.scene.load_scene`.
    luts : dict[str, xr.Dataset]
        King PSF look-up table per species, see
        :func:`~maja_environment_effects.psf.mean_king_psf`.
    device : torch.device or str, optional
        Device the computation runs on.

    Returns
    -------
    xr.Dataset
        ``rho_unif``, ``rho_env`` and ``rho_s`` over ``(band, y, x)``.
    """
    species = dict(
        zip(scene.species.values.tolist(), scene.species_fraction.values.tolist())
    )
    res = float(scene.x[1] - scene.x[0])
    out: dict[str, list[torch.Tensor]] = {"rho_unif": [], "rho_env": [], "rho_s": []}
    for band in scene.band.values:
        ds = scene.sel(band=band)

        def get(name: str, ds: xr.Dataset = ds) -> torch.Tensor:
            return torch.as_tensor(ds[name].values, dtype=torch.float32, device=device)

        psf = mean_king_psf(
            luts,
            species,
            aot=float(scene.aot.mean()),
            rh=float(scene.rh.mean()),
            wl=float(ds.wl),
            n=scene.sizes["x"],
            res=res,
            device=device,
        )
        rho_unif = toa_to_unif(
            get("rho_toa"),
            get("rho_atm"),
            get("tdir_down"),
            get("tdif_down"),
            get("tdir_up"),
            get("tdif_up"),
            get("sph_alb"),
        )
        rho_env = fft_convolve_2d(rho_unif, psf)
        rho_s = unif_to_surface(
            rho_unif, rho_env, get("tdir_up"), get("tdif_up"), get("sph_alb")
        )
        for name, value in zip(out, (rho_unif, rho_env, rho_s)):
            out[name].append(value.cpu())
    dims = ("band", "y", "x")
    return xr.Dataset(
        {name: (dims, torch.stack(values).numpy()) for name, values in out.items()},
        coords={c: scene[c] for c in dims},
    )


def toa_to_unif(
    rho_toa: torch.Tensor,
    rho_atm: torch.Tensor,
    tdir_down: torch.Tensor,
    tdif_down: torch.Tensor,
    tdir_up: torch.Tensor,
    tdif_up: torch.Tensor,
    sph_alb: torch.Tensor,
) -> torch.Tensor:
    """Invert the 5S model assuming a uniform surface (``rho_s = rho_env``).

    The formula is:

        :: rho_unif = rho_star / (sph_alb * rho_star + t_up * t_down)

    with:

        :: rho_star = rho_toa - rho_atm
        :: t_up = tdir_up + tdif_up
        :: t_down = tdir_down + tdif_down

    Returns
    -------
    torch.Tensor
        Uniform surface reflectance ``rho_unif``.
    """
    rho_star = rho_toa - rho_atm
    t_up_down = (tdir_up + tdif_up) * (tdir_down + tdif_down)
    return rho_star / (sph_alb * rho_star + t_up_down)


def unif_to_surface(
    rho_unif: torch.Tensor,
    rho_env: torch.Tensor,
    tdir_up: torch.Tensor,
    tdif_up: torch.Tensor,
    sph_alb: torch.Tensor,
) -> torch.Tensor:
    """Retrieve the surface reflectance from ``rho_unif`` and ``rho_env``.

    The formula is:

        :: rho_s = (rho_unif * (tdir_up + tdif_up) * frac - rho_env * tdif_up) / tdir_up

    with:

        :: frac = (1 - sph_alb * rho_env) / (1 - sph_alb * rho_unif)

    Returns
    -------
    torch.Tensor
        Surface reflectance ``rho_s``.
    """
    frac = (1 - rho_env * sph_alb) / (1 - rho_unif * sph_alb)
    return (rho_unif * (tdir_up + tdif_up) * frac - rho_env * tdif_up) / tdir_up


def surface_to_toa(
    rho_s: torch.Tensor,
    rho_env: torch.Tensor,
    rho_atm: torch.Tensor,
    tdir_down: torch.Tensor,
    tdif_down: torch.Tensor,
    tdir_up: torch.Tensor,
    tdif_up: torch.Tensor,
    sph_alb: torch.Tensor,
) -> torch.Tensor:
    """Apply the 5S forward model with environment effects.

    The formula is:

        :: rho_toa = rho_atm + t_down * reflected / (1 - sph_alb * rho_env)

    with:

        :: t_down = tdir_down + tdif_down
        :: reflected = tdir_up * rho_s + tdif_up * rho_env

    Returns
    -------
    torch.Tensor
        Top-of-atmosphere reflectance ``rho_toa``.
    """
    t_down = tdir_down + tdif_down
    reflected = tdir_up * rho_s + tdif_up * rho_env
    return rho_atm + t_down * reflected / (1 - sph_alb * rho_env)
