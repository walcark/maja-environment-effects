"""Generate a synthetic Sentinel-2 scene at 120 m and a synthetic King PSF LUT.

Values are physically plausible orders of magnitude, not radiative transfer
outputs: they stand in until a real MAJA product and SMART-G LUT are used.
"""

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter

from maja_environment_effects.convolution import fft_convolve_2d
from maja_environment_effects.correction import surface_to_toa
from maja_environment_effects.psf import mean_king_psf
from maja_environment_effects.scene import DATA_DIR

RNG = np.random.default_rng(0)
N, RES = 915, 0.12  # Sentinel-2 tile (109.8 km) at 120 m
BANDS = {"B2": 490.0, "B3": 560.0, "B4": 665.0, "B8": 842.0}
SZA, VZA = 35.0, 5.0
SPECIES = {
    "ammonium": 0.07,
    "blackcar": 0.03,
    "dust": 0.15,
    "nitrate": 0.08,
    "organicm": 0.20,
    "seasalt": 0.10,
    "sulphate": 0.35,
    "secondar": 0.02,
}
# Core width [km] and power-law index of the pure aerosol PSF: coarse
# particles scatter more forward, hence a narrower kernel.
SPECIES_KING = {
    "ammonium": (0.6, 2.2),
    "blackcar": (0.5, 2.5),
    "dust": (0.25, 1.8),
    "nitrate": (0.6, 2.2),
    "organicm": (0.55, 2.2),
    "seasalt": (0.3, 1.8),
    "sulphate": (0.6, 2.3),
    "secondar": (0.6, 2.2),
}
RAYLEIGH_KING = (1.5, 1.3)
# Land cover reflectances for B2, B3, B4, B8.
COVERS = np.array(
    [
        [0.05, 0.04, 0.02, 0.01],  # water
        [0.03, 0.07, 0.04, 0.35],  # vegetation
        [0.10, 0.14, 0.18, 0.25],  # bare soil
        [0.20, 0.22, 0.24, 0.28],  # urban
    ]
)


def smooth_field(sigma_px: float, low: float, high: float) -> np.ndarray:
    """Return a smooth random field rescaled to ``[low, high]``."""
    f = gaussian_filter(RNG.standard_normal((N, N)), sigma_px, mode="reflect")
    f = (f - f.min()) / (f.max() - f.min())
    return low + (high - low) * f


def tau_rayleigh(wl: np.ndarray | float) -> np.ndarray | float:
    """Return the Rayleigh optical thickness at sea level (wl in nm)."""
    um = np.asarray(wl) / 1000.0
    return 0.008569 * um**-4 * (1 + 0.0113 * um**-2 + 0.00013 * um**-4)


def tau_aerosol(aot550: np.ndarray | float, wl: np.ndarray | float) -> np.ndarray:
    """Return the aerosol optical thickness with an Angstrom exponent of 1.3."""
    return np.asarray(aot550) * (np.asarray(wl) / 550.0) ** -1.3


def make_luts() -> dict[str, xr.Dataset]:
    """Build sigma/gamma over (aot, rh, wl) per species as a Rayleigh/aerosol mix."""
    aot = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5])
    rh = np.array([30.0, 50.0, 70.0, 80.0, 90.0, 95.0])
    wl = np.array([443.0, 490.0, 560.0, 665.0, 842.0, 1610.0, 2190.0])
    a, h, w = np.meshgrid(aot, rh, wl, indexing="ij")
    tr, ta = tau_rayleigh(w), tau_aerosol(a, w)
    f = ta / (ta + tr)  # aerosol share of the scattering
    growth = 1 - 0.2 * (h - 30) / 65  # hygroscopic growth narrows the kernel
    dims = ("aot", "rh", "wl")
    luts = {}
    for name, (s_a, g_a) in SPECIES_KING.items():
        sigma = (1 - f) * RAYLEIGH_KING[0] + f * s_a * growth
        gamma = np.clip((1 - f) * RAYLEIGH_KING[1] + f * g_a, 1.02, 5.0)
        luts[name] = xr.Dataset(
            {"sigma": (dims, sigma), "gamma": (dims, gamma)},
            coords={"aot": aot, "rh": rh, "wl": wl},
            attrs={"description": f"Synthetic King PSF parameters of {name}, sigma in km."},
        )
    return luts


def make_scene(luts: dict[str, xr.Dataset]) -> xr.Dataset:
    """Build surface, atmosphere and TOA reflectance of a synthetic scene."""
    cover = smooth_field(25, 0, 1)
    classes = np.digitize(cover, [0.3, 0.55, 0.8])  # water, veg, soil, urban
    texture = 1 + 0.1 * gaussian_filter(RNG.standard_normal((N, N)), 1.5)
    rho_s = COVERS[classes].transpose(2, 0, 1) * texture
    aot = smooth_field(150, 0.05, 0.4)
    rh = smooth_field(200, 50.0, 90.0)

    mu_s, mu_v = np.cos(np.radians(SZA)), np.cos(np.radians(VZA))
    wl = np.array(list(BANDS.values()))[:, None, None]
    tr, ta = tau_rayleigh(wl), tau_aerosol(aot, wl)
    tdir_down, tdir_up = np.exp(-(tr + ta) / mu_s), np.exp(-(tr + ta) / mu_v)
    # Total transmittance: half of Rayleigh and most aerosol light goes forward.
    t_down = np.exp(-(0.5 * tr + 0.16 * ta) / mu_s)
    t_up = np.exp(-(0.5 * tr + 0.16 * ta) / mu_v)
    sph_alb = 0.9 * tr + 0.15 * ta
    rho_atm = (tr + 0.25 * ta) / (4 * mu_s * mu_v)
    radiative = {
        "rho_atm": rho_atm,
        "tdir_down": tdir_down,
        "tdif_down": t_down - tdir_down,
        "tdir_up": tdir_up,
        "tdif_up": t_up - tdir_up,
        "sph_alb": sph_alb,
    }

    rho_toa = []
    for i, w in enumerate(BANDS.values()):
        psf = mean_king_psf(luts, SPECIES, float(aot.mean()), float(rh.mean()), w, N, RES)
        band = {k: torch.as_tensor(v[i], dtype=torch.float32) for k, v in radiative.items()}
        surface = torch.as_tensor(rho_s[i], dtype=torch.float32)
        rho_env = fft_convolve_2d(surface, psf)
        rho_toa.append(surface_to_toa(surface, rho_env, **band).numpy())

    coord = (np.arange(N) + 0.5) * RES
    dims = ("band", "y", "x")
    return xr.Dataset(
        {
            "rho_s": (dims, rho_s),
            "rho_toa": (dims, np.array(rho_toa)),
            **{k: (dims, v) for k, v in radiative.items()},
            "aot": (("y", "x"), aot),
            "rh": (("y", "x"), rh),
            "species_fraction": ("species", list(SPECIES.values())),
        },
        coords={
            "band": list(BANDS),
            "wl": ("band", list(BANDS.values())),
            "y": coord,
            "x": coord,
            "species": list(SPECIES),
        },
        attrs={"description": "Synthetic scene, x/y in km, wl in nm, aot at 550 nm."},
    )


def main() -> None:
    """Write ``data/king/<species>.nc`` and ``data/scene.nc``."""
    (DATA_DIR / "king").mkdir(parents=True, exist_ok=True)
    luts = make_luts()
    for name, lut in luts.items():
        lut.to_netcdf(DATA_DIR / "king" / f"{name}.nc")
    scene = make_scene(luts)
    # int16 packing, as in MAJA products, keeps the file small.
    packed = {"dtype": "int16", "scale_factor": 1e-4, "zlib": True, "_FillValue": -32768}
    encoding = {v: packed for v in scene.data_vars if scene[v].ndim > 1}
    encoding["rh"] = {**packed, "scale_factor": 1e-2}
    scene.to_netcdf(DATA_DIR / "scene.nc", encoding=encoding)


if __name__ == "__main__":
    main()
