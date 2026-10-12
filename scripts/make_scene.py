"""Write a synthetic Sentinel-2 scene at 120 m, from the two look-up tables.

What a real product carries is what this writes: a surface reflectance, the
maps of aerosol optical thickness, relative humidity and ground elevation, the
angles, and the aerosol fractions of the date.  The TOA reflectance is then
the forward 5S formula, convolved with the King PSF of that atmosphere, so the
scene is consistent with the tables the correction will read back.

``rho_s`` is kept as the truth to compare against, which a real product does
not have.  Nothing else in the scene is privileged information.

    pixi run python scripts/make_radiative_lut.py
    pixi run python scripts/make_scene.py
"""

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter

from maja_environment_effects.convolution import fft_convolve_2d
from maja_environment_effects.correction import surface_to_toa
from maja_environment_effects.psf import mean_king_psf
from maja_environment_effects.radiative import radiative_terms
from maja_environment_effects.scene import DATA_DIR, load_luts, load_radiative_lut

RNG = np.random.default_rng(0)
N, RES = 915, 0.12  # Sentinel-2 tile (109.8 km) at 120 m
BANDS = {"B2": 490.0, "B3": 560.0, "B4": 665.0, "B8": 842.0}
SZA, VZA = 35.0, 5.0

#: Fractions of the AOT, as a CAMS mixture of a polluted continental date.
SPECIES = {
    "ammonium": 0.07,
    "blackcar": 0.03,
    "dust": 0.15,
    "nitrate": 0.08,
    "organicm": 0.20,
    "seasalt": 0.10,
    "secondar": 0.02,
    "sulphate": 0.35,
}

#: Land cover reflectances for B2, B3, B4, B8.
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


def surface() -> np.ndarray:
    """Return the surface reflectance of four land cover classes, with texture."""
    classes = np.digitize(smooth_field(80, 0.0, 1.0), [0.35, 0.55, 0.8])
    texture = 1.0 + 0.15 * gaussian_filter(RNG.standard_normal((N, N)), 2.0)
    return COVERS[classes].transpose(2, 0, 1) * texture


def main() -> None:
    """Write ``data/scene.nc``."""
    king, radiative = load_luts(), load_radiative_lut()
    coord = (np.arange(N) - N // 2) * RES
    dims2 = (("y", "x"), )
    maps = {
        "aot": xr.DataArray(smooth_field(150, 0.05, 0.4), dims=dims2[0]),
        "rh": xr.DataArray(smooth_field(200, 50.0, 90.0), dims=dims2[0]),
        "h": xr.DataArray(smooth_field(120, 0.0, 1.2), dims=dims2[0]),
    }
    rho_s = surface()

    rho_toa = []
    for index, wl in enumerate(BANDS.values()):
        terms = radiative_terms(
            radiative, SPECIES, wl=wl, aot=maps["aot"], h=maps["h"],
            sza=SZA, vza=VZA,
        )
        psf = mean_king_psf(
            king, SPECIES, aot=float(maps["aot"].mean()),
            rh=float(maps["rh"].mean()), wl=wl, n=N, res=RES,
            h=float(maps["h"].mean()),
        )
        band = {
            k: torch.as_tensor(v.values, dtype=torch.float32)
            for k, v in terms.items()
        }
        truth = torch.as_tensor(rho_s[index], dtype=torch.float32)
        rho_env = fft_convolve_2d(truth, psf)
        rho_toa.append(surface_to_toa(truth, rho_env, **band).numpy())

    scene = xr.Dataset(
        {
            "rho_s": (("band", "y", "x"), rho_s.astype("float32")),
            "rho_toa": (("band", "y", "x"), np.stack(rho_toa)),
            **{k: (dims2[0], v.values) for k, v in maps.items()},
            "species_fraction": ("species", list(SPECIES.values())),
        },
        coords={
            "band": list(BANDS),
            "wl": ("band", list(BANDS.values())),
            "y": coord,
            "x": coord,
            "species": list(SPECIES),
            "sza": SZA,
            "vza": VZA,
        },
        attrs={
            "description": (
                "Synthetic scene, x/y in km, wl in nm, aot at 550 nm, h in km, "
                "angles in degrees. rho_toa is the forward 5S formula of rho_s."
            ),
            "source": "scripts/make_scene.py",
        },
    )
    # int16 packing, as in MAJA products, keeps the file small.
    packed = {"dtype": "int16", "scale_factor": 1e-4, "zlib": True,
              "_FillValue": -32768}
    encoding = {v: packed for v in scene.data_vars if scene[v].ndim > 1}
    encoding["rh"] = {**packed, "scale_factor": 1e-2}
    path = DATA_DIR / "scene.nc"
    scene.to_netcdf(path, encoding=encoding)
    print(f"wrote {path}: {dict(scene.sizes)}")


if __name__ == "__main__":
    main()
