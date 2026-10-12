"""Write a synthetic radiative look-up table with the shape MAJA reads.

The values are not radiative transfer outputs: they are the single-scattering
approximation evaluated over the grid, which puts every term at a realistic
order of magnitude and gives it the right dependence on wavelength, aerosol
optical thickness, elevation and angle.  They stand in until the table is
recomputed with SMART-G, and nothing but this file changes when it is.

The axes are the ones the MAJA technical note interpolates over:

    tdir, tdif   (species, band, aot, h, theta)
    sph_alb      (species, band, aot, h)
    rho_atm      (species, band, aot, h, sza, vza)

``theta`` serves both directions: the downward transmittance is read at the
solar zenith angle, the upward one at the viewing zenith angle.  The relative
azimuth is fixed at 0, as in the deposited King PSF tables.
"""

import numpy as np
import xarray as xr

from maja_environment_effects.scene import DATA_DIR

#: Sentinel-2 band centres [nm], the 1380 nm cirrus band excepted.  Same
#: nodes as the King PSF tables, so one wavelength lands on both.
BANDS = np.array(
    [443.0, 490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 945.0,
     1610.0, 2190.0]
)
AOT = np.round(np.arange(0.0, 0.81, 0.1), 2)
H = np.array([0.0, 1.0, 2.0, 3.0])
THETA = np.arange(0.0, 71.0, 10.0)
SZA = np.arange(0.0, 71.0, 10.0)
VZA = np.arange(0.0, 41.0, 10.0)

#: Angstrom exponent, single scattering albedo at 550 nm and asymmetry
#: parameter of each CAMS species.  Coarse species scatter forward and
#: barely absorb; black carbon is the opposite.
SPECIES = {
    "ammonium": (1.6, 1.00, 0.65),
    "blackcar": (1.0, 0.25, 0.35),
    "dust": (0.2, 0.92, 0.75),
    "nitrate": (1.6, 1.00, 0.65),
    "organicm": (1.4, 0.96, 0.68),
    "seasalt": (0.3, 1.00, 0.78),
    "secondar": (1.5, 0.97, 0.66),
    "sulphate": (1.5, 1.00, 0.67),
}

#: Scale heights [km] of the molecules and of the aerosols.  The aerosol one
#: is the 2 km the King tables are built at.
H_RAYLEIGH, H_AEROSOL = 8.0, 2.0


def tau_rayleigh(wl_nm: np.ndarray, h_km: np.ndarray) -> np.ndarray:
    """Return the Rayleigh optical thickness above *h_km*.

    Bodhaine et al. (1999) at sea level, scaled by the barometric factor of
    an 8 km scale height.
    """
    um = wl_nm / 1000.0
    tau0 = 0.008569 * um**-4 * (1 + 0.0113 * um**-2 + 0.00013 * um**-4)
    return tau0 * np.exp(-h_km / H_RAYLEIGH)


def tau_aerosol(aot550: np.ndarray, wl_nm: np.ndarray, alpha: float,
                h_km: np.ndarray) -> np.ndarray:
    """Return the aerosol optical thickness above *h_km*, from its Angstrom law."""
    return aot550 * (wl_nm / 550.0) ** -alpha * np.exp(-h_km / H_AEROSOL)


def scattering_angle(sza: np.ndarray, vza: np.ndarray) -> np.ndarray:
    """Return cos of the scattering angle, in the principal plane at azimuth 0."""
    mu_s, mu_v = np.cos(np.radians(sza)), np.cos(np.radians(vza))
    sin_s, sin_v = np.sin(np.radians(sza)), np.sin(np.radians(vza))
    return -mu_s * mu_v + sin_s * sin_v


def main() -> None:
    """Write ``data/radiative.nc``."""
    wl = BANDS[:, None, None]
    aot = AOT[None, :, None]
    h = H[None, None, :]
    tau_r = tau_rayleigh(wl, h)  # (band, 1, h)

    tdir, tdif, salb, ratm = [], [], [], []
    for alpha, omega, g in SPECIES.values():
        tau_a = tau_aerosol(aot, wl, alpha, h)  # (band, aot, h)

        # Transmittances. The direct one is Beer-Lambert; the total one
        # discounts the forward-scattered fraction, half for molecules and
        # (1 + g) / 2 for aerosols, and the diffuse one is the difference.
        mu = np.cos(np.radians(THETA))[None, None, None, :]
        tau_r_b, tau_a_b = tau_r[..., None], tau_a[..., None]
        direct = np.exp(-(tau_r_b + tau_a_b) / mu)
        total = np.exp(
            -(0.5 * tau_r_b + (1.0 - omega * (1.0 + g) / 2.0) * tau_a_b) / mu
        )
        tdir.append(direct)
        tdif.append(np.clip(total - direct, 0.0, None))

        # Spherical albedo, the 5S parameterisation.
        salb.append(
            0.92 * tau_r * np.exp(-tau_r) + 0.33 * (1.0 - np.exp(-tau_a * omega))
        )

        # Path reflectance, single scattering, molecules plus aerosols.
        s, v = SZA[None, None, None, :, None], VZA[None, None, None, None, :]
        cos_theta = scattering_angle(s, v)
        phase_r = 0.75 * (1.0 + cos_theta**2)
        phase_a = (1 - g**2) / (1 + g**2 - 2 * g * cos_theta) ** 1.5
        mu_s, mu_v = np.cos(np.radians(s)), np.cos(np.radians(v))
        ratm.append(
            (tau_r[..., None, None] * phase_r
             + tau_a[..., None, None] * omega * phase_a)
            / (4.0 * mu_s * mu_v)
        )

    coords = {
        "species": list(SPECIES),
        "band": BANDS,
        "aot": AOT,
        "h": H,
        "theta": THETA,
        "sza": SZA,
        "vza": VZA,
    }
    ds = xr.Dataset(
        {
            "tdir": (("species", "band", "aot", "h", "theta"), np.stack(tdir)),
            "tdif": (("species", "band", "aot", "h", "theta"), np.stack(tdif)),
            "sph_alb": (("species", "band", "aot", "h"), np.stack(salb)),
            "rho_atm": (
                ("species", "band", "aot", "h", "sza", "vza"),
                np.stack(ratm),
            ),
        },
        coords=coords,
        attrs={
            "description": (
                "Synthetic radiative LUT, single-scattering approximation. "
                "band in nm, aot at 550 nm, h and angles in km and degrees, "
                "relative azimuth fixed at 0."
            ),
            "source": "scripts/make_radiative_lut.py",
        },
    )
    for name in ("tdir", "tdif", "sph_alb", "rho_atm"):
        ds[name].attrs["units"] = "1"
    ds.h.attrs["units"] = "km"
    for name in ("theta", "sza", "vza"):
        ds[name].attrs["units"] = "degree"
    ds.band.attrs["units"] = "nm"
    path = DATA_DIR / "radiative.nc"
    ds.to_netcdf(path)
    print(f"wrote {path}: {dict(ds.sizes)}")


if __name__ == "__main__":
    main()
