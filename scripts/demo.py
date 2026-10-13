"""Correct environment effects on the demo scene and compare to the truth.

    pixi run python scripts/demo.py

Prints what the scene holds, then the error of the retrieved reflectance with
and without the correction.  ``scripts/plot.py`` draws the same thing.
"""

import time

import numpy as np
import torch
import xarray as xr

from maja_environment_effects.correction import correct_scene
from maja_environment_effects.scene import (
    load_luts,
    load_radiative_lut,
    load_scene,
)


def describe(scene: xr.Dataset) -> None:
    """Print the scene, so that what is being corrected is on screen."""
    res = float(scene.x[1] - scene.x[0])
    size = scene.sizes["x"]
    print(
        f"scene     {size} x {size} pixels of {res * 1000:.0f} m, "
        f"{size * res:.1f} km across"
    )
    print(
        "bands     "
        + ", ".join(
            f"{b} ({float(scene.wl.sel(band=b)):.0f} nm)"
            for b in scene.band.values
        )
    )
    print(f"geometry  sza {float(scene.sza):.0f}°, vza {float(scene.vza):.0f}°")
    for name, unit in (("aot", ""), ("rh", " %"), ("h", " km")):
        field = scene[name]
        print(
            f"{name:<9} {float(field.min()):.2f} to {float(field.max()):.2f}"
            f"{unit}, mean {float(field.mean()):.2f}{unit}"
        )
    fractions = sorted(
        zip(scene.species.values.tolist(), scene.species_fraction.values.tolist()),
        key=lambda pair: -pair[1],
    )
    print(
        "aerosol   " + ", ".join(f"{name} {share:.0%}" for name, share in fractions)
    )
    print(
        f"rho_toa   {float(scene.rho_toa.min()):.3f} to "
        f"{float(scene.rho_toa.max()):.3f}, against {float(scene.rho_s.min()):.3f} "
        f"to {float(scene.rho_s.max()):.3f} at the surface"
    )


def warm_up(device: str) -> None:
    """Pay the one-shot cost of the device before the timer starts.

    The first CUDA call builds a context and compiles the FFT plans, which
    takes longer than the correction itself and says nothing about it.
    """
    torch.fft.rfft2(torch.zeros(8, 8, device=device))
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    """Describe the scene, correct it on every device, report the error."""
    scene = load_scene()
    describe(scene)

    king, radiative = load_luts(), load_radiative_lut()
    print()
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("correcting on cuda  ... unavailable, this torch has no CUDA")
            continue
        warm_up(device)
        start = time.perf_counter()
        result = correct_scene(scene, king, radiative, device=device)
        print(f"correcting on {device:<5} ... {time.perf_counter() - start:.1f} s")
    print()

    print(f"{'band':<5} {'RMSE rho_unif':>14} {'RMSE rho_s':>11} {'gain':>7}")
    for band in scene.band.values:
        truth = scene.rho_s.sel(band=band).values
        rmse = [
            np.sqrt(np.mean((result[v].sel(band=band).values - truth) ** 2))
            for v in ("rho_unif", "rho_s")
        ]
        print(
            f"{band:<5} {rmse[0]:>14.5f} {rmse[1]:>11.5f} "
            f"{rmse[0] / rmse[1]:>6.1f}x"
        )


if __name__ == "__main__":
    main()
