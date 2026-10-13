"""Draw the correction of one band, and the PSF it was corrected with.

    pixi run python scripts/plot.py          # B4, into figures/correction.png
    pixi run python scripts/plot.py B8

Five maps and one curve: the measured reflectance, the truth, the retrieved
surface reflectance, the error before and after the correction, and the radial
profile of the kernel.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from maja_environment_effects.correction import correct_scene
from maja_environment_effects.psf import mean_king_psf
from maja_environment_effects.scene import load_luts, load_radiative_lut, load_scene

FIGURES = Path(__file__).parents[1] / "figures"


def main() -> None:
    """Write ``figures/correction.png`` for the band given, B4 by default."""
    band = sys.argv[1] if len(sys.argv) > 1 else "B4"
    scene = load_scene()
    king = load_luts()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = correct_scene(scene, king, load_radiative_lut(), device=device)

    truth = scene.rho_s.sel(band=band).values
    rho_toa = scene.rho_toa.sel(band=band).values
    rho_unif = result.rho_unif.sel(band=band).values
    rho_s = result.rho_s.sel(band=band).values
    extent = [float(scene.x[0]), float(scene.x[-1]),
              float(scene.y[0]), float(scene.y[-1])]

    # The kernel of this band's atmosphere, on a small grid: the profile is
    # what the figure shows, not the 915 x 915 map.
    res = float(scene.x[1] - scene.x[0])
    kernel = mean_king_psf(
        king,
        dict(zip(scene.species.values.tolist(),
                 scene.species_fraction.values.tolist())),
        aot=float(scene.aot.mean()),
        rh=float(scene.rh.mean()),
        wl=float(scene.wl.sel(band=band)),
        n=301,
        res=res,
        h=float(scene.h.mean()),
    ).numpy()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    vmin, vmax = float(truth.min()), float(truth.max())
    maps = [
        (axes[0, 0], rho_toa, r"$\rho_{toa}$, measured", vmin, vmax, "viridis"),
        (axes[0, 1], truth, r"$\rho_s$, truth", vmin, vmax, "viridis"),
        (axes[0, 2], rho_s, r"$\rho_s$, retrieved", vmin, vmax, "viridis"),
    ]
    error = np.max(np.abs(rho_unif - truth))
    maps += [
        (axes[1, 0], rho_unif - truth,
         r"error without the correction, $\rho_{unif} - \rho_s$",
         -error, error, "RdBu_r"),
        (axes[1, 1], rho_s - truth,
         r"error with it, $\hat{\rho_s} - \rho_s$", -error, error, "RdBu_r"),
    ]
    for axis, data, title, low, high, cmap in maps:
        image = axis.imshow(data, origin="lower", extent=extent,
                            vmin=low, vmax=high, cmap=cmap)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("x [km]")
        axis.set_ylabel("y [km]")
        fig.colorbar(image, ax=axis, fraction=0.046)

    # Radial profile of the kernel, from its centre outwards.
    half = kernel.shape[0] // 2
    radius = np.arange(half + 1) * res
    axis = axes[1, 2]
    axis.plot(radius[1:], kernel[half, half:][1:])
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("distance to the pixel [km]")
    axis.set_ylabel("kernel weight")
    axis.set_title("King PSF of the scene, radial cut", fontsize=11)
    axis.grid(alpha=0.3, which="both")

    rmse = [np.sqrt(np.mean((v - truth) ** 2)) for v in (rho_unif, rho_s)]
    fig.suptitle(
        f"{band} at {float(scene.wl.sel(band=band)):.0f} nm, "
        f"AOT {float(scene.aot.mean()):.2f}, RH {float(scene.rh.mean()):.0f} %. "
        f"RMSE {rmse[0]:.5f} without the correction, {rmse[1]:.5f} with it.",
        fontsize=12,
    )
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "correction.png"
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
