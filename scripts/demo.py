"""Correct environment effects on the demo scene and compare to the truth."""

import numpy as np
import torch

from maja_environment_effects.correction import correct_scene
from maja_environment_effects.scene import load_luts, load_scene


def main():
    # Load and correct the scene
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene = load_scene()
    result = correct_scene(scene, load_luts(), device=device)

    # Print results
    print(f"{'band':<5} {'RMSE rho_unif':>14} {'RMSE rho_s':>11}")
    for band in scene.band.values:
        truth = scene.rho_s.sel(band=band).values
        rmse = [
            np.sqrt(np.mean((result[v].sel(band=band).values - truth) ** 2))
            for v in ("rho_unif", "rho_s")
        ]
        print(f"{band:<5} {rmse[0]:>14.5f} {rmse[1]:>11.5f}")


if __name__ == "__main__":
    main()
