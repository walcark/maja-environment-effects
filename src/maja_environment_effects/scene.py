"""Loading of the pre-recorded demo scene and PSF look-up table."""

from pathlib import Path

import xarray as xr

DATA_DIR = Path(__file__).parents[2] / "data"


def load_scene(path: Path = DATA_DIR / "scene.nc") -> xr.Dataset:
    """Load a scene with its surface reflectance and atmospheric terms."""
    return xr.load_dataset(path)


def load_luts(directory: Path = DATA_DIR / "king") -> dict[str, xr.Dataset]:
    """Load one King PSF look-up table per species, keyed by file stem."""
    return {p.stem: xr.load_dataset(p) for p in sorted(directory.glob("*.nc"))}
