"""Assert what the convolution and the PSF are supposed to guarantee.

    pixi run python scripts/check.py

Three properties, each of which fails loudly if the logic breaks: a masked
convolution leaves No_data out, an empty mask gives No_data, and the kernel
read from the tables is normalised.
"""

import torch

from maja_environment_effects.convolution import fft_convolve_2d
from maja_environment_effects.psf import mean_king_psf
from maja_environment_effects.scene import load_luts

N = 101


def check_mask_renormalises() -> None:
    """A quarter of No_data must not drag a flat image down."""
    image = torch.full((N, N), 0.3)
    mask = torch.ones_like(image)
    mask[: N // 2, : N // 2] = 0.0
    kernel = torch.ones(N, N) / (N * N)

    biased = fft_convolve_2d(image * mask, kernel)
    assert float(biased.min()) < 0.2, "the unmasked convolution should be biased"

    out = fft_convolve_2d(image, kernel, mask)
    assert torch.allclose(out, torch.full_like(out, 0.3), atol=1e-4), out.min()


def check_empty_mask_is_nodata() -> None:
    """No valid pixel in the footprint gives ``nan``, not a division by zero."""
    out = fft_convolve_2d(
        torch.full((N, N), 0.3), torch.ones(N, N) / (N * N), torch.zeros(N, N)
    )
    assert bool(out.isnan().all()), "an empty mask must give No_data"


def check_kernel_is_normalised() -> None:
    """The kernel read from the deposited tables sums to 1."""
    luts = load_luts()
    fractions = {name: 1.0 / len(luts) for name in luts}
    kernel = mean_king_psf(
        luts, fractions, aot=0.23, rh=72.0, wl=560.0, n=N, res=0.12, h=0.4
    )
    assert abs(float(kernel.sum()) - 1.0) < 1e-5, float(kernel.sum())
    assert float(kernel.min()) >= 0.0, float(kernel.min())


if __name__ == "__main__":
    for check in (
        check_mask_renormalises,
        check_empty_mask_is_nodata,
        check_kernel_is_normalised,
    ):
        check()
        print(f"ok   {check.__name__}")
