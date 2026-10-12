"""Linear 2D convolution in the Fourier domain, on CPU or GPU."""

import torch
import torch.nn.functional as F

#: Below this weight, the kernel sees no valid pixel and the output is No_data.
MIN_WEIGHT = 1e-6


def _spectrum(kernel: torch.Tensor, ny: int, nx: int, h: int) -> torch.Tensor:
    """Return the FFT of *kernel*, zero-padded to ``(ny, nx)`` and centred on 0."""
    extended = F.pad(kernel, (0, nx - 2 * h - 1, 0, ny - 2 * h - 1))
    return torch.fft.rfft2(extended.roll(shifts=(-h, -h), dims=(0, 1)))


def _convolve(field: torch.Tensor, spectrum: torch.Tensor, h: int) -> torch.Tensor:
    """Mirror-pad *field*, convolve it with *spectrum*, and crop back."""
    padded = F.pad(field[None, None], (h, h, h, h), mode="reflect")[0, 0]
    ny, nx = padded.shape
    out = torch.fft.irfft2(torch.fft.rfft2(padded) * spectrum, s=(ny, nx))
    return out[h : ny - h, h : nx - h]


def fft_convolve_2d(
    image: torch.Tensor,
    kernel: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convolve an image with a centered kernel, using mirror padding.

    The image is mirror-padded by ``k // 2`` on each side, so the circular
    convolution the FFT computes equals the linear convolution over the
    image.

    With a *mask*, the result is the mean over the valid pixels: ``image *
    mask`` convolved with the kernel, divided by ``mask`` convolved with the
    same kernel. Without it a No_data pixel counts as a zero reflectance,
    and since the kernel covers the whole image, every output pixel
    integrates every No_data of the scene. Where the mask is zero over the
    whole kernel, the output is ``nan``.

    Parameters
    ----------
    image : torch.Tensor
        2D input of shape ``(H, W)``.
    kernel : torch.Tensor
        2D kernel of odd shape ``(k, k)``, centered on ``(k // 2, k // 2)``.
        ``k // 2`` must be smaller than ``H`` and ``W``.
    mask : torch.Tensor or None, optional
        1 on the valid pixels and 0 elsewhere, same shape as *image*.

    Returns
    -------
    torch.Tensor
        Convolved image of shape ``(H, W)``.
    """
    h = kernel.shape[-1] // 2
    spectrum = _spectrum(kernel, image.shape[0] + 2 * h, image.shape[1] + 2 * h, h)
    if mask is None:
        return _convolve(image, spectrum, h)
    weighted = _convolve(image * mask, spectrum, h)
    weight = _convolve(mask, spectrum, h)
    return torch.where(
        weight > MIN_WEIGHT,
        weighted / weight.clamp_min(MIN_WEIGHT),
        torch.nan,
    )
