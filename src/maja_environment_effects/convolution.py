"""Linear 2D convolution computed in the Fourier domain.

Pytorch allows for fast FFT transforms on either CPU or GPU.
"""

import torch
import torch.nn.functional as F


def fft_convolve_2d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve an image with a centered kernel, using mirror padding.

    The image is mirror-padded by ``k // 2`` on every side. The circular FFT
    thus never wraps real pixels. The result is a true linear convolution.

    Parameters
    ----------
    image : torch.Tensor
        2D input of shape ``(H, W)``.
    kernel : torch.Tensor
        2D kernel of odd shape ``(k, k)``, centered on ``(k // 2, k // 2)``.
        ``k // 2`` must be smaller than ``H`` and ``W``.

    Returns
    -------
    torch.Tensor
        Convolved image of shape ``(H, W)``.
    """
    h = kernel.shape[-1] // 2
    padded = F.pad(image[None, None], (h, h, h, h), mode="reflect")[0, 0]
    ny, nx = padded.shape
    # Move the kernel center to the origin so the output is not shifted.
    kernel_ext = F.pad(kernel, (0, nx - 2 * h - 1, 0, ny - 2 * h - 1))
    kernel_ext = kernel_ext.roll(shifts=(-h, -h), dims=(0, 1))
    out = torch.fft.irfft2(
        torch.fft.rfft2(padded) * torch.fft.rfft2(kernel_ext),
        s=(ny, nx),
    )
    return out[h : ny - h, h : nx - h]
