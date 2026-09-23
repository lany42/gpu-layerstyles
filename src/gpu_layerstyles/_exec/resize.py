# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Chunked execution of image downscaling."""

import torch
from comfy import model_management

from .._resize import ImageResizer
from .core import _clear_exception_frames, _execute, _validate_image


def _process_resize_chunk(
    source: torch.Tensor,
    destination: torch.Tensor,
    resizer: ImageResizer,
    device: torch.device,
) -> None:
    chunk = source.to(device=device, dtype=torch.float32)
    destination.copy_(
        resizer(chunk, model_management.throw_exception_if_processing_interrupted)
    )


@torch.no_grad()
def process_image_resize(
    image: torch.Tensor,
    width: int,
    height: int,
    method: str = "bicubic",
    output_device: str = "cpu",
    batch_size: int = 0,
) -> torch.Tensor:
    """Downscale complete RGB/RGBA chunks into a differently sized destination."""
    _validate_image(image)
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    source_height, source_width = image.shape[1:3]
    if width > source_width or height > source_height:
        raise ValueError(
            f"ImageScaleDown cannot upscale: source is {source_width}x{source_height}, "
            f"requested {width}x{height}. Neither dimension may exceed the source."
        )
    if method not in ("bicubic", "lanczos"):
        raise ValueError("method must be 'bicubic' or 'lanczos'.")

    resizer = ImageResizer(
        (source_height, source_width), (height, width), image.shape[-1], method
    )

    def process_chunk(start, count, destination, device):
        _process_resize_chunk(
            image[start : start + count],
            destination[start : start + count],
            resizer,
            device,
        )

    try:
        return _execute(
            image,
            process_chunk,
            resizer.working_memory,
            output_device,
            batch_size,
            output_shape=(len(image), height, width, image.shape[-1]),
        )
    except BaseException as error:
        _clear_exception_frames(error)
        raise
    finally:
        resizer.clear()
