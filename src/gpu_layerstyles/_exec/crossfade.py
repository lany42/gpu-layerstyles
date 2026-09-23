# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Chunked execution of image batch crossfades."""

import torch

from .core import (
    _WORKING_MULTIPLIER,
    _clear_exception_frames,
    _execute,
    _validate_image,
)


def _process_crossfade_chunk(
    images_1: torch.Tensor,
    images_2: torch.Tensor,
    start_index: int,
    frames: int,
    start: int,
    count: int,
    destination: torch.Tensor,
    device: torch.device,
) -> None:
    # Output chunks may span both seams. Each write is safe to repeat after OOM.
    end = start + count
    prefix_end = min(end, start_index)
    if start < prefix_end:
        destination[start:prefix_end].copy_(
            images_1[start:prefix_end].to(device=device, dtype=torch.float32)
        )

    blend_start = max(start, start_index)
    blend_end = min(end, start_index + frames)
    if blend_start < blend_end:
        first = images_1[blend_start:blend_end].to(device=device, dtype=torch.float32)
        second = images_2[blend_start - start_index : blend_end - start_index].to(
            device=device, dtype=torch.float32
        )
        # Absolute transition positions keep weights independent of chunk boundaries.
        alpha = torch.arange(
            blend_start - start_index,
            blend_end - start_index,
            device=first.device,
            dtype=torch.float32,
        ).reshape(-1, 1, 1, 1) / (frames - 1)
        destination[blend_start:blend_end].copy_(first * (1 - alpha) + second * alpha)

    suffix_start = max(start, start_index + frames)
    if suffix_start < end:
        destination[suffix_start:end].copy_(
            images_2[suffix_start - start_index : end - start_index].to(
                device=device, dtype=torch.float32
            )
        )


@torch.no_grad()
def process_image_crossfade(
    images_1: torch.Tensor,
    images_2: torch.Tensor,
    start_index: int = 0,
    frames: int = 2,
    output_device: str = "cpu",
    batch_size: int = 0,
) -> torch.Tensor:
    """Assemble a prefix, an exact full fade, and the second batch's suffix."""
    _validate_image(images_1, "images_1")
    _validate_image(images_2, "images_2")
    if images_1.shape[1:] != images_2.shape[1:]:
        raise ValueError(
            "CrossFade requires matching height, width, and channel count in "
            "images_1 and images_2."
        )
    for name, value, minimum in (
        ("start_index", start_index, 0),
        ("frames", frames, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if start_index >= len(images_1):
        raise ValueError(
            f"start_index {start_index} is out of range for images_1 with "
            f"{len(images_1)} frames."
        )
    remaining = len(images_1) - start_index
    if frames > remaining or frames > len(images_2):
        raise ValueError(
            f"CrossFade requires exactly {frames} transition frames: images_1 has "
            f"{remaining} frames remaining at start_index {start_index}, and "
            f"images_2 has {len(images_2)} frames. Reduce frames or start_index."
        )

    def process_chunk(start, count, destination, device):
        _process_crossfade_chunk(
            images_1, images_2, start_index, frames, start, count, destination, device
        )

    # Reserve for both float32 input chunks and their blend/copy temporaries.
    frame_bytes = images_1[0].numel() * 4
    try:
        return _execute(
            images_1,
            process_chunk,
            lambda count: _WORKING_MULTIPLIER * count * 2 * frame_bytes,
            output_device,
            batch_size,
            output_shape=(start_index + len(images_2), *images_1.shape[1:]),
        )
    except BaseException as error:
        _clear_exception_frames(error)
        raise
