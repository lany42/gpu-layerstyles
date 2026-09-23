# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Chunked execution of single-image color operations."""

from collections.abc import Callable

import torch

from .core import _WORKING_MULTIPLIER, _execute, _validate_image

ColorOperation = Callable[[torch.Tensor], torch.Tensor]


def _process_chunk(
    source: torch.Tensor,
    destination: torch.Tensor,
    operation: ColorOperation,
    device: torch.device,
) -> None:
    # A separate function releases all temporary tensors before an OOM retry.
    chunk = source.to(device=device, dtype=torch.float32)
    rgb = operation(chunk[..., :3])
    # Blocking copies make each CPU chunk ready before reporting progress.
    destination[..., :3].copy_(rgb)
    if chunk.shape[-1] == 4:
        destination[..., 3:].copy_(chunk[..., 3:])


@torch.no_grad()
def process_image(
    image: torch.Tensor,
    operation: ColorOperation,
    output_device: str = "cpu",
    batch_size: int = 0,
) -> torch.Tensor:
    _validate_image(image)

    def process_chunk(start, count, destination, device):
        _process_chunk(
            image[start : start + count],
            destination[start : start + count],
            operation,
            device,
        )

    frame_bytes = image[0].numel() * 4
    return _execute(
        image,
        process_chunk,
        lambda count: _WORKING_MULTIPLIER * count * frame_bytes,
        output_device,
        batch_size,
    )
