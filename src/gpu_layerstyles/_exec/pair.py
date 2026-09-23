# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Chunked execution of paired-image color operations."""

from collections.abc import Callable

import torch

from .core import (
    _WORKING_MULTIPLIER,
    _clear_exception_frames,
    _execute,
    _validate_image,
)
from .image import _process_chunk


def _prepare_reference_chunk[Reference](
    image_ref: torch.Tensor,
    prepare_reference: Callable[[torch.Tensor], Reference],
    device: torch.device,
) -> Reference:
    # Conversion and preparation-only tensors leave scope before cache publication.
    chunk = image_ref.to(device=device, dtype=torch.float32)
    return prepare_reference(chunk[..., :3])


@torch.no_grad()
def process_image_pair[Reference](
    image: torch.Tensor,
    image_ref: torch.Tensor,
    prepare_reference: Callable[[torch.Tensor], Reference],
    operation: Callable[[torch.Tensor, Reference], torch.Tensor],
    output_device: str = "cpu",
    batch_size: int = 0,
    *,
    matching_dimensions: bool = False,
    active: bool = True,
) -> torch.Tensor:
    """Execute paired RGB operations with an invocation-local singleton reference.

    Callables must leave inputs and prepared references untouched. Inactive calls
    validate both images but only copy the target, without preparing references.
    """
    _validate_image(image)
    _validate_image(image_ref, "image_ref")
    if len(image_ref) not in (1, len(image)):
        raise ValueError(
            "image_ref must contain one frame or match the image batch length."
        )
    if matching_dimensions and image.shape[1:3] != image_ref.shape[1:3]:
        raise ValueError("MVGD requires matching target/reference height and width.")

    singleton = len(image_ref) == 1
    unprepared = object()
    cached_reference = unprepared

    def process_chunk(start, count, destination, device):
        nonlocal cached_reference
        prepared = None
        if active:
            if singleton:
                if cached_reference is unprepared:
                    # Publish only after successful preparation; retain across
                    # subsequent target OOM retries at the same frame indices.
                    cached_reference = _prepare_reference_chunk(
                        image_ref, prepare_reference, device
                    )
                prepared = cached_reference
            else:
                prepared = _prepare_reference_chunk(
                    image_ref[start : start + count], prepare_reference, device
                )

        def color_operation(rgb):
            return operation(rgb, prepared) if active else rgb

        try:
            _process_chunk(
                image[start : start + count],
                destination[start : start + count],
                color_operation,
                device,
            )
        finally:
            # Paired preparation belongs only to this attempt. The singleton
            # survives in cached_reference until the invocation's finally block.
            color_operation = prepared = None

    target_frame_bytes = image[0].numel() * 4
    reference_frame_bytes = image_ref[0].numel() * 4

    def working_memory(count):
        return _WORKING_MULTIPLIER * (
            count * target_frame_bytes
            + (1 if singleton else count) * reference_frame_bytes
        )

    try:
        return _execute(image, process_chunk, working_memory, output_device, batch_size)
    except BaseException as error:
        # A caller may retain the exception and its chained causes. Keep the
        # diagnostic tracebacks, but release their tensors and prepared state.
        _clear_exception_frames(error)
        raise
    finally:
        cached_reference = unprepared
