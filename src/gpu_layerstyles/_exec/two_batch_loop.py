# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Chunked assembly of two image batches with crossfaded loop boundaries."""

import torch

from .._batch import parse_batch_selection
from .core import (
    _WORKING_MULTIPLIER,
    _clear_exception_frames,
    _execute,
    _validate_image,
)
from .crossfade import _process_crossfade_chunk


@torch.no_grad()
def process_two_batch_loop(
    images_1: torch.Tensor,
    images_2: torch.Tensor,
    blend_target: int = 15,
    append_first_frame: bool = False,
    output_device: str = "cpu",
    batch_size: int = 0,
) -> torch.Tensor:
    """Blend both boundaries, preserving the nonempty middle of each batch."""
    _validate_image(images_1, "images_1")
    _validate_image(images_2, "images_2")
    if images_1.shape[1:] != images_2.shape[1:]:
        raise ValueError(
            "TwoBatchLoop requires matching height, width, and channel count in "
            "images_1 and images_2."
        )
    if (
        isinstance(blend_target, bool)
        or not isinstance(blend_target, int)
        or not 2 <= blend_target <= 2**31 - 1
    ):
        raise ValueError("blend_target must be an integer between 2 and 2147483647.")
    if not isinstance(append_first_frame, bool):
        raise TypeError("append_first_frame must be a boolean.")
    minimum = 2 * blend_target + 1
    for name, images in (("images_1", images_1), ("images_2", images_2)):
        if len(images) < minimum:
            raise ValueError(
                f"TwoBatchLoop requires at least {minimum} frames in {name} for "
                f"blend_target {blend_target} and a nonempty middle; "
                f"{name} has {len(images)} frames. Reduce blend_target."
            )

    def partition(images):
        selections = (
            parse_batch_selection(expression, len(images))
            for expression in (
                f":{blend_target}",
                f"{blend_target}:-{blend_target}",
                f"-{blend_target}:",
            )
        )
        return tuple(images[slice(s.start, s.stop, s.step)] for s in selections)

    first_a, middle_a, last_a = partition(images_1)
    first_b, middle_b, last_b = partition(images_2)
    segments = (
        (last_b, first_a),
        (middle_a, None),
        (last_a, first_b),
        (middle_b, None),
    )
    loop_frames = len(images_1) + len(images_2) - 2 * blend_target

    def process_chunk(start, count, destination, device):
        end = start + count
        offset = 0
        for first, second in segments:
            segment_end = offset + len(first)
            local_start = max(start, offset) - offset
            local_end = min(end, segment_end) - offset
            if local_start < local_end:
                target = destination[offset:segment_end]
                if second is None:
                    target[local_start:local_end].copy_(
                        first[local_start:local_end].to(
                            device=device, dtype=torch.float32
                        )
                    )
                else:
                    _process_crossfade_chunk(
                        first,
                        second,
                        0,
                        blend_target,
                        local_start,
                        local_end - local_start,
                        target,
                        device,
                    )
            offset = segment_end

        # Earlier segments finish first, including frame zero in a single chunk.
        # Repeating this copy after a partially written chunk fails is safe.
        if append_first_frame and start <= loop_frames < end:
            destination[loop_frames : loop_frames + 1].copy_(destination[:1])

    # Use CrossFade's reservation for both float32 inputs and blend temporaries.
    frame_bytes = images_1[0].numel() * 4
    try:
        return _execute(
            images_1,
            process_chunk,
            lambda count: _WORKING_MULTIPLIER * count * 2 * frame_bytes,
            output_device,
            batch_size,
            output_shape=(loop_frames + int(append_first_frame), *images_1.shape[1:]),
        )
    except BaseException as error:
        _clear_exception_frames(error)
        raise
