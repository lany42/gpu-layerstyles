# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Chunked execution with ComfyUI's device, memory, and progress management."""

import logging
import traceback
from collections.abc import Callable

import torch
from comfy import model_management
from comfy.utils import ProgressBar

ChunkOperation = Callable[[int, int, torch.Tensor, torch.device], None]
_MIB = 1024 * 1024
_WORKING_MULTIPLIER = 8
_LOGGER = logging.getLogger(__name__)


def _validate_image(image: torch.Tensor, name: str = "IMAGE") -> None:
    if not isinstance(image, torch.Tensor) or not image.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor.")
    if (
        image.layout != torch.strided
        or image.ndim != 4
        or image.shape[-1] not in (3, 4)
        or any(size == 0 for size in image.shape)
    ):
        raise ValueError(
            f"{name} must have nonempty shape [B, H, W, 3] or [B, H, W, 4]."
        )


def _is_allocation_error(error: Exception) -> bool:
    if isinstance(error, MemoryError) or model_management.is_oom(error):
        return True
    # CPU allocation failures are RuntimeErrors rather than CUDA OOM exceptions.
    message = str(error).lower()
    return (
        isinstance(error, RuntimeError)
        and "defaultcpuallocator" in message
        and ("not enough memory" in message or "can't allocate memory" in message)
    )


def _output_memory_error(output_bytes: int, device: torch.device) -> RuntimeError:
    advice = "Reduce the input batch or resolution, or free cached outputs."
    if device.type != "cpu":
        advice += " Select output_device='cpu' to keep the complete output in host RAM."
    return RuntimeError(
        f"GPU LayerStyles cannot allocate the complete {output_bytes / _MIB:.1f} MiB "
        f"output on {device}. {advice} batch_size only limits temporary memory."
    )


def _clear_exception_frames(error: BaseException) -> None:
    # Retained exceptions must not keep failed temporaries or prepared state.
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        current = current.__cause__ or current.__context__


def _execute(
    image: torch.Tensor,
    process_chunk: ChunkOperation,
    working_memory: Callable[[int], int],
    output_device: str,
    batch_size: int,
    *,
    output_shape: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    """Shared destination allocation, retry, progress, and cancellation loop."""
    if output_device not in ("gpu", "cpu"):
        raise ValueError("output_device must be 'gpu' or 'cpu'.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 0
    ):
        raise ValueError(
            "batch_size must be an integer >= 0 (0 selects automatically)."
        )

    model_management.throw_exception_if_processing_interrupted()
    device = model_management.get_torch_device()
    destination_device = device if output_device == "gpu" else torch.device("cpu")
    output_shape = image.shape if output_shape is None else output_shape
    frames, height, width, channels = output_shape
    frame_bytes = height * width * channels * 4
    output_bytes = frames * frame_bytes
    # A CPU destination also needs reservation when compute itself is on CPU.
    reservation = output_bytes if destination_device == device else 0
    chunk_size = min(frames, batch_size or 64)
    model_management.free_memory(reservation + working_memory(chunk_size), device)
    available = int(model_management.get_free_memory(device))
    if reservation > available:
        raise _output_memory_error(output_bytes, destination_device)

    try:
        destination = torch.empty(
            output_shape, dtype=torch.float32, device=destination_device
        )
    except Exception as error:
        if not _is_allocation_error(error):
            raise
        raise _output_memory_error(output_bytes, destination_device) from error

    progress = ProgressBar(frames)
    start = 0
    while start < frames:
        model_management.throw_exception_if_processing_interrupted()
        count = min(chunk_size, frames - start)
        try:
            process_chunk(start, count, destination, device)
        except Exception as error:
            if not _is_allocation_error(error):
                raise
            if count == 1:
                advice = "Reduce image resolution or free memory and cached outputs."
                if destination_device.type != "cpu":
                    advice += (
                        " Try output_device='cpu' to release space for processing."
                    )
                raise RuntimeError(
                    f"GPU LayerStyles ran out of memory processing one frame on "
                    f"{device}. {advice}"
                ) from error
            chunk_size = max(1, count // 2)
            _LOGGER.warning(
                "GPU LayerStyles: reducing chunk size from %d to %d after an "
                "allocation failure on %s.",
                count,
                chunk_size,
                device,
            )
        else:
            start += count
            progress.update_absolute(start)
            continue

        # Exit the except block first: its traceback can retain GPU tensors.
        model_management.soft_empty_cache()
        model_management.free_memory(working_memory(chunk_size), device)

    model_management.throw_exception_if_processing_interrupted()
    return destination
