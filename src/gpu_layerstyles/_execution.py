# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Chunked execution with ComfyUI's device, memory, and progress management."""

import logging
import traceback
from collections.abc import Callable

import torch
from comfy import model_management
from comfy.utils import ProgressBar
from comfy_api.latest import io

from ._resize import ImageResizer

ColorOperation = Callable[[torch.Tensor], torch.Tensor]
ChunkOperation = Callable[[int, int, torch.Tensor, torch.device], None]
_MIB = 1024 * 1024
_WORKING_MULTIPLIER = 8
_LOGGER = logging.getLogger(__name__)


def execution_inputs() -> list:
    return [
        io.Combo.Input(
            "output_device",
            options=["gpu", "cpu"],
            default="cpu",
            tooltip=(
                "Return output on CPU (default) after float32 processing on ComfyUI's "
                "selected compute device. GPU keeps output on that device to avoid "
                "transfers between compatible nodes. ComfyUI CPU mode returns CPU output."
            ),
        ),
        io.Int.Input(
            "batch_size",
            default=0,
            min=0,
            max=2**31 - 1,
            step=1,
            tooltip=(
                "Frames per processing chunk. 0 starts at up to 64 frames. Positive "
                "values can exceed 64, capped by the output length. Allocation failures "
                "halve the chunk size for this run; each run starts fresh. Complete "
                "inputs and outputs still need memory. Pass smaller batches through "
                "the entire workflow to reduce complete-batch memory."
            ),
        ),
    ]


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


def _prepare_reference_chunk[Reference](
    image_ref: torch.Tensor,
    prepare_reference: Callable[[torch.Tensor], Reference],
    device: torch.device,
) -> Reference:
    # Conversion and preparation-only tensors leave scope before cache publication.
    chunk = image_ref.to(device=device, dtype=torch.float32)
    return prepare_reference(chunk[..., :3])


def _clear_exception_frames(error: BaseException) -> None:
    # Retained exceptions must not keep failed temporaries or prepared state.
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        current = current.__cause__ or current.__context__


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
