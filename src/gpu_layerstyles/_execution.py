"""Chunked execution with ComfyUI's device, memory, and progress management."""

import logging
from collections.abc import Callable

import torch
from comfy import model_management
from comfy.utils import ProgressBar
from comfy_api.latest import io

ColorOperation = Callable[[torch.Tensor], torch.Tensor]
_MIB = 1024 * 1024
_WORKING_MULTIPLIER = 8
_LOGGER = logging.getLogger(__name__)


def execution_inputs() -> list:
    return [
        io.Combo.Input(
            "output_device",
            options=["gpu", "cpu"],
            default="gpu",
            tooltip=(
                "Keep output on ComfyUI's selected compute device, or return it on CPU. "
                "ComfyUI CPU mode always uses CPU."
            ),
        ),
        io.Int.Input(
            "batch_size",
            default=0,
            min=0,
            max=2**31 - 1,
            step=1,
            tooltip=(
                "Frames per processing chunk. 0 selects automatically; "
                "reduced if memory runs out."
            ),
        ),
    ]


def _validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor) or not image.is_floating_point():
        raise TypeError("IMAGE must be a floating-point torch.Tensor.")
    if (
        image.layout != torch.strided
        or image.ndim != 4
        or image.shape[-1] not in (3, 4)
        or any(size == 0 for size in image.shape)
    ):
        raise ValueError("IMAGE must have nonempty shape [B, H, W, 3] or [B, H, W, 4].")


def _automatic_chunk_size(frames: int, frame_bytes: int, available: int) -> int:
    budget = min(max(available, 0) // 4, 512 * _MIB)
    return max(1, min(frames, 32, budget // (_WORKING_MULTIPLIER * frame_bytes)))


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
    output_device: str = "gpu",
    batch_size: int = 0,
) -> torch.Tensor:
    _validate_image(image)
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
    frames, height, width, channels = image.shape
    frame_bytes = height * width * channels * 4
    output_bytes = frames * frame_bytes
    working_bytes = _WORKING_MULTIPLIER * frame_bytes
    # A CPU destination also needs reservation when compute itself is on CPU.
    reservation = output_bytes if destination_device == device else 0
    chunk_size = min(frames, batch_size) if batch_size else 1
    model_management.free_memory(reservation + chunk_size * working_bytes, device)
    available = int(model_management.get_free_memory(device))
    if reservation > available:
        raise _output_memory_error(output_bytes, destination_device)
    if batch_size == 0:
        chunk_size = _automatic_chunk_size(frames, frame_bytes, available - reservation)

    try:
        destination = torch.empty(
            image.shape, dtype=torch.float32, device=destination_device
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
            _process_chunk(
                image[start : start + count],
                destination[start : start + count],
                operation,
                device,
            )
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
        model_management.free_memory(chunk_size * working_bytes, device)

    model_management.throw_exception_if_processing_interrupted()
    return destination
