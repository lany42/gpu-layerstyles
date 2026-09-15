import weakref
from itertools import accumulate

import pytest
import torch

from gpu_layerstyles import _execution
from gpu_layerstyles._execution import process_image

from .conftest import InterruptProcessingException


@pytest.mark.parametrize(
    "shape",
    [(3, 4, 3), (1, 3, 4, 1), (1, 3, 4, 5), (0, 2, 3, 3), (1, 0, 3, 3), (1, 2, 0, 4)],
)
def test_rejects_unsupported_or_empty_shapes(shape, runtime):
    with pytest.raises(ValueError, match="nonempty shape"):
        process_image(torch.empty(shape), lambda rgb: rgb)
    assert not runtime.free_requests


@pytest.mark.parametrize("image", [None, [], torch.ones(1, 2, 3, 3, dtype=torch.uint8)])
def test_rejects_nonfloating_inputs(image):
    with pytest.raises(TypeError, match="floating-point"):
        process_image(image, lambda rgb: rgb)


def test_rejects_sparse_images():
    image = torch.ones(1, 2, 3, 3).to_sparse()
    with pytest.raises(ValueError, match="nonempty shape"):
        process_image(image, lambda rgb: rgb)


@pytest.mark.parametrize("batch_size", [-1, 1.5, True, "2"])
def test_rejects_invalid_chunk_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        process_image(torch.ones(1, 1, 1, 3), lambda rgb: rgb, batch_size=batch_size)


def test_rejects_invalid_output_device():
    with pytest.raises(ValueError, match="output_device"):
        process_image(torch.ones(1, 1, 1, 3), lambda rgb: rgb, output_device="cuda")


@pytest.mark.parametrize(
    "frames,expected",
    [
        (1, [1]),
        (17, [17]),
        (63, [63]),
        (64, [64]),
        (65, [64, 1]),
        (128, [64, 64]),
        (131, [64, 64, 3]),
    ],
)
@pytest.mark.parametrize("working_headroom", [0, 1024**3])
def test_automatic_chunks_reserve_initial_work_and_complete_output(
    runtime, frames, expected, working_headroom
):
    image = torch.rand(frames, 2, 3, 4, dtype=torch.float64)
    frame_bytes = 2 * 3 * 4 * 4
    output_bytes = frames * frame_bytes
    runtime.available = output_bytes + working_headroom
    calls = []

    def operation(rgb):
        calls.append(len(rgb))
        return rgb

    result = process_image(image, operation)
    assert calls == expected
    assert runtime.free_requests == [
        (output_bytes + expected[0] * 8 * frame_bytes, runtime.device)
    ]
    assert runtime.progress[0].total == frames
    assert runtime.progress[0].updates == list(accumulate(expected))
    assert torch.equal(result, image.float())


def test_preallocation_follows_memory_management_and_partial_chunks_report_progress(
    runtime, monkeypatch
):
    image = torch.rand(7, 3, 5, 4)
    allocated = []
    original_empty = torch.empty

    def empty(shape, **kwargs):
        assert runtime.free_requests
        allocated.append(tuple(shape))
        return original_empty(shape, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    result = process_image(image, lambda rgb: rgb * 0.5, batch_size=3)
    assert allocated == [tuple(image.shape)]
    assert runtime.progress[0].total == 7
    assert runtime.progress[0].updates == [3, 6, 7]
    assert torch.equal(result[..., :3], image[..., :3] * 0.5)
    assert torch.equal(result[..., 3], image[..., 3])


@pytest.mark.parametrize("batch_size", [0, 7])
def test_oom_halves_chunk_size_retries_same_frames_and_releases_temporaries(
    runtime, monkeypatch, batch_size
):
    image = torch.rand(7, 2, 3, 4)
    calls = []
    failed_tensors = []
    before = image.clone()

    def operation(rgb):
        calls.append(len(rgb))
        if len(rgb) > 1:
            temporary = rgb.clone()
            failed_tensors.append(weakref.ref(temporary))
            raise torch.OutOfMemoryError("simulated allocation failure")
        return rgb + 0.125

    def empty_cache():
        assert all(reference() is None for reference in failed_tensors)
        runtime.cache_clears += 1

    monkeypatch.setattr(_execution.model_management, "soft_empty_cache", empty_cache)
    result = process_image(image, operation, batch_size=batch_size)
    assert calls == [7, 3, *([1] * 7)]
    assert runtime.cache_clears == 2
    assert runtime.progress[0].updates == list(range(1, 8))
    assert torch.equal(result[..., :3], image[..., :3] + 0.125)
    assert torch.equal(result[..., 3], image[..., 3])
    assert torch.equal(image, before)


@pytest.mark.parametrize("output_device", [None, "cpu", "gpu"])
def test_automatic_oom_retries_64_32_16_and_restarts_fresh(
    runtime, gpu_routing, monkeypatch, output_device
):
    image = torch.arange(149 * 4, dtype=torch.float64).reshape(149, 1, 1, 4)
    before = image.clone()
    calls, failed_tensors = [], []
    fail = True

    def operation(rgb):
        start = int(rgb[0, 0, 0, 0].item()) // 4
        calls.append((start, len(rgb)))
        if fail and start >= 64 and len(rgb) > 16:
            temporary = rgb.clone()
            failed_tensors.extend((weakref.ref(rgb), weakref.ref(temporary)))
            raise torch.OutOfMemoryError("simulated allocation failure")
        return rgb + 0.125

    def empty_cache():
        assert all(reference() is None for reference in failed_tensors)
        runtime.cache_clears += 1

    monkeypatch.setattr(_execution.model_management, "soft_empty_cache", empty_cache)
    options = {} if output_device is None else {"output_device": output_device}
    result = process_image(image, operation, **options)
    assert calls == [
        (0, 64),
        (64, 64),
        (64, 32),
        (64, 16),
        (80, 16),
        (96, 16),
        (112, 16),
        (128, 16),
        (144, 5),
    ]
    assert runtime.cache_clears == 2
    assert runtime.progress[0].updates == [64, 80, 96, 112, 128, 144, 149]
    reservation = image.numel() * 4 if output_device == "gpu" else 0
    working_bytes = 8 * 4 * 4
    assert runtime.free_requests == [
        (reservation + 64 * working_bytes, runtime.device),
        (32 * working_bytes, runtime.device),
        (16 * working_bytes, runtime.device),
    ]
    destination = runtime.device if output_device == "gpu" else torch.device("cpu")
    allocation = (tuple(image.shape), destination, torch.float32)
    assert gpu_routing.allocations == [allocation]
    assert torch.equal(result[..., :3], image[..., :3].float() + 0.125)
    assert torch.equal(result[..., 3], image[..., 3].float())

    fail = False
    calls.clear()
    fresh_result = process_image(image, operation, **options)
    assert calls == [(0, 64), (64, 64), (128, 21)]
    assert runtime.progress[1].updates == [64, 128, 149]
    assert gpu_routing.allocations == [allocation, allocation]
    assert runtime.free_requests[-1] == runtime.free_requests[0]
    assert torch.equal(fresh_result, result)
    assert torch.equal(image, before)


def test_oom_in_final_partial_chunk_retries_only_that_chunk(runtime):
    image = torch.arange(7 * 3, dtype=torch.float32).reshape(7, 1, 1, 3)
    starts = []

    def operation(rgb):
        starts.append((rgb[0, 0, 0, 0].item(), len(rgb)))
        if len(starts) == 2:
            raise torch.OutOfMemoryError("simulated")
        return rgb

    assert torch.equal(process_image(image, operation, batch_size=4), image)
    assert starts == [(0, 4), (12, 3), (12, 1), (15, 1), (18, 1)]
    assert runtime.progress[0].updates == [4, 5, 6, 7]


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("simulated"),
        MemoryError("simulated"),
        RuntimeError("DefaultCPUAllocator: not enough memory"),
    ],
)
def test_one_frame_oom_is_actionable_and_does_not_loop(error, runtime):
    def operation(rgb):
        raise error

    with pytest.raises(RuntimeError, match="one frame.*Reduce image resolution"):
        process_image(torch.rand(1, 2, 3, 3), operation)
    assert runtime.progress[0].updates == []
    assert runtime.cache_clears == 0


def test_output_that_cannot_fit_fails_before_allocation(runtime, monkeypatch):
    runtime.available = 1

    def unexpected(*args, **kwargs):
        pytest.fail(
            "Allocation should not be attempted when the complete output cannot fit"
        )

    monkeypatch.setattr(torch, "empty", unexpected)
    with pytest.raises(RuntimeError, match="complete.*output.*batch_size"):
        process_image(torch.ones(1, 1, 1, 3), lambda rgb: rgb)


@pytest.mark.parametrize("output_device", [None, "cpu", "gpu"])
def test_output_allocation_failure_does_not_retry_or_change_placement(
    runtime, monkeypatch, output_device
):
    runtime.device = torch.device("cuda:2")
    calls = []

    def empty(shape, **kwargs):
        calls.append(kwargs["device"])
        raise torch.OutOfMemoryError("simulated fragmentation")

    monkeypatch.setattr(torch, "empty", empty)
    options = {} if output_device is None else {"output_device": output_device}
    destination = runtime.device if output_device == "gpu" else torch.device("cpu")
    with pytest.raises(
        RuntimeError, match=f"complete.*{destination}.*batch_size"
    ) as caught:
        process_image(torch.ones(2, 1, 1, 3), lambda rgb: rgb, **options)
    if output_device == "gpu":
        assert "output_device='cpu'" in str(caught.value)
    assert calls == [destination]
    assert runtime.cache_clears == 0


@pytest.mark.parametrize("output_device", [None, "gpu", "cpu"])
@pytest.mark.parametrize(
    "batch_size,expected", [(0, [64, 64, 3]), (80, [80, 51]), (200, [131])]
)
def test_device_selection_and_gpu_reservation_with_cpu_backed_allocation_spy(
    runtime, gpu_routing, output_device, batch_size, expected
):
    """Verify routing policy without claiming actual GPU execution."""
    image = torch.rand(131, 2, 3, 4, dtype=torch.float64)
    if output_device != "gpu":
        runtime.available = 0  # CPU output does not reserve the full batch on GPU.
    options = {} if output_device is None else {"output_device": output_device}
    output = process_image(image, lambda rgb: rgb, batch_size=batch_size, **options)
    expected_device = runtime.device if output_device == "gpu" else torch.device("cpu")
    assert gpu_routing.allocations == [
        (tuple(image.shape), expected_device, torch.float32)
    ]
    assert gpu_routing.transfers == [
        (count, runtime.device, torch.float32) for count in expected
    ]
    reservation = image.numel() * 4 if output_device == "gpu" else 0
    assert runtime.free_requests == [
        (reservation + expected[0] * 8 * 2 * 3 * 4 * 4, runtime.device)
    ]
    assert runtime.progress[0].updates == list(accumulate(expected))
    assert torch.equal(output, image.float())


def test_nongpu_allocation_errors_propagate_without_retry(runtime):
    error = RuntimeError("unrelated kernel or shape failure")

    def operation(rgb):
        raise error

    with pytest.raises(RuntimeError) as caught:
        process_image(torch.rand(3, 1, 1, 3), operation, batch_size=3)
    assert caught.value is error
    assert runtime.cache_clears == 0


@pytest.mark.parametrize("interrupt_on,updates", [(1, []), (3, [2]), (5, [2, 4, 5])])
def test_cancellation_before_allocation_between_chunks_and_before_return(
    runtime, interrupt_on, updates
):
    runtime.interrupt_on = interrupt_on
    with pytest.raises(InterruptProcessingException):
        process_image(torch.rand(5, 2, 3, 3), lambda rgb: rgb, batch_size=2)
    if updates:
        assert runtime.progress[0].updates == updates
    else:
        assert not runtime.progress
        assert not runtime.free_requests
