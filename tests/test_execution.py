import weakref

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
    "frames,frame_bytes,available,expected",
    [
        (240, 12, 1024**3, 32),  # Frame count cap.
        (5, 12, 1024**3, 5),
        (240, 720 * 1280 * 3 * 4, 16 * 1024**3, 6),  # 512 MiB cap.
        (240, 720 * 1280 * 3 * 4, 512 * 1024**2, 1),  # Quarter of free memory.
        (50, 1024, 8 * 1024 * 4 * 7, 7),
        (50, 1024, 0, 1),
        (50, 1024, -1, 1),
    ],
)
def test_auto_chunk_budget(frames, frame_bytes, available, expected):
    assert _execution._automatic_chunk_size(frames, frame_bytes, available) == expected


def test_auto_chunk_reserves_complete_output_first(runtime):
    image = torch.rand(12, 2, 3, 4)
    frame_bytes = 2 * 3 * 4 * 4
    runtime.available = image.numel() * 4 + 4 * 8 * frame_bytes * 3
    calls = []

    def operation(rgb):
        calls.append(len(rgb))
        return rgb

    result = process_image(image, operation)
    assert calls == [3, 3, 3, 3]
    assert runtime.free_requests == [
        (image.numel() * 4 + 8 * frame_bytes, runtime.device)
    ]
    assert torch.equal(result, image)


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


def test_output_allocation_failure_does_not_retry_or_change_placement(
    runtime, monkeypatch
):
    runtime.device = torch.device("cuda:2")
    calls = []

    def empty(shape, **kwargs):
        calls.append(kwargs["device"])
        raise torch.OutOfMemoryError("simulated fragmentation")

    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(RuntimeError, match="complete.*cuda:2.*output_device='cpu'"):
        process_image(torch.ones(2, 1, 1, 3), lambda rgb: rgb)
    assert calls == [torch.device("cuda:2")]
    assert runtime.cache_clears == 0


@pytest.mark.parametrize("output_device", ["gpu", "cpu"])
def test_device_selection_and_gpu_reservation_with_cpu_backed_allocation_spy(
    runtime, monkeypatch, output_device
):
    """Verify routing policy without claiming actual GPU execution."""
    runtime.device = torch.device("cuda:2")
    image = torch.rand(5, 2, 3, 4)
    original_empty = torch.empty
    original_process = _execution._process_chunk
    placements, chunks = [], []

    def empty(shape, *, device, dtype):
        placements.append(device)
        return original_empty(shape, dtype=dtype, device="cpu")

    def process(source, destination, operation, device):
        chunks.append((len(source), device, destination.device))
        original_process(source, destination, operation, torch.device("cpu"))

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(_execution, "_process_chunk", process)
    output = process_image(image, lambda rgb: rgb, output_device, 2)
    expected_device = runtime.device if output_device == "gpu" else torch.device("cpu")
    assert placements == [expected_device]
    assert [count for count, _, _ in chunks] == [2, 2, 1]
    assert all(device == runtime.device for _, device, _ in chunks)
    reservation = image.numel() * 4 if output_device == "gpu" else 0
    assert runtime.free_requests[0] == (
        reservation + 2 * 8 * 2 * 3 * 4 * 4,
        runtime.device,
    )
    assert torch.equal(output, image)


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
