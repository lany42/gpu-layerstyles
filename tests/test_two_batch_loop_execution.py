# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""TwoBatchLoop's allocation, transfers, retries, progress, and cleanup."""

import weakref
from itertools import accumulate

import pytest
import torch

from gpu_layerstyles._exec import core, two_batch_loop
from gpu_layerstyles.nodes.two_batch_loop import TwoBatchLoop

from .conftest import InterruptProcessingException


@pytest.mark.parametrize("output_device", [None, "cpu", "gpu"])
@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize(
    "batch_size,counts,transfers",
    [
        (0, [64, 64, 4], [15, 15, 49, 2, 15, 15, 47, 4]),
        (132, [132], [15, 15, 51, 15, 15, 51]),
        (200, [132], [15, 15, 51, 15, 15, 51]),
    ],
)
def test_routing_reservation_and_progress_include_closing_frame(
    output_device, append, batch_size, counts, transfers, runtime, gpu_routing
):
    first = torch.rand(81, 2, 3, 4, dtype=torch.float64)
    second = torch.rand(81, 2, 3, 4, dtype=torch.float16)
    options = {} if output_device is None else {"output_device": output_device}
    if output_device != "gpu":
        runtime.available = 0
    counts = counts.copy()
    if append:
        if batch_size == 132:
            counts.append(1)
        else:
            counts[-1] += 1
    output = TwoBatchLoop.execute(
        first, second, append_first_frame=append, batch_size=batch_size, **options
    ).result[0]
    destination = runtime.device if output_device == "gpu" else torch.device("cpu")
    assert gpu_routing.allocations == [
        ((132 + int(append), 2, 3, 4), destination, torch.float32)
    ]
    assert gpu_routing.transfers == [
        (count, runtime.device, torch.float32) for count in transfers
    ]
    working_bytes = 8 * 2 * counts[0] * 2 * 3 * 4 * 4
    reservation = output.numel() * 4 if output_device == "gpu" else 0
    assert runtime.free_requests == [(working_bytes + reservation, runtime.device)]
    assert runtime.progress[0].total == 132 + int(append)
    assert runtime.progress[0].updates == list(accumulate(counts))
    assert output.dtype == torch.float32
    assert torch.equal(output[15:66], first[15:66].float())
    assert torch.equal(output[81:132], second[15:66].float())
    if append:
        assert torch.equal(output[-1], output[0])


def test_one_output_and_bounded_transfers_and_weights(runtime, monkeypatch):
    first = torch.rand(101, 2, 3, 3, dtype=torch.float64)
    second = torch.rand(149, 2, 3, 3, dtype=torch.float64)
    allocations, weight_counts = [], []
    original_empty, original_to, original_arange = (
        torch.empty,
        torch.Tensor.to,
        torch.arange,
    )

    def empty(shape, **kwargs):
        assert runtime.free_requests
        allocations.append(tuple(shape))
        return original_empty(shape, **kwargs)

    def to(tensor, *args, **kwargs):
        assert runtime.free_requests
        assert len(tensor) <= 7
        return original_to(tensor, *args, **kwargs)

    def arange(start, end, **kwargs):
        assert runtime.free_requests
        weight_counts.append(end - start)
        return original_arange(start, end, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("TwoBatchLoop must write directly to its preallocated destination")

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch, "arange", arange)
    monkeypatch.setattr(torch, "cat", unexpected)
    monkeypatch.setattr(torch, "stack", unexpected)
    output = TwoBatchLoop.execute(first, second, 33, True, batch_size=7).result[0]
    assert allocations == [(185, 2, 3, 3)]
    assert max(weight_counts) == 7
    assert sum(weight_counts) == 66
    assert runtime.free_requests == [
        (output.numel() * 4 + 8 * 2 * 7 * 2 * 3 * 3 * 4, runtime.device)
    ]
    assert runtime.progress[0].total == 185
    assert runtime.progress[0].updates[-1] == 185
    assert torch.equal(output[-1], output[0])


@pytest.mark.parametrize(
    "batch_size,attempts,updates,retry_count",
    [
        (133, [(0, 133), (0, 66), (66, 66), (132, 1)], [66, 132, 133], 66),
        (
            0,
            [(0, 64), (64, 64), (128, 5), (128, 2), (130, 2), (132, 1)],
            [64, 128, 130, 132, 133],
            2,
        ),
    ],
)
def test_partially_written_closing_chunk_retries_then_copies_a_single_frame(
    batch_size, attempts, updates, retry_count, runtime, monkeypatch
):
    first = torch.rand(81, 2, 3, 4, dtype=torch.float64)
    second = torch.rand(81, 2, 3, 4, dtype=torch.float64)
    expected = TwoBatchLoop.execute(first, second, append_first_frame=True).result[0]
    runtime.progress.clear()
    runtime.free_requests.clear()
    original_execute = two_batch_loop._execute
    original_copy = torch.Tensor.copy_
    original_empty = torch.empty
    original_to = torch.Tensor.to
    seen, allocations, converted = [], [], []
    fail = True

    def execute(image, process, *args, **kwargs):
        def record(start, count, destination, device):
            seen.append((start, count))
            process(start, count, destination, device)

        return original_execute(image, record, *args, **kwargs)

    def copy(tensor, source, *args, **kwargs):
        nonlocal fail
        result = original_copy(tensor, source, *args, **kwargs)
        if fail and tensor.storage_offset() == 132 * 2 * 3 * 4:
            fail = False
            raise torch.OutOfMemoryError("simulated failure after closing-frame write")
        return result

    def empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append(weakref.ref(result))
        return result

    def to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        converted.append(weakref.ref(result))
        return result

    def empty_cache():
        assert converted and all(reference() is None for reference in converted)
        runtime.cache_clears += 1

    monkeypatch.setattr(two_batch_loop, "_execute", execute)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(core.model_management, "soft_empty_cache", empty_cache)
    output = TwoBatchLoop.execute(
        first, second, append_first_frame=True, batch_size=batch_size
    ).result[0]
    assert seen == attempts
    assert len(allocations) == 1
    assert allocations[0]() is output
    assert torch.equal(output, expected)
    assert runtime.progress[0].updates == updates
    assert runtime.progress[0].total == 133
    assert runtime.cache_clears == 1
    frame_bytes = 2 * 3 * 4 * 4
    assert runtime.free_requests == [
        ((133 + 8 * 2 * attempts[0][1]) * frame_bytes, runtime.device),
        (8 * 2 * retry_count * frame_bytes, runtime.device),
    ]
    seen.clear()
    again = TwoBatchLoop.execute(
        first, second, append_first_frame=True, batch_size=batch_size
    ).result[0]
    assert seen[0] == attempts[0]  # A fresh run restores the requested chunk size.
    assert runtime.free_requests[-1] == runtime.free_requests[0]
    assert torch.equal(again, expected)


def test_oom_on_second_input_transfer_releases_failed_temporaries(runtime, monkeypatch):
    first = torch.zeros(31, 1, 1, 3, dtype=torch.float64)
    second = torch.ones(31, 1, 1, 3, dtype=torch.float64)
    expected = TwoBatchLoop.execute(first, second, append_first_frame=True).result[0]
    original_to = torch.Tensor.to
    converted = []

    def to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        converted.append(weakref.ref(result))
        if tensor[0, 0, 0, 0] == 0 and len(tensor) > 4:
            raise MemoryError("simulated input allocation failure")
        return result

    def empty_cache():
        assert converted and all(reference() is None for reference in converted)
        runtime.cache_clears += 1

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(core.model_management, "soft_empty_cache", empty_cache)
    output = TwoBatchLoop.execute(first, second, append_first_frame=True).result[0]
    assert torch.equal(output, expected)
    assert runtime.cache_clears == 3  # 33 -> 16 -> 8 -> 4 frames.
    assert runtime.progress[-1].updates == [4, 8, 12, 16, 20, 24, 28, 32, 33]


@pytest.mark.parametrize(
    "error_type",
    [torch.OutOfMemoryError, MemoryError, RuntimeError, InterruptProcessingException],
)
@pytest.mark.parametrize("stage", ["transition", "closing"])
def test_retained_terminal_exception_releases_output_and_temporaries(
    error_type, stage, runtime, monkeypatch
):
    first = torch.zeros(5, 1, 1, 3, dtype=torch.float64)
    second = torch.ones(5, 1, 1, 3, dtype=torch.float64)
    original_empty, original_to, original_copy = (
        torch.empty,
        torch.Tensor.to,
        torch.Tensor.copy_,
    )
    references = []

    def empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    def to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        references.append(weakref.ref(result))
        if stage == "transition" and tensor[0, 0, 0, 0] == 0:
            raise error_type("simulated terminal transfer failure")
        return result

    def copy(tensor, source, *args, **kwargs):
        if stage == "closing" and tensor.storage_offset() == 6 * 3:
            raise error_type("simulated terminal closing-frame failure")
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    expected_error = (
        RuntimeError
        if error_type in (torch.OutOfMemoryError, MemoryError)
        else error_type
    )
    with pytest.raises(expected_error) as caught:
        TwoBatchLoop.execute(first, second, 2, True, batch_size=1)
    assert caught.value is not None
    assert references and all(reference() is None for reference in references)
    assert runtime.cache_clears == 0
    assert runtime.progress[0].updates == (
        [] if stage == "transition" else [1, 2, 3, 4, 5, 6]
    )
    if error_type in (torch.OutOfMemoryError, MemoryError):
        assert "processing one frame" in str(caught.value)


def test_closing_frame_reservation_fails_before_allocation(runtime, monkeypatch):
    first, second = torch.ones(5, 2, 3, 3), torch.ones(5, 2, 3, 3)
    runtime.available = 6 * 2 * 3 * 3 * 4  # Exactly enough for the unclosed output.

    def unexpected(*args, **kwargs):
        pytest.fail("The complete output must fit before allocation")

    monkeypatch.setattr(torch, "empty", unexpected)
    with pytest.raises(RuntimeError, match="complete.*output on cpu.*batch_size"):
        TwoBatchLoop.execute(first, second, 2, True)
    assert not runtime.progress


@pytest.mark.parametrize("output_device", [None, "cpu", "gpu"])
def test_destination_failure_preserves_placement_without_retries(
    output_device, runtime, monkeypatch
):
    first, second = torch.ones(5, 2, 3, 3), torch.ones(5, 2, 3, 3)
    runtime.device = torch.device("cuda:2")
    allocations = []

    def empty(shape, **kwargs):
        assert runtime.free_requests
        allocations.append((tuple(shape), kwargs["device"], kwargs["dtype"]))
        raise torch.OutOfMemoryError("simulated output allocation failure")

    monkeypatch.setattr(torch, "empty", empty)
    options = {} if output_device is None else {"output_device": output_device}
    destination = runtime.device if output_device == "gpu" else torch.device("cpu")
    with pytest.raises(
        RuntimeError, match=f"complete.*output on {destination}.*batch_size"
    ):
        TwoBatchLoop.execute(first, second, 2, True, **options)
    assert allocations == [((7, 2, 3, 3), destination, torch.float32)]
    assert not runtime.progress
    assert runtime.cache_clears == 0


@pytest.mark.parametrize(
    "interrupt_on,updates", [(1, []), (3, [2]), (5, [2, 4, 6]), (6, [2, 4, 6, 7])]
)
def test_cancellation_before_allocation_chunks_closing_frame_and_return(
    interrupt_on, updates, runtime, monkeypatch
):
    first, second = torch.ones(5, 1, 1, 3), torch.ones(5, 1, 1, 3)
    runtime.interrupt_on = interrupt_on
    original_empty = torch.empty
    references = []

    def empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(InterruptProcessingException) as caught:
        TwoBatchLoop.execute(first, second, 2, True, batch_size=2)
    assert caught.value is not None
    assert all(reference() is None for reference in references)
    if updates:
        assert len(references) == 1
        assert runtime.progress[0].updates == updates
    else:
        assert not references
        assert not runtime.progress
        assert not runtime.free_requests
