# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""CrossFade's output sizing, device routing, retries, and cancellation."""

import weakref
from itertools import accumulate

import pytest
import torch

from gpu_layerstyles._exec import core, crossfade
from gpu_layerstyles.nodes.cross_fade import CrossFade

from .conftest import InterruptProcessingException


@pytest.mark.parametrize("output_device", [None, "cpu", "gpu"])
@pytest.mark.parametrize(
    "batch_size,counts,transfers",
    [
        (0, [64, 64, 8], [5, 59, 59, 8, 8, 56, 8]),
        (80, [80, 56], [5, 67, 67, 8, 56]),
        (200, [136], [5, 67, 67, 64]),
    ],
)
def test_routing_output_chunk_sizes_and_both_input_reservations(
    output_device, batch_size, counts, transfers, runtime, gpu_routing
):
    first = torch.rand(77, 2, 3, 4, dtype=torch.float64)
    second = torch.rand(131, 2, 3, 4, dtype=torch.float16)
    options = {} if output_device is None else {"output_device": output_device}
    if output_device != "gpu":
        runtime.available = 0  # CPU output does not reserve the whole result on GPU.
    output = CrossFade.execute(
        first, second, 5, 67, batch_size=batch_size, **options
    ).result[0]
    destination = runtime.device if output_device == "gpu" else torch.device("cpu")
    assert gpu_routing.allocations == [((136, 2, 3, 4), destination, torch.float32)]
    assert gpu_routing.transfers == [
        (count, runtime.device, torch.float32) for count in transfers
    ]
    working_bytes = 8 * 2 * counts[0] * 2 * 3 * 4 * 4
    reservation = output.numel() * 4 if output_device == "gpu" else 0
    assert runtime.free_requests == [(working_bytes + reservation, runtime.device)]
    assert runtime.progress[0].total == 136
    assert runtime.progress[0].updates == list(accumulate(counts))
    assert output.dtype == torch.float32
    assert torch.equal(output[:5], first[:5].float())
    assert torch.equal(output[72:], second[67:].float())


def test_memory_management_precedes_work_with_one_output_and_bounded_weights(
    runtime, monkeypatch
):
    first = torch.rand(101, 2, 3, 3, dtype=torch.float64)
    second = torch.rand(149, 2, 3, 3, dtype=torch.float64)
    allocations, weight_counts = [], []
    original_empty = torch.empty
    original_to = torch.Tensor.to
    original_arange = torch.arange

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
        pytest.fail("CrossFade must write directly to its preallocated destination")

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch, "arange", arange)
    monkeypatch.setattr(torch, "cat", unexpected)
    monkeypatch.setattr(torch, "stack", unexpected)
    output = CrossFade.execute(first, second, 5, 93, batch_size=7).result[0]
    assert allocations == [(154, 2, 3, 3)]
    assert max(weight_counts) == 7
    assert sum(weight_counts) == 93
    assert runtime.free_requests == [
        (output.numel() * 4 + 8 * 2 * 7 * 2 * 3 * 3 * 4, runtime.device)
    ]


def test_oom_during_second_input_transfer_retries_seams_and_restarts_fresh(
    runtime, monkeypatch
):
    first = torch.arange(149, dtype=torch.float64)[:, None, None, None].expand(
        -1, 2, 3, 4
    )
    second = (1000 + torch.arange(101, dtype=torch.float64))[
        :, None, None, None
    ].expand(-1, 2, 3, 4)
    before = first.clone(), second.clone()
    original_process = crossfade._process_crossfade_chunk
    original_to = torch.Tensor.to
    attempts, converted = [], []
    fail = True

    def process(
        images_1, images_2, start_index, frames, start, count, destination, device
    ):
        attempts.append((start, count))
        return original_process(
            images_1, images_2, start_index, frames, start, count, destination, device
        )

    def to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        converted.append(weakref.ref(result))
        if fail and tensor[0, 0, 0, 0] >= 1000 and len(tensor) > 16:
            raise torch.OutOfMemoryError("simulated second-input allocation failure")
        return result

    def empty_cache():
        # Both first-input and failed second-input conversions must be released.
        assert converted and all(reference() is None for reference in converted)
        runtime.cache_clears += 1

    monkeypatch.setattr(crossfade, "_process_crossfade_chunk", process)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(core.model_management, "soft_empty_cache", empty_cache)
    output = CrossFade.execute(first, second, 70, 73).result[0]
    assert attempts == [
        (0, 64),
        (64, 64),
        (64, 32),
        (64, 16),
        (80, 16),
        (96, 16),
        (112, 16),
        (128, 16),
        (144, 16),
        (160, 11),
    ]
    assert runtime.progress[0].updates == [64, 80, 96, 112, 128, 144, 160, 171]
    assert runtime.cache_clears == 2
    frame_bytes = 2 * 3 * 4 * 4
    assert runtime.free_requests == [
        ((171 + 8 * 2 * 64) * frame_bytes, runtime.device),
        (8 * 2 * 32 * frame_bytes, runtime.device),
        (8 * 2 * 16 * frame_bytes, runtime.device),
    ]
    expected_values = [
        *range(70),
        *(70 + j + 930 * j / 72 for j in range(73)),
        *range(1073, 1101),
    ]
    expected = torch.tensor(expected_values)[:, None, None, None].expand_as(output)
    torch.testing.assert_close(output, expected)
    assert torch.equal(first, before[0])
    assert torch.equal(second, before[1])

    fail = False
    attempts.clear()
    again = CrossFade.execute(first, second, 70, 73).result[0]
    assert attempts == [(0, 64), (64, 64), (128, 43)]
    assert runtime.progress[1].updates == [64, 128, 171]
    assert runtime.free_requests[-1] == runtime.free_requests[0]
    assert torch.equal(output, again)


@pytest.mark.parametrize(
    "error_type",
    [torch.OutOfMemoryError, MemoryError, RuntimeError, InterruptProcessingException],
)
def test_retained_terminal_exception_releases_output_and_temporaries(
    error_type, runtime, monkeypatch
):
    first = torch.zeros(2, 1, 1, 3, dtype=torch.float64)
    second = torch.ones(2, 1, 1, 3, dtype=torch.float64)
    original_empty = torch.empty
    original_to = torch.Tensor.to
    references = []

    def empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    def to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        references.append(weakref.ref(result))
        if tensor[0, 0, 0, 0] == 1:
            raise error_type("simulated terminal transfer failure")
        return result

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    expected_error = (
        RuntimeError
        if error_type in (torch.OutOfMemoryError, MemoryError)
        else error_type
    )
    with pytest.raises(expected_error) as caught:
        CrossFade.execute(first, second, batch_size=1)
    assert caught.value is not None  # Keep the exception and its tracebacks alive.
    assert len(references) == 3
    assert all(reference() is None for reference in references)
    assert runtime.cache_clears == 0
    assert runtime.progress[0].updates == []
    if error_type in (torch.OutOfMemoryError, MemoryError):
        assert "processing one frame" in str(caught.value)


def test_complete_output_too_large_fails_before_allocation(runtime, monkeypatch):
    first, second = torch.ones(7, 2, 3, 3), torch.ones(13, 2, 3, 3)
    runtime.available = 16 * 2 * 3 * 3 * 4 - 1  # Output is longer than either input.

    def unexpected(*args, **kwargs):
        pytest.fail("The complete output must fit before allocation")

    monkeypatch.setattr(torch, "empty", unexpected)
    with pytest.raises(RuntimeError, match="complete.*output on cpu.*batch_size"):
        CrossFade.execute(first, second, 3, 3)
    assert not runtime.progress


@pytest.mark.parametrize("output_device", [None, "cpu", "gpu"])
def test_destination_failure_preserves_output_placement_without_retries(
    output_device, runtime, monkeypatch
):
    first, second = torch.ones(7, 2, 3, 3), torch.ones(13, 2, 3, 3)
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
        CrossFade.execute(first, second, 3, 3, **options)
    assert allocations == [((16, 2, 3, 3), destination, torch.float32)]
    assert not runtime.progress
    assert runtime.cache_clears == 0


@pytest.mark.parametrize("interrupt_on,updates", [(1, []), (3, [2]), (5, [2, 4, 5])])
def test_cancellation_before_allocation_between_chunks_and_before_return(
    interrupt_on, updates, runtime, monkeypatch
):
    first, second = torch.ones(5, 1, 1, 3), torch.ones(4, 1, 1, 3)
    runtime.interrupt_on = interrupt_on
    original_empty = torch.empty
    references = []

    def empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(InterruptProcessingException) as caught:
        CrossFade.execute(first, second, 1, 2, batch_size=2)
    assert caught.value is not None
    assert all(reference() is None for reference in references)
    if updates:
        assert len(references) == 1
        assert runtime.progress[0].updates == updates
    else:
        assert not references
        assert not runtime.progress
        assert not runtime.free_requests
