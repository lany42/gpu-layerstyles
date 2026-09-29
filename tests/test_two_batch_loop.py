# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""TwoBatchLoop's sequence, numerics, validation, and chunked execution."""

import weakref
from itertools import accumulate

import pytest
import torch

from gpu_layerstyles._exec import core, two_batch_loop
from gpu_layerstyles.nodes.two_batch_loop import TwoBatchLoop

from .conftest import InterruptProcessingException


def reference_loop(first, second, frames):
    # Per-frame float64 arithmetic, independent of the chunked float32 kernel.
    def fade(source, target):
        return [
            source[len(source) - frames + j].double() * (1 - j / (frames - 1))
            + target[j].double() * (j / (frames - 1))
            for j in range(frames)
        ]

    return torch.stack(
        fade(second, first)
        + [first[j].double() for j in range(frames, len(first) - frames)]
        + fade(first, second)
        + [second[j].double() for j in range(frames, len(second) - frames)]
    ).float()


# Chunks may split a transition, a middle, or isolate the closing frame.
@pytest.mark.parametrize("batch_size", [1, 15, 66, 132])
def test_complete_81_frame_sequence_and_optional_closing_frame(batch_size):
    first = torch.arange(81, dtype=torch.float32)[:, None, None, None].expand(
        -1, 2, 3, 3
    )
    second = first + 1000
    output = TwoBatchLoop.execute(first, second, batch_size=batch_size).result[0]
    assert output.shape == (132, 2, 3, 3)
    torch.testing.assert_close(output, reference_loop(first, second, 15))
    assert torch.equal(output[0], second[66])
    assert torch.equal(output[14], first[14])
    assert torch.equal(output[15:66], first[15:66])
    assert torch.equal(output[66], first[66])
    assert torch.equal(output[80], second[14])
    assert torch.equal(output[81:132], second[15:66])
    # The middle's last frame immediately precedes the tail at loop frame zero.
    assert torch.equal(output[-1], second[65])

    closed = TwoBatchLoop.execute(
        first, second, append_first_frame=True, batch_size=batch_size
    ).result[0]
    assert closed.shape == (133, 2, 3, 3)
    assert torch.equal(closed[:-1], output)
    assert torch.equal(closed[-1], output[0])
    closed[-1].zero_()
    assert torch.equal(closed[0], output[0])  # A copy, not shared frame storage.


@pytest.mark.parametrize("n1,n2,frames", [(5, 5, 2), (5, 9, 2), (47, 31, 15)])
@pytest.mark.parametrize("append", [False, True])
def test_unequal_and_minimum_lengths(n1, n2, frames, append):
    first = torch.rand(n1, 2, 3, 4)
    second = torch.rand(n2, 2, 3, 4)
    output = TwoBatchLoop.execute(first, second, frames, append, batch_size=3).result[0]
    assert len(output) == n1 + n2 - 2 * frames + int(append)
    torch.testing.assert_close(
        output[: n1 + n2 - 2 * frames], reference_loop(first, second, frames)
    )
    if append:
        assert torch.equal(output[-1], output[0])


def test_rgba_blends_alpha_independently_without_clamping():
    first = torch.tensor([-2.0, 0.5, 1.5, 0.0]).expand(7, 1, 1, 4)
    second = torch.tensor([2.0, 0.0, 0.5, 1.0]).expand(7, 1, 1, 4)
    output = TwoBatchLoop.execute(first, second, 3).result[0]
    expected = torch.tensor(
        [
            [2.0, 0.0, 0.5, 1.0],
            [0.0, 0.25, 1.0, 0.5],
            [-2.0, 0.5, 1.5, 0.0],
            [-2.0, 0.5, 1.5, 0.0],
            [-2.0, 0.5, 1.5, 0.0],
            [0.0, 0.25, 1.0, 0.5],
            [2.0, 0.0, 0.5, 1.0],
            [2.0, 0.0, 0.5, 1.0],
        ]
    )[:, None, None, :]
    assert torch.equal(output, expected)


@pytest.mark.parametrize("batch_size", [0, 7])
def test_chunked_mixed_precision_loop_matches_a_float64_reference(batch_size):
    generator = torch.Generator().manual_seed(137)
    first = (torch.rand(77, 3, 5, 4, generator=generator) * 3 - 1).half()
    second = (torch.rand(91, 3, 5, 4, generator=generator) * 3 - 1).double()
    output = TwoBatchLoop.execute(
        first, second, 33, True, batch_size=batch_size
    ).result[0]
    torch.testing.assert_close(
        output[:-1], reference_loop(first, second, 33), rtol=2e-6, atol=4e-7
    )
    assert output.shape == (103, 3, 5, 4)
    assert torch.equal(output[-1], output[0])


def test_overlapping_inputs_are_read_without_aliasing_the_output():
    image = torch.rand(9, 2, 3, 4, generator=torch.Generator().manual_seed(2))
    before = image.clone()
    output = TwoBatchLoop.execute(image, image[2:], 3, True, batch_size=2).result[0]
    torch.testing.assert_close(output[:-1], reference_loop(before, before[2:], 3))
    output.zero_()
    assert torch.equal(image, before)


@pytest.fixture
def forbid_memory_management(runtime):
    yield
    assert not runtime.free_requests


@pytest.mark.parametrize("value", [True, 1, 2.0, 2**31])
def test_blend_target_requires_an_integer_in_range(value, forbid_memory_management):
    with pytest.raises(ValueError, match="^blend_target must be an integer between 2"):
        TwoBatchLoop.execute(torch.ones(5, 1, 1, 3), torch.ones(5, 1, 1, 3), value)


@pytest.mark.parametrize("value", [1, "False"])
def test_closing_frame_requires_a_boolean(value, forbid_memory_management):
    with pytest.raises(TypeError, match="append_first_frame"):
        TwoBatchLoop.execute(torch.ones(5, 1, 1, 3), torch.ones(5, 1, 1, 3), 2, value)


@pytest.mark.parametrize("name", ["images_1", "images_2"])
@pytest.mark.parametrize("frames,length", [(2, 4), (15, 30)])
def test_short_batches_and_empty_middles_are_rejected(
    name, frames, length, forbid_memory_management
):
    inputs = {"images_1": torch.ones(81, 1, 1, 3), "images_2": torch.ones(81, 1, 1, 3)}
    inputs[name] = torch.ones(length, 1, 1, 3)
    with pytest.raises(
        ValueError, match=f"at least {2 * frames + 1} frames in {name}.*nonempty middle"
    ):
        TwoBatchLoop.execute(**inputs, blend_target=frames)


@pytest.mark.parametrize("shape", [(31, 1, 3, 3), (31, 2, 1, 3), (31, 2, 3, 4)])
def test_mismatched_dimensions_and_channels_are_rejected(
    shape, forbid_memory_management
):
    with pytest.raises(ValueError, match="matching height, width, and channel count"):
        TwoBatchLoop.execute(torch.ones(31, 2, 3, 3), torch.ones(shape))


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize(
    "batch_size,counts,transfers",
    [
        (0, [64, 64, 4], [15, 15, 49, 2, 15, 15, 47, 4]),
        (132, [132], [15, 15, 51, 15, 15, 51]),
    ],
)
def test_chunks_transfer_only_needed_frames_and_reserve_the_closing_frame(
    append, batch_size, counts, transfers, runtime, gpu_routing
):
    first = torch.rand(81, 2, 3, 4, dtype=torch.float64)
    second = torch.rand(81, 2, 3, 4, dtype=torch.float16)
    counts = counts.copy()
    if append:
        if batch_size == 132:
            counts.append(1)
        else:
            counts[-1] += 1
    output = TwoBatchLoop.execute(
        first,
        second,
        append_first_frame=append,
        output_device="gpu",
        batch_size=batch_size,
    ).result[0]
    assert gpu_routing.allocations == [
        ((132 + int(append), 2, 3, 4), runtime.device, torch.float32)
    ]
    assert gpu_routing.transfers == [
        (count, runtime.device, torch.float32) for count in transfers
    ]
    working_bytes = 8 * 2 * counts[0] * 2 * 3 * 4 * 4
    assert runtime.free_requests == [
        (working_bytes + output.numel() * 4, runtime.device)
    ]
    assert runtime.progress[0].total == 132 + int(append)
    assert runtime.progress[0].updates == list(accumulate(counts))
    assert output.dtype == torch.float32
    assert torch.equal(output[15:66], first[15:66].float())
    assert torch.equal(output[81:132], second[15:66].float())
    if append:
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


@pytest.mark.parametrize(
    "error_type",
    [torch.OutOfMemoryError, InterruptProcessingException],
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
        RuntimeError if error_type is torch.OutOfMemoryError else error_type
    )
    with pytest.raises(expected_error) as caught:
        TwoBatchLoop.execute(first, second, 2, True, batch_size=1)
    assert caught.value is not None
    assert references and all(reference() is None for reference in references)
    assert runtime.cache_clears == 0
    assert runtime.progress[0].updates == (
        [] if stage == "transition" else [1, 2, 3, 4, 5, 6]
    )
    if error_type is torch.OutOfMemoryError:
        assert "processing one frame" in str(caught.value)
