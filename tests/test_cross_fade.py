# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""CrossFade's sequence assembly, numerics, validation, and chunked execution."""

import weakref
from itertools import accumulate

import pytest
import torch

from gpu_layerstyles._exec import core, crossfade
from gpu_layerstyles.nodes.cross_fade import CrossFade

from .conftest import InterruptProcessingException


@pytest.mark.parametrize("batch_size", [0, 1, 2, 3, 4, 99])
@pytest.mark.parametrize(
    "start,frames,expected",
    [
        (0, 2, [0, 12, 16, 20, 24]),
        (0, 5, [0, 3.75, 9, 15.75, 24]),
        (2, 3, [0, 1, 2, 7.5, 16, 20, 24]),
        (4, 3, [0, 1, 2, 3, 4, 8.5, 16, 20, 24]),
    ],
)
def test_exact_sequence_endpoints_and_first_batch_tail_discard(
    start, frames, expected, batch_size
):
    images_1 = torch.arange(7, dtype=torch.float32)[:, None, None, None].expand(
        -1, 2, 3, 3
    )
    images_2 = torch.tensor([8, 12, 16, 20, 24], dtype=torch.float32)[
        :, None, None, None
    ].expand(-1, 2, 3, 3)
    output = CrossFade.execute(
        images_1, images_2, start, frames, batch_size=batch_size
    ).result[0]
    values = torch.tensor(expected)[:, None, None, None].expand(-1, 2, 3, 3)
    assert torch.equal(output, values)
    assert len(output) == start + len(images_2)


def test_defaults_make_a_two_frame_transition():
    first = torch.full((2, 1, 1, 3), -2.0)
    second = torch.full((2, 1, 1, 3), 3.0)
    output = CrossFade.execute(first, second).result[0]
    assert torch.equal(
        output, torch.tensor([-2.0, 3.0])[:, None, None, None].expand_as(output)
    )


def test_rgba_channels_fade_independently_without_premultiplication_or_clamping():
    first = torch.tensor([-2.0, 0.5, 1.5, 0.0]).expand(3, 1, 1, 4)
    second = torch.tensor([2.0, 0.0, 0.5, 1.0]).expand(3, 1, 1, 4)
    output = CrossFade.execute(first, second, frames=3).result[0]
    expected = torch.tensor(
        [
            [-2.0, 0.5, 1.5, 0.0],
            [0.0, 0.25, 1.0, 0.5],
            [2.0, 0.0, 0.5, 1.0],
        ]
    )[:, None, None, :]
    assert torch.equal(output, expected)


@pytest.mark.parametrize("batch_size", [0, 3])
def test_chunked_mixed_precision_fade_matches_a_float64_reference(batch_size):
    generator = torch.Generator().manual_seed(137)
    first = (torch.rand(77, 3, 5, 4, generator=generator) * 3 - 1).half()
    second = (torch.rand(131, 3, 5, 4, generator=generator) * 3 - 1).double()
    output = CrossFade.execute(first, second, 5, 67, batch_size=batch_size).result[0]

    # Use a per-frame float64 reference, independent of the chunked float32 kernel.
    expected = [frame.double() for frame in first[:5]]
    for j in range(67):
        weight = j / 66
        expected.append(
            first[5 + j].double() * (1 - weight) + second[j].double() * weight
        )
    expected.extend(frame.double() for frame in second[67:])
    expected = torch.stack(expected).float()
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=4e-7)
    assert output.shape == (136, 3, 5, 4)
    assert torch.equal(output[5], first[5].float())
    assert torch.equal(output[71], second[66].float())


def test_overlapping_inputs_are_read_without_aliasing_the_output():
    image = torch.rand(9, 2, 3, 4, generator=torch.Generator().manual_seed(2))
    before = image.clone()
    output = CrossFade.execute(image, image[2:], 1, 3, batch_size=2).result[0]
    torch.testing.assert_close(output[2], (before[2] + before[3]) / 2)
    output.zero_()
    assert torch.equal(image, before)


@pytest.fixture
def forbid_memory_management(runtime):
    yield
    assert not runtime.free_requests


@pytest.mark.parametrize("name", ["start_index", "frames"])
@pytest.mark.parametrize("value", [True, -1, 2.0])
def test_controls_must_be_exact_integers(name, value, forbid_memory_management):
    with pytest.raises(ValueError, match=name):
        CrossFade.execute(
            torch.ones(5, 1, 1, 3), torch.ones(5, 1, 1, 3), **{name: value}
        )


@pytest.mark.parametrize(
    "start,frames,n1,n2,match",
    [
        (0, 0, 5, 5, "frames must be an integer >= 2"),
        (0, 1, 5, 5, "frames must be an integer >= 2"),
        (5, 2, 5, 5, "start_index 5 is out of range"),
        (6, 2, 5, 5, "start_index 6 is out of range"),
        (4, 2, 5, 5, "exactly 2.*images_1 has 1.*images_2 has 5"),
        (1, 5, 5, 5, "exactly 5.*images_1 has 4.*images_2 has 5"),
        (0, 5, 7, 3, "exactly 5.*images_1 has 7.*images_2 has 3"),
        (0, 2, 1, 5, "exactly 2.*images_1 has 1.*images_2 has 5"),
        (0, 2, 5, 1, "exactly 2.*images_1 has 5.*images_2 has 1"),
    ],
)
def test_invalid_ranges_are_rejected_without_shortening(
    start, frames, n1, n2, match, forbid_memory_management
):
    with pytest.raises(ValueError, match=match):
        CrossFade.execute(
            torch.ones(n1, 1, 1, 3), torch.ones(n2, 1, 1, 3), start, frames
        )


@pytest.mark.parametrize("shape", [(2, 1, 3, 3), (2, 2, 1, 3), (2, 2, 3, 4)])
def test_mismatched_dimensions_and_channels_are_rejected(
    shape, forbid_memory_management
):
    with pytest.raises(ValueError, match="matching height, width, and channel count"):
        CrossFade.execute(torch.ones(2, 2, 3, 3), torch.ones(shape))


@pytest.mark.parametrize(
    "batch_size,counts,transfers",
    [
        (0, [64, 64, 8], [5, 59, 59, 8, 8, 56, 8]),
        (80, [80, 56], [5, 67, 67, 8, 56]),
        (200, [136], [5, 67, 67, 64]),
    ],
)
def test_chunks_transfer_only_the_frames_each_seam_needs(
    batch_size, counts, transfers, runtime, gpu_routing
):
    first = torch.rand(77, 2, 3, 4, dtype=torch.float64)
    second = torch.rand(131, 2, 3, 4, dtype=torch.float16)
    output = CrossFade.execute(first, second, 5, 67, "gpu", batch_size).result[0]
    assert gpu_routing.allocations == [((136, 2, 3, 4), runtime.device, torch.float32)]
    assert gpu_routing.transfers == [
        (count, runtime.device, torch.float32) for count in transfers
    ]
    # Both float32 input chunks and their blend temporaries are reserved.
    working_bytes = 8 * 2 * counts[0] * 2 * 3 * 4 * 4
    assert runtime.free_requests == [
        (working_bytes + output.numel() * 4, runtime.device)
    ]
    assert runtime.progress[0].total == 136
    assert runtime.progress[0].updates == list(accumulate(counts))
    assert output.dtype == torch.float32
    assert torch.equal(output[:5], first[:5].float())
    assert torch.equal(output[72:], second[67:].float())


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
    [torch.OutOfMemoryError, InterruptProcessingException],
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
        RuntimeError if error_type is torch.OutOfMemoryError else error_type
    )
    with pytest.raises(expected_error) as caught:
        CrossFade.execute(first, second, batch_size=1)
    assert caught.value is not None  # Keep the exception and its tracebacks alive.
    assert len(references) == 3
    assert all(reference() is None for reference in references)
    assert runtime.cache_clears == 0
    assert runtime.progress[0].updates == []
    if error_type is torch.OutOfMemoryError:
        assert "processing one frame" in str(caught.value)
