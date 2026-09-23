# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Resized destination accounting, reusable coefficients, and failure cleanup."""

import weakref
from itertools import accumulate

import pytest
import torch

from gpu_layerstyles import _resize
from gpu_layerstyles._exec import core
from gpu_layerstyles.nodes.image_scale_down import ImageScaleDown

from .conftest import InterruptProcessingException


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
@pytest.mark.parametrize("output_device", [None, "cpu", "gpu"])
@pytest.mark.parametrize(
    "batch_size,counts", [(0, [64, 64, 3]), (80, [80, 51]), (200, [131])]
)
def test_routing_chunk_sizes_and_resized_reservation(
    method, output_device, batch_size, counts, runtime, gpu_routing
):
    image = torch.rand(131, 5, 7, 4, dtype=torch.float64)
    options = {} if output_device is None else {"output_device": output_device}
    if output_device != "gpu":
        runtime.available = 0  # The CPU destination is not reserved on the GPU.
    output = ImageScaleDown.execute(
        image, 3, 2, method, batch_size=batch_size, **options
    ).result[0]
    destination = runtime.device if output_device == "gpu" else torch.device("cpu")
    assert gpu_routing.allocations == [((131, 2, 3, 4), destination, torch.float32)]
    assert gpu_routing.transfers == [
        (count, runtime.device, torch.float32) for count in counts
    ]
    assert runtime.progress[0].updates == list(accumulate(counts))
    working = _resize.ImageResizer((5, 7), (2, 3), 4, method).working_memory(counts[0])
    reservation = output.numel() * 4 if output_device == "gpu" else 0
    assert runtime.free_requests == [(working + reservation, runtime.device)]


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_cpu_reservation_uses_smaller_output_and_memory_is_freed_first(
    method, runtime, monkeypatch
):
    image = torch.rand(3, 17, 29, 3)
    runtime.available = 3 * 5 * 7 * 3 * 4  # Enough for the output, not the source.
    allocations = []
    original = torch.empty

    def empty(shape, **kwargs):
        assert runtime.free_requests
        allocations.append(tuple(shape))
        return original(shape, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    output = ImageScaleDown.execute(image, 7, 5, method).result[0]
    assert allocations == [(3, 5, 7, 3)]
    assert output.numel() * 4 == runtime.available
    assert runtime.progress[0].updates == [3]


def test_output_allocation_failure_preserves_placement_and_reports_resized_shape(
    runtime, monkeypatch
):
    runtime.device = torch.device("cuda:2")
    allocations = []

    def empty(shape, **kwargs):
        allocations.append((tuple(shape), kwargs["device"]))
        raise torch.OutOfMemoryError("simulated destination allocation failure")

    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(RuntimeError, match="complete.*output on cuda:2.*batch_size"):
        ImageScaleDown.execute(torch.ones(5, 7, 11, 3), 3, 2, "lanczos", "gpu")
    assert allocations == [((5, 2, 3, 3), runtime.device)]
    assert runtime.cache_clears == 0


def test_lanczos_oom_retries_same_frames_reuses_coefficients_and_restarts_fresh(
    runtime, monkeypatch
):
    image = (
        (torch.arange(149, dtype=torch.float32) / 256)[:, None, None, None]
        .expand(149, 3, 7, 3)
        .clone()
    )
    before = image.clone()
    attempts, failed_tensors, coefficient_refs = [], [], []
    original_resize = _resize.ImageResizer.__call__
    original_axis = _resize._resample_axis
    original_coefficients = _resize._lanczos_coefficients
    fail = True

    def observe(self, chunk, check_interrupt):
        attempts.append((round(float(chunk[0, 0, 0, 0]) * 256), len(chunk)))
        return original_resize(self, chunk, check_interrupt)

    def coefficients(*args):
        assert runtime.free_requests
        result = original_coefficients(*args)
        coefficient_refs.extend(
            (weakref.ref(result.indices), weakref.ref(result.weights))
        )
        return result

    def axis(values, axis, coefficients, check_interrupt):
        if (
            fail
            and axis == 2
            and len(values) > 16
            and float(values[0, 0, 0, 0]) >= 0.249
        ):
            temporary = values.clone()
            failed_tensors.extend((weakref.ref(values), weakref.ref(temporary)))
            raise torch.OutOfMemoryError("simulated second-pass failure")
        return original_axis(values, axis, coefficients, check_interrupt)

    def empty_cache():
        assert all(ref() is None for ref in failed_tensors)
        assert all(ref() is not None for ref in coefficient_refs)
        runtime.cache_clears += 1

    monkeypatch.setattr(_resize.ImageResizer, "__call__", observe)
    monkeypatch.setattr(_resize, "_resample_axis", axis)
    monkeypatch.setattr(_resize, "_lanczos_coefficients", coefficients)
    monkeypatch.setattr(core.model_management, "soft_empty_cache", empty_cache)
    output = ImageScaleDown.execute(image, 3, 2, "lanczos").result[0]
    assert attempts == [
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
    assert runtime.progress[0].updates == [64, 80, 96, 112, 128, 144, 149]
    assert runtime.cache_clears == 2
    assert len(coefficient_refs) == 4  # Only one pair of axis tables was prepared.
    assert all(ref() is None for ref in coefficient_refs)
    torch.testing.assert_close(output, image[:, :2, :3])
    assert torch.equal(image, before)

    fail = False
    attempts.clear()
    again = ImageScaleDown.execute(image, 3, 2, "lanczos").result[0]
    assert attempts == [(0, 64), (64, 64), (128, 21)]
    assert len(coefficient_refs) == 8
    assert all(ref() is None for ref in coefficient_refs)
    assert torch.equal(output, again)


def test_oom_during_coefficient_preparation_releases_partial_tables(
    runtime, monkeypatch
):
    original = _resize._lanczos_coefficients
    calls = 0
    failed_tensors = []

    def coefficients(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            temporary = torch.ones(13)
            failed_tensors.append(weakref.ref(temporary))
            raise torch.OutOfMemoryError("simulated coefficient allocation failure")
        result = original(*args)
        if calls == 1:
            failed_tensors.extend(
                (weakref.ref(result.indices), weakref.ref(result.weights))
            )
        return result

    def empty_cache():
        assert all(ref() is None for ref in failed_tensors)
        runtime.cache_clears += 1

    monkeypatch.setattr(_resize, "_lanczos_coefficients", coefficients)
    monkeypatch.setattr(core.model_management, "soft_empty_cache", empty_cache)
    image = torch.full((7, 7, 11, 3), 0.5001)
    output = ImageScaleDown.execute(image, 3, 2, "lanczos").result[0]
    assert calls == 4
    assert runtime.cache_clears == 1
    assert runtime.progress[0].updates == [3, 6, 7]
    torch.testing.assert_close(output, torch.full_like(output, 0.5001))


@pytest.mark.parametrize(
    "error_type", [torch.OutOfMemoryError, RuntimeError, InterruptProcessingException]
)
def test_retained_terminal_exception_releases_prepared_and_temporary_tensors(
    error_type, runtime, monkeypatch
):
    references = []

    def fail(values, axis, coefficients, check_interrupt):
        temporary = values.clone()
        references.extend(
            (
                weakref.ref(temporary),
                weakref.ref(coefficients.weights),
                weakref.ref(coefficients.indices),
            )
        )
        raise error_type("simulated terminal failure")

    monkeypatch.setattr(_resize, "_resample_axis", fail)
    expected_error = (
        RuntimeError if error_type is torch.OutOfMemoryError else error_type
    )
    with pytest.raises(expected_error) as caught:
        ImageScaleDown.execute(torch.ones(1, 7, 11, 3), 3, 2, "lanczos")
    assert caught.value is not None  # Keep its traceback alive for the assertions.
    assert references and all(ref() is None for ref in references)
    assert runtime.cache_clears == 0
    assert runtime.progress[0].updates == []
    if error_type is torch.OutOfMemoryError:
        assert "processing one frame" in str(caught.value)


def test_cancellation_between_gather_tiles_releases_output_and_tables(
    runtime, monkeypatch
):
    runtime.interrupt_on = 5  # Allocation, chunk, two tiles, then cancellation.
    monkeypatch.setattr(_resize, "_GATHER_BYTES", 64)
    references = []
    original_empty = torch.empty
    original_coefficients = _resize._lanczos_coefficients

    def empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    def coefficients(*args):
        result = original_coefficients(*args)
        references.extend((weakref.ref(result.weights), weakref.ref(result.indices)))
        return result

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(_resize, "_lanczos_coefficients", coefficients)
    with pytest.raises(InterruptProcessingException) as caught:
        ImageScaleDown.execute(torch.ones(3, 7, 11, 3), 3, 2, "lanczos")
    assert caught.value is not None
    assert runtime.checks == 5
    assert runtime.progress[0].updates == []
    assert references and all(ref() is None for ref in references)
    assert runtime.cache_clears == 0


@pytest.mark.parametrize(
    "size,expected", [((2, 3), [3, 2]), ((1, 6), [2, 3]), ((3, 3), [3])]
)
def test_axis_order_minimizes_intermediate_size(size, expected, monkeypatch):
    axes = []
    original = _resize._resample_axis

    def axis(values, axis, coefficients, check_interrupt):
        axes.append(axis)
        return original(values, axis, coefficients, check_interrupt)

    monkeypatch.setattr(_resize, "_resample_axis", axis)
    height, width = size
    ImageScaleDown.execute(torch.ones(1, 3, 7, 3), width, height, "lanczos")
    assert axes == expected
