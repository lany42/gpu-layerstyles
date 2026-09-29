# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""ImageScaleDown validation, alpha handling, chunking, and memory behavior.

Lanczos parity with Pillow lives in test_resize_reference.py.
"""

import weakref

import pytest
import torch
import torch.nn.functional as F

from gpu_layerstyles import _resize
from gpu_layerstyles._exec import core
from gpu_layerstyles.nodes.image_scale_down import ImageScaleDown

from .conftest import InterruptProcessingException

METHODS = ["bicubic", "lanczos"]


def rand(*shape, seed=21):
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("name", ["width", "height"])
@pytest.mark.parametrize("value", [0, True, 1.5])
def test_invalid_dimensions_fail_before_allocation(name, value, runtime):
    arguments = {"width": 2, "height": 2, name: value}
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        ImageScaleDown.execute(torch.ones(1, 7, 9, 3), **arguments)
    assert not runtime.free_requests


@pytest.mark.parametrize("width,height", [(10, 7), (9, 8)])
def test_rejects_either_upscaled_dimension(width, height, runtime):
    with pytest.raises(ValueError, match=f"source is 9x7, requested {width}x{height}"):
        ImageScaleDown.execute(torch.ones(1, 7, 9, 3), width, height)
    assert not runtime.free_requests


def test_rejects_invalid_method_even_for_unchanged_size(runtime):
    with pytest.raises(ValueError, match="method"):
        ImageScaleDown.execute(torch.ones(1, 2, 3, 3), 3, 2, "nearest")
    assert not runtime.free_requests


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("channels", [3, 4])
def test_batches_match_single_frames(method, channels):
    image = rand(5, 9, 11, channels) * 0.8 + 0.1
    output = ImageScaleDown.execute(image, 6, 5, method, batch_size=2).result[0]
    singles = torch.cat(
        [ImageScaleDown.execute(frame[None], 6, 5, method).result[0] for frame in image]
    )
    assert output.shape == (5, 5, 6, channels)
    torch.testing.assert_close(output, singles, rtol=1e-6, atol=2e-7)


def test_default_bicubic_is_antialiased_and_clamped():
    image = rand(2, 9, 11, 3)
    expected = F.interpolate(
        image.movedim(-1, 1),
        size=(7, 5),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    output = ImageScaleDown.execute(image, 5, 7).result[0]
    torch.testing.assert_close(
        output, expected.clamp(0, 1).movedim(1, -1), rtol=0, atol=0
    )


@pytest.mark.parametrize("method", METHODS)
def test_unchanged_dimensions_copy_without_filter_or_clamp(method):
    image = rand(2, 3, 5, 4) * 2 - 0.5
    output = ImageScaleDown.execute(image, 5, 3, method, batch_size=1).result[0]
    assert torch.equal(output, image)


@pytest.mark.parametrize("method", METHODS)
def test_transparent_colors_do_not_bleed_into_visible_edges(method):
    image = torch.zeros(1, 7, 19, 4)
    image[..., :9, 0] = 1
    image[..., :9, 3] = 1
    image[..., 9:, 1] = 1  # Hidden green next to opaque red.
    output = ImageScaleDown.execute(image, 8, 3, method).result[0]
    visible = output[..., 3] > 0
    assert torch.any((output[..., 3] > 0) & (output[..., 3] < 1))
    torch.testing.assert_close(
        output[..., 0][visible], torch.ones_like(output[..., 0][visible])
    )
    assert torch.count_nonzero(output[..., 1:3]) == 0
    assert torch.count_nonzero(output[..., :3][~visible]) == 0
    assert torch.all((output >= 0) & (output <= 1))


@pytest.mark.parametrize("method", METHODS)
def test_ringing_that_cancels_alpha_leaves_no_color(method):
    # Opaque white beside opaque black rings to zero alpha with nonzero color.
    image = torch.zeros(1, 1, 24, 4)
    image[..., 11:13, 3] = 1
    image[..., 11, :3] = 1
    output = ImageScaleDown.execute(image, 9, 1, method).result[0]
    transparent = output[..., 3] == 0
    assert torch.any(transparent)
    assert torch.count_nonzero(output[..., :3][transparent]) == 0


@pytest.mark.parametrize("method", METHODS)
def test_constant_color_survives_partial_alpha_and_fully_transparent_is_zero(method):
    image = torch.tensor([0.125, 0.5, 0.875, 0.0]).expand(2, 11, 17, 4).clone()
    image[0, ..., 3] = torch.linspace(0.2, 0.8, 17)
    output = ImageScaleDown.execute(image, 7, 5, method).result[0]
    torch.testing.assert_close(output[0, ..., :3], image[0, :5, :7, :3])
    assert torch.all((output[0, ..., 3] >= 0.2) & (output[0, ..., 3] <= 0.8))
    assert torch.count_nonzero(output[1]) == 0


@pytest.mark.parametrize("method", METHODS)
def test_alpha_overshoot_is_clamped_after_unpremultiplication(method):
    color = torch.tensor([0.25, 0.5, 0.75])
    image = torch.cat((color, torch.zeros(1))).expand(1, 17, 29, 4).clone()
    image[..., 14:, 3] = 1
    output = ImageScaleDown.execute(image, 13, 7, method).result[0]
    visible = output[..., 3] > 0
    torch.testing.assert_close(
        output[..., :3][visible], color.expand_as(output[..., :3][visible])
    )
    assert output[..., 3].min() == 0
    assert output[..., 3].max() == 1


@pytest.mark.parametrize("capacity", [20, 1024])
def test_bounded_gathers_and_split_support_preserve_result(capacity, monkeypatch):
    image = rand(2, 7, 11, 3)
    expected = ImageScaleDown.execute(image, 4, 3, "lanczos").result[0]
    original = torch.index_select
    sizes = []

    def index_select(*args, **kwargs):
        result = original(*args, **kwargs)
        sizes.append(result.numel() * result.element_size())
        assert sizes[-1] <= capacity
        return result

    monkeypatch.setattr(_resize, "_GATHER_BYTES", capacity)
    monkeypatch.setattr(torch, "index_select", index_select)
    output = ImageScaleDown.execute(image, 4, 3, "lanczos").result[0]
    assert len(sizes) > 2
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=2e-7)


@pytest.mark.parametrize("method", METHODS)
def test_chunks_run_on_the_compute_device_with_reserved_working_memory(
    method, runtime, gpu_routing
):
    output = ImageScaleDown.execute(rand(131, 5, 7, 4), 3, 2, method, "gpu").result[0]
    assert gpu_routing.transfers == [
        (count, runtime.device, torch.float32) for count in (64, 64, 3)
    ]
    ((amount, device),) = runtime.free_requests
    working = amount - output.numel() * 4
    # Eight float32 copies of each source chunk; Lanczos adds tables and gathers.
    allowance = 8 * 64 * 5 * 7 * 4 * 4
    assert device == runtime.device
    assert working == allowance if method == "bicubic" else working > allowance


def test_reservation_is_for_the_resized_output(runtime):
    runtime.available = 3 * 5 * 7 * 3 * 4  # Enough for the output, not the source.
    output = ImageScaleDown.execute(torch.rand(3, 17, 29, 3), 7, 5).result[0]
    assert output.shape == (3, 5, 7, 3)
    assert runtime.progress[0].updates == [3]


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
    "error_type", [torch.OutOfMemoryError, InterruptProcessingException]
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
