# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Color node output against independent float64, SciPy, and Pillow references."""

import weakref

import numpy as np
import pytest
import torch
from PIL import Image, ImageEnhance
from scipy.interpolate import CubicSpline

from gpu_layerstyles.nodes.color_correct_brightness_and_contrast import (
    BrightnessContrastV2,
)
from gpu_layerstyles.nodes.color_correct_color_balance import ColorBalance
from gpu_layerstyles.nodes.color_correct_color_temperature import ColorTemperature

from .conftest import InterruptProcessingException

NODES = [
    pytest.param(ColorBalance, (0.3, -0.2, 0.4), id="ColorBalance"),
    pytest.param(BrightnessContrastV2, (1.2, 0.8, 1.3), id="BrightnessContrast"),
    pytest.param(ColorTemperature, (-37,), id="ColorTemperature"),
]


def balance(image, *sliders):
    return ColorBalance.execute(image, *sliders).result[0]


def enhance(image, *adjustments):
    return BrightnessContrastV2.execute(image, *adjustments).result[0]


def temperature(image, value):
    return ColorTemperature.execute(image, value).result[0]


def spline_balance(image, sliders):
    """Independent float64 reference using interpolated three-point curves."""
    source = np.asarray(image, dtype=np.float64)
    result = source.copy()
    for channel, slider in enumerate(sliders):
        for center, maximum in ((0.15, 0.1), (0.5, 1), (0.8, 0.2)):
            curve = CubicSpline([0, center, 1], [0, center + slider * maximum, 1])
            result[..., channel] = np.clip(curve(result[..., channel]), 0, 1)
    weights = np.array([0.2126, 0.7152, 0.0722])
    original = np.sum(source * weights, axis=-1, keepdims=True)
    adjusted = np.sum(result * weights, axis=-1, keepdims=True)
    # Upstream divides by the actual positive luminance. Pixels with zero adjusted
    # luminance become black at the final uint8 conversion; represent that explicitly.
    ratio = np.divide(
        original, adjusted, out=np.zeros_like(original), where=adjusted > 0
    )
    return np.clip(result * ratio, 0, 1)


def enhancement_reference(image, brightness, contrast, saturation):
    """Float64 reference, intentionally processing each frame separately."""
    frames = []
    weights = np.array([0.299, 0.587, 0.114])
    for frame in np.asarray(image, dtype=np.float64):
        result = frame.copy()
        if brightness != 1:
            result = np.clip(brightness * result, 0, 1)
        if contrast != 1:
            mean = np.sum(result * weights, axis=-1).mean()
            result = np.clip(contrast * result + (1 - contrast) * mean, 0, 1)
        if saturation != 1:
            gray = np.sum(result * weights, axis=-1, keepdims=True)
            result = np.clip(saturation * result + (1 - saturation) * gray, 0, 1)
        frames.append(result)
    return np.stack(frames)


@pytest.mark.parametrize("node,controls", NODES)
@pytest.mark.parametrize("batch_size", [0, 3])
def test_frames_are_processed_independently_and_alpha_is_copied(
    node, controls, batch_size
):
    generator = torch.Generator().manual_seed(22)
    image = torch.rand(7, 4, 9, 4, dtype=torch.float64, generator=generator)
    # Distinct frame brightness makes per-frame contrast means differ.
    image *= torch.linspace(0.05, 1.0, len(image)).reshape(-1, 1, 1, 1)
    image[..., 3] = torch.linspace(-0.5, 1.5, 7 * 4 * 9).reshape(7, 4, 9)
    output = node.execute(image, *controls, "cpu", batch_size).result[0]
    singles = torch.cat(
        [node.execute(frame[None], *controls).result[0] for frame in image]
    )
    torch.testing.assert_close(output, singles, rtol=1e-6, atol=2e-7)
    assert torch.equal(output[..., 3], image[..., 3].float())


@pytest.mark.parametrize("node,controls", NODES)
@pytest.mark.parametrize(
    "error_type", [torch.OutOfMemoryError, InterruptProcessingException]
)
def test_retained_terminal_exception_releases_output_and_temporaries(
    node, controls, error_type, runtime, monkeypatch
):
    original_empty, original_to = torch.empty, torch.Tensor.to
    references = []

    def empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    def to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        references.append(weakref.ref(result))
        if len(references) == 3:  # Output, first frame, then the second frame.
            raise error_type("simulated terminal failure after one frame")
        return result

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    expected_error = (
        RuntimeError if error_type is torch.OutOfMemoryError else error_type
    )
    image = torch.rand(3, 2, 3, 4, dtype=torch.float64)
    with pytest.raises(expected_error) as caught:
        node.execute(image, *controls, batch_size=1)
    assert caught.value is not None  # Keep the exception and its tracebacks alive.
    assert len(references) == 3
    assert all(reference() is None for reference in references)
    assert runtime.progress[0].updates == [1]


@pytest.mark.parametrize(
    "pixel,sliders",
    [
        ((202, 0, 0), (-0.801, 0, 0)),
        ((0, 202, 0), (0, -0.801, 0)),
        ((0, 0, 202), (0, 0, -0.801)),
        ((11, 11, 11), (-0.259, -0.259, -0.259)),
    ],
)
def test_balance_restores_small_positive_adjusted_luminance(pixel, sliders):
    # Upstream returns these same byte values. The curves leave positive RGB,
    # so luminosity restoration must recover a primary color or equal-channel gray.
    image = torch.tensor(pixel, dtype=torch.float32).reshape(1, 1, 1, 3) / 255
    torch.testing.assert_close(balance(image, *sliders), image, rtol=1e-6, atol=0)


@pytest.mark.parametrize("green", [1e-9, 1e-18])
def test_balance_restores_luminosity_when_only_a_tiny_channel_survives(green):
    image = torch.tensor([[[[0.3, green, 0.3]]]])
    # Red and blue are clipped to zero by their curves; green remains positive.
    # Preserving input luminance therefore determines the green output directly.
    original_luminance = 0.3 * 0.2126 + green * 0.7152 + 0.3 * 0.0722
    expected = torch.tensor([[[[0, original_luminance / 0.7152, 0]]]])
    result = balance(image, -1, -0.25, -1)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "sliders",
    [(-1, -1, -1), (1, 1, 1), (-1, 0, 1), (1, -1, 0), (0, 1, -1), (-0.7, 0.5, -0.001)],
)
def test_color_balance_order_clipping_and_luminosity_match_spline(sliders):
    image = torch.rand((2, 7, 11, 3), generator=torch.Generator().manual_seed(32))
    expected = spline_balance(image.numpy(), sliders)
    actual = balance(image, *sliders)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=1e-6)
    assert actual.min() >= 0
    assert actual.max() <= 1


def test_balance_preserves_luminosity_without_clipping():
    image = torch.tensor([[[[0.2, 0.3, 0.4], [0.4, 0.2, 0.3]]]])
    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    result = balance(image, 0.1, -0.1, 0.15)
    torch.testing.assert_close((result * weights).sum(-1), (image * weights).sum(-1))


def test_black_adjusted_pixels_remain_black():
    image = torch.full((1, 2, 3, 3), 0.3)
    assert torch.equal(balance(image, -1, -1, -1), torch.zeros_like(image))


@pytest.mark.parametrize("sliders", [(-1, -1, -1), (1, 1, 1), (-0.5, 0.8, 0.2)])
def test_zero_and_near_zero_luminance(sliders):
    image = torch.tensor([0, 1e-9, 1e-7, 1e-6, 1e-5]).reshape(1, 1, 5, 1)
    image = image * torch.tensor([0.2, 0.5, 1])
    result = balance(image, *sliders)
    assert torch.isfinite(result).all()
    assert torch.equal(result[..., 0, :], torch.zeros_like(result[..., 0, :]))
    np.testing.assert_allclose(
        result.numpy(), spline_balance(image.numpy(), sliders), rtol=2e-5, atol=1e-12
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (-100, [1, 0.7, 0.5]),
        (-25, [0.625, 0.55, 0.5]),
        (0, [0.5, 0.5, 0.5]),
        (25, [0.475, 0.5, 0.625]),
        (100, [0.4, 0.5, 1]),
    ],
)
def test_temperature_direction_and_gains(value, expected):
    result = temperature(torch.full((1, 1, 1, 3), 0.5), value)
    torch.testing.assert_close(result.flatten(), torch.tensor(expected))


@pytest.mark.parametrize("value", [-100, 100])
def test_temperature_clips_rgb(value):
    result = temperature(torch.tensor([[[[-0.1, 0.9, 1.2]]]]), value)
    assert result.min() >= 0
    assert result.max() <= 1


@pytest.mark.parametrize(
    "adjustments",
    [
        (0, 1, 1),
        (3, 1, 1),
        (1, 0, 1),
        (1, 3, 1),
        (1, 1, 0),
        (1, 1, 3),
        (3, 3, 3),
        (3, 0, 3),
        (1.23, 0.76, 1.12),
    ],
)
def test_enhancements_match_float_reference(adjustments):
    image = torch.rand((3, 5, 9, 3), generator=torch.Generator().manual_seed(17))
    image[0] *= 0.1
    image[2] = 0.8 + image[2] * 0.2
    np.testing.assert_allclose(
        enhance(image, *adjustments).numpy(),
        enhancement_reference(image.numpy(), *adjustments),
        rtol=2e-5,
        atol=1e-6,
    )


def test_contrast_uses_each_frames_mean_grayscale_after_brightness_clipping():
    image = torch.tensor([[[[1, 0, 0], [0, 0.1, 0]]], [[[0.1, 0.2, 0.3], [1, 1, 1]]]])
    means = torch.tensor(
        [(0.299 + 0.2 * 0.587) / 2, (0.2 * 0.299 + 0.4 * 0.587 + 0.6 * 0.114 + 1) / 2]
    )
    torch.testing.assert_close(
        enhance(image, 2, 0, 1), means.reshape(2, 1, 1, 1).expand_as(image)
    )


def test_saturation_uses_post_contrast_pixel_luminance():
    image = torch.tensor([[[[1.0, 0.2, 0.0], [0.1, 0.7, 0.9]]]])
    contrasted = enhance(image, 1.3, 1.4, 1)
    expected = (contrasted * torch.tensor([0.299, 0.587, 0.114])).sum(-1, keepdim=True)
    result = enhance(image, 1.3, 1.4, 0)
    torch.testing.assert_close(result, expected.expand_as(result))


@pytest.mark.parametrize(
    "adjustments", [(1, 1, 1), (1.23, 1, 1), (1, 1.4, 1), (1, 1, 0.6), (1.2, 0.8, 1.1)]
)
def test_pillow_reference_allows_removed_integer_rounding(adjustments):
    pixels = np.random.default_rng(31).integers(0, 256, (17, 19, 3), dtype=np.uint8)
    reference = Image.fromarray(pixels)
    for enhancer, factor in zip(
        (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color),
        adjustments,
    ):
        reference = enhancer(reference).enhance(factor)
    image = torch.from_numpy(pixels.astype(np.float32) / 255).unsqueeze(0)
    np.testing.assert_allclose(
        enhance(image, *adjustments)[0].numpy(),
        np.asarray(reference) / 255,
        rtol=0,
        atol=4 / 255,
    )


@pytest.mark.parametrize(
    "node,controls",
    [
        (ColorBalance, (0, 0, 0)),
        (BrightnessContrastV2, (1, 1, 1)),
        (ColorTemperature, (0,)),
    ],
)
def test_neutral_controls_preserve_tiny_and_out_of_range_values(node, controls):
    image = torch.tensor([[[[-0.2, 1e-9, 1.2]]]])
    assert torch.equal(node.execute(image, *controls).result[0], image)
