from itertools import product

import numpy as np
import pytest
import torch
from PIL import Image, ImageEnhance
from scipy.interpolate import CubicSpline

from gpu_layerstyles._color import (
    _adjust_curve,
    brightness_contrast,
    color_balance,
    color_temperature,
)


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


@pytest.mark.parametrize("center,maximum", [(0.15, 0.1), (0.5, 1), (0.8, 0.2)])
@pytest.mark.parametrize("slider", [-1, -0.7, -0.001, 0, 0.5, 1])
def test_parabolic_curve_matches_scipy_on_dense_ramp(center, maximum, slider):
    ramp = torch.cat([torch.linspace(0, 1, 4097), torch.tensor([center])])
    rgb = ramp.reshape(1, 1, -1, 1).expand(-1, -1, -1, 3)
    actual = _adjust_curve(rgb, torch.tensor([slider] * 3), center, maximum)
    curve = CubicSpline([0, center, 1], [0, center + slider * maximum, 1])
    expected = np.clip(curve(rgb.numpy()), 0, 1)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=3e-7)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual[0, 0, 0], torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(actual[0, 0, -2], torch.ones(3), rtol=0, atol=0)
    torch.testing.assert_close(
        actual[0, 0, -1],
        torch.full((3,), np.clip(center + slider * maximum, 0, 1)),
    )


@pytest.mark.parametrize("center,maximum,slider", [(0.5, 1.0, -0.25), (0.8, 0.2, -0.8)])
def test_curve_preserves_small_values_when_control_points_define_x_squared(
    center, maximum, slider
):
    # These control points lie on y=x^2. An absolute tolerance would hide
    # cancellation of the small positive outputs that luminosity later restores.
    ramp = torch.tensor([0, 1e-9, 1e-7, 1e-5, 1e-3, 1])
    rgb = ramp.reshape(1, 1, -1, 1).expand(-1, -1, -1, 3)
    result = _adjust_curve(rgb, torch.tensor([slider] * 3), center, maximum)
    torch.testing.assert_close(
        result.double(), rgb.double().square(), rtol=2e-7, atol=0
    )


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
    result = color_balance(image, *sliders)
    torch.testing.assert_close(result, image, rtol=1e-6, atol=0)


@pytest.mark.parametrize("green", [1e-9, 1e-18])
def test_balance_restores_luminosity_when_only_a_tiny_channel_survives(green):
    image = torch.tensor([[[[0.3, green, 0.3]]]])
    # Red and blue are clipped to zero by their curves; green remains positive.
    # Preserving input luminance therefore determines the green output directly.
    original_luminance = 0.3 * 0.2126 + green * 0.7152 + 0.3 * 0.0722
    expected = torch.tensor([[[[0, original_luminance / 0.7152, 0]]]])
    result = color_balance(image, -1, -0.25, -1)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("sliders", list(product([-1, 0, 1], repeat=3)))
def test_color_balance_order_clipping_and_luminosity_match_spline(sliders):
    image = torch.rand((2, 7, 11, 3), generator=torch.Generator().manual_seed(32))
    expected = spline_balance(image.numpy(), sliders)
    actual = color_balance(image, *sliders)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=1e-6)
    assert actual.min() >= 0
    assert actual.max() <= 1


def test_balance_preserves_luminosity_without_clipping():
    image = torch.tensor([[[[0.2, 0.3, 0.4], [0.4, 0.2, 0.3]]]])
    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    result = color_balance(image, 0.1, -0.1, 0.15)
    torch.testing.assert_close((result * weights).sum(-1), (image * weights).sum(-1))


def test_black_adjusted_pixels_remain_black():
    image = torch.full((1, 2, 3, 3), 0.3)
    result = color_balance(image, -1, -1, -1)
    assert torch.equal(result, torch.zeros_like(image))


@pytest.mark.parametrize("sliders", [(-1, -1, -1), (1, 1, 1), (-0.5, 0.8, 0.2)])
def test_zero_and_near_zero_luminance(sliders):
    image = torch.tensor([0, 1e-9, 1e-7, 1e-6, 1e-5]).reshape(1, 1, 5, 1)
    image = image * torch.tensor([0.2, 0.5, 1])
    result = color_balance(image, *sliders)
    assert torch.isfinite(result).all()
    assert torch.equal(result[..., 0, :], torch.zeros_like(result[..., 0, :]))
    np.testing.assert_allclose(
        result.numpy(), spline_balance(image.numpy(), sliders), rtol=2e-5, atol=1e-12
    )


@pytest.mark.parametrize(
    "temperature,expected",
    [
        (-100, [1, 0.7, 0.5]),
        (-25, [0.625, 0.55, 0.5]),
        (0, [0.5, 0.5, 0.5]),
        (25, [0.475, 0.5, 0.625]),
        (100, [0.4, 0.5, 1]),
    ],
)
def test_temperature_direction_and_gains(temperature, expected):
    result = color_temperature(torch.full((1, 1, 1, 3), 0.5), temperature)
    torch.testing.assert_close(result.flatten(), torch.tensor(expected))


@pytest.mark.parametrize("temperature", [-100, 100])
def test_temperature_clips_rgb(temperature):
    image = torch.tensor([[[[-0.1, 0.9, 1.2]]]])
    result = color_temperature(image, temperature)
    assert result.min() >= 0
    assert result.max() <= 1


@pytest.mark.parametrize(
    "adjustments", [*product([0, 1, 3], repeat=3), (1.23, 0.76, 1.12)]
)
def test_enhancements_match_float_reference(adjustments):
    image = torch.rand((3, 5, 9, 3), generator=torch.Generator().manual_seed(17))
    image[0] *= 0.1
    image[2] = 0.8 + image[2] * 0.2
    result = brightness_contrast(image, *adjustments)
    np.testing.assert_allclose(
        result.numpy(),
        enhancement_reference(image.numpy(), *adjustments),
        rtol=2e-5,
        atol=1e-6,
    )


def test_contrast_uses_each_frames_mean_grayscale_after_brightness_clipping():
    image = torch.tensor([[[[1, 0, 0], [0, 0.1, 0]]], [[[0.1, 0.2, 0.3], [1, 1, 1]]]])
    result = brightness_contrast(image, 2, 0, 1)
    means = torch.tensor(
        [(0.299 + 0.2 * 0.587) / 2, (0.2 * 0.299 + 0.4 * 0.587 + 0.6 * 0.114 + 1) / 2]
    )
    torch.testing.assert_close(result, means.reshape(2, 1, 1, 1).expand_as(image))


def test_saturation_uses_post_contrast_pixel_luminance():
    image = torch.tensor([[[[1.0, 0.2, 0.0], [0.1, 0.7, 0.9]]]])
    contrasted = brightness_contrast(image, 1.3, 1.4, 1)
    expected = (contrasted * torch.tensor([0.299, 0.587, 0.114])).sum(-1, keepdim=True)
    result = brightness_contrast(image, 1.3, 1.4, 0)
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
    result = brightness_contrast(image, *adjustments)[0].numpy()
    np.testing.assert_allclose(
        result, np.asarray(reference) / 255, rtol=0, atol=4 / 255
    )


@pytest.mark.parametrize(
    "operation,args",
    [
        (color_balance, (0, 0, 0)),
        (brightness_contrast, (1, 1, 1)),
        (color_temperature, (0,)),
    ],
)
def test_neutral_math_preserves_tiny_values_and_out_of_range_values(operation, args):
    image = torch.tensor([[[[-0.2, 1e-9, 1.2]]]])
    assert torch.equal(operation(image, *args), image)
