# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Color operations on float32 RGB chunks shaped [B, H, W, 3].

All operations leave their input untouched. Neutral operations return the input;
the executor copies it into a separately allocated output.
"""

import torch


def _luminance(rgb: torch.Tensor, weights: tuple[float, float, float]) -> torch.Tensor:
    # Elementwise operations avoid reduced-precision matrix multiplication.
    return (rgb * rgb.new_tensor(weights)).sum(dim=-1, keepdim=True)


def color_temperature(rgb: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return rgb
    t = -temperature / 100.0
    gains = (1 + t, 1 + 0.4 * t, 1) if t > 0 else (1 + 0.2 * t, 1, 1 - t)
    return (rgb * rgb.new_tensor(gains)).clamp_(0, 1)


def _adjust_curve(
    rgb: torch.Tensor, sliders: torch.Tensor, center: float, maximum: float
) -> torch.Tensor:
    # The three-point spline is x * (x + (1 + a) * (1 - x)), where
    # a = slider * maximum / (center * (1 - center)). This form retains
    # small x^2 terms when a is -1 instead of subtracting nearly equal values.
    linear = 1 + sliders * (maximum / (center * (1 - center)))
    return (rgb * (rgb + linear * (1 - rgb))).clamp_(0, 1)


def color_balance(
    rgb: torch.Tensor, cyan_red: float, magenta_green: float, yellow_blue: float
) -> torch.Tensor:
    if cyan_red == magenta_green == yellow_blue == 0:
        return rgb
    weights = (0.2126, 0.7152, 0.0722)
    original_luminance = _luminance(rgb, weights)
    sliders = rgb.new_tensor((cyan_red, magenta_green, yellow_blue))
    for center, maximum in ((0.15, 0.1), (0.5, 1.0), (0.8, 0.2)):
        rgb = _adjust_curve(rgb, sliders, center, maximum)
    adjusted_luminance = _luminance(rgb, weights)
    # Protect only zero luminance; a floor would darken positive adjusted RGB.
    # Normalize first to avoid an overflowing source/adjusted luminance ratio.
    adjusted_luminance.masked_fill_(adjusted_luminance == 0, 1)
    return rgb.div_(adjusted_luminance).mul_(original_luminance).clamp_(0, 1)


def brightness_contrast(
    rgb: torch.Tensor, brightness: float, contrast: float, saturation: float
) -> torch.Tensor:
    weights = (0.299, 0.587, 0.114)
    if brightness != 1:
        rgb = (rgb * brightness).clamp_(0, 1)
    if contrast != 1:
        mean = _luminance(rgb, weights).mean(dim=(1, 2), keepdim=True)
        rgb = (rgb - mean).mul_(contrast).add_(mean).clamp_(0, 1)
    if saturation != 1:
        gray = _luminance(rgb, weights)
        rgb = (rgb - gray).mul_(saturation).add_(gray).clamp_(0, 1)
    return rgb
