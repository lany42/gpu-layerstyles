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
    return (rgb + sliders * maximum * rgb * (1 - rgb) / (center * (1 - center))).clamp_(
        0, 1
    )


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
    ratio = original_luminance / _luminance(rgb, weights).clamp_min_(1e-6)
    return rgb.mul_(ratio).clamp_(0, 1)


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
