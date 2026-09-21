# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Float32 downsampling, with Lanczos-3 adapted from Pillow's Resample.c.

The filter, sample bounds, and normalization follow Pillow commit
7d67d3764871ebfb4cbbd04459eede0bbc37966e. Tensor execution and bounded
gathers are implemented here for CPU and GPU. See COPYRIGHT and the retained
upstream notices in LICENSES/Pillow-LICENSE.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_GATHER_BYTES = 64 * 1024 * 1024


def _tap_count(source: int, target: int) -> int:
    # ceil(3 * scale) * 2 + 1, capped at the actual source length.
    return min(source, 2 * ((3 * source + target - 1) // target) + 1)


@dataclass
class AxisCoefficients:
    indices: torch.Tensor
    weights: torch.Tensor


def _lanczos_coefficients(
    source: int, target: int, device: torch.device
) -> AxisCoefficients:
    output = torch.arange(target, dtype=torch.int64, device=device)
    centers = (2 * output + 1) * source
    # Pillow rounds center +/- support + 0.5, then clips to the source.
    # Flooring negative lower bounds is equivalent after clipping at zero.
    first = torch.div(
        centers - 6 * source + target, 2 * target, rounding_mode="floor"
    ).clamp_(min=0)
    end = torch.div(
        centers + 6 * source + target, 2 * target, rounding_mode="floor"
    ).clamp_(max=source)
    indices = first[:, None] + torch.arange(
        _tap_count(source, target), dtype=torch.int64, device=device
    )
    # Form the relative distance exactly before converting to float32. This
    # avoids subtracting large rounded pixel centers at near-identity scales.
    distance = ((2 * indices + 1) * target - centers[:, None]).float() / (2 * source)
    weights = torch.sinc(distance) * torch.sinc(distance / 3)
    valid = (indices < end[:, None]) & (distance >= -3) & (distance < 3)
    weights.masked_fill_(~valid, 0)
    weights /= weights.sum(dim=1, keepdim=True)
    # Invalid gathers use a safe index but carry zero weight; boundary pixels
    # are not replicated into the filter's normalization.
    return AxisCoefficients(indices.clamp_(max=source - 1), weights)


def _resample_axis(
    image: torch.Tensor,
    axis: int,
    coefficients: AxisCoefficients,
    check_interrupt: Callable[[], None],
) -> torch.Tensor:
    moved = image.movedim(axis, -1)
    lines = moved.reshape(-1, moved.shape[-1])
    target, taps = coefficients.indices.shape
    result = lines.new_empty((len(lines), target))
    capacity = max(1, _GATHER_BYTES // image.element_size())
    tap_step = min(taps, capacity)
    column_step = min(target, max(1, capacity // tap_step))
    line_step = min(len(lines), max(1, capacity // (column_step * tap_step)))

    for row in range(0, len(lines), line_step):
        source = lines[row : row + line_step]
        for column in range(0, target, column_step):
            columns = slice(column, column + column_step)
            destination = result[row : row + line_step, columns]
            for tap in range(0, taps, tap_step):
                check_interrupt()
                indices = coefficients.indices[columns, tap : tap + tap_step]
                weights = coefficients.weights[columns, tap : tap + tap_step]
                samples = torch.index_select(source, 1, indices.reshape(-1)).reshape(
                    len(source), *indices.shape
                )
                samples.mul_(weights)
                if tap == 0:
                    torch.sum(samples, dim=-1, dtype=torch.float32, out=destination)
                else:
                    destination.add_(samples.sum(dim=-1, dtype=torch.float32))
                # Do not retain the last gather while allocating its successor.
                del samples

    return result.reshape(*moved.shape[:-1], target).movedim(-1, axis)


class ImageResizer:
    """Geometry and invocation-local coefficients for complete float32 chunks."""

    def __init__(
        self,
        source_size: tuple[int, int],
        size: tuple[int, int],
        channels: int,
        method: str,
    ) -> None:
        self.source_size = source_size
        self.size = size
        self.channels = channels
        self.method = method
        height, width = source_size
        target_height, target_width = size
        axes = [(3, width, target_width), (2, height, target_height)]
        if target_height * width < height * target_width:
            axes.reverse()
        self.axes = [(axis, n, m) for axis, n, m in axes if n != m]
        self.coefficients: tuple[AxisCoefficients, ...] | None = None

    def working_memory(self, count: int) -> int:
        height, width = self.source_size
        # Conservative allowance for converted input, premultiplication, layout
        # copies, pass outputs, and unpremultiplication, all bounded by input size.
        working = 8 * count * height * width * self.channels * 4
        if self.method != "lanczos" or not self.axes:
            return working
        # Include preparation-only index, distance, sinc, and masking tensors,
        # not just the cached int64 indices and float32 weights (12 bytes/tap).
        working += 64 * sum(m * _tap_count(n, m) for _, n, m in self.axes)
        gather = 0
        dimensions = [height, width]
        for axis, n, m in self.axes:
            other = dimensions[1 if axis == 2 else 0]
            gather = max(
                gather, count * self.channels * other * m * _tap_count(n, m) * 4
            )
            dimensions[axis - 2] = m
        return working + min(_GATHER_BYTES, gather)

    def clear(self) -> None:
        self.coefficients = None

    def _prepare(self, device: torch.device) -> tuple[AxisCoefficients, ...]:
        # Publish only a complete set. An OOM during preparation releases all
        # partial tables when the failed attempt's traceback goes out of scope.
        return tuple(_lanczos_coefficients(n, m, device) for _, n, m in self.axes)

    def __call__(
        self, image: torch.Tensor, check_interrupt: Callable[[], None]
    ) -> torch.Tensor:
        if not self.axes:
            return image

        with torch.autocast(device_type=image.device.type, enabled=False):
            if self.method == "lanczos" and self.coefficients is None:
                self.coefficients = self._prepare(image.device)

            values = image.movedim(-1, 1)
            if self.channels == 4:
                values = torch.cat(
                    (values[:, :3] * values[:, 3:], values[:, 3:]), dim=1
                )
            if self.method == "bicubic":
                values = F.interpolate(
                    values,
                    size=self.size,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            else:
                for (axis, _, _), coefficients in zip(self.axes, self.coefficients):
                    values = _resample_axis(values, axis, coefficients, check_interrupt)

            if self.channels == 4:
                alpha = values[:, 3:]
                positive = alpha > 0
                divisor = torch.where(positive, alpha, 1.0)
                rgb = torch.where(positive, values[:, :3] / divisor, 0.0)
                values = torch.cat((rgb, alpha), dim=1)
            return values.clamp_(0, 1).movedim(1, -1)
