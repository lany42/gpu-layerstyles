# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Color adjustment, downscaling, and image batch nodes for ComfyUI."""

from comfy_api.latest import ComfyExtension, io

from .color_correct_brightness_and_contrast import BrightnessContrastV2
from .color_correct_color_balance import ColorBalance
from .color_correct_color_temperature import ColorTemperature
from .color_match import ColorMatch
from .image_scale_down import ImageScaleDown
from .slice_image_batch import SliceImageBatch


class GPULayerStylesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ColorBalance,
            BrightnessContrastV2,
            ColorTemperature,
            ColorMatch,
            ImageScaleDown,
            SliceImageBatch,
        ]


async def comfy_entrypoint() -> GPULayerStylesExtension:
    return GPULayerStylesExtension()


__all__ = ["GPULayerStylesExtension", "comfy_entrypoint"]
