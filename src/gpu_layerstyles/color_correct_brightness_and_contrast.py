"""Brightness Contrast V2 node."""

from functools import partial

import torch
from comfy_api.latest import io

from ._color import brightness_contrast
from ._execution import execution_inputs, process_image


class BrightnessContrastV2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_BrightnessContrastV2",
            display_name="GPU LayerStyles Brightness Contrast V2",
            category="GPU LayerStyles/Color",
            description="Adjust brightness, per-frame contrast, then saturation.",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("brightness", default=1, min=0.0, max=3, step=0.01),
                io.Float.Input("contrast", default=1, min=0.0, max=3, step=0.01),
                io.Float.Input("saturation", default=1, min=0.0, max=3, step=0.01),
                *execution_inputs(),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        brightness: float,
        contrast: float,
        saturation: float,
        output_device: str = "cpu",
        batch_size: int = 0,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            process_image(
                image,
                partial(
                    brightness_contrast,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                ),
                output_device,
                batch_size,
            )
        )
