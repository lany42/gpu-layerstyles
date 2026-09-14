"""ColorTemperature node."""

from functools import partial

import torch
from comfy_api.latest import io

from ._color import color_temperature
from ._execution import execution_inputs, process_image


class ColorTemperature(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_ColorTemperature",
            display_name="GPU LayerStyles ColorTemperature",
            category="GPU LayerStyles/Color",
            description="Adjust color temperature. Negative values warm; positive values cool.",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("temperature", default=0, min=-100, max=100, step=1),
                *execution_inputs(),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        temperature: float,
        output_device: str = "gpu",
        batch_size: int = 0,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            process_image(
                image,
                partial(color_temperature, temperature=temperature),
                output_device,
                batch_size,
            )
        )
