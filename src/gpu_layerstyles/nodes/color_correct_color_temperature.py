# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""ColorTemperature node."""

from functools import partial

import torch
from comfy_api.latest import io

from .._color import color_temperature
from .._exec.image import process_image
from .._exec.inputs import execution_inputs


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
        output_device: str = "cpu",
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
