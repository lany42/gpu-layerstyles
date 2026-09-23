# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""ColorBalance node."""

from functools import partial

import torch
from comfy_api.latest import io

from .._color import color_balance
from .._exec.image import process_image
from .._exec.inputs import execution_inputs


class ColorBalance(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_ColorBalance",
            display_name="GPU LayerStyles ColorBalance",
            category="GPU LayerStyles/Color",
            description="Balance RGB in shadows, midtones, and highlights while preserving luminosity.",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("cyan_red", default=0, min=-1.0, max=1.0, step=0.001),
                io.Float.Input(
                    "magenta_green", default=0, min=-1.0, max=1.0, step=0.001
                ),
                io.Float.Input("yellow_blue", default=0, min=-1.0, max=1.0, step=0.001),
                *execution_inputs(),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        cyan_red: float,
        magenta_green: float,
        yellow_blue: float,
        output_device: str = "cpu",
        batch_size: int = 0,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            process_image(
                image,
                partial(
                    color_balance,
                    cyan_red=cyan_red,
                    magenta_green=magenta_green,
                    yellow_blue=yellow_blue,
                ),
                output_device,
                batch_size,
            )
        )
