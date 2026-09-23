# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Float32 ImageScaleDown node."""

import torch
from comfy_api.latest import io

from .._exec.inputs import execution_inputs
from .._exec.resize import process_image_resize


class ImageScaleDown(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_ImageScaleDown",
            display_name="GPU LayerStyles ImageScaleDown",
            category="GPU LayerStyles/Image",
            description=(
                "Downscale to exact dimensions with antialiased bicubic or Lanczos-3 "
                "in float32. Either dimension exceeding the input is rejected. "
                "RGBA uses premultiplied alpha while filtering. Resized values are "
                "clamped to [0, 1]; unchanged sizes are copied without filtering."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=512, min=1, max=16384, step=1),
                io.Int.Input("height", default=512, min=1, max=16384, step=1),
                io.Combo.Input(
                    "method", options=["bicubic", "lanczos"], default="bicubic"
                ),
                *execution_inputs(),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        width: int,
        height: int,
        method: str = "bicubic",
        output_device: str = "cpu",
        batch_size: int = 0,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            process_image_resize(
                image, width, height, method, output_device, batch_size
            )
        )
