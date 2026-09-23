# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Linear full-range crossfades between image batches."""

import torch
from comfy_api.latest import io

from .._exec.crossfade import process_image_crossfade
from .._exec.inputs import execution_inputs


class CrossFade(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_CrossFade",
            display_name="GPU LayerStyles CrossFade",
            category="GPU LayerStyles/Batch",
            description=(
                "Keep images_1 before start_index, linearly crossfade into images_2 "
                "starting at its first frame, then keep the rest of images_2. Later "
                "frames of images_1 are discarded. The first transition frame is "
                "entirely images_1 and the last entirely images_2. Matching RGB or "
                "RGBA dimensions are required; every channel, including alpha, fades "
                "in float32 without clamping. Output has start_index + images_2's "
                "frame count. CPU output needs RAM for the complete result."
            ),
            inputs=[
                io.Image.Input("images_1"),
                io.Image.Input("images_2"),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                    tooltip=(
                        "Zero-based start of the transition in images_1. Negative "
                        "indices are rejected. images_2 always starts at index 0."
                    ),
                ),
                io.Int.Input(
                    "frames",
                    default=2,
                    min=2,
                    max=2**31 - 1,
                    step=1,
                    tooltip=(
                        "Exact number of frames from each batch used for the fade, "
                        "including both endpoints. At least 2; must fit after "
                        "start_index in images_1 and within images_2."
                    ),
                ),
                *execution_inputs(),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        images_1: torch.Tensor,
        images_2: torch.Tensor,
        start_index: int = 0,
        frames: int = 2,
        output_device: str = "cpu",
        batch_size: int = 0,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            process_image_crossfade(
                images_1, images_2, start_index, frames, output_device, batch_size
            )
        )
