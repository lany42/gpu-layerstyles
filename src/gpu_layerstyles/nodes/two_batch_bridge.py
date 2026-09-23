# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Split the boundary batches and outer endpoints for bridge generation."""

import torch
from comfy_api.latest import io

from .._exec.core import _validate_image


class TwoBatchBridge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_TwoBatchBridge",
            display_name="GPU LayerStyles TwoBatchBridge",
            category="GPU LayerStyles/Batch",
            description=(
                "Prepare bridge generation from a batch's ends. bridge_first is "
                "the last blend_target frames; bridge_last is the first "
                "blend_target frames, both in original order. first/last contains "
                "bridge_first's first frame followed by bridge_last's last frame. "
                "The input must have more than blend_target frames; the slices "
                "may overlap. Outputs are independent contiguous copies with "
                "the input's dtype, device, and pixel values."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Int.Input(
                    "blend_target",
                    default=15,
                    min=1,
                    max=2**31 - 1,
                    step=1,
                    tooltip=(
                        "Number of frames to take from each end. At least 1 and "
                        "less than the input batch length. Overlap is allowed."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="bridge_first"),
                io.Image.Output(display_name="bridge_last"),
                io.Image.Output(display_name="first/last"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, images: torch.Tensor, blend_target: int = 15) -> io.NodeOutput:
        _validate_image(images, "images")
        if (
            isinstance(blend_target, bool)
            or not isinstance(blend_target, int)
            or not 1 <= blend_target <= 2**31 - 1
        ):
            raise ValueError(
                "blend_target must be an integer between 1 and 2147483647."
            )
        if len(images) <= blend_target:
            raise ValueError(
                f"TwoBatchBridge requires more than {blend_target} frames in images "
                f"for blend_target {blend_target}; images has {len(images)} frames. "
                "Reduce blend_target."
            )

        bridge_first = images[-blend_target:].clone(
            memory_format=torch.contiguous_format
        )
        bridge_last = images[:blend_target].clone(memory_format=torch.contiguous_format)
        first_last = torch.stack((bridge_first[0], bridge_last[-1]))
        return io.NodeOutput(bridge_first, bridge_last, first_last)
