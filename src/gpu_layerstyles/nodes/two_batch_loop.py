# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Loop two image batches with exact crossfades at both boundaries."""

import torch
from comfy_api.latest import io

from .._exec.inputs import execution_inputs
from .._exec.two_batch_loop import process_two_batch_loop


class TwoBatchLoop(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_TwoBatchLoop",
            display_name="GPU LayerStyles TwoBatchLoop",
            category="GPU LayerStyles/Batch",
            description=(
                "Loop two batches in this order: images_2's tail fades into "
                "images_1's head, images_1's middle, images_1's tail fades into "
                "images_2's head, images_2's middle. Each batch needs at least "
                "2 * blend_target + 1 frames. Unequal lengths are allowed; RGB or "
                "RGBA dimensions must match. All channels, including alpha, fade "
                "in float32 without clamping. Output has the combined input length "
                "minus 2 * blend_target, plus one if append first frame is enabled. "
                "CPU output needs RAM for the complete result."
            ),
            inputs=[
                io.Image.Input("images_1"),
                io.Image.Input("images_2"),
                io.Int.Input(
                    "blend_target",
                    default=15,
                    min=2,
                    max=2**31 - 1,
                    step=1,
                    tooltip=(
                        "Exact number of frames in each transition, including both "
                        "endpoints. At least 2; both batches must retain a nonempty "
                        "middle after removing this many frames from each end."
                    ),
                ),
                io.Boolean.Input(
                    "append_first_frame",
                    display_name="append first frame",
                    default=False,
                    tooltip=(
                        "Append an exact copy of the completed loop's first frame "
                        "to enable interpolation from the last frame to the first. "
                        "After interpolation, remove the redundant endpoint with "
                        "an external SliceImageBatch node using 0:-1. TwoBatchLoop "
                        "does not interpolate or trim frames."
                    ),
                ),
                *execution_inputs(),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(
        cls,
        images_1: torch.Tensor,
        images_2: torch.Tensor,
        blend_target: int = 15,
        append_first_frame: bool = False,
        output_device: str = "cpu",
        batch_size: int = 0,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            process_two_batch_loop(
                images_1,
                images_2,
                blend_target,
                append_first_frame,
                output_device,
                batch_size,
            )
        )
