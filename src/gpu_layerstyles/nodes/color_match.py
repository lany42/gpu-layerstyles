# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native PyTorch ColorMatch node using MKL or analytical MVGD."""

import math
from functools import partial
from numbers import Real

import torch
from comfy_api.latest import io

from .._color_match import color_match, prepare_reference
from .._exec.inputs import execution_inputs
from .._exec.pair import process_image_pair


class ColorMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_ColorMatch",
            display_name="GPU LayerStyles ColorMatch",
            category="GPU LayerStyles/Color",
            description=(
                "Match RGB using color-matcher's MKL or analytical MVGD algorithms. "
                "Use one reference or one per target frame. MKL accepts any reference "
                "resolution; MVGD requires equal height and width and uses corresponding "
                "pixel positions. Target alpha is preserved."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Image.Input("image_ref"),
                io.Combo.Input("method", options=["mkl", "mvgd"], default="mkl"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                *execution_inputs(),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        image_ref: torch.Tensor,
        method: str = "mkl",
        strength: float = 1.0,
        output_device: str = "cpu",
        batch_size: int = 0,
    ) -> io.NodeOutput:
        if method not in ("mkl", "mvgd"):
            raise ValueError("method must be 'mkl' or 'mvgd'.")
        if (
            isinstance(strength, bool)
            or not isinstance(strength, Real)
            or not math.isfinite(strength)
            or not 0 <= strength <= 1
        ):
            raise ValueError("strength must be a finite number from 0 to 1.")
        return io.NodeOutput(
            process_image_pair(
                image,
                image_ref,
                partial(prepare_reference, method=method),
                partial(color_match, method=method, strength=strength),
                output_device,
                batch_size,
                matching_dimensions=method == "mvgd",
                active=strength != 0,
            )
        )
