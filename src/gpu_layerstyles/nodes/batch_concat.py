# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Concatenate images and batches in socket order without pixel conversion."""

import torch
from comfy_api.latest import io

from .._exec.core import _validate_image


class BatchConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_BatchConcat",
            display_name="GPU LayerStyles BatchConcat",
            category="GPU LayerStyles/Batch",
            description=(
                "Concatenate images and image batches in socket order into one "
                "independent, contiguous batch. Inputs must have matching height, "
                "width, channel count, dtype, and device. Pixel values and frame order "
                "are preserved. A single connected input is also copied. Copies need "
                "memory for both the inputs and output."
            ),
            inputs=[
                io.Autogrow.Input(
                    id="images",
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            id="image",
                            tooltip=(
                                "A nonempty floating-point RGB or RGBA image [H, W, C] "
                                "or batch [B, H, W, C]."
                            ),
                        ),
                        names=[f"image_{index}" for index in range(1, 101)],
                        min=1,
                    ),
                    tooltip=(
                        "Connect images or batches in concatenation order; a new "
                        "socket appears after each connection, up to 100 inputs."
                    ),
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, images: dict[str, torch.Tensor]) -> io.NodeOutput:
        if not images:
            raise ValueError("At least one image or image batch must be connected.")

        # Socket order must not depend on dictionary insertion order.
        names = sorted(images, key=lambda name: int(name.removeprefix("image_")))
        batches: list[torch.Tensor] = []
        for name in names:
            image = images[name]
            if isinstance(image, torch.Tensor) and image.ndim == 3:
                image = image.unsqueeze(0)
            _validate_image(image, name)
            if batches:
                first = batches[0]
                if image.shape[1:] != first.shape[1:]:
                    raise ValueError(
                        f"{name} must match {names[0]}'s height, width, and channel "
                        f"count; got {tuple(image.shape[1:])}, "
                        f"expected {tuple(first.shape[1:])}."
                    )
                if image.dtype != first.dtype:
                    raise ValueError(
                        f"{name} must match {names[0]}'s dtype; "
                        f"got {image.dtype}, expected {first.dtype}."
                    )
                if image.device != first.device:
                    raise ValueError(
                        f"{name} must match {names[0]}'s device; "
                        f"got {image.device}, expected {first.device}."
                    )
            batches.append(image)

        return io.NodeOutput(torch.cat(batches, dim=0).contiguous())
