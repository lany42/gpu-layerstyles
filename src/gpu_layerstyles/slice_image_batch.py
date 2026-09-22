# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Strict Python-style image batch selection without pixel conversion."""

import re

import torch
from comfy_api.latest import io

from ._execution import _validate_image


def _parse_selection(expression: str, length: int) -> range:
    if not isinstance(expression, str):
        raise TypeError("slice must be a string containing an index or slice.")

    parts = expression.split(":")
    if not expression.strip() or len(parts) > 3:
        raise ValueError(
            f"Invalid slice {expression!r}; expected an integer or START:END[:STRIDE]."
        )

    names = ("index",) if len(parts) == 1 else ("START", "END", "STRIDE")
    values = []
    for name, part in zip(names, parts):
        token = part.strip()
        if not token and len(parts) > 1:
            values.append(None)
            continue
        if re.fullmatch(r"[+-]?[0-9]+", token) is None:
            raise ValueError(
                f"Invalid {name} {token!r} in slice {expression!r}; "
                "expected a signed decimal integer."
            )
        value = int(token)
        if name != "STRIDE":
            upper = length if name == "END" else length - 1
            if not -length <= value <= upper:
                raise IndexError(
                    f"{name} {value} in slice {expression!r} is out of range for "
                    f"a batch of {length} images; expected {-length} <= {name} <= {upper}."
                )
        elif value == 0:
            raise ValueError(f"STRIDE must be nonzero in slice {expression!r}.")
        values.append(value)

    if len(values) == 1:
        index = range(length)[values[0]]
        return range(index, index + 1)

    # Keep omitted bounds as None: reverse slices treat them differently from -1.
    selection = range(length)[slice(*values)]
    if not selection:
        raise ValueError(
            f"Slice {expression!r} selects no images from a batch of {length} images; "
            "the selection must be nonempty."
        )
    return selection


class SliceImageBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GPULayerStyles_SliceImageBatch",
            display_name="GPU LayerStyles SliceImageBatch",
            category="GPU LayerStyles/Batch",
            description=(
                "Select images using a Python index or START:END[:STRIDE]. "
                "END is exclusive; negative indices count from the end and negative "
                "strides reverse selection order. Out-of-range bounds and empty "
                "selections are rejected. : (the default) and :: return the input tensor "
                "unchanged. Other selections return an independent contiguous batch "
                "with the same device, dtype, and pixel values, including for a "
                "single image. Copies need memory for both the input and output."
            ),
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "slice",
                    default=":",
                    multiline=False,
                    dynamic_prompts=False,
                    tooltip=(
                        "One index or START:END[:STRIDE]. : and :: pass the input "
                        "through unchanged; :15 selects the first 15; -15: selects "
                        "the last 15 in forward order; ::-1 reverses the batch. "
                        "For N images, index/START "
                        "must be between -N and N-1, and END between -N and N. "
                        "Omit END to reverse through index 0. STRIDE cannot be zero."
                    ),
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, image: torch.Tensor, slice: str = ":") -> io.NodeOutput:
        if image is None:
            raise TypeError("image is None; connect a nonempty IMAGE batch.")
        _validate_image(image, "image")
        if isinstance(slice, str) and slice.strip() in (":", "::"):
            return io.NodeOutput(image)
        selection = _parse_selection(slice, len(image))
        indices = torch.tensor(list(selection), dtype=torch.long, device=image.device)
        return io.NodeOutput(torch.index_select(image, 0, indices).contiguous())
