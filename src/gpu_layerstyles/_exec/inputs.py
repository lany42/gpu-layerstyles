# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Shared execution controls for ComfyUI node schemas."""

from comfy_api.latest import io


def execution_inputs() -> list:
    return [
        io.Combo.Input(
            "output_device",
            options=["gpu", "cpu"],
            default="cpu",
            tooltip=(
                "Return output on CPU (default) after float32 processing on ComfyUI's "
                "selected compute device. GPU keeps output on that device to avoid "
                "transfers between compatible nodes. ComfyUI CPU mode returns CPU output."
            ),
        ),
        io.Int.Input(
            "batch_size",
            default=0,
            min=0,
            max=2**31 - 1,
            step=1,
            tooltip=(
                "Frames per processing chunk. 0 starts at up to 64 frames. Positive "
                "values can exceed 64, capped by the output length. Allocation failures "
                "halve the chunk size for this run; each run starts fresh. Complete "
                "inputs and outputs still need memory. Pass smaller batches through "
                "the entire workflow to reduce complete-batch memory."
            ),
        ),
    ]
