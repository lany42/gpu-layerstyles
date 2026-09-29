# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Lanczos checked against Pillow F-mode filtering without uint8 conversion."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gpu_layerstyles.nodes.image_scale_down import ImageScaleDown


def pillow_resize(image, width, height):
    # Each float32 plane becomes mode F; Pillow RGB/RGBA would quantize to 8 bits.
    return np.stack(
        [
            np.stack(
                [
                    np.asarray(
                        Image.fromarray(frame[..., channel]).resize(
                            (width, height), Image.Resampling.LANCZOS
                        )
                    )
                    for channel in range(frame.shape[-1])
                ],
                axis=-1,
            )
            for frame in image
        ]
    )


@pytest.mark.parametrize(
    "source,target",
    [
        ((13, 17), (7, 11)),
        ((91, 101), (1, 2)),
        ((1, 37), (1, 13)),
        ((37, 1), (13, 1)),
        ((71, 11), (3, 10)),
        ((11, 71), (10, 3)),
        ((9, 13), (9, 5)),
        ((9, 13), (4, 13)),
        ((3, 4096), (2, 3999)),
        ((2, 8192), (1, 1023)),
        ((2, 4096), (1, 1)),
    ],
)
def test_lanczos_matches_pillow_float_reference(source, target):
    rng = np.random.default_rng(17)
    image = rng.random((2, *source, 3), dtype=np.float32)
    height, width = target
    expected = pillow_resize(image, width, height)
    output = ImageScaleDown.execute(
        torch.from_numpy(image), width, height, "lanczos"
    ).result[0]
    np.testing.assert_allclose(
        output.numpy(), expected.clip(0, 1), atol=2e-6, rtol=1e-5
    )


@pytest.mark.parametrize(
    "pattern", ["constant", "gradient", "impulse", "checkerboard", "edge"]
)
def test_patterns_and_boundary_normalization(pattern):
    rows, columns = np.indices((17, 29))
    if pattern == "constant":
        plane = np.full((17, 29), 0.5001, dtype=np.float32)
    elif pattern == "gradient":
        plane = (rows / 16 + columns / 28).astype(np.float32) / 2
    elif pattern == "impulse":
        plane = np.zeros((17, 29), dtype=np.float32)
        plane[0, 0] = plane[8, 14] = plane[-1, -1] = 1
    elif pattern == "checkerboard":
        plane = ((rows + columns) % 2).astype(np.float32)
    else:
        plane = (columns >= 14).astype(np.float32)
    image = np.repeat(plane[None, ..., None], 3, axis=-1)
    expected = pillow_resize(image, 13, 7)
    output = ImageScaleDown.execute(torch.from_numpy(image), 13, 7, "lanczos").result[0]
    np.testing.assert_allclose(
        output.numpy(), expected.clip(0, 1), atol=2e-6, rtol=1e-5
    )
    if pattern == "edge":
        assert expected.min() < 0 and expected.max() > 1
        assert output.min() == 0 and output.max() == 1


@pytest.mark.parametrize("name", ["house", "plain"])
def test_existing_photographs_match_pillow(name):
    path = Path(__file__).parent / "data" / "color_matcher" / f"scotland_{name}.png"
    with Image.open(path) as photograph:
        image = np.asarray(photograph).astype(np.float32)[None] / 255
    height, width = image.shape[1:3]
    height, width = height // 3, width // 4
    expected = pillow_resize(image, width, height).clip(0, 1)
    output = ImageScaleDown.execute(
        torch.from_numpy(image), width, height, "lanczos"
    ).result[0]
    np.testing.assert_allclose(output.numpy(), expected, atol=2e-6, rtol=1e-5)


def test_premultiplied_rgba_matches_float_pillow_planes():
    rng = np.random.default_rng(29)
    image = rng.random((2, 19, 31, 4), dtype=np.float32)
    image[..., 3] = image[..., 3] * 0.8 + 0.1
    premultiplied = image.copy()
    premultiplied[..., :3] *= premultiplied[..., 3:]
    expected = pillow_resize(premultiplied, 13, 7)
    expected[..., :3] /= expected[..., 3:]
    output = ImageScaleDown.execute(torch.from_numpy(image), 13, 7, "lanczos").result[0]
    np.testing.assert_allclose(
        output.numpy(), expected.clip(0, 1), atol=2e-6, rtol=1e-5
    )
