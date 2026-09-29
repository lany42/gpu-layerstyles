# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Photographic and synthetic transfers compared directly with color-matcher 0.6.0.

Original reference implementation and fixture source: Christopher Hahne's
color-matcher 0.6.0. Original upstream test copyright (c) 2020 Christopher Hahne
<inbox@christopherhahne.de>.

Both implementations receive identical interleaved float32 inputs; photographs
are passed as uint8/255.
These tests are GPU LayerStyles project code; see COPYRIGHT for attribution
and data/color_matcher/README.md for fixture hashes, source URLs, and provenance.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from color_matcher import ColorMatcher
from PIL import Image

from gpu_layerstyles.nodes.color_match import ColorMatch

# Allow minor numerical differences: about 0.13 of one 8-bit channel level.
MAX_ABSOLUTE_ERROR = 5e-4


@pytest.fixture(scope="module")
def photographs():
    images = {}
    for name in ("house", "plain"):
        path = Path(__file__).parent / "data" / "color_matcher" / f"scotland_{name}.png"
        with Image.open(path) as image:
            images[name] = np.asarray(image).astype(np.float32) / 255
    # MKL accepts any reference resolution; a small textured crop also makes
    # the sample-covariance normalization visible.
    images["plain_crop"] = np.ascontiguousarray(images["plain"][264:270, 24:31])
    return images


@pytest.mark.parametrize(
    "method,source_name,reference_name,strength",
    [
        ("mkl", "house", "plain", 1.0),
        ("mvgd", "house", "plain", 1.0),
        ("mkl", "plain", "house", 0.5),
        ("mvgd", "plain", "house", 0.5),
        ("mkl", "house", "plain_crop", 1.0),
    ],
)
def test_photographic_transfer_matches_upstream(
    photographs, method, source_name, reference_name, strength
):
    source, reference = photographs[source_name], photographs[reference_name]
    expected = np.real_if_close(
        ColorMatcher(src=source, ref=reference, method=method).main()
    )
    assert not np.iscomplexobj(expected)
    target, palette = torch.from_numpy(source)[None], torch.from_numpy(reference)[None]
    output = ColorMatch.execute(target, palette, method, strength).result[0][0].numpy()
    final = np.clip((source + strength * (expected - source)).astype(np.float32), 0, 1)
    np.testing.assert_allclose(output, final, rtol=0, atol=MAX_ABSOLUTE_ERROR)


def near_gray(rng, size, noisy_channels):
    image = np.repeat(rng.random((size, size, 1), dtype=np.float32), 3, axis=-1)
    image[..., noisy_channels] += 0.01 * rng.random(
        (size, size, len(noisy_channels)), dtype=np.float32
    )
    return image


@pytest.mark.parametrize(
    "side,size,noisy_channels",
    [
        # Identical red and green leave a weak blue tint that must survive the
        # rank tolerance while the exact degeneracy is truncated.
        ("target", 13, [2]),
        # Weak independent noise must survive the reference pseudoinverse,
        # whose default tolerance would grow with the pixel count.
        ("reference", 256, [0, 1, 2]),
    ],
)
def test_weak_color_variation_is_matched_like_upstream_mvgd(side, size, noisy_channels):
    rng = np.random.default_rng(5)
    weak = near_gray(rng, size, noisy_channels)
    other = rng.random((size, size, 3), dtype=np.float32)
    source, reference = (weak, other) if side == "target" else (other, weak)
    expected = np.real_if_close(
        ColorMatcher(src=source, ref=reference, method="mvgd").main()
    )
    target, palette = torch.from_numpy(source)[None], torch.from_numpy(reference)[None]
    output = ColorMatch.execute(target, palette, "mvgd").result[0][0].numpy()
    np.testing.assert_allclose(
        output, np.clip(expected, 0, 1), rtol=0, atol=MAX_ABSOLUTE_ERROR
    )
