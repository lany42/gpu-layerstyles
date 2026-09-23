# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Two photographic transfers compared directly with color-matcher 0.6.0.

Original reference implementation and fixture source: Christopher Hahne's
color-matcher 0.6.0. Original upstream test copyright (c) 2020 Christopher Hahne
<inbox@christopherhahne.de>.

Both implementations receive identical interleaved float32 uint8/255 inputs.
These tests are GPU LayerStyles project code; see COPYRIGHT for attribution
and data/color_matcher/README.md for fixture hashes, source URLs, and provenance.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from color_matcher import ColorMatcher
from PIL import Image

from gpu_layerstyles._color_match import match_rgb, prepare_reference
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
    return images


@pytest.mark.parametrize("method", ["mkl", "mvgd"])
@pytest.mark.parametrize(
    "source_name,reference_name,strength",
    [
        ("house", "plain", 1.0),
        ("plain", "house", 0.5),
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
    raw = match_rgb(target, prepare_reference(palette, method), method)[0].numpy()
    np.testing.assert_allclose(raw, expected, rtol=0, atol=MAX_ABSOLUTE_ERROR)

    output = ColorMatch.execute(target, palette, method, strength).result[0][0].numpy()
    final = np.clip((source + strength * (expected - source)).astype(np.float32), 0, 1)
    np.testing.assert_allclose(output, final, rtol=0, atol=MAX_ABSOLUTE_ERROR)
