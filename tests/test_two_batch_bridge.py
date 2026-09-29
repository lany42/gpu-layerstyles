# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""TwoBatchBridge's exact selections, storage, and validation."""

import pytest
import torch

from gpu_layerstyles.nodes.two_batch_bridge import TwoBatchBridge


def test_complete_81_frame_example():
    images = torch.arange(81, dtype=torch.float32)[:, None, None, None].expand(
        -1, 2, 3, 3
    )
    bridge_first, bridge_last, first_last = TwoBatchBridge.execute(images).result
    assert torch.equal(bridge_first, images[66:81])
    assert torch.equal(bridge_last, images[0:15])
    assert torch.equal(first_last, images[[66, 14]])


@pytest.mark.parametrize(
    "length,blend_target", [(2, 1), (81, 1), (3, 2), (16, 15), (20, 15), (30, 15)]
)
def test_minimum_lengths_single_frames_and_overlapping_slices(length, blend_target):
    images = torch.arange(length * 4, dtype=torch.float64).reshape(length, 1, 1, 4)
    bridge_first, bridge_last, first_last = TwoBatchBridge.execute(
        images, blend_target
    ).result
    assert torch.equal(bridge_first, images[length - blend_target :])
    assert torch.equal(bridge_last, images[:blend_target])
    assert torch.equal(first_last, images[[length - blend_target, blend_target - 1]])


@pytest.mark.parametrize("layout", ["contiguous", "expanded"])
def test_overlapping_outputs_are_independent_copies(layout):
    storage = torch.arange(9 * 2 * 3 * 4, dtype=torch.float32).reshape(9, 2, 3, 4)
    images = (
        storage[1:9] if layout == "contiguous" else storage[1:2].expand(8, -1, -1, -1)
    )
    before = images.clone()
    outputs = TwoBatchBridge.execute(images, 7).result
    expected = (before[-7:], before[:7], before[[1, 6]])
    # Mutating any output must leave the input and the other outputs intact.
    for index, output in enumerate(outputs):
        assert output.is_contiguous()
        output.fill_(123)
        for other in range(index + 1, len(outputs)):
            assert torch.equal(outputs[other], expected[other])
        assert torch.equal(images, before)


@pytest.mark.parametrize("value", [True, 0, 1.0, 2**31])
def test_blend_target_requires_an_integer_in_range(value):
    with pytest.raises(ValueError, match="blend_target must be an integer between 1"):
        TwoBatchBridge.execute(torch.ones(81, 1, 1, 3), value)


@pytest.mark.parametrize("length,blend_target", [(1, 1), (15, 15)])
def test_insufficient_frames_are_rejected(length, blend_target):
    with pytest.raises(
        ValueError,
        match=f"more than {blend_target} frames in images.*images has {length} frames",
    ):
        TwoBatchBridge.execute(torch.ones(length, 1, 1, 3), blend_target)
