# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Public BatchConcat ordering, storage, and input compatibility behavior."""

import pytest
import torch

from gpu_layerstyles.nodes.batch_concat import BatchConcat


def test_numeric_socket_order_preserves_frames_gaps_and_duplicates():
    first = torch.tensor([1.0, 2.0])[:, None, None, None].expand(-1, 2, 3, 3)
    single = torch.full((2, 3, 3), 3.0)
    last = torch.tensor([4.0, 5.0, 6.0])[:, None, None, None].expand(-1, 2, 3, 3)
    inputs = {
        "image_100": last,
        "image_10": single,
        "image_4": first,
        "image_1": first,
    }
    before = list(inputs.items())

    output = BatchConcat.execute(inputs).result[0]

    expected = torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert output.shape == (8, 2, 3, 3)
    assert torch.equal(output, expected[:, None, None, None].expand_as(output))
    assert list(inputs) == [name for name, _ in before]
    assert all(inputs[name] is image for name, image in before)


@pytest.mark.parametrize("shape", [(2, 3, 3), (4, 2, 3, 3)])
def test_one_connected_input_is_still_copied(shape):
    image = torch.rand(shape)
    output = BatchConcat.execute({"image_7": image}).result[0]
    assert torch.equal(output, image if image.ndim == 4 else image[None])
    assert output.untyped_storage().data_ptr() != image.untyped_storage().data_ptr()


@pytest.mark.parametrize("layout", ["strided", "expanded", "channels_last"])
def test_copies_are_compact_and_contiguous_for_any_input_layout(layout):
    batches = []
    for count in (2, 3):
        image = torch.arange(count * 2 * 3 * 4, dtype=torch.float16)
        image = image.reshape(count, 2, 3, 4) / 8 - 2  # Values beyond [0, 1].
        if layout == "strided":
            image = image.transpose(1, 2)
        elif layout == "expanded":
            image = image[:1].expand(count, -1, -1, -1)
        else:
            image = image.contiguous(memory_format=torch.channels_last)
        batches.append(image)

    output = BatchConcat.execute(dict(zip(("image_1", "image_2"), batches))).result[0]

    assert torch.equal(
        output, torch.stack([frame for image in batches for frame in image])
    )
    assert output.is_contiguous()
    assert output.untyped_storage().nbytes() == output.numel() * output.element_size()


def test_requires_at_least_one_connected_input():
    with pytest.raises(ValueError, match="At least one image or image batch"):
        BatchConcat.execute({})


@pytest.mark.parametrize(
    "image,error",
    [
        ("not an image", TypeError),
        (torch.ones(2, 3, 2), ValueError),
        (torch.ones(2, 3, 3).to_sparse(), ValueError),
    ],
)
def test_single_images_are_validated_after_gaining_a_batch_dimension(image, error):
    with pytest.raises(error, match="image_2 must"):
        BatchConcat.execute({"image_1": torch.ones(1, 2, 3, 3), "image_2": image})


@pytest.mark.parametrize(
    "image,property_name",
    [
        (torch.ones(1, 4, 3, 3), "height, width, and channel count"),
        (torch.ones(2, 4, 3), "height, width, and channel count"),
        (torch.ones(1, 2, 3, 4), "height, width, and channel count"),
        (torch.ones(2, 3, 3, dtype=torch.float64), "dtype"),
        (torch.ones(1, 2, 3, 3, device="meta"), "device"),
    ],
)
def test_rejects_mismatches_against_the_first_socket(image, property_name):
    with pytest.raises(
        ValueError, match=f"image_10 must match image_2's {property_name}"
    ):
        BatchConcat.execute({"image_10": image, "image_2": torch.ones(1, 2, 3, 3)})
