# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Public BatchConcat ordering, storage, and input compatibility behavior."""

import inspect
import math

import pytest
import torch
from comfy_api.latest import io

from gpu_layerstyles.nodes.batch_concat import BatchConcat


def test_schema_has_expanding_images_and_no_execution_controls():
    schema = BatchConcat.define_schema()
    assert schema.node_id == "GPULayerStyles_BatchConcat"
    assert schema.display_name == "GPU LayerStyles BatchConcat"
    assert schema.category == "GPU LayerStyles/Batch"
    assert [field.id for field in schema.inputs] == ["images"]
    assert list(inspect.signature(BatchConcat.execute).parameters) == ["images"]
    field = schema.inputs[0]
    assert isinstance(field, io.Autogrow.Input)
    assert isinstance(field.template, io.Autogrow.TemplateNames)
    assert isinstance(field.template.input, io.Image.Input)
    assert field.template.input.id == "image"
    assert field.template.names == [f"image_{index}" for index in range(1, 101)]
    assert field.template.min == 1
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "image"
    for name in ("define_schema", "execute"):
        assert isinstance(inspect.getattr_static(BatchConcat, name), classmethod)


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

    result = BatchConcat.execute(inputs)

    assert isinstance(result, io.NodeOutput)
    assert len(result.result) == 1
    output = result.result[0]
    expected = torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert output.shape == (8, 2, 3, 3)
    assert torch.equal(output, expected[:, None, None, None].expand_as(output))
    assert list(inputs) == [name for name, _ in before]
    assert all(inputs[name] is image for name, image in before)


@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 2, 3, 3), (4, 2, 3, 3)])
@pytest.mark.parametrize("strided", [False, True])
def test_one_connected_input_returns_an_independent_contiguous_batch(shape, strided):
    image = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    if strided:
        image = image.transpose(-3, -2)
    image.requires_grad_()
    before = image.detach().clone()

    output = BatchConcat.execute({"image_7": image}).result[0]

    expected = before.unsqueeze(0) if image.ndim == 3 else before
    assert torch.equal(output, expected)
    assert output.shape == expected.shape
    assert output.is_contiguous()
    assert output.untyped_storage().data_ptr() != image.untyped_storage().data_ptr()
    assert not output.requires_grad
    assert output.grad_fn is None
    output.zero_()
    assert torch.equal(image, before)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "expanded", "channels_last"]
)
def test_copies_preserve_pixels_dtype_device_and_inputs(
    dtype, channels, layout, runtime
):
    # The host's compute device and execution machinery must not affect a copy.
    runtime.device = torch.device("cuda:2")
    runtime.available = 0
    runtime.interrupt_on = 1
    batches = []
    for count in (2, 3):
        image = torch.arange(count * 2 * 3 * channels, dtype=dtype)
        image = image.reshape(count, 2, 3, channels) / 8 - 2
        if layout == "strided":
            image = image.transpose(1, 2)
        elif layout == "expanded":
            image = image[:1].expand(count, -1, -1, -1)
        elif layout == "channels_last":
            image = image.contiguous(memory_format=torch.channels_last)
        batches.append(image.requires_grad_())
    originals = [image.detach().clone() for image in batches]
    versions = [image._version for image in batches]

    output = BatchConcat.execute(dict(zip(("image_1", "image_2"), batches))).result[0]

    expected = torch.stack([frame for original in originals for frame in original])
    assert torch.equal(output, expected)
    assert output.dtype == dtype
    assert output.device == batches[0].device
    assert output.is_contiguous()
    assert output.untyped_storage().nbytes() == output.numel() * output.element_size()
    assert not output.requires_grad
    assert output.grad_fn is None
    for image in batches:
        assert output.untyped_storage().data_ptr() != image.untyped_storage().data_ptr()
    output.fill_(123)
    for image, original, version in zip(batches, originals, versions, strict=True):
        assert torch.equal(image, original)
        assert image._version == version
    assert runtime.free_requests == []
    assert runtime.progress == []
    assert runtime.checks == runtime.cache_clears == 0


def test_special_pixel_values_are_preserved():
    image = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
    image = image.reshape(1, 1, 4)

    output = BatchConcat.execute({"image_1": image}).result[0]

    assert torch.equal(output.view(torch.int32), image.unsqueeze(0).view(torch.int32))


def test_requires_at_least_one_connected_input():
    with pytest.raises(ValueError, match="At least one image or image batch"):
        BatchConcat.execute({})


@pytest.mark.parametrize(
    "image,error",
    [
        (None, TypeError),
        ([], TypeError),
        ("not an image", TypeError),
        (torch.ones(1, 2, 3, 3, dtype=torch.uint8), TypeError),
        (torch.ones(2, 3, 3, dtype=torch.complex64), TypeError),
        (torch.ones(()), ValueError),
        (torch.ones(2, 3), ValueError),
        (torch.ones(1, 1, 2, 3, 3), ValueError),
        (torch.ones(1, 2, 3, 1), ValueError),
        (torch.ones(2, 3, 2), ValueError),
        (torch.ones(0, 2, 3, 3), ValueError),
        (torch.ones(1, 0, 3, 3), ValueError),
        (torch.ones(1, 2, 0, 4), ValueError),
        (torch.ones(0, 3, 3), ValueError),
        (torch.ones(2, 0, 3), ValueError),
        (torch.ones(2, 3, 0), ValueError),
        (torch.ones(1, 2, 3, 3).to_sparse(), ValueError),
        (torch.ones(2, 3, 3).to_sparse(), ValueError),
    ],
)
def test_rejects_invalid_connected_inputs_by_socket_name(image, error):
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
