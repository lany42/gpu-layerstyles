# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Public SliceImageBatch behavior against Python list selection."""

import inspect

import pytest
import torch
from comfy_api.latest import io

from gpu_layerstyles.nodes.slice_image_batch import SliceImageBatch


def test_schema_and_defaults():
    schema = SliceImageBatch.define_schema()
    assert schema.node_id == "GPULayerStyles_SliceImageBatch"
    assert schema.display_name == "GPU LayerStyles SliceImageBatch"
    assert schema.category == "GPU LayerStyles/Batch"
    assert [field.id for field in schema.inputs] == ["image", "slice"]
    parameters = inspect.signature(SliceImageBatch.execute).parameters
    assert list(parameters) == ["image", "slice"]
    assert parameters["slice"].default == ":"
    field = schema.inputs[1]
    assert field.default == ":"
    assert field.multiline is False
    assert field.dynamic_prompts is False
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "image"
    for name in ("define_schema", "execute"):
        assert isinstance(inspect.getattr_static(SliceImageBatch, name), classmethod)


@pytest.mark.parametrize(
    "expression,key",
    [
        (":15", slice(None, 15)),
        ("15:-15", slice(15, -15)),
        ("-15:", slice(-15, None)),
        ("42", 42),
        ("-3", -3),
        ("0", 0),
        ("-0", 0),
        ("99", 99),
        ("-100", -100),
        (":", slice(None)),
        (":100", slice(None, 100)),
        ("-100:", slice(-100, None)),
        ("::", slice(None)),
        ("1::", slice(1, None)),
        ("1:10:", slice(1, 10)),
        ("::2", slice(None, None, 2)),
        ("1:10:3", slice(1, 10, 3)),
        ("::-1", slice(None, None, -1)),
        ("8:2:-1", slice(8, 2, -1)),
        ("8:2:-2", slice(8, 2, -2)),
        ("99::-3", slice(99, None, -3)),
        (":0:-1", slice(None, 0, -1)),
        (":-100:-1", slice(None, -100, -1)),
        ("-1:-100:-2", slice(-1, -100, -2)),
        ("-100::-1", slice(-100, None, -1)),
        (" +1 : +10 : +3 ", slice(1, 10, 3)),
        ("\t -3 \t", -3),
        (f"::{10**100}", slice(None, None, 10**100)),
        (f"::-{10**100}", slice(None, None, -(10**100))),
    ],
)
def test_selections_match_python_lists(expression, key):
    image = torch.arange(100 * 2 * 3 * 4, dtype=torch.float32).reshape(100, 2, 3, 4)
    selected = list(range(len(image)))[key]
    if isinstance(selected, int):
        selected = [selected]
    expected = torch.stack([image[index] for index in selected])
    result = SliceImageBatch.execute(image, expression)
    assert isinstance(result, io.NodeOutput)
    assert len(result.result) == 1
    assert torch.equal(result.result[0], expected)


@pytest.mark.parametrize("expression", [None, ":", " \t: \t", "::", " \t:: \t"])
@pytest.mark.parametrize("frames", [1, 7])
@pytest.mark.parametrize("strided", [False, True])
def test_noop_slices_pass_through_the_original_tensor(
    expression, frames, strided, forbid_selection
):
    image = torch.rand(frames, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    if strided:
        image = image.transpose(1, 2)
    before = image.detach().clone()
    version = image._version
    arguments = () if expression is None else (expression,)
    result = SliceImageBatch.execute(image, *arguments)
    assert isinstance(result, io.NodeOutput)
    assert len(result.result) == 1
    assert result.result[0] is image
    assert torch.equal(image.detach(), before)
    assert image._version == version


@pytest.mark.parametrize("expression", ["0", "-1", ":1", "-1:", "::-1"])
def test_one_image_input_keeps_batch_dimension(expression):
    image = torch.rand(1, 2, 3, 3)
    output = SliceImageBatch.execute(image, expression).result[0]
    assert output.shape == (1, 2, 3, 3)
    assert torch.equal(output, image)
    assert output.data_ptr() != image.data_ptr()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "expression,key",
    [
        ("0:", slice(0, None)),
        ("::-1", slice(None, None, -1)),
        ("1::3", slice(1, None, 3)),
        ("-3", -3),
    ],
)
def test_copies_preserve_pixels_dtype_device_and_input(
    dtype, channels, strided, expression, key, runtime
):
    # The selected ComfyUI compute device must not affect the input's placement.
    runtime.device = torch.device("cuda:2")
    runtime.available = 0
    runtime.interrupt_on = 1
    image = torch.arange(131 * 2 * 3 * channels).reshape(131, 2, 3, channels).to(dtype)
    image = image / 8 - 2  # Include values outside [0, 1], including alpha.
    if strided:
        image = image.transpose(1, 2)
    image.requires_grad_()
    before = image.detach().clone()
    version = image._version
    output = SliceImageBatch.execute(image, expression).result[0]
    selected = list(range(len(image)))[key]
    if isinstance(selected, int):
        selected = [selected]
    expected = torch.stack([before[index] for index in selected])

    assert torch.equal(output, expected)
    assert output.dtype == image.dtype
    assert output.device == image.device
    assert output.is_contiguous()
    assert output.untyped_storage().data_ptr() != image.untyped_storage().data_ptr()
    assert not output.requires_grad
    assert output.grad_fn is None
    output.fill_(123)
    assert torch.equal(image.detach(), before)
    assert image._version == version
    assert runtime.free_requests == []
    assert runtime.progress == []
    assert runtime.checks == runtime.cache_clears == 0


@pytest.fixture
def forbid_selection(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("This call must not allocate indices or select images")

    monkeypatch.setattr(torch, "tensor", unexpected)
    monkeypatch.setattr(torch, "index_select", unexpected)


@pytest.mark.parametrize(
    "expression,field,upper",
    [
        ("100", "index", 99),
        ("-101", "index", 99),
        (str(10**100), "index", 99),
        ("100:", "START", 99),
        ("-101:", "START", 99),
        ("100:0:-1", "START", 99),
        (":101", "END", 100),
        (":-101", "END", 100),
        (":101:1000", "END", 100),
        (":-101:-1", "END", 100),
    ],
)
def test_bounds_are_checked_before_python_can_clip(
    expression, field, upper, forbid_selection
):
    with pytest.raises(IndexError) as caught:
        SliceImageBatch.execute(torch.ones(100, 1, 1, 3), expression)
    message = str(caught.value)
    assert repr(expression) in message
    assert "batch of 100 images" in message
    assert f"-100 <= {field} <= {upper}" in message


@pytest.mark.parametrize(
    "expression",
    [
        "",
        " ",
        "1:2:3:4",
        "1,2",
        "[1:2]",
        "1+2",
        "None",
        "1:None",
        "1.0",
        ":2.5",
        "::x",
        "1_0",
        "--1",
        "- 1",
        "1:2:1.5",
        "::0",
        "1:2:-0",
    ],
)
def test_rejects_malformed_slices_and_zero_stride(expression, forbid_selection):
    with pytest.raises(ValueError) as caught:
        SliceImageBatch.execute(torch.ones(100, 1, 1, 3), expression)
    assert repr(expression) in str(caught.value)


@pytest.mark.parametrize("expression", [None, 3, 1.5, True, [], slice(None)])
def test_slice_must_be_a_string(expression, forbid_selection):
    with pytest.raises(TypeError, match="slice must be a string"):
        SliceImageBatch.execute(torch.ones(100, 1, 1, 3), expression)


@pytest.mark.parametrize(
    "length,expression,error",
    [
        (100, "5:5", ValueError),
        (100, "8:2", ValueError),
        (100, ":0", ValueError),
        (100, "0:-100", ValueError),
        (100, "5:8:-1", ValueError),
        (100, ":-1:-1", ValueError),
        (100, "99:100:-1", ValueError),
        (30, "15:-15", ValueError),
        (20, "15:-15", ValueError),
        (15, "15:-15", IndexError),
        (10, ":15", IndexError),
        (10, "-15:", IndexError),
        (42, "42", IndexError),
        (2, "-3", IndexError),
    ],
)
def test_empty_or_short_batches_never_return_a_silent_subslice(
    length, expression, error, forbid_selection
):
    with pytest.raises(error) as caught:
        SliceImageBatch.execute(torch.ones(length, 1, 1, 3), expression)
    message = str(caught.value)
    assert repr(expression) in message
    assert f"batch of {length} images" in message
    if error is ValueError:
        assert "selects no images" in message


@pytest.mark.parametrize(
    "image,error,match",
    [
        (None, TypeError, "image is None"),
        ([], TypeError, "floating-point"),
        (torch.ones(1, 2, 3, 3, dtype=torch.uint8), TypeError, "floating-point"),
        (torch.ones(2, 3, 3), ValueError, "nonempty shape"),
        (torch.ones(1, 2, 3, 1), ValueError, "nonempty shape"),
        (torch.ones(0, 2, 3, 3), ValueError, "nonempty shape"),
        (torch.ones(1, 0, 3, 3), ValueError, "nonempty shape"),
        (torch.ones(1, 2, 0, 4), ValueError, "nonempty shape"),
        (torch.ones(1, 2, 3, 3).to_sparse(), ValueError, "nonempty shape"),
    ],
)
def test_rejects_missing_or_invalid_images(image, error, match, forbid_selection):
    with pytest.raises(error, match=match):
        SliceImageBatch.execute(image)
