# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""TwoBatchLoop's public controls, sequence, and numerical behavior."""

import inspect

import pytest
import torch
from comfy_api.latest import io

from gpu_layerstyles._exec.two_batch_loop import process_two_batch_loop
from gpu_layerstyles.nodes.two_batch_loop import TwoBatchLoop


def reference_loop(first, second, frames):
    # Per-frame float64 arithmetic, independent of the chunked float32 kernel.
    def fade(source, target):
        return [
            source[len(source) - frames + j].double() * (1 - j / (frames - 1))
            + target[j].double() * (j / (frames - 1))
            for j in range(frames)
        ]

    return torch.stack(
        fade(second, first)
        + [first[j].double() for j in range(frames, len(first) - frames)]
        + fade(first, second)
        + [second[j].double() for j in range(frames, len(second) - frames)]
    ).float()


def test_schema_and_execution_contract():
    schema = TwoBatchLoop.define_schema()
    names = [
        "images_1",
        "images_2",
        "blend_target",
        "append_first_frame",
        "output_device",
        "batch_size",
    ]
    assert [field.id for field in schema.inputs] == names
    for execute in (TwoBatchLoop.execute, process_two_batch_loop):
        parameters = inspect.signature(execute).parameters
        assert list(parameters) == names
        assert [parameters[name].default for name in names[2:]] == [15, False, "cpu", 0]
        assert parameters["append_first_frame"].default is False
    for name in ("define_schema", "execute"):
        assert isinstance(inspect.getattr_static(TwoBatchLoop, name), classmethod)
    assert schema.node_id == "GPULayerStyles_TwoBatchLoop"
    assert schema.display_name == "GPU LayerStyles TwoBatchLoop"
    assert schema.category == "GPU LayerStyles/Batch"
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "images"
    blend, append, output, batch = schema.inputs[2:]
    assert (blend.default, blend.min, blend.max, blend.step) == (15, 2, 2**31 - 1, 1)
    assert isinstance(append, io.Boolean.Input)
    assert append.display_name == "append first frame"
    assert append.default is False
    assert "After interpolation" in append.tooltip
    assert "SliceImageBatch" in append.tooltip
    assert "0:-1" in append.tooltip
    assert output.options == ["gpu", "cpu"]
    assert output.default == "cpu"
    assert (batch.default, batch.min, batch.max, batch.step) == (0, 0, 2**31 - 1, 1)


@pytest.mark.parametrize("batch_size", [0, 1, 7, 15, 66, 132, 133, 200])
def test_complete_81_frame_sequence_and_optional_closing_frame(batch_size):
    first = torch.arange(81, dtype=torch.float32)[:, None, None, None].expand(
        -1, 2, 3, 3
    )
    second = first + 1000
    result = TwoBatchLoop.execute(first, second, batch_size=batch_size)
    assert isinstance(result, io.NodeOutput)
    assert len(result.result) == 1
    output = result.result[0]
    assert output.shape == (132, 2, 3, 3)
    torch.testing.assert_close(output, reference_loop(first, second, 15))
    assert torch.equal(output[0], second[66])
    assert torch.equal(output[14], first[14])
    assert torch.equal(output[15:66], first[15:66])
    assert torch.equal(output[66], first[66])
    assert torch.equal(output[80], second[14])
    assert torch.equal(output[81:132], second[15:66])
    # The middle's last frame immediately precedes the tail at loop frame zero.
    assert torch.equal(output[-1], second[65])

    disabled = TwoBatchLoop.execute(
        first, second, append_first_frame=False, batch_size=batch_size
    ).result[0]
    closed = TwoBatchLoop.execute(
        first, second, append_first_frame=True, batch_size=batch_size
    ).result[0]
    assert torch.equal(disabled, output)
    assert closed.shape == (133, 2, 3, 3)
    assert torch.equal(closed[:-1], output)
    assert torch.equal(closed[-1], output[0])
    closed[-1].zero_()
    assert torch.equal(closed[0], output[0])  # A copy, not shared frame storage.


@pytest.mark.parametrize("n1,n2,frames", [(5, 5, 2), (5, 9, 2), (47, 31, 15)])
@pytest.mark.parametrize("append", [False, True])
def test_unequal_and_minimum_lengths(n1, n2, frames, append):
    first = torch.rand(n1, 2, 3, 4)
    second = torch.rand(n2, 2, 3, 4)
    output = process_two_batch_loop(first, second, frames, append, batch_size=3)
    assert len(output) == n1 + n2 - 2 * frames + int(append)
    torch.testing.assert_close(
        output[: n1 + n2 - 2 * frames], reference_loop(first, second, frames)
    )
    if append:
        assert torch.equal(output[-1], output[0])


def test_rgba_blends_alpha_independently_without_clamping():
    first = torch.tensor([-2.0, 0.5, 1.5, 0.0]).expand(7, 1, 1, 4)
    second = torch.tensor([2.0, 0.0, 0.5, 1.0]).expand(7, 1, 1, 4)
    output = TwoBatchLoop.execute(first, second, 3).result[0]
    expected = torch.tensor(
        [
            [2.0, 0.0, 0.5, 1.0],
            [0.0, 0.25, 1.0, 0.5],
            [-2.0, 0.5, 1.5, 0.0],
            [-2.0, 0.5, 1.5, 0.0],
            [-2.0, 0.5, 1.5, 0.0],
            [0.0, 0.25, 1.0, 0.5],
            [2.0, 0.0, 0.5, 1.0],
            [2.0, 0.0, 0.5, 1.0],
        ]
    )[:, None, None, :]
    assert torch.equal(output, expected)


@pytest.mark.parametrize(
    "dtype_1,dtype_2",
    [
        (torch.float16, torch.float64),
        (torch.bfloat16, torch.float16),
        (torch.float32, torch.float32),
        (torch.float64, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("batch_size", [0, 1, 7, 200])
def test_chunk_independence_float32_and_input_preservation(
    dtype_1, dtype_2, channels, batch_size
):
    generator = torch.Generator().manual_seed(137)
    first = (
        (torch.rand(77, 3, 5, channels, generator=generator) * 3 - 1)
        .to(dtype_1)
        .transpose(1, 2)
        .requires_grad_()
    )
    second = (
        (torch.rand(91, 3, 5, channels, generator=generator) * 3 - 1)
        .to(dtype_2)
        .transpose(1, 2)
        .requires_grad_()
    )
    before = first.detach().clone(), second.detach().clone()
    versions = first._version, second._version
    output = TwoBatchLoop.execute(
        first, second, 33, True, batch_size=batch_size
    ).result[0]
    expected = reference_loop(*before, 33)
    torch.testing.assert_close(output[:-1], expected, rtol=2e-6, atol=4e-7)
    whole = TwoBatchLoop.execute(first, second, 33, True, batch_size=200).result[0]
    assert torch.equal(output, whole)
    assert output.shape == (103, 5, 3, channels)
    assert torch.equal(output[-1], output[0])
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert output.is_contiguous()
    assert not output.requires_grad
    assert output.grad_fn is None
    assert output.untyped_storage().data_ptr() not in (
        first.untyped_storage().data_ptr(),
        second.untyped_storage().data_ptr(),
    )
    assert torch.equal(first, before[0])
    assert torch.equal(second, before[1])
    assert (first._version, second._version) == versions


@pytest.mark.parametrize("output_device", ["cpu", "gpu"])
def test_overlapping_inputs_and_comfy_cpu_mode(output_device):
    image = torch.rand(9, 2, 3, 4, generator=torch.Generator().manual_seed(2))
    before = image.clone()
    output = TwoBatchLoop.execute(image, image[2:], 3, True, output_device, 2).result[0]
    assert output.device.type == "cpu"
    torch.testing.assert_close(output[:-1], reference_loop(before, before[2:], 3))
    assert torch.equal(output[-1], output[0])
    assert torch.equal(image, before)
    output.zero_()
    assert torch.equal(image, before)


def test_float32_inside_autocast():
    first = torch.rand(7, 2, 3, 4)
    second = torch.rand(9, 2, 3, 4)
    expected = TwoBatchLoop.execute(first, second, 3, True).result[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = TwoBatchLoop.execute(first, second, 3, True).result[0]
    assert output.dtype == torch.float32
    assert torch.equal(output, expected)


@pytest.fixture
def forbid_work(monkeypatch, runtime):
    def unexpected(*args, **kwargs):
        pytest.fail("TwoBatchLoop must validate before allocating or processing")

    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch.Tensor, "to", unexpected)
    monkeypatch.setattr(torch, "arange", unexpected)
    yield
    assert not runtime.free_requests
    assert not runtime.progress


@pytest.mark.parametrize(
    "value", [None, True, False, -1, 0, 1, 2.0, "2", float("inf"), float("nan"), 2**31]
)
def test_blend_target_requires_an_integer_in_range(value, forbid_work):
    with pytest.raises(ValueError, match="blend_target"):
        TwoBatchLoop.execute(torch.ones(5, 1, 1, 3), torch.ones(5, 1, 1, 3), value)


@pytest.mark.parametrize("value", [None, 0, 1, -1, 0.0, "False", [], float("nan")])
def test_closing_frame_requires_a_boolean(value, forbid_work):
    with pytest.raises(TypeError, match="append_first_frame"):
        TwoBatchLoop.execute(torch.ones(5, 1, 1, 3), torch.ones(5, 1, 1, 3), 2, value)


@pytest.mark.parametrize("name", ["images_1", "images_2"])
@pytest.mark.parametrize(
    "frames,length", [(2, 4), (15, 1), (15, 14), (15, 15), (15, 16), (15, 29), (15, 30)]
)
def test_short_batches_and_empty_middles_are_rejected(
    name, frames, length, forbid_work
):
    inputs = {"images_1": torch.ones(81, 1, 1, 3), "images_2": torch.ones(81, 1, 1, 3)}
    inputs[name] = torch.ones(length, 1, 1, 3)
    with pytest.raises(
        ValueError, match=f"at least {2 * frames + 1} frames in {name}.*nonempty middle"
    ):
        TwoBatchLoop.execute(**inputs, blend_target=frames)


@pytest.mark.parametrize("name", ["images_1", "images_2"])
@pytest.mark.parametrize(
    "image,error",
    [
        (None, TypeError),
        ([], TypeError),
        (torch.ones(31, 2, 3, 3, dtype=torch.uint8), TypeError),
        (torch.ones(31, 2, 3), ValueError),
        (torch.ones(0, 2, 3, 3), ValueError),
        (torch.ones(31, 0, 3, 3), ValueError),
        (torch.ones(31, 2, 0, 3), ValueError),
        (torch.ones(31, 2, 3, 1), ValueError),
        (torch.ones(31, 2, 3, 5), ValueError),
        (torch.ones(31, 2, 3, 3).to_sparse(), ValueError),
    ],
)
def test_invalid_images_are_named_and_rejected(name, image, error, forbid_work):
    inputs = {"images_1": torch.ones(31, 2, 3, 3), "images_2": torch.ones(31, 2, 3, 3)}
    inputs[name] = image
    with pytest.raises(error, match=name):
        TwoBatchLoop.execute(**inputs)


@pytest.mark.parametrize("shape", [(31, 1, 3, 3), (31, 2, 1, 3), (31, 2, 3, 4)])
def test_mismatched_dimensions_and_channels_are_rejected(shape, forbid_work):
    with pytest.raises(ValueError, match="matching height, width, and channel count"):
        TwoBatchLoop.execute(torch.ones(31, 2, 3, 3), torch.ones(shape))


@pytest.mark.parametrize(
    "options,match",
    [
        ({"batch_size": -1}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 2.0}, "batch_size"),
        ({"output_device": "cuda"}, "output_device"),
    ],
)
def test_execution_controls_are_validated(options, match, forbid_work):
    with pytest.raises(ValueError, match=match):
        TwoBatchLoop.execute(
            torch.ones(31, 1, 1, 3), torch.ones(31, 1, 1, 3), **options
        )
