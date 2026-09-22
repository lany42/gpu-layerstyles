# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""CrossFade's public controls, sequence assembly, and numerical behavior."""

import inspect

import pytest
import torch
from comfy_api.latest import io

from gpu_layerstyles.cross_fade import CrossFade


def test_schema_and_execute_contract():
    schema = CrossFade.define_schema()
    names = [
        "images_1",
        "images_2",
        "start_index",
        "frames",
        "output_device",
        "batch_size",
    ]
    assert [field.id for field in schema.inputs] == names
    parameters = inspect.signature(CrossFade.execute).parameters
    assert list(parameters) == names
    assert [parameters[name].default for name in names[2:]] == [0, 2, "cpu", 0]
    assert isinstance(inspect.getattr_static(CrossFade, "define_schema"), classmethod)
    assert isinstance(inspect.getattr_static(CrossFade, "execute"), classmethod)
    assert schema.node_id == "GPULayerStyles_CrossFade"
    assert schema.display_name == "GPU LayerStyles CrossFade"
    assert schema.category == "GPU LayerStyles/Batch"
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "image"
    start, frames, output, batch = schema.inputs[2:]
    assert (start.default, start.min, start.max, start.step) == (0, 0, 2**31 - 1, 1)
    assert (frames.default, frames.min, frames.max, frames.step) == (2, 2, 2**31 - 1, 1)
    assert output.options == ["gpu", "cpu"]
    assert output.default == "cpu"
    assert (batch.default, batch.min, batch.max, batch.step) == (0, 0, 2**31 - 1, 1)


@pytest.mark.parametrize("batch_size", [0, 1, 2, 3, 4, 99])
@pytest.mark.parametrize(
    "start,frames,expected",
    [
        (0, 2, [0, 12, 16, 20, 24]),
        (0, 5, [0, 3.75, 9, 15.75, 24]),
        (2, 3, [0, 1, 2, 7.5, 16, 20, 24]),
        (4, 3, [0, 1, 2, 3, 4, 8.5, 16, 20, 24]),
    ],
)
def test_exact_sequence_endpoints_and_first_batch_tail_discard(
    start, frames, expected, batch_size
):
    images_1 = torch.arange(7, dtype=torch.float32)[:, None, None, None].expand(
        -1, 2, 3, 3
    )
    images_2 = torch.tensor([8, 12, 16, 20, 24], dtype=torch.float32)[
        :, None, None, None
    ].expand(-1, 2, 3, 3)
    result = CrossFade.execute(images_1, images_2, start, frames, batch_size=batch_size)
    assert isinstance(result, io.NodeOutput)
    assert len(result.result) == 1
    output = result.result[0]
    values = torch.tensor(expected)[:, None, None, None].expand(-1, 2, 3, 3)
    assert torch.equal(output, values)
    assert len(output) == start + len(images_2)


def test_defaults_make_a_two_frame_transition():
    first = torch.full((2, 1, 1, 3), -2.0)
    second = torch.full((2, 1, 1, 3), 3.0)
    output = CrossFade.execute(first, second).result[0]
    assert torch.equal(
        output, torch.tensor([-2.0, 3.0])[:, None, None, None].expand_as(output)
    )


def test_rgba_channels_fade_independently_without_premultiplication_or_clamping():
    first = torch.tensor([-2.0, 0.5, 1.5, 0.0]).expand(3, 1, 1, 4)
    second = torch.tensor([2.0, 0.0, 0.5, 1.0]).expand(3, 1, 1, 4)
    output = CrossFade.execute(first, second, frames=3).result[0]
    expected = torch.tensor(
        [
            [-2.0, 0.5, 1.5, 0.0],
            [0.0, 0.25, 1.0, 0.5],
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
@pytest.mark.parametrize("batch_size", [0, 1, 3, 80, 999])
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
        (torch.rand(131, 3, 5, channels, generator=generator) * 3 - 1)
        .to(dtype_2)
        .transpose(1, 2)
        .requires_grad_()
    )
    before = first.detach().clone(), second.detach().clone()
    versions = first._version, second._version
    output = CrossFade.execute(first, second, 5, 67, batch_size=batch_size).result[0]

    # Use a per-frame float64 reference, independent of the chunked float32 kernel.
    expected = [frame.double() for frame in before[0][:5]]
    for j in range(67):
        weight = j / 66
        expected.append(
            before[0][5 + j].double() * (1 - weight) + before[1][j].double() * weight
        )
    expected.extend(frame.double() for frame in before[1][67:])
    expected = torch.stack(expected).float()
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=4e-7)
    assert output.shape == (136, 5, 3, channels)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert output.is_contiguous()
    assert not output.requires_grad
    assert output.grad_fn is None
    assert output.data_ptr() not in (first.data_ptr(), second.data_ptr())
    assert torch.equal(output[5], before[0][5].float())
    assert torch.equal(output[71], before[1][66].float())
    assert torch.equal(first, before[0])
    assert torch.equal(second, before[1])
    assert (first._version, second._version) == versions


@pytest.mark.parametrize("output_device", ["cpu", "gpu"])
def test_overlapping_inputs_are_safe_and_honor_comfy_cpu_mode(output_device):
    image = torch.rand(9, 2, 3, 4, generator=torch.Generator().manual_seed(2))
    before = image.clone()
    output = CrossFade.execute(image, image[2:], 1, 3, output_device, 2).result[0]
    assert output.device.type == "cpu"
    assert torch.equal(image, before)
    output.zero_()
    assert torch.equal(image, before)


def test_float32_inside_autocast():
    first = torch.rand(7, 2, 3, 4)
    second = torch.rand(5, 2, 3, 4)
    expected = CrossFade.execute(first, second, 2, 3).result[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = CrossFade.execute(first, second, 2, 3).result[0]
    assert output.dtype == torch.float32
    assert torch.equal(output, expected)


@pytest.fixture
def forbid_work(monkeypatch, runtime):
    def unexpected(*args, **kwargs):
        pytest.fail("CrossFade must validate before allocating or processing")

    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch.Tensor, "to", unexpected)
    monkeypatch.setattr(torch, "arange", unexpected)
    yield
    assert not runtime.free_requests
    assert not runtime.progress


@pytest.mark.parametrize("name", ["start_index", "frames"])
@pytest.mark.parametrize(
    "value", [None, True, False, -1, 2.0, "2", float("inf"), float("nan")]
)
def test_controls_must_be_exact_integers(name, value, forbid_work):
    with pytest.raises(ValueError, match=name):
        CrossFade.execute(
            torch.ones(5, 1, 1, 3), torch.ones(5, 1, 1, 3), **{name: value}
        )


@pytest.mark.parametrize(
    "start,frames,n1,n2,match",
    [
        (0, 0, 5, 5, "frames must be an integer >= 2"),
        (0, 1, 5, 5, "frames must be an integer >= 2"),
        (5, 2, 5, 5, "start_index 5 is out of range"),
        (6, 2, 5, 5, "start_index 6 is out of range"),
        (4, 2, 5, 5, "exactly 2.*images_1 has 1.*images_2 has 5"),
        (1, 5, 5, 5, "exactly 5.*images_1 has 4.*images_2 has 5"),
        (0, 5, 7, 3, "exactly 5.*images_1 has 7.*images_2 has 3"),
        (0, 2, 1, 5, "exactly 2.*images_1 has 1.*images_2 has 5"),
        (0, 2, 5, 1, "exactly 2.*images_1 has 5.*images_2 has 1"),
    ],
)
def test_invalid_ranges_are_rejected_without_shortening(
    start, frames, n1, n2, match, forbid_work
):
    with pytest.raises(ValueError, match=match):
        CrossFade.execute(
            torch.ones(n1, 1, 1, 3), torch.ones(n2, 1, 1, 3), start, frames
        )


@pytest.mark.parametrize("name", ["images_1", "images_2"])
@pytest.mark.parametrize(
    "image,error",
    [
        (None, TypeError),
        ([], TypeError),
        (torch.ones(2, 2, 3, 3, dtype=torch.uint8), TypeError),
        (torch.ones(2, 2, 3), ValueError),
        (torch.ones(0, 2, 3, 3), ValueError),
        (torch.ones(2, 0, 3, 3), ValueError),
        (torch.ones(2, 2, 0, 3), ValueError),
        (torch.ones(2, 2, 3, 1), ValueError),
        (torch.ones(2, 2, 3, 5), ValueError),
        (torch.ones(2, 2, 3, 3).to_sparse(), ValueError),
    ],
)
def test_invalid_images_are_named_and_rejected(name, image, error, forbid_work):
    inputs = {"images_1": torch.ones(2, 2, 3, 3), "images_2": torch.ones(2, 2, 3, 3)}
    inputs[name] = image
    with pytest.raises(error, match=name):
        CrossFade.execute(**inputs)


@pytest.mark.parametrize("shape", [(2, 1, 3, 3), (2, 2, 1, 3), (2, 2, 3, 4)])
def test_mismatched_dimensions_and_channels_are_rejected(shape, forbid_work):
    with pytest.raises(ValueError, match="matching height, width, and channel count"):
        CrossFade.execute(torch.ones(2, 2, 3, 3), torch.ones(shape))


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
        CrossFade.execute(torch.ones(2, 1, 1, 3), torch.ones(2, 1, 1, 3), **options)
