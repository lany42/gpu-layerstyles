# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""TwoBatchBridge's public contract, exact selections, and storage behavior."""

import inspect

import pytest
import torch
from comfy_api.latest import io

from gpu_layerstyles.nodes.two_batch_bridge import TwoBatchBridge


def test_schema_and_execution_contract():
    schema = TwoBatchBridge.define_schema()
    assert schema.node_id == "GPULayerStyles_TwoBatchBridge"
    assert schema.display_name == "GPU LayerStyles TwoBatchBridge"
    assert schema.category == "GPU LayerStyles/Batch"
    assert [field.id for field in schema.inputs] == ["images", "blend_target"]
    assert isinstance(schema.inputs[0], io.Image.Input)
    blend = schema.inputs[1]
    assert isinstance(blend, io.Int.Input)
    assert (blend.default, blend.min, blend.max, blend.step) == (15, 1, 2**31 - 1, 1)
    assert [field.display_name for field in schema.outputs] == [
        "bridge_first",
        "bridge_last",
        "first/last",
    ]
    assert all(isinstance(field, io.Image.Output) for field in schema.outputs)
    parameters = inspect.signature(TwoBatchBridge.execute).parameters
    assert list(parameters) == ["images", "blend_target"]
    assert parameters["images"].default is inspect.Parameter.empty
    assert parameters["blend_target"].default == 15
    for name in ("define_schema", "execute"):
        assert isinstance(inspect.getattr_static(TwoBatchBridge, name), classmethod)


@pytest.mark.parametrize("options", [{}, {"blend_target": 15}])
def test_complete_81_frame_example(options):
    images = torch.arange(81, dtype=torch.float32)[:, None, None, None].expand(
        -1, 2, 3, 3
    )
    result = TwoBatchBridge.execute(images, **options)
    assert isinstance(result, io.NodeOutput)
    assert len(result.result) == 3
    bridge_first, bridge_last, first_last = result.result
    assert bridge_first.shape == bridge_last.shape == (15, 2, 3, 3)
    assert first_last.shape == (2, 2, 3, 3)
    assert torch.equal(bridge_first, images[66:81])
    assert torch.equal(bridge_last, images[0:15])
    assert torch.equal(first_last[0], images[66])
    assert torch.equal(first_last[1], images[14])


@pytest.mark.parametrize(
    "length,blend_target", [(2, 1), (81, 1), (3, 2), (16, 15), (20, 15), (30, 15)]
)
def test_minimum_lengths_single_frames_and_overlapping_slices(length, blend_target):
    images = torch.arange(length * 4, dtype=torch.float64).reshape(length, 1, 1, 4)
    bridge_first, bridge_last, first_last = TwoBatchBridge.execute(
        images, blend_target
    ).result
    assert bridge_first.shape == bridge_last.shape == (blend_target, 1, 1, 4)
    assert first_last.shape == (2, 1, 1, 4)
    assert torch.equal(bridge_first, images[length - blend_target :])
    assert torch.equal(bridge_last, images[:blend_target])
    assert torch.equal(first_last[0], images[length - blend_target])
    assert torch.equal(first_last[1], images[blend_target - 1])


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "expanded"])
@pytest.mark.parametrize("blend_target", [1, 7])
def test_copies_preserve_pixels_dtype_device_and_input(
    dtype, channels, layout, blend_target, runtime
):
    # ComfyUI's selected compute device must not affect this splitting utility.
    runtime.device = torch.device("cuda:2")
    runtime.available = 0
    runtime.interrupt_on = 1
    storage = torch.arange(17 * 2 * 3 * channels).reshape(17, 2, 3, channels).to(dtype)
    storage = storage / 8 - 2  # Include values outside [0, 1], including alpha.
    if layout == "strided":
        images = storage[1::2].transpose(1, 2)
    elif layout == "expanded":
        images = storage[1:2].expand(8, -1, -1, -1)
    else:
        images = storage[1:9]
    images.requires_grad_()
    before = images.detach().clone()
    storage_before = storage.clone()
    storage_pointer = storage.untyped_storage().data_ptr()
    version = images._version
    outputs = TwoBatchBridge.execute(images, blend_target).result
    expected = (
        before[-blend_target:],
        before[:blend_target],
        torch.stack((before[-blend_target], before[blend_target - 1])),
    )
    pointers = {storage_pointer}
    for output, selected in zip(outputs, expected, strict=True):
        assert torch.equal(output, selected)
        assert output.dtype == images.dtype
        assert output.device == images.device
        assert output.is_contiguous()
        pointer = output.untyped_storage().data_ptr()
        assert pointer not in pointers
        pointers.add(pointer)
        assert not output.requires_grad
        assert output.grad_fn is None

    # Mutating any output must leave the input and the other outputs intact.
    for index, output in enumerate(outputs):
        output.fill_(123)
        for other in range(index + 1, len(outputs)):
            assert torch.equal(outputs[other], expected[other])
        assert torch.equal(images.detach(), before)
        assert torch.equal(storage, storage_before)
        assert images.untyped_storage().data_ptr() == storage_pointer
        assert images._version == version
    assert runtime.free_requests == []
    assert runtime.progress == []
    assert runtime.checks == runtime.cache_clears == 0


def test_stack_contains_only_the_two_endpoint_frames(monkeypatch):
    images = torch.rand(81, 2, 3, 4, requires_grad=True)
    original_stack = torch.stack
    calls = []

    def stack(frames, *args, **kwargs):
        assert not torch.is_grad_enabled()
        assert len(frames) == 2
        assert torch.equal(frames[0], images[66])
        assert torch.equal(frames[1], images[14])
        calls.append(len(frames))
        return original_stack(frames, *args, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail(
            "TwoBatchBridge must slice directly without concatenation or transfer"
        )

    monkeypatch.setattr(torch, "stack", stack)
    monkeypatch.setattr(torch, "cat", unexpected)
    monkeypatch.setattr(torch, "index_select", unexpected)
    monkeypatch.setattr(torch.Tensor, "to", unexpected)
    TwoBatchBridge.execute(images)
    assert calls == [2]
    assert torch.is_grad_enabled()


@pytest.fixture
def forbid_output_allocation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("TwoBatchBridge must validate before allocating outputs")

    monkeypatch.setattr(torch.Tensor, "clone", unexpected)
    monkeypatch.setattr(torch.Tensor, "contiguous", unexpected)
    monkeypatch.setattr(torch, "stack", unexpected)
    monkeypatch.setattr(torch, "cat", unexpected)
    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch, "index_select", unexpected)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        -1,
        0,
        1.0,
        "1",
        [],
        float("inf"),
        float("nan"),
        2**31,
        10**100,
    ],
)
def test_invalid_controls_are_rejected_before_allocation(
    value, forbid_output_allocation
):
    with pytest.raises(ValueError, match="blend_target must be an integer between 1"):
        TwoBatchBridge.execute(torch.ones(81, 1, 1, 3), value)


@pytest.mark.parametrize(
    "length,blend_target", [(1, 1), (1, 15), (14, 15), (15, 15), (1, 2**31 - 1)]
)
def test_insufficient_frames_are_rejected(
    length, blend_target, forbid_output_allocation
):
    with pytest.raises(
        ValueError,
        match=f"more than {blend_target} frames in images.*images has {length} frames",
    ):
        TwoBatchBridge.execute(torch.ones(length, 1, 1, 3), blend_target)


@pytest.mark.parametrize(
    "images,error,match",
    [
        (None, TypeError, "floating-point"),
        ([], TypeError, "floating-point"),
        (torch.ones(16, 2, 3, 3, dtype=torch.uint8), TypeError, "floating-point"),
        (torch.ones(16, 2, 3, 3, dtype=torch.complex64), TypeError, "floating-point"),
        (torch.ones(16, 2, 3), ValueError, "nonempty shape"),
        (torch.ones(16, 2, 3, 1), ValueError, "nonempty shape"),
        (torch.ones(16, 2, 3, 5), ValueError, "nonempty shape"),
        (torch.ones(0, 2, 3, 3), ValueError, "nonempty shape"),
        (torch.ones(16, 0, 3, 3), ValueError, "nonempty shape"),
        (torch.ones(16, 2, 0, 4), ValueError, "nonempty shape"),
        (torch.ones(16, 2, 3, 3).to_sparse(), ValueError, "nonempty shape"),
    ],
)
def test_invalid_images_are_rejected(images, error, match, forbid_output_allocation):
    with pytest.raises(error, match=f"images.*{match}"):
        TwoBatchBridge.execute(images)
