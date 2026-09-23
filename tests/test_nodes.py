# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch
from comfy_api.latest import ComfyExtension, io

from gpu_layerstyles import comfy_entrypoint
from gpu_layerstyles.nodes.color_correct_brightness_and_contrast import (
    BrightnessContrastV2,
)
from gpu_layerstyles.nodes.color_correct_color_balance import ColorBalance
from gpu_layerstyles.nodes.color_correct_color_temperature import ColorTemperature
from gpu_layerstyles.nodes.color_match import ColorMatch
from gpu_layerstyles.nodes.cross_fade import CrossFade
from gpu_layerstyles.nodes.image_scale_down import ImageScaleDown
from gpu_layerstyles.nodes.slice_image_batch import SliceImageBatch
from gpu_layerstyles.nodes.two_batch_bridge import TwoBatchBridge
from gpu_layerstyles.nodes.two_batch_loop import TwoBatchLoop

NODES = [ColorBalance, BrightnessContrastV2, ColorTemperature]
NEUTRAL = [(0, 0, 0), (1, 1, 1), (0,)]
ACTIVE = [(0.3, -0.2, 0.4), (1.2, 0.8, 1.3), (-37,)]


def test_extension_registers_exactly_nine_v3_nodes():
    extension = asyncio.run(comfy_entrypoint())
    assert isinstance(extension, ComfyExtension)
    assert asyncio.run(extension.get_node_list()) == [
        *NODES,
        ColorMatch,
        ImageScaleDown,
        SliceImageBatch,
        CrossFade,
        TwoBatchLoop,
        TwoBatchBridge,
    ]
    assert [
        node.define_schema().node_id
        for node in [
            *NODES,
            ColorMatch,
            ImageScaleDown,
            SliceImageBatch,
            CrossFade,
            TwoBatchLoop,
            TwoBatchBridge,
        ]
    ] == [
        "GPULayerStyles_ColorBalance",
        "GPULayerStyles_BrightnessContrastV2",
        "GPULayerStyles_ColorTemperature",
        "GPULayerStyles_ColorMatch",
        "GPULayerStyles_ImageScaleDown",
        "GPULayerStyles_SliceImageBatch",
        "GPULayerStyles_CrossFade",
        "GPULayerStyles_TwoBatchLoop",
        "GPULayerStyles_TwoBatchBridge",
    ]


def test_root_loader_supports_clone_and_zip_installations(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gpu_layerstyles_clone",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    extension = asyncio.run(module.comfy_entrypoint())
    assert [
        node.define_schema().node_id for node in asyncio.run(extension.get_node_list())
    ] == [
        node.define_schema().node_id
        for node in [
            *NODES,
            ColorMatch,
            ImageScaleDown,
            SliceImageBatch,
            CrossFade,
            TwoBatchLoop,
            TwoBatchBridge,
        ]
    ]


@pytest.mark.parametrize(
    "node,names,default,minimum,maximum,step",
    [
        (ColorBalance, ["cyan_red", "magenta_green", "yellow_blue"], 0, -1, 1, 0.001),
        (BrightnessContrastV2, ["brightness", "contrast", "saturation"], 1, 0, 3, 0.01),
        (ColorTemperature, ["temperature"], 0, -100, 100, 1),
    ],
)
def test_schema_and_execute_controls(node, names, default, minimum, maximum, step):
    schema = node.define_schema()
    expected_names = ["image", *names, "output_device", "batch_size"]
    assert [item.id for item in schema.inputs] == expected_names
    assert list(inspect.signature(node.execute).parameters) == expected_names
    assert isinstance(inspect.getattr_static(node, "execute"), classmethod)
    assert isinstance(inspect.getattr_static(node, "define_schema"), classmethod)
    assert schema.category == "GPU LayerStyles/Color"
    assert schema.display_name.startswith("GPU LayerStyles ")
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "image"
    for field in schema.inputs[1:-2]:
        assert (field.default, field.min, field.max, field.step) == (
            default,
            minimum,
            maximum,
            step,
        )
    output, batch = schema.inputs[-2:]
    assert output.options == ["gpu", "cpu"]
    assert output.default == "cpu"
    assert (batch.default, batch.min, batch.max, batch.step) == (0, 0, 2**31 - 1, 1)
    parameters = inspect.signature(node.execute).parameters
    assert parameters["output_device"].default == "cpu"
    assert parameters["batch_size"].default == 0


@pytest.mark.parametrize("node,controls", list(zip(NODES, ACTIVE)))
@pytest.mark.parametrize("output_device", [None, "cpu", "gpu"])
def test_node_output_placement_with_mocked_gpu(
    node, controls, output_device, runtime, gpu_routing
):
    image = torch.rand(5, 2, 3, 4, dtype=torch.float64)
    before = image.clone()
    options = {} if output_device is None else {"output_device": output_device}
    result = node.execute(image, *controls, **options).result[0]
    destination = runtime.device if output_device == "gpu" else torch.device("cpu")
    assert gpu_routing.allocations == [(tuple(image.shape), destination, torch.float32)]
    assert gpu_routing.transfers == [(5, runtime.device, torch.float32)]
    assert result.dtype == torch.float32
    assert torch.equal(result[..., 3], image[..., 3].float())
    assert torch.equal(image, before)


@pytest.mark.parametrize("node,controls", list(zip(NODES, NEUTRAL)))
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("output_device", ["gpu", "cpu"])
def test_neutral_identity_float32_no_grad_and_no_alias(
    node, controls, dtype, output_device
):
    image = torch.rand((3, 2, 7, 4), dtype=dtype).requires_grad_()
    image_before = image.detach().clone()
    version = image._version
    result = node.execute(image, *controls, output_device, 2)
    assert isinstance(result, io.NodeOutput)
    output = result.result[0]
    assert output.shape == image.shape
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"  # Honors ComfyUI CPU mode for either setting.
    assert not output.requires_grad
    assert output.grad_fn is None
    assert output.data_ptr() != image.data_ptr()
    assert torch.equal(output, image_before.float())
    assert torch.equal(image.detach(), image_before)
    assert image._version == version


@pytest.mark.parametrize("node,controls", list(zip(NODES, ACTIVE)))
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 3), (5, 1, 9, 4), (5, 7, 1, 3), (7, 4, 9, 4)]
)
@pytest.mark.parametrize("chunk_size", [0, 1, 3, 99])
def test_batches_match_independent_frames_and_preserve_input(
    node, controls, shape, chunk_size
):
    image = torch.rand(shape, dtype=torch.float64)
    image *= torch.linspace(0.05, 1.0, shape[0]).reshape(-1, 1, 1, 1)
    if shape[-1] == 4:
        image[..., 3] = torch.linspace(-0.5, 1.5, image[..., 3].numel()).reshape(
            shape[:3]
        )
    image = image.transpose(1, 2)  # Also exercise strided, noncontiguous IMAGE inputs.
    before = image.clone()
    version = image._version
    output = node.execute(image, *controls, "cpu", chunk_size).result[0]
    singles = torch.cat(
        [
            node.execute(frame.unsqueeze(0), *controls, "cpu", 1).result[0]
            for frame in image
        ]
    )
    torch.testing.assert_close(output, singles, rtol=1e-6, atol=2e-7)
    assert output.shape == image.shape
    assert output.dtype == torch.float32
    assert torch.equal(image, before)
    assert image._version == version
    if shape[-1] == 4:
        assert torch.equal(output[..., 3], image[..., 3].float())


@pytest.mark.parametrize("use_defaults", [True, False], ids=["defaults", "explicit"])
def test_normal_three_node_chain_matches_frame_processing(runtime, use_defaults):
    image = torch.rand(
        (67, 3, 7, 4), dtype=torch.float64, generator=torch.Generator().manual_seed(22)
    )
    image *= torch.linspace(0.05, 1.0, len(image)).reshape(-1, 1, 1, 1)
    before = image.clone()
    options = (
        [{}, {}, {}]
        if use_defaults
        else [
            {"output_device": "gpu", "batch_size": 2},
            {"output_device": "gpu", "batch_size": 3},
            {"output_device": "cpu", "batch_size": 2},
        ]
    )

    def chain(frames):
        frames = ColorTemperature.execute(frames, -25, **options[0]).result[0]
        frames = ColorBalance.execute(frames, 0.3, -0.2, 0.1, **options[1]).result[0]
        return BrightnessContrastV2.execute(frames, 1.2, 1.1, 0.9, **options[2]).result[
            0
        ]

    output = chain(image)
    if use_defaults:
        assert [progress.updates for progress in runtime.progress] == [[64, 67]] * 3
    singles = torch.cat([chain(frame[None]) for frame in image])
    torch.testing.assert_close(output, singles)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert torch.equal(output[..., 3], image[..., 3].float())
    assert torch.equal(image, before)


def test_processing_stays_float32_inside_autocast():
    image = torch.rand((2, 3, 4, 3), generator=torch.Generator().manual_seed(9))
    expected = ColorBalance.execute(image, 0.4, -0.2, 0.1).result[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = ColorBalance.execute(image, 0.4, -0.2, 0.1).result[0]
    assert result.dtype == torch.float32
    assert torch.equal(result, expected)
