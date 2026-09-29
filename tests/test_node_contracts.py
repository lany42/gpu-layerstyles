# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Contracts every node shares with ComfyUI, checked once across all nodes."""

import asyncio
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from comfy_api.latest import ComfyExtension, io

from gpu_layerstyles import comfy_entrypoint
from gpu_layerstyles.nodes.batch_concat import BatchConcat
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

# Saved workflows reference these IDs.
NODE_IDS = [
    "GPULayerStyles_ColorBalance",
    "GPULayerStyles_BrightnessContrastV2",
    "GPULayerStyles_ColorTemperature",
    "GPULayerStyles_ColorMatch",
    "GPULayerStyles_ImageScaleDown",
    "GPULayerStyles_SliceImageBatch",
    "GPULayerStyles_BatchConcat",
    "GPULayerStyles_CrossFade",
    "GPULayerStyles_TwoBatchLoop",
    "GPULayerStyles_TwoBatchBridge",
]

# Saved workflows store widget values by position and validate them against
# these ranges and options, so narrowing or reordering them breaks old graphs.
# Numbers are (type, default, min, max, step): defaults set what a newly added
# node does, and steps set how its widgets round values.
LIMIT = 2**31 - 1
EXECUTION = [("output_device", ["gpu", "cpu"]), ("batch_size", ("INT", 0, 0, LIMIT, 1))]
LAYOUTS = {
    ColorBalance: [
        ("image", "IMAGE"),
        ("cyan_red", ("FLOAT", 0, -1, 1, 0.001)),
        ("magenta_green", ("FLOAT", 0, -1, 1, 0.001)),
        ("yellow_blue", ("FLOAT", 0, -1, 1, 0.001)),
        *EXECUTION,
    ],
    BrightnessContrastV2: [
        ("image", "IMAGE"),
        ("brightness", ("FLOAT", 1, 0, 3, 0.01)),
        ("contrast", ("FLOAT", 1, 0, 3, 0.01)),
        ("saturation", ("FLOAT", 1, 0, 3, 0.01)),
        *EXECUTION,
    ],
    ColorTemperature: [
        ("image", "IMAGE"),
        ("temperature", ("FLOAT", 0, -100, 100, 1)),
        *EXECUTION,
    ],
    ColorMatch: [
        ("image", "IMAGE"),
        ("image_ref", "IMAGE"),
        ("method", ["mkl", "mvgd"]),
        ("strength", ("FLOAT", 1, 0, 1, 0.01)),
        *EXECUTION,
    ],
    ImageScaleDown: [
        ("image", "IMAGE"),
        ("width", ("INT", 512, 1, 16384, 1)),
        ("height", ("INT", 512, 1, 16384, 1)),
        ("method", ["bicubic", "lanczos"]),
        *EXECUTION,
    ],
    SliceImageBatch: [("image", "IMAGE"), ("slice", str)],
    BatchConcat: [("images", [f"image_{index}" for index in range(1, 101)])],
    CrossFade: [
        ("images_1", "IMAGE"),
        ("images_2", "IMAGE"),
        ("start_index", ("INT", 0, 0, LIMIT, 1)),
        ("frames", ("INT", 2, 2, LIMIT, 1)),
        *EXECUTION,
    ],
    TwoBatchLoop: [
        ("images_1", "IMAGE"),
        ("images_2", "IMAGE"),
        ("blend_target", ("INT", 15, 2, LIMIT, 1)),
        ("append_first_frame", bool),
        *EXECUTION,
    ],
    TwoBatchBridge: [("images", "IMAGE"), ("blend_target", ("INT", 15, 1, LIMIT, 1))],
}
# Multiple outputs are told apart only by their labels.
OUTPUTS = {TwoBatchBridge: ["bridge_first", "bridge_last", "first/last"]}


def describe(item):
    if isinstance(item, io.Image.Input):
        return "IMAGE"
    if isinstance(item, io.Autogrow.Input):
        assert isinstance(item.template.input, io.Image.Input)
        assert item.template.min == 1
        return item.template.names
    if isinstance(item, io.Combo.Input):
        return item.options
    if isinstance(item, io.Boolean.Input):
        return bool
    if isinstance(item, io.String.Input):
        return str
    kind = "INT" if isinstance(item, io.Int.Input) else "FLOAT"
    return (kind, item.default, item.min, item.max, item.step)


def test_extension_registers_every_node_under_its_persistent_id():
    extension = asyncio.run(comfy_entrypoint())
    assert isinstance(extension, ComfyExtension)
    nodes = asyncio.run(extension.get_node_list())
    assert [node.define_schema().node_id for node in nodes] == NODE_IDS


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
    nodes = asyncio.run(asyncio.run(module.comfy_entrypoint()).get_node_list())
    assert [node.define_schema().node_id for node in nodes] == NODE_IDS


@pytest.mark.parametrize("node", LAYOUTS, ids=lambda node: node.__name__)
def test_schema_matches_execute_and_saved_workflow_layout(node):
    schema = node.define_schema()
    parameters = inspect.signature(node.execute).parameters
    # ComfyUI calls execute with keyword arguments named by the schema.
    assert [item.id for item in schema.inputs] == list(parameters)
    for item in schema.inputs:
        default = parameters[item.id].default
        if default is not inspect.Parameter.empty:
            assert default == item.default, item.id
    assert [(item.id, describe(item)) for item in schema.inputs] == LAYOUTS[node]
    names = [output.display_name for output in schema.outputs]
    assert names == OUTPUTS[node] if node in OUTPUTS else len(names) == 1


@dataclass(frozen=True)
class Case:
    node: type
    sockets: dict[str, int]
    controls: dict = field(default_factory=dict)

    @property
    def processes(self):
        """Whether the node computes float32 output through the chunked executor."""
        return self.node not in (SliceImageBatch, BatchConcat, TwoBatchBridge)

    def images(self, dtype=torch.float32):
        generator = torch.Generator().manual_seed(7)
        # Transposed storage also exercises strided, noncontiguous IMAGE inputs.
        return {
            socket: torch.rand(frames, 5, 4, 4, generator=generator)
            .transpose(1, 2)
            .to(dtype)
            for socket, frames in self.sockets.items()
        }

    def run(self, images, **options):
        if self.node is BatchConcat:
            return BatchConcat.execute(images).result
        return self.node.execute(**images, **self.controls, **options).result


def case(node, sockets, label="", **controls):
    name = "-".join(filter(None, [node.__name__, controls.get("method"), label]))
    return pytest.param(Case(node, sockets, controls), id=name)


CASES = [
    case(ColorBalance, {"image": 3}, cyan_red=0.3, magenta_green=-0.2, yellow_blue=0.4),
    case(
        BrightnessContrastV2,
        {"image": 3},
        brightness=1.2,
        contrast=0.8,
        saturation=1.3,
    ),
    case(ColorTemperature, {"image": 3}, temperature=-37),
    # Neutral controls must still return a separate float32 copy.
    case(
        ColorBalance,
        {"image": 3},
        "neutral",
        cyan_red=0,
        magenta_green=0,
        yellow_blue=0,
    ),
    case(
        BrightnessContrastV2,
        {"image": 3},
        "neutral",
        brightness=1,
        contrast=1,
        saturation=1,
    ),
    case(ColorTemperature, {"image": 3}, "neutral", temperature=0),
    case(ColorMatch, {"image": 3, "image_ref": 1}, method="mkl", strength=0.7),
    case(ColorMatch, {"image": 3, "image_ref": 3}, method="mvgd"),
    case(ImageScaleDown, {"image": 3}, width=3, height=2, method="bicubic"),
    case(ImageScaleDown, {"image": 3}, width=3, height=2, method="lanczos"),
    case(CrossFade, {"images_1": 4, "images_2": 3}, start_index=1, frames=2),
    case(
        TwoBatchLoop,
        {"images_1": 5, "images_2": 6},
        blend_target=2,
        append_first_frame=True,
    ),
    # A full selection of a single image is still an independent copy.
    case(SliceImageBatch, {"image": 1}, slice="0:"),
    case(BatchConcat, {"image_1": 2, "image_2": 3}),
    case(TwoBatchBridge, {"images": 4}, blend_target=2),
]
PROCESSING = [param for param in CASES if param.values[0].processes]
# The error names the input: single-image executor nodes call theirs IMAGE.
SOCKETS = [
    pytest.param(
        param.values[0],
        socket,
        "IMAGE" if socket == "image" and param.values[0].processes else socket,
        id=f"{param.id}-{socket}",
    )
    for param in CASES
    if param.id not in ("ColorMatch-mvgd", "ImageScaleDown-lanczos")
    and not param.id.endswith("-neutral")
    for socket in param.values[0].sockets
]


@pytest.mark.parametrize("case", CASES)
def test_outputs_are_new_tensors_and_inputs_are_untouched(case, runtime):
    if not case.processes:
        # Batch utilities copy on the input's device without memory management.
        runtime.device = torch.device("cuda:2")
    inputs = {name: image.requires_grad_() for name, image in case.images().items()}
    before = {name: image.detach().clone() for name, image in inputs.items()}
    versions = {name: image._version for name, image in inputs.items()}
    outputs = case.run(inputs)
    storages = {image.untyped_storage().data_ptr() for image in inputs.values()}
    for output in outputs:
        assert output.dtype == torch.float32
        assert output.device.type == "cpu"
        assert output.is_contiguous()
        assert not output.requires_grad
        assert output.grad_fn is None
        assert output.untyped_storage().data_ptr() not in storages
        storages.add(output.untyped_storage().data_ptr())
        output.fill_(123)
    for name, image in inputs.items():
        assert torch.equal(image.detach(), before[name])
        assert image._version == versions[name]
    if not case.processes:
        assert runtime.free_requests == runtime.progress == []
        assert runtime.checks == 0


@pytest.mark.parametrize("case", CASES)
def test_executors_compute_in_float32_and_utilities_preserve_dtype(case):
    half = case.images(torch.float16)
    expected = case.run({name: image.float() for name, image in half.items()})
    outputs = case.run(half)
    for output, reference in zip(outputs, expected, strict=True):
        if case.processes:
            assert output.dtype == torch.float32
            assert torch.equal(output, reference)
        else:
            assert output.dtype == torch.float16
            assert torch.equal(output.float(), reference)


@pytest.mark.parametrize("case", PROCESSING)
def test_processing_stays_float32_under_autocast(case):
    inputs = case.images()
    expected = case.run(inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = case.run(inputs)
    assert all(map(torch.equal, outputs, expected))


@pytest.mark.parametrize("case", PROCESSING)
def test_output_device_and_batch_size_reach_the_executor(case, runtime, gpu_routing):
    inputs = case.images()
    default = case.run(inputs)[0]
    explicit = case.run(inputs, output_device="gpu", batch_size=2)[0]
    frames = len(default)
    assert [device for _, device, _ in gpu_routing.allocations] == [
        torch.device("cpu"),
        runtime.device,
    ]
    assert [progress.updates for progress in runtime.progress] == [
        [frames],
        [*range(2, frames, 2), frames],
    ]
    torch.testing.assert_close(explicit, default, rtol=1e-6, atol=2e-7)


@pytest.mark.parametrize(("case", "socket", "label"), SOCKETS)
def test_invalid_images_name_their_input_before_memory_management(
    case, socket, label, runtime
):
    inputs = case.images()
    inputs[socket] = torch.ones(2, 4, 5, 2)
    with pytest.raises(ValueError, match=f"^{label} must have nonempty shape"):
        case.run(inputs)
    assert runtime.free_requests == []
