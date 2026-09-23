# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""ColorMatch's public node interface; numerical comparisons live separately."""

import inspect

import pytest
import torch
from comfy_api.latest import io

from gpu_layerstyles.nodes.color_match import ColorMatch


def test_schema_and_execute_contract():
    schema = ColorMatch.define_schema()
    names = ["image", "image_ref", "method", "strength", "output_device", "batch_size"]
    assert [item.id for item in schema.inputs] == names
    assert list(inspect.signature(ColorMatch.execute).parameters) == names
    assert isinstance(inspect.getattr_static(ColorMatch, "execute"), classmethod)
    assert isinstance(inspect.getattr_static(ColorMatch, "define_schema"), classmethod)
    assert schema.node_id == "GPULayerStyles_ColorMatch"
    assert schema.display_name == "GPU LayerStyles ColorMatch"
    assert schema.category == "GPU LayerStyles/Color"
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "image"
    method, strength, output, batch = schema.inputs[2:]
    assert method.options == ["mkl", "mvgd"]
    assert method.default == "mkl"
    assert (strength.default, strength.min, strength.max, strength.step) == (
        1,
        0,
        1,
        0.01,
    )
    assert output.options == ["gpu", "cpu"]
    assert output.default == "cpu"
    assert (batch.default, batch.min, batch.max, batch.step) == (0, 0, 2**31 - 1, 1)
    parameters = inspect.signature(ColorMatch.execute).parameters
    assert [parameters[name].default for name in names[2:]] == ["mkl", 1, "cpu", 0]


@pytest.mark.parametrize(
    "options,message",
    [
        ({"method": "unknown"}, "method"),
        ({"strength": -0.1}, "strength"),
        ({"strength": 1.1}, "strength"),
        ({"strength": float("nan")}, "strength"),
        ({"image_ref": torch.zeros(3, 3, 5, 3)}, "one frame.*batch length"),
        ({"method": "mvgd", "image_ref": torch.zeros(1, 5, 3, 3)}, "height and width"),
    ],
)
def test_invalid_controls_and_reference_structure(options, message, runtime):
    inputs = {"image": torch.zeros(2, 3, 5, 3), "image_ref": torch.zeros(1, 3, 5, 3)}
    inputs.update(options)
    with pytest.raises(ValueError, match=message):
        ColorMatch.execute(**inputs)
    assert not runtime.free_requests


@pytest.mark.parametrize("output_device", ["cpu", "gpu"])
def test_image_output_contract_and_placement(output_device, runtime, gpu_routing):
    image = torch.rand(2, 3, 5, 4, dtype=torch.float64).requires_grad_()
    reference = torch.rand(1, 3, 5, 3)
    before = image.detach().clone(), reference.clone()
    result = ColorMatch.execute(image, reference, output_device=output_device)
    assert isinstance(result, io.NodeOutput)
    assert len(result.result) == 1
    output = result.result[0]
    assert output.shape == image.shape
    assert output.dtype == torch.float32
    assert not output.requires_grad
    assert output.data_ptr() != image.data_ptr()
    assert torch.equal(output[..., 3], image[..., 3].float())
    assert torch.equal(image, before[0])
    assert torch.equal(reference, before[1])
    destination = runtime.device if output_device == "gpu" else torch.device("cpu")
    assert gpu_routing.allocations == [(tuple(image.shape), destination, torch.float32)]
