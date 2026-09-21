# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Public ImageScaleDown behavior on real CPU tensors."""

import inspect

import pytest
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from gpu_layerstyles import _resize
from gpu_layerstyles.image_scale_down import ImageScaleDown


def test_schema_and_defaults():
    schema = ImageScaleDown.define_schema()
    names = ["image", "width", "height", "method", "output_device", "batch_size"]
    assert schema.node_id == "GPULayerStyles_ImageScaleDown"
    assert schema.display_name == "GPU LayerStyles ImageScaleDown"
    assert schema.category == "GPU LayerStyles/Image"
    assert [item.id for item in schema.inputs] == names
    assert list(inspect.signature(ImageScaleDown.execute).parameters) == names
    for name in ("define_schema", "execute"):
        assert isinstance(inspect.getattr_static(ImageScaleDown, name), classmethod)
    for field in schema.inputs[1:3]:
        assert (field.default, field.min, field.max, field.step) == (512, 1, 16384, 1)
    method, output, batch = schema.inputs[3:]
    assert (method.options, method.default) == (["bicubic", "lanczos"], "bicubic")
    assert (output.options, output.default) == (["gpu", "cpu"], "cpu")
    assert (batch.default, batch.min, batch.max, batch.step) == (0, 0, 2**31 - 1, 1)
    parameters = inspect.signature(ImageScaleDown.execute).parameters
    assert parameters["method"].default == "bicubic"
    assert parameters["output_device"].default == "cpu"
    assert parameters["batch_size"].default == 0
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "image"


@pytest.mark.parametrize("name", ["width", "height"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2", None])
def test_invalid_dimensions_fail_before_allocation(name, value, runtime):
    arguments = {"width": 2, "height": 2, name: value}
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        ImageScaleDown.execute(torch.ones(1, 7, 9, 3), **arguments)
    assert not runtime.free_requests


@pytest.mark.parametrize("width,height", [(10, 7), (9, 8), (10, 8), (10, 3), (2, 8)])
def test_rejects_either_upscaled_dimension(width, height, runtime):
    with pytest.raises(ValueError, match=f"source is 9x7, requested {width}x{height}"):
        ImageScaleDown.execute(torch.ones(1, 7, 9, 3), width, height)
    assert not runtime.free_requests


@pytest.mark.parametrize("method", ["nearest", "Lanczos", "", None])
def test_rejects_invalid_method_even_for_unchanged_size(method, runtime):
    with pytest.raises(ValueError, match="method"):
        ImageScaleDown.execute(torch.ones(1, 2, 3, 3), 3, 2, method)
    assert not runtime.free_requests


@pytest.mark.parametrize(
    "image,error",
    [
        (None, TypeError),
        (torch.ones(1, 2, 3, 3, dtype=torch.uint8), TypeError),
        (torch.ones(2, 3, 3), ValueError),
        (torch.ones(1, 2, 3, 1), ValueError),
        (torch.ones(0, 2, 3, 3), ValueError),
        (torch.ones(1, 0, 3, 3), ValueError),
        (torch.ones(1, 2, 3, 3).to_sparse(), ValueError),
    ],
)
def test_reuses_image_validation(image, error, runtime):
    with pytest.raises(error):
        ImageScaleDown.execute(image, 1, 1)
    assert not runtime.free_requests


@pytest.mark.parametrize(
    "options,match",
    [
        ({"output_device": "cuda"}, "output_device"),
        ({"batch_size": -1}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 1.5}, "batch_size"),
    ],
)
def test_reuses_execution_validation(options, match, runtime):
    with pytest.raises(ValueError, match=match):
        ImageScaleDown.execute(torch.ones(1, 3, 4, 3), 2, 2, **options)
    assert not runtime.free_requests


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("output_device", ["cpu", "gpu"])
def test_batches_match_frames_without_mutation_or_grad(
    method, channels, dtype, output_device
):
    image = torch.rand(5, 9, 11, channels, dtype=dtype) * 0.8 + 0.1
    image = image.transpose(1, 2).requires_grad_()
    before = image.detach().clone()
    version = image._version
    result = ImageScaleDown.execute(image, 6, 5, method, output_device, 2)
    assert isinstance(result, io.NodeOutput)
    output = result.result[0]
    singles = torch.cat(
        [ImageScaleDown.execute(frame[None], 6, 5, method).result[0] for frame in image]
    )
    torch.testing.assert_close(output, singles, rtol=1e-6, atol=2e-7)
    assert output.shape == (5, 5, 6, channels)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"  # Both settings respect ComfyUI CPU mode.
    assert not output.requires_grad
    assert output.grad_fn is None
    assert output.data_ptr() != image.data_ptr()
    assert torch.equal(image.detach(), before)
    assert image._version == version


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_fractional_colors_survive_without_uint8_quantization(method):
    colors = torch.tensor([0.5001, 0.123456, 0.876543])
    image = colors.expand(2, 13, 17, 3).clone()
    output = ImageScaleDown.execute(image, 7, 5, method).result[0]
    torch.testing.assert_close(output, colors.expand_as(output), rtol=0, atol=2e-7)
    assert torch.max(torch.abs(output * 255 - torch.round(output * 255))) > 0.4


@pytest.mark.parametrize("size", [(5, 7), (1, 1), (11, 3), (4, 9)])
def test_default_bicubic_matches_antialiased_torch(size):
    image = torch.rand(2, 9, 11, 3, generator=torch.Generator().manual_seed(21))
    width, height = size
    expected = (
        F.interpolate(
            image.movedim(-1, 1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .clamp(0, 1)
        .movedim(1, -1)
    )
    output = ImageScaleDown.execute(image, width, height).result[0]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_unchanged_dimensions_copy_without_filter_or_clamp(method, dtype, monkeypatch):
    image = (torch.rand(2, 3, 5, 4, dtype=dtype) * 2 - 0.5).requires_grad_()
    before = image.detach().clone()

    def unexpected(*args, **kwargs):
        pytest.fail("Unchanged sizes must not prepare or execute a resampler")

    monkeypatch.setattr(_resize, "_lanczos_coefficients", unexpected)
    monkeypatch.setattr(F, "interpolate", unexpected)
    output = ImageScaleDown.execute(image, 5, 3, method, batch_size=1).result[0]
    assert torch.equal(output, before.float())
    assert torch.equal(image.detach(), before)
    assert output.dtype == torch.float32
    assert output.data_ptr() != image.data_ptr()
    assert not output.requires_grad


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_transparent_colors_do_not_bleed_into_visible_edges(method):
    image = torch.zeros(1, 7, 19, 4)
    image[..., :9, 0] = 1
    image[..., :9, 3] = 1
    image[..., 9:, 1] = 1  # Hidden green next to opaque red.
    output = ImageScaleDown.execute(image, 8, 3, method).result[0]
    visible = output[..., 3] > 0
    assert torch.any((output[..., 3] > 0) & (output[..., 3] < 1))
    torch.testing.assert_close(
        output[..., 0][visible], torch.ones_like(output[..., 0][visible])
    )
    assert torch.count_nonzero(output[..., 1:3]) == 0
    assert torch.count_nonzero(output[..., :3][~visible]) == 0
    assert torch.all((output >= 0) & (output <= 1))


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_constant_color_survives_partial_alpha_and_fully_transparent_is_zero(method):
    image = torch.tensor([0.125, 0.5, 0.875, 0.0]).expand(2, 11, 17, 4).clone()
    image[0, ..., 3] = torch.linspace(0.2, 0.8, 17)
    output = ImageScaleDown.execute(image, 7, 5, method).result[0]
    torch.testing.assert_close(output[0, ..., :3], image[0, :5, :7, :3])
    assert torch.all((output[0, ..., 3] >= 0.2) & (output[0, ..., 3] <= 0.8))
    assert torch.count_nonzero(output[1]) == 0


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_opaque_rgba_matches_rgb(method):
    rgb = torch.rand(2, 13, 19, 3)
    rgba = torch.cat((rgb, torch.ones_like(rgb[..., :1])), dim=-1)
    expected = ImageScaleDown.execute(rgb, 8, 5, method).result[0]
    output = ImageScaleDown.execute(rgba, 8, 5, method).result[0]
    torch.testing.assert_close(output[..., :3], expected, rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(output[..., 3], torch.ones_like(output[..., 3]))


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_alpha_overshoot_is_clamped_after_unpremultiplication(method):
    color = torch.tensor([0.25, 0.5, 0.75])
    image = torch.cat((color, torch.zeros(1))).expand(1, 17, 29, 4).clone()
    image[..., 14:, 3] = 1
    output = ImageScaleDown.execute(image, 13, 7, method).result[0]
    visible = output[..., 3] > 0
    torch.testing.assert_close(
        output[..., :3][visible], color.expand_as(output[..., :3][visible])
    )
    assert output[..., 3].min() == 0
    assert output[..., 3].max() == 1


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_autocast_does_not_reduce_precision(method):
    image = torch.rand(2, 13, 17, 4)
    expected = ImageScaleDown.execute(image, 7, 5, method).result[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = ImageScaleDown.execute(image, 7, 5, method).result[0]
    assert output.dtype == torch.float32
    assert torch.equal(output, expected)


@pytest.mark.parametrize("capacity", [20, 1024])
def test_bounded_gathers_and_split_support_preserve_result(capacity, monkeypatch):
    image = torch.rand(2, 7, 11, 3)
    expected = ImageScaleDown.execute(image, 4, 3, "lanczos").result[0]
    original = torch.index_select
    sizes = []

    def index_select(*args, **kwargs):
        result = original(*args, **kwargs)
        sizes.append(result.numel() * result.element_size())
        assert sizes[-1] <= capacity
        return result

    monkeypatch.setattr(_resize, "_GATHER_BYTES", capacity)
    monkeypatch.setattr(torch, "index_select", index_select)
    output = ImageScaleDown.execute(image, 4, 3, "lanczos").result[0]
    assert len(sizes) > 2
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=2e-7)
