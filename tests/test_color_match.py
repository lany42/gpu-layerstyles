# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""ColorMatch controls, reference modes, degenerate statistics, and cleanup.

Numerical parity with color-matcher lives in test_color_match_reference.py.
"""

import weakref

import pytest
import torch

from gpu_layerstyles.nodes import color_match as color_match_node
from gpu_layerstyles.nodes.color_match import ColorMatch

from .conftest import InterruptProcessingException

METHODS = ["mkl", "mvgd"]


def rand(*shape, seed=5):
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize(
    "options,message",
    [
        ({"method": "unknown"}, "method"),
        ({"strength": -0.1}, "strength"),
        ({"strength": 1.1}, "strength"),
        ({"strength": float("nan")}, "strength"),
        ({"image_ref": torch.zeros(3, 3, 5, 3)}, "one frame.*batch length"),
        ({"method": "mvgd", "image_ref": torch.zeros(1, 5, 3, 3)}, "height and width"),
        (
            {"method": "mvgd", "strength": 0.0, "image_ref": torch.zeros(1, 5, 3, 3)},
            "height and width",
        ),
    ],
)
def test_invalid_controls_and_reference_structure(options, message, runtime):
    inputs = {"image": torch.zeros(2, 3, 5, 3), "image_ref": torch.zeros(1, 3, 5, 3)}
    inputs.update(options)
    with pytest.raises(ValueError, match=message):
        ColorMatch.execute(**inputs)
    assert not runtime.free_requests


@pytest.mark.parametrize("socket", ["image", "image_ref"])
def test_active_matching_rejects_nonfinite_rgb(socket):
    inputs = {"image": rand(2, 3, 5, 3), "image_ref": rand(1, 3, 5, 3)}
    inputs[socket][0, 1, 2, 0] = float("nan")
    with pytest.raises(ValueError, match="finite RGB"):
        ColorMatch.execute(**inputs)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("references", [1, 3], ids=["shared", "per-frame"])
def test_references_follow_their_targets_across_chunks(method, references):
    image, image_ref = rand(3, 4, 5, 4), rand(references, 4, 5, 3, seed=6)
    output = ColorMatch.execute(image, image_ref, method, 0.8, batch_size=2).result[0]
    singles = torch.cat(
        [
            ColorMatch.execute(frame[None], reference[None], method, 0.8).result[0]
            for frame, reference in zip(image, image_ref.expand(3, -1, -1, -1))
        ]
    )
    torch.testing.assert_close(output, singles, rtol=1e-6, atol=2e-7)
    assert torch.equal(output[..., 3], image[..., 3])


@pytest.mark.parametrize("method", METHODS)
def test_zero_strength_copies_the_target_exactly(method):
    image = rand(2, 3, 5, 4) * 2 - 0.5  # Includes values outside [0, 1].
    output = ColorMatch.execute(image, rand(1, 3, 5, 3, seed=6), method, 0.0)
    assert torch.equal(output.result[0], image)


@pytest.mark.parametrize(
    "references,strength,transfers",
    [(1, 1.0, [1, 2, 2]), (4, 1.0, [2, 2, 2, 2]), (1, 0.0, [2, 2])],
    ids=["singleton", "paired", "inactive"],
)
def test_references_reach_the_compute_device_once_per_use(
    references, strength, transfers, runtime, gpu_routing
):
    image, image_ref = rand(4, 3, 5, 3), rand(references, 3, 5, 3, seed=6)
    ColorMatch.execute(image, image_ref, strength=strength, batch_size=2)
    # One reference is prepared once per run; strength 0 prepares none.
    assert gpu_routing.transfers == [
        (count, runtime.device, torch.float32) for count in transfers
    ]
    frame = 3 * 5 * 3 * 4
    reference_frames = 1 if references == 1 else 2
    # The CPU output is not reserved on the compute device.
    assert runtime.free_requests == [
        (8 * (2 + reference_frames) * frame, runtime.device)
    ]


def test_singleton_reference_is_reused_across_oom_retries(monkeypatch):
    image, image_ref = rand(4, 3, 5, 3), rand(1, 3, 5, 3, seed=6)
    expected = ColorMatch.execute(image, image_ref).result[0]
    prepared, attempts = [], []
    original_prepare = color_match_node.prepare_reference
    original_match = color_match_node.color_match

    def prepare(rgb, method):
        prepared.append(len(rgb))
        return original_prepare(rgb, method)

    def match(rgb, reference, method, strength):
        attempts.append(len(rgb))
        if len(rgb) > 1:
            raise torch.OutOfMemoryError("simulated target allocation failure")
        return original_match(rgb, reference, method, strength)

    monkeypatch.setattr(color_match_node, "prepare_reference", prepare)
    monkeypatch.setattr(color_match_node, "color_match", match)
    output = ColorMatch.execute(image, image_ref).result[0]
    assert attempts == [4, 2, 1, 1, 1, 1]
    assert prepared == [1]
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=2e-7)


@pytest.mark.parametrize(
    "error_type", [torch.OutOfMemoryError, InterruptProcessingException]
)
def test_retained_terminal_exception_releases_the_prepared_reference(
    error_type, monkeypatch
):
    references = []
    original = color_match_node.prepare_reference

    def prepare(rgb, method):
        result = original(rgb, method)
        references.extend((weakref.ref(result.mean), weakref.ref(result.covariance)))
        return result

    def fail(rgb, reference, method, strength):
        raise error_type("simulated terminal failure")

    monkeypatch.setattr(color_match_node, "prepare_reference", prepare)
    monkeypatch.setattr(color_match_node, "color_match", fail)
    expected_error = (
        RuntimeError if error_type is torch.OutOfMemoryError else error_type
    )
    with pytest.raises(expected_error) as caught:
        ColorMatch.execute(rand(1, 3, 5, 3), rand(1, 3, 5, 3, seed=6))
    assert caught.value is not None  # Keep its traceback alive for the assertions.
    assert references and all(reference() is None for reference in references)


@pytest.mark.parametrize("method", METHODS)
def test_constant_target_takes_the_reference_mean(method):
    image = torch.tensor([0.1, 0.7, 0.3]).expand(1, 13, 17, 3).clone()
    image_ref = rand(1, 13, 17, 3)
    output = ColorMatch.execute(image, image_ref, method).result[0]
    mean = image_ref.mean(dim=(1, 2), keepdim=True)
    torch.testing.assert_close(output, mean.expand_as(output))


@pytest.mark.parametrize(
    "method,height,width", [("mkl", 13, 17), ("mvgd", 13, 17), ("mkl", 1, 1)]
)
def test_constant_reference_paints_its_color(method, height, width):
    # A one-pixel swatch has zero covariance without a sample to divide by.
    color = torch.tensor([0.1, 0.7, 0.3])
    image_ref = color.expand(1, height, width, 3).clone()
    output = ColorMatch.execute(rand(2, 13, 17, 3), image_ref, method).result[0]
    torch.testing.assert_close(output, color.expand_as(output))


@pytest.mark.parametrize("method", METHODS)
def test_gray_images_stay_gray_with_the_reference_tone(method):
    # Rank-one covariances exercise the rank tolerance on both sides.
    image = rand(1, 13, 17, 1).expand(-1, -1, -1, 3)
    image_ref = (rand(1, 13, 17, 1, seed=6) * 0.4 + 0.3).expand(-1, -1, -1, 3)
    output = ColorMatch.execute(image, image_ref, method).result[0]
    torch.testing.assert_close(output, output[..., :1].expand_as(output))
    torch.testing.assert_close(output.mean(), image_ref.mean())
    if method == "mkl":
        torch.testing.assert_close(output[..., 0].std(), image_ref[..., 0].std())


@pytest.mark.parametrize("method", METHODS)
def test_rounding_noise_in_a_gray_target_is_not_amplified(method):
    # The rank tolerance treats one-ulp channel differences as noise, not color.
    gray = rand(1, 13, 17, 1).expand(-1, -1, -1, 3).clone()
    noisy = gray.clone()
    noisy[..., 1] = torch.nextafter(noisy[..., 1], torch.ones(()))
    noisy[..., 2] *= 1 - 2**-23
    image_ref = rand(1, 13, 17, 3, seed=6)
    expected = ColorMatch.execute(gray, image_ref, method).result[0]
    output = ColorMatch.execute(noisy, image_ref, method).result[0]
    torch.testing.assert_close(output, expected, rtol=0, atol=2e-6)
