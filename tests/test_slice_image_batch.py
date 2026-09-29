# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Public SliceImageBatch behavior against Python list selection."""

import pytest
import torch

from gpu_layerstyles.nodes.slice_image_batch import SliceImageBatch


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
        (":100", slice(None, 100)),
        ("-100:", slice(-100, None)),
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
    assert torch.equal(SliceImageBatch.execute(image, expression).result[0], expected)


@pytest.mark.parametrize("expression", [None, ":", " \t: \t", "::", " \t:: \t"])
def test_noop_slices_pass_through_the_original_tensor(expression):
    image = torch.rand(7, 2, 3, 4)
    arguments = () if expression is None else (expression,)
    assert SliceImageBatch.execute(image, *arguments).result[0] is image


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
def test_bounds_are_checked_before_python_can_clip(expression, field, upper):
    with pytest.raises(IndexError) as caught:
        SliceImageBatch.execute(torch.ones(100, 1, 1, 3), expression)
    message = str(caught.value)
    assert repr(expression) in message
    assert "batch of 100 images" in message
    assert f"-100 <= {field} <= {upper}" in message


@pytest.mark.parametrize(
    "expression",
    # int() accepts "1_0"; the parser requires plain signed decimal digits.
    [" ", "1:2:3:4", "1,2", "None", "1.0", "1_0", "::0", "1:2:-0"],
)
def test_rejects_malformed_slices_and_zero_stride(expression):
    with pytest.raises(ValueError) as caught:
        SliceImageBatch.execute(torch.ones(100, 1, 1, 3), expression)
    assert repr(expression) in str(caught.value)


def test_slice_must_be_a_string():
    with pytest.raises(TypeError, match="slice must be a string"):
        SliceImageBatch.execute(torch.ones(100, 1, 1, 3), 3)


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
    length, expression, error
):
    with pytest.raises(error) as caught:
        SliceImageBatch.execute(torch.ones(length, 1, 1, 3), expression)
    message = str(caught.value)
    assert repr(expression) in message
    assert f"batch of {length} images" in message
    if error is ValueError:
        assert "selects no images" in message


def test_missing_image_has_an_actionable_error():
    with pytest.raises(TypeError, match="image is None; connect"):
        SliceImageBatch.execute(None)
