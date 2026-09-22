# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Small ComfyUI test doubles; these tests exercise real CPU PyTorch tensors.

ComfyUI installation and CUDA execution are checked separately in the host app.
"""

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch


@dataclass
class ImageInput:
    id: str


@dataclass
class ImageOutput:
    display_name: str | None = None


@dataclass
class NumberInput:
    id: str
    default: float | int
    min: float | int
    max: float | int
    step: float | int
    tooltip: str | None = None


@dataclass
class ComboInput:
    id: str
    options: list[str]
    default: str
    tooltip: str | None = None


@dataclass
class StringInput:
    id: str
    default: str
    multiline: bool = False
    dynamic_prompts: bool | None = None
    tooltip: str | None = None


@dataclass
class Schema:
    node_id: str
    display_name: str
    category: str
    description: str
    inputs: list
    outputs: list


class ComfyNode:
    pass


class ComfyExtension:
    pass


class NodeOutput:
    def __init__(self, *args):
        self.result = args


class InterruptProcessingException(BaseException):
    pass


comfy = ModuleType("comfy")
management = ModuleType("comfy.model_management")
utils = ModuleType("comfy.utils")
api = ModuleType("comfy_api")
latest = ModuleType("comfy_api.latest")
latest.ComfyExtension = ComfyExtension
latest.io = SimpleNamespace(
    ComfyNode=ComfyNode,
    Schema=Schema,
    Image=SimpleNamespace(Input=ImageInput, Output=ImageOutput),
    Float=SimpleNamespace(Input=NumberInput),
    Int=SimpleNamespace(Input=NumberInput),
    Combo=SimpleNamespace(Input=ComboInput),
    String=SimpleNamespace(Input=StringInput),
    NodeOutput=NodeOutput,
)
comfy.model_management = management
comfy.utils = utils
api.latest = latest
# Imports during collection need a ProgressBar; the fixture sets its behavior.
utils.ProgressBar = lambda total: None
sys.modules.update(
    {
        "comfy": comfy,
        "comfy.model_management": management,
        "comfy.utils": utils,
        "comfy_api": api,
        "comfy_api.latest": latest,
    }
)


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    from gpu_layerstyles import _execution

    state = SimpleNamespace(
        device=torch.device("cpu"),
        available=8 * 1024**3,
        free_requests=[],
        cache_clears=0,
        checks=0,
        interrupt_on=None,
        progress=[],
    )

    def check_interrupt():
        state.checks += 1
        if state.checks == state.interrupt_on:
            raise InterruptProcessingException

    def empty_cache():
        state.cache_clears += 1

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.updates = []
            state.progress.append(self)

        def update_absolute(self, value):
            self.updates.append(value)

    monkeypatch.setattr(
        management, "get_torch_device", lambda: state.device, raising=False
    )
    monkeypatch.setattr(
        management, "get_free_memory", lambda device: state.available, raising=False
    )
    monkeypatch.setattr(
        management,
        "free_memory",
        lambda amount, device: state.free_requests.append((amount, device)),
        raising=False,
    )
    monkeypatch.setattr(management, "soft_empty_cache", empty_cache, raising=False)
    monkeypatch.setattr(
        management,
        "is_oom",
        lambda error: isinstance(error, torch.OutOfMemoryError),
        raising=False,
    )
    monkeypatch.setattr(
        management,
        "throw_exception_if_processing_interrupted",
        check_interrupt,
        raising=False,
    )
    monkeypatch.setattr(_execution, "ProgressBar", ProgressBar)
    return state


@pytest.fixture
def gpu_routing(runtime, monkeypatch):
    """Record GPU routing requests while running tensor operations on real CPU tensors."""
    runtime.device = torch.device("cuda:2")
    state = SimpleNamespace(allocations=[], transfers=[])
    original_empty = torch.empty
    original_to = torch.Tensor.to

    def empty(shape, *, device, dtype):
        assert runtime.free_requests
        state.allocations.append((tuple(shape), device, dtype))
        return original_empty(shape, dtype=dtype, device="cpu")

    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == runtime.device:
            state.transfers.append((len(tensor), kwargs["device"], kwargs["dtype"]))
            kwargs["device"] = torch.device("cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    return state
