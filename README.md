# GPU LayerStyles

Three independent ComfyUI V3 nodes for float32 color adjustments on image batches:

| Display name | Node ID | Adjustment controls, in order |
| --- | --- | --- |
| GPU LayerStyles ColorBalance | `GPULayerStyles_ColorBalance` | `cyan_red`, `magenta_green`, `yellow_blue` |
| GPU LayerStyles Brightness Contrast V2 | `GPULayerStyles_BrightnessContrastV2` | `brightness`, `contrast`, `saturation` |
| GPU LayerStyles ColorTemperature | `GPULayerStyles_ColorTemperature` | `temperature` |

All three appear under **GPU LayerStyles/Color**. Each accepts a floating-point
ComfyUI `IMAGE` with shape `[B,H,W,3]` or `[B,H,W,4]` and returns the same shape in
float32. Use `B=1` for a single image. Empty dimensions, other shapes, and integer
tensors are rejected. RGB is adjusted; alpha is preserved after float32 conversion.
Inputs are never modified, and neutral controls preserve identity.

## Install

Copy or clone this repository into `ComfyUI/custom_nodes/gpu-layerstyles`, then
restart ComfyUI. The root `__init__.py` loads the package from `src/gpu_layerstyles`
and registers exactly three nodes through `ComfyExtension`.

Use a recent ComfyUI with the V3 API (`comfy_api.latest`). The target environment is
comfpod's Python 3.13, PyTorch 2.14.0, and CUDA 13.0. The nodes use ComfyUI's installed
PyTorch; there are no additional runtime dependencies to install. ComfyUI's selected
compute device is used, including CPU when ComfyUI runs in CPU mode.

Source archives in `dist/` can also be extracted into that custom-node directory.
The wheel provides the Python package. For a wheel installation in ComfyUI's Python
environment, add `custom_nodes/gpu-layerstyles/__init__.py` containing:

```python
from gpu_layerstyles import comfy_entrypoint
```

Restart ComfyUI after adding the loader. Installing a wheel alone does not create a
custom-node directory.

## Controls and behavior

Every node takes `image` first and appends `output_device` and `batch_size` after its
adjustment controls.

- **ColorBalance:** all three sliders default to `0`, range from `-1` to `1`, and
  step by `0.001`. Each RGB slider controls three successive curves: shadows,
  midtones, then highlights. Each curve evaluates
  `clamp(x + slider * maximum * x * (1-x) / (center * (1-center)), 0, 1)`, with
  `(center, maximum)` values `(0.15, 0.1)`, `(0.5, 1.0)`, and `(0.8, 0.2)`.
  The equivalent polynomial is evaluated to retain small positive values near
  black. Luminosity is restored using weights `(0.2126, 0.7152, 0.0722)` and the
  actual positive adjusted luminance, followed by clipping. Fully black adjusted
  pixels remain black. All-zero sliders skip the curves and luminosity restoration.
- **Brightness Contrast V2:** brightness, contrast, and saturation default to `1`,
  range from `0` to `3`, and step by `0.01`. Brightness multiplies RGB. Contrast is
  centered on each frame's mean grayscale luminance after brightness. Saturation
  is centered on each pixel's grayscale luminance after contrast. Each active stage
  clamps RGB to `[0,1]`. Grayscale weights are `(0.299, 0.587, 0.114)`; frames are
  never averaged together.
- **ColorTemperature:** temperature defaults to `0`, ranges from `-100` to `100`,
  and steps by `1`. Negative values warm the image; positive values cool it. With
  `t = -temperature / 100`, positive `t` multiplies red by `1+t` and green by
  `1+0.4t`; negative `t` multiplies red by `1+0.2t` and blue by `1-t`. RGB is then
  clamped to `[0,1]`. Zero skips the adjustment.

All processing uses float32 without 8-bit conversion or integer grayscale rounding.
Small differences from an 8-bit image-editing pipeline are expected.

## Output placement and memory

- **`output_device`:** `gpu` (default) keeps output on ComfyUI's selected compute
  device. `cpu` transfers each finished chunk directly into the CPU destination.
  In ComfyUI CPU mode, both settings return CPU tensors.
- **`batch_size`:** `0` (default) chooses a chunk size automatically. A positive
  integer requests that many frames per chunk, limited by the number of input
  frames. Allocation failures halve the chunk size down to one frame.

The executor asks ComfyUI to free memory before allocation, including space for the
complete output when it resides on the compute device. Automatic chunks use an
eight-times-frame-size working estimate and up to 25% of remaining available memory,
capped at 512 MiB and 32 frames, with a one-frame minimum. The destination is
preallocated and filled by chunks. Progress and cancellation are checked between
chunks. Processing uses ordinary PyTorch operations with gradient tracking disabled.

If the complete output cannot fit, or processing a single frame still fails, the
node reports an error with steps to reduce memory use. Output placement is never
changed automatically.

ComfyUI can cache complete GPU batches. Chunking bounds temporary allocations;
cached outputs still occupy memory. A float32 RGB batch of **240 frames at
720×1280** occupies about **2.47 GiB** per output (RGBA: **3.30 GiB**), before
temporary tensors and other cached data.

For the usual chain, set **ColorTemperature → ColorBalance** to GPU output. Set
**Brightness Contrast V2** to CPU output when its downstream consumer requires CPU
tensors. Free cached outputs, reduce the input batch/resolution, or choose CPU
output if the complete GPU destination cannot fit.

## Development and verification

Create an isolated CPU test environment with Python 3.13 and `uv`:

```bash
uv sync --locked --group dev
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
uv build
```

The `dev` dependency group in `pyproject.toml` declares PyTorch 2.14.0, and
`tool.uv.sources` selects its CPU wheel index. `uv.lock` records the resolved test
dependencies, so a fresh clone needs only the sync command above. NumPy, Pillow,
and SciPy are used only by reference tests. These development dependencies are
excluded from the wheel's runtime requirements; ComfyUI supplies its own PyTorch.

Tests compare dense ramps and extreme ColorBalance sliders against SciPy, and
enhancements against float references and Pillow with quantization tolerance. They
also cover curve cancellation near black, restoration of tiny positive adjusted
luminance, neutral identity, clipping, zero/near-zero luminance, alpha, input
immutability, mixed-brightness batches, narrow/noncontiguous inputs, partial chunks,
allocation retries, progress, cancellation, and loader/schema registration.

The automated suite runs real CPU tensors with ComfyUI API/memory test doubles.
Device-routing tests simulate allocation requests; they do not execute CUDA.
Installation testing in ComfyUI, GPU execution, visual comparisons, and benchmarking
are manual follow-ups, including the 240-frame Temperature → ColorBalance →
Brightness Contrast chain. No benchmark harness is included.

## License

[AGPL-3.0-only](LICENSE).
