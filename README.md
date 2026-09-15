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

- **`output_device`:** `cpu` (default) processes float32 chunks on ComfyUI's selected
  compute device and copies each completed chunk into a preallocated CPU output.
  `gpu` keeps output on the selected compute device to avoid transfers between
  compatible nodes. In ComfyUI CPU mode, both settings return CPU tensors. New nodes
  and calls omitting this control use CPU output; saved workflows explicitly
  selecting `gpu` keep that placement.
- **`batch_size`:** `0` (default) starts with `min(input_frame_count, 64)` frames per
  chunk. A positive integer requests that many frames, capped only by the input
  length; values above 64 are supported. Allocation failures halve the failing
  chunk size down to one frame and retry the same frames. The reduced size applies
  for the rest of that invocation; each new invocation starts fresh.

Before allocating the destination, the executor asks ComfyUI to free memory for
working space estimated at eight times the initial chunk's FP32 size, plus the
complete output when it shares the compute device. This includes the CPU output
when ComfyUI runs in CPU mode. Automatic sizing starts at up to 64 frames regardless
of reported free memory and adapts through allocation retries. Progress and
cancellation are checked between chunks. Processing and output stay in float32,
with gradient tracking disabled.

If the complete output cannot fit, or processing a single frame still fails, the
node reports an error with steps to reduce memory use. Output placement is never
changed automatically.

### Complete batches and system RAM

CPU output stores complete batches in system RAM. A batch of **2048 frames
at 1024×1024, RGB float32** requires `2048 × 1024 × 1024 × 3 × 4` bytes:

| Tensors retained in CPU memory | RAM before other allocations |
| --- | --- |
| One complete batch | 24 GiB |
| Input and one output | 48 GiB |
| Original input and all three node outputs | 96 GiB |

Models, temporary tensors, other cached data, and the rest of ComfyUI require
additional memory. Internal chunk sizing controls working memory. Passing smaller
image batches through the entire workflow reduces complete-batch storage needs.
GPU output can speed up compatible node chains by avoiding CPU transfers, provided
the complete outputs fit in VRAM.

### ComfyUI cache controls

Current upstream ComfyUI defaults to RAM-pressure caching, which can evict cached
results as available system RAM falls below its thresholds. `--cache-ram` values
specify **free-memory headroom thresholds**, not maximum cache sizes: the first
sets the active-cache threshold, and the optional second sets the inactive-cache
and pin threshold. For example, `--cache-ram 8` requests 8 GB of active-cache
headroom; it does not limit the cache to 8 GB. See the
[upstream cache controls](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/cli_args.py#L123-L128).

The global launch option `--cache-none` reduces RAM/VRAM used for cached results at
the cost of recomputing every node on each run. It still requires memory for inputs
and outputs needed by the running workflow. Cache options, accepted arguments, and
defaults depend on the installed ComfyUI version; check `python main.py --help`
from your ComfyUI installation and update if the RAM-pressure options are missing.
This pack does not change global cache settings automatically or add a per-node
cache-eviction control.

## Development and verification

Create an isolated CPU test environment with Python 3.13 and `uv`:

```bash
uv python find
uv sync --locked
uv run --offline --locked ruff check --select I --fix .
uv run --offline --locked ruff check --fix .
uv run --offline --locked ruff format .
uv run --offline --locked ruff check .
uv run --locked pytest
uv lock --check
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
immutability, mixed-brightness batches, narrow/noncontiguous inputs, the 64-frame
automatic boundary, larger explicit chunks, allocation retries, default CPU and
explicit GPU routing, complete-output reservations, progress, cancellation, and
loader/schema registration. A three-node chain with default output settings is
compared against independent-frame processing.

The automated suite runs real CPU tensors with ComfyUI API/memory test doubles.
Device-routing tests simulate allocation requests; they do not execute CUDA.
Installation testing in ComfyUI, GPU execution, visual comparisons, and benchmarking
are manual follow-ups, including the Temperature → ColorBalance → Brightness
Contrast chain and a 32-versus-64-frame chunk speed comparison. No benchmark harness
or performance threshold is required for delivery.

## License

[AGPL-3.0-only](LICENSE).
