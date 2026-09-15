# GPU LayerStyles

Four independent ComfyUI V3 nodes for float32 color adjustments and matching on image batches:

| Display name | Node ID | Adjustment controls, in order |
| --- | --- | --- |
| GPU LayerStyles ColorBalance | `GPULayerStyles_ColorBalance` | `cyan_red`, `magenta_green`, `yellow_blue` |
| GPU LayerStyles Brightness Contrast V2 | `GPULayerStyles_BrightnessContrastV2` | `brightness`, `contrast`, `saturation` |
| GPU LayerStyles ColorTemperature | `GPULayerStyles_ColorTemperature` | `temperature` |
| GPU LayerStyles ColorMatch | `GPULayerStyles_ColorMatch` | `image_ref`, `method`, `strength` |

All four appear under **GPU LayerStyles/Color**. Each accepts a floating-point
ComfyUI `IMAGE` with shape `[B,H,W,3]` or `[B,H,W,4]` and returns the same shape in
float32. Use `B=1` for a single image. Empty dimensions, other shapes, and integer
tensors are rejected. RGB is adjusted; alpha is preserved after float32 conversion.
Inputs are never modified, and neutral controls preserve identity.

## Performance on image and video batches

Informal, user-reported benchmarks of a three-node video adjustment chain show
**97.6% less elapsed time (42.11× speedup)** with outputs kept on the GPU, compared
with upstream ComfyUI_LayerStyle. The same workload with CPU offloading showed
**83.7% less elapsed time (6.12× speedup)**.

In both comparisons below, **A** is GPU LayerStyles and **B** is upstream
ComfyUI_LayerStyle. Times are total **milliseconds**. Calculated in Python: elapsed
time reduction is `(B - A) / B × 100%`; speedup is `B / A`. Percentages are rounded
to one decimal place and speedups to two.

### Typical video workload: all three nodes chained

The input was a **720×1280 (720p) video with 525 frames**, equivalent to **8.75 s
at 60 fps**. All three nodes were chained together for each pack. Every run used a
**fresh workflow after clearing the execution cache**. GPU LayerStyles was measured
in two separate passes: CPU offloading on every node (`output_device="cpu"`), then
GPU output on every node (`output_device="gpu"`). Times cover the complete
three-node adjustment chain.

| Pipeline | Total time (ms) | Elapsed time reduction vs. upstream | Speedup vs. upstream |
| --- | ---: | ---: | ---: |
| A: CPU offloading | 11,157 | 83.7% | 6.12× |
| A: GPU output | 1,621 | 97.6% | 42.11× |
| B: upstream | 68,255 | 0.0% | 1.00× |

With GPU output, the reported chain time fell from **68.255 s to 1.621 s**, saving
**66.634 s per 525-frame pass**. CPU offloading saved **57.098 s**. Comparing the two
GPU LayerStyles passes directly, GPU output took **85.5% less time** than CPU
offloading, a **6.88× speedup** that saved another **9.536 s**.

Both GPU LayerStyles passes use GPU processing. Keeping outputs on the GPU avoids
repeated transfers to and from system RAM between compatible nodes, illustrating
the benefit of a GPU-only adjustment chain when complete outputs fit in VRAM. See
[output placement and memory](#output-placement-and-memory) for the tradeoffs.
The timing procedure did not specify GPU synchronization or whether a final
transfer to CPU was included.

### Individual nodes: 1024×1024 images

All inputs were **1024×1024 images (approximately 1 MP)**, with the same color
settings used for each pair of nodes. Times were measured from each node's input
to output. All GPU LayerStyles nodes used CPU offloading, reportedly with the
default chunk settings, likely up to 64 frames per chunk; actual chunk sizes were
not confirmed. The image counts below describe complete input batches.

| Images | Node | A: GPU LayerStyles (ms) | B: upstream (ms) | Elapsed time reduction | Speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | ColorTemperature | 8 | 16 | 50.0% | 2.00× |
| 1 | ColorBalance | 45 | 200 | 77.5% | 4.44× |
| 1 | BrightnessAndContrast | 8 | 22 | 63.6% | 2.75× |
| 256 | ColorTemperature | 2,897 | 3,698 | 21.7% | 1.28× |
| 256 | ColorBalance | 3,264 | 54,295 | 94.0% | 16.63× |
| 256 | BrightnessAndContrast | 3,399 | 5,465 | 37.8% | 1.61× |
| 1,024 | ColorBalance | 13,402 | 214,959 | 93.8% | 16.04× |

BrightnessAndContrast is the benchmark label for this pack's Brightness Contrast
V2 comparison. Only ColorBalance was measured at 1,024 images.

Across the batched ColorBalance runs, elapsed time fell by **93.8–94.0%**, a
**16.04–16.63× speedup**. For **256 frames**, ColorBalance fell from **54.295 s to
3.264 s**, saving **51.031 s**. At **1,024 frames**, it fell from **214.959 s to
13.402 s**, saving **201.557 s**. The 256-frame runs also reduced ColorTemperature
time by **21.7%** and BrightnessAndContrast time by **37.8%**.

### Measurement context

Single-image timings were reported to vary substantially. Hardware, software
versions, warm-up procedure, and repeat counts were not supplied, so the results
are indicative and may vary across setups. These measurements cover the adjustment
nodes; whole-workflow gains depend on how much time these adjustments contribute.

## Visual fidelity and float32 precision

All three nodes perform their color adjustments in **float32**, retaining
fractional RGB values through the output. In user comparisons, ColorBalance and
ColorTemperature appeared indistinguishable from upstream. Brightness Contrast V2
also matched visually when brightness, contrast, or saturation was adjusted alone.

With combined adjustments, GPU LayerStyles can appear slightly lighter at identical
settings. Upstream's Brightness Contrast V2 processes 8-bit Pillow images,
truncating fractional RGB values after each active adjustment and rounding grayscale
values and the contrast mean. These losses can accumulate into a small darkening
bias. Retaining the fractions improves numerical fidelity to the underlying
adjustment formulas; exact pixel parity with upstream's quantized output is not
expected.

A manual output comparison supported this analysis: GPU LayerStyles at brightness
**0.97**, contrast **1.11**, and saturation **1.10** closely matched upstream at
**0.98**, **1.10**, and **1.10**, respectively. Keeping `brightness × contrast`
approximately constant preserves RGB channel differences before clipping while
lowering brightness. This compensation depends on the image and is not a universal
mapping between the nodes.

## Install

Copy or clone this repository into `ComfyUI/custom_nodes/gpu-layerstyles`, then
restart ComfyUI. The root `__init__.py` loads the package from `src/gpu_layerstyles`
and registers exactly four nodes through `ComfyExtension`.

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
node-specific controls.

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
- **ColorMatch:** `image_ref` supplies one reference frame for the whole batch or
  one per target frame. `method` offers `mkl` (default) and `mvgd`; `strength`
  defaults to `1`, ranges from `0` to `1`, and steps by `0.01`. See the method
  matrix below.

All processing uses float32 without 8-bit conversion or integer grayscale rounding.
Small differences from an 8-bit image-editing pipeline are expected.

### ColorMatch methods and references

| Method | Transfer behavior | Reference dimensions |
| --- | --- | --- |
| `mkl` | Matches RGB means and covariance through Monge–Kantorovich linearization | Any nonempty height and width |
| `mvgd` | Analytical multivariate Gaussian transfer; also depends on corresponding pixel positions | Target and reference must have the same height **and** width |

The algorithms are adapted from Christopher Hahne's
[color-matcher 0.6.0](https://pypi.org/project/color-matcher/0.6.0/) to run in
PyTorch on ComfyUI's selected compute device. Matching uses float32 throughout;
upstream covariance calculations use NumPy's float64. Stable centering and
tolerances tuned for float32 account for rounding and images with little color
variation, so small numerical differences from upstream are expected.

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
working space estimated at eight times the initial chunk's FP32 size (target plus
reference for ColorMatch), plus the complete output when it shares the compute
device. This includes the CPU output when ComfyUI runs in CPU mode. Automatic sizing
starts at up to 64 frames regardless of reported free memory and adapts through
allocation retries. Progress and cancellation are checked between chunks.
Processing and output stay in float32, with gradient tracking disabled.

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
| Original input and three adjustment-node outputs | 96 GiB |

Models, temporary tensors, other cached data, and the rest of ComfyUI require
additional memory. Internal chunk sizing controls working memory. Passing smaller
image batches through the entire workflow reduces complete-batch storage needs.
GPU output can speed up compatible node chains by avoiding CPU transfers, provided
the complete outputs fit in VRAM.

ColorMatch also retains its complete reference input. For the example above, a
paired reference batch adds 24 GiB, bringing target, reference, and output to
72 GiB when all three are in system RAM. Chunk sizing does not reduce this storage.

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
SciPy, and `color-matcher==0.6.0` are used only by reference tests. These development
dependencies are excluded from the wheel's runtime requirements; ComfyUI supplies
its own PyTorch.

Tests compare dense ramps and extreme ColorBalance sliders against SciPy, and
enhancements against float references and Pillow with quantization tolerance. They
also cover curve cancellation near black, restoration of tiny positive adjusted
luminance, neutral identity, clipping, zero/near-zero luminance, alpha, input
immutability, mixed-brightness batches, narrow/noncontiguous inputs, the 64-frame
automatic boundary, larger explicit chunks, allocation retries, default CPU and
explicit GPU routing, complete-output reservations, progress, cancellation, and
loader/schema registration. A three-node chain with default output settings is
compared against independent-frame processing.

ColorMatch tests cover its public interface and compare MKL and MVGD against
color-matcher 0.6.0 using two upstream photographs and identical float32 inputs,
allowing minor numerical error. See
[fixture provenance and licensing](tests/data/color_matcher/README.md).

The automated suite runs real CPU tensors with ComfyUI API/memory test doubles.
Device-routing tests simulate allocation requests; they do not execute CUDA.
Installation testing in ComfyUI, GPU execution, visual comparisons, and benchmarking
are manual follow-ups, including the Temperature → ColorBalance → Brightness
Contrast chain and a 32-versus-64-frame chunk speed comparison. Synchronize the GPU
when timing runs. No benchmark harness or performance threshold is required for
delivery.

## License

Copyright © 2026 Lany Atwood <lany@colorized.life>. The project's Python source,
including the native PyTorch ColorMatch implementation and tests, is licensed
under [AGPL-3.0-only](LICENSE).

Christopher Hahne's color-matcher served as the original source for the MKL and
analytical MVGD algorithms and supplies the numerical reference used by our tests.
See [COPYRIGHT](COPYRIGHT) for project and original-source attribution.

The two vendored test photographs retain their upstream provenance and applicable
licensing; see the [fixture notices](tests/data/color_matcher/README.md). The
[original upstream GPL text](LICENSES/color-matcher-LICENSE) and attribution
notices ship alongside the project license in source and wheel distributions.
