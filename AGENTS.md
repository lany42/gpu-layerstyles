Do NOT update the README unless explicitly requested.

# Development workflow

Run commands from the repository root.

```sh
# Install Python per .python-version with your OS package manager, then check selection.
uv python find
# Install the locked development environment, including CPU PyTorch for tests.
# Rerun after dependency updates.
uv sync --locked
# Add or remove a development dependency.
uv add --dev package
uv remove --dev package
# Resolve changes; use --upgrade or --upgrade-package package to update locked versions.
uv lock

# After every Python task, run these checks in order.
# Sort and format import blocks.
uv run --offline --locked ruff check --select I --fix .
# Apply safe lint fixes across the project.
uv run --offline --locked ruff check --fix .
# Format Python code.
uv run --offline --locked ruff format .
# Check lint rules, including import ordering.
uv run --offline --locked ruff check .
# Run the CPU correctness tests.
uv run --locked pytest
# Verify the lockfile matches project metadata.
uv lock --check
# Build source and wheel distributions.
uv build
```

Use Conventional Commits: `<type>[optional scope][!]: <summary>`. Write an
imperative subject of at most 50 characters, with no trailing period. Follow
it with one blank line and a single short paragraph explaining what changed
and why in plain language. Limit the body to four lines, each at most 72
characters; avoid lists and exhaustive change logs.

## Dependencies and packaging

- ComfyUI supplies runtime PyTorch and `comfy_api`. Do not add a PyTorch runtime
  dependency that could replace ComfyUI's configured CUDA build.
- Keep CPU PyTorch and the NumPy, Pillow, SciPy, and color-matcher reference-test
  dependencies in the `dev` group. Keep the explicit CPU wheel index for test PyTorch.
- Update `pyproject.toml` and `uv.lock` together. There are currently no additional
  runtime dependencies and no `requirements.txt`. If additional runtime dependencies
  are introduced, maintain `requirements.txt` by hand to match only
  `[project].dependencies` and their constraints. Never generate it from `uv.lock`
  or `uv export`; commit affected dependency files together.
- Keep the implementation in `src/gpu_layerstyles` and retain the root ComfyUI
  loader for clone and ZIP installations. Include the loader, tests, lockfile, and
  this file in source distributions.

## Licensing and attribution

- Begin every project Python source file, including the root loader and tests,
  with these headers:

  ```python
  # SPDX-License-Identifier: AGPL-3.0-only
  # SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
  ```

- Keep project-code and original-algorithm-source attribution in `COPYRIGHT`,
  and include it in source and wheel distributions. Qualify color-matcher's
  attribution as the original algorithm source. Preserve the upstream fixture
  provenance and license notices separately from the project's Python code.

## Node behavior and validation

- Register exactly five V3 nodes through `ComfyExtension`, with their existing
  node IDs and control names, order, ranges, steps, and adjustment defaults.
- Default `output_device` to `"cpu"` in the shared schema, all five node methods,
  and the executor; retain option order `["gpu", "cpu"]`. CPU output still processes
  chunks on ComfyUI's selected compute device and copies them into a preallocated
  CPU destination. Explicit GPU output stays on the selected compute device.
- Keep `batch_size=0` as the default and start automatic chunks at
  `min(input_frame_count, 64)`. Positive sizes are capped only by input length and
  can exceed 64. On allocation failure, release failed temporaries, halve the
  failing chunk count to a minimum of one, and retry the same frames. Keep the
  reduced size for the rest of that invocation; start fresh on the next invocation.
- Process RGB in float32 and leave input tensors untouched. Color nodes preserve
  alpha unchanged. ImageScaleDown filters premultiplied RGB and alpha, restores
  straight RGBA, and clamps only the final resize. Keep contrast statistics local
  to each frame and skip neutral adjustments.
- ImageScaleDown accepts exact positive integer dimensions and rejects either
  dimension exceeding the source. Bicubic uses Torch antialiasing with
  `align_corners=False`; Lanczos-3 follows Pillow's floating-point filter and
  boundary normalization. Keep gather buffers bounded and coefficients local to
  the invocation. Unchanged sizes return a float32 copy without filtering or
  clamping. Keep Pillow's source attribution and license notices in distributions.
- Ask ComfyUI to free memory before allocating working space and outputs,
  accounting for all inputs and the output device. Preserve output placement,
  allocation retries, clear memory errors, cancellation, and progress.
- Preserve ColorMatch's documented methods and reference requirements. Validate
  its public behavior and compare representative images against upstream, allowing
  minor float32 numerical differences.
- Use ComfyUI's existing global cache controls for accumulated outputs. Do not
  change global cache settings automatically or add per-node cache-eviction or
  precision controls. Document complete-batch RAM needs and distinguish internal
  chunk sizing from passing smaller batches through the entire workflow.
- Automated tests use real CPU tensors with ComfyUI API and memory test doubles.
  ComfyUI installation, GPU execution, visual comparisons, and benchmarking are
  manual follow-ups, including a 32-versus-64-frame chunk speed comparison. Do not
  add a benchmark harness or a performance threshold as a prerequisite for delivery.
