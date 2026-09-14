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
- Keep CPU PyTorch and the NumPy, Pillow, and SciPy reference-test dependencies in
  the `dev` group. Keep the explicit CPU wheel index configured for test PyTorch.
- Update `pyproject.toml` and `uv.lock` together. There are currently no additional
  runtime dependencies and no `requirements.txt`. If additional runtime dependencies
  are introduced, maintain `requirements.txt` by hand to match only
  `[project].dependencies` and their constraints. Never generate it from `uv.lock`
  or `uv export`; commit affected dependency files together.
- Keep the implementation in `src/gpu_layerstyles` and retain the root ComfyUI
  loader for clone and ZIP installations. Include the loader, tests, lockfile, and
  this file in source distributions.

## Node behavior and validation

- Register exactly three V3 nodes through `ComfyExtension`, with their existing
  node IDs and control names, order, ranges, steps, and defaults.
- Process RGB in float32, preserve alpha, and leave input tensors untouched. Keep
  contrast statistics local to each frame and skip neutral adjustments.
- Use ComfyUI's selected device and memory management. Account for the complete
  destination as well as chunk working memory, and preserve explicit output
  placement when retrying allocation failures.
- Automated tests use real CPU tensors with ComfyUI API and memory test doubles.
  ComfyUI installation, GPU execution, visual comparisons, and benchmarking are
  manual follow-ups. Do not add a benchmark harness or a performance threshold as
  a prerequisite for delivery.
