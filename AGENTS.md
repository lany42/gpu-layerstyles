Do NOT update the README unless explicitly requested.

# Development workflow

Run from the repository root. Use the Python version in `.python-version`
and uv; `uv sync --locked` installs the development environment. Update
`pyproject.toml` and `uv.lock` together. After Python changes, run these
checks in order:

```sh
uv run --offline --locked ruff check --select I --fix .
uv run --offline --locked ruff check --fix .
uv run --offline --locked ruff format .
uv run --offline --locked ruff check .
uv run --locked pytest
uv lock --check
uv build
```

Keep implementation in `src/gpu_layerstyles` and the root ComfyUI loader
for clone and ZIP installations.

ComfyUI supplies runtime PyTorch and `comfy_api`. Keep runtime dependencies
empty unless a new dependency is needed; do not replace the host's CUDA
build. Keep CPU PyTorch, NumPy, Pillow, SciPy, and color-matcher in the
`dev` group, with an explicit CPU wheel index for PyTorch.

If runtime dependencies are added, maintain `requirements.txt` by hand to
match only `[project].dependencies` and their constraints. Never generate
it from `uv.lock` or `uv export`; commit affected dependency files together.

Include the root loader, tests, lockfile, and AGENTS.md in source
distributions; include COPYRIGHT in both source and wheel distributions.

Every Python source and test starts with:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
```

Keep project-code and original-algorithm-source attribution in COPYRIGHT.
Qualify color-matcher's attribution as the original algorithm source.
Preserve upstream source and fixture provenance and license notices,
including Pillow's, separately from project-code notices in distributions.

Use Conventional Commits: `<type>[optional scope][!]: <summary>`. Write an
imperative subject of at most 50 characters, with no trailing period. Follow
it with one blank line and a single short paragraph explaining what changed
and why in plain language. Limit the body to four lines, each at most 72
characters; avoid lists and exhaustive change logs.
