# Development Setup

This page is for contributors. If you only want to **use** CGMPy, see
[Installation](../getting-started/installation.md) instead.

## Prerequisites

- **Python ≥ 3.10**
- **Git**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`
- Optional but recommended: **VS Code** with the
  [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and
  [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) extensions.

## First-time setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/cgmpy.git
cd cgmpy

# 2. Create a virtual environment
uv venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# 3. Install with all extras
uv pip install -e ".[dev,docs,agata]"

# 4. Install pre-commit hooks
pre-commit install
```

### Alternative: dev container (Docker / VSCode)

If you have Docker and the **Dev Containers** VSCode extension
(`ms-vscode-remote.remote-containers`), the project ships a
pre-configured container with all dependencies pre-installed:

1. Open the project in VSCode.
2. Run **Dev Containers: Reopen in Container** (F1).
3. Wait for `scripts/container-init.sh` to finish.

The container uses `mcr.microsoft.com/devcontainers/python:3.11-bullseye`
plus the `uv` feature. See [`.devcontainer/devcontainer.json`](https://github.com/javierpa95/cgmpy/blob/main/.devcontainer/devcontainer.json)
for the full configuration.

## Verify the setup

```bash
# Tests
pytest -q

# Lint + format
ruff check .
ruff format --check .

# Type check (relaxed)
mypy cgmpy/

# Build the docs (optional)
mkdocs serve
```

If all of the above pass, you are ready to start contributing.

## Project layout

```
cgmpy/
├── cgmpy/                # Source code
│   ├── data/             # Loaders, processors, analyzers, exporters
│   ├── metrics/          # Basic, time-in-range, variability, pregnancy, targets
│   ├── plotting/         # AGP, daily traces, statistical plots
│   ├── analysis/         # GlucoseAnalysis facade
│   ├── agata/            # AGATA reference integration
│   └── utils/            # Helpers
├── tests/                # Test suite
│   ├── unit/             # Per-module unit tests
│   ├── integration/      # Cross-module integration tests
│   ├── clinical/         # Clinical regression tests
│   └── fixtures/         # Synthetic / anonymized data
├── examples/             # Numbered runnable examples
├── docs/                 # Documentation (this site)
├── scripts/              # Dev / build / anonymization scripts
├── pyproject.toml        # Package metadata and tool config
└── Makefile              # Cross-platform task runner
```

## Common tasks

| Task                         | Command                                |
|------------------------------|----------------------------------------|
| Run all tests                | `make test` or `pytest`                |
| Quick smoke test             | `make test-fast`                       |
| Unit tests only              | `make test-unit`                       |
| With coverage                | `make test-coverage`                   |
| Lint                         | `make lint`                            |
| Auto-fix lint                | `make lint-fix`                        |
| Format                       | `make format`                          |
| Type check                   | `make typecheck`                       |
| Security audit               | `make security`                        |
| Live-reload docs             | `make docs-serve`                      |
| Build docs (CI mode)         | `make docs-build`                      |
| Build distribution           | `make build`                           |
| Test PyPI publish            | `make publish-test`                    |

## Working with pre-commit

Pre-commit runs a set of hooks (Ruff, interrogate, commitlint, …) on
every commit. If a hook fails, the commit is **blocked** until you fix
the issue.

```bash
# Run on staged files (default on commit)
pre-commit run

# Run on every file
pre-commit run --all-files
```

To bypass a hook (use sparingly, document why in the commit message):

```bash
git commit --no-verify -m "fix: emergency hotfix"
```

## AI-assisted development

CGMPy is designed to be friendly to AI coding agents while keeping
humans in control. See [`AGENTS.md`](https://github.com/javierpa95/cgmpy/blob/main/AGENTS.md)
for the full agent conventions.

The OpenCode harness in `.opencode/` is the primary supported flow. It
includes:

- An orchestrator agent (`cgmpy-architect`).
- Specialist agents for data, metrics, plotting, AGATA.
- Skills for git, docs, security, post-coding check, Python library.
- Commands: `/start`, `/end`, `/test`, `/release`.
- Rules for structure, development, security, testing, documentation.

## Next steps

- [Testing](testing.md) — how to write and run tests.
- [Git workflow](git-workflow.md) — branches, commits, PRs.
- [Architecture](architecture.md) — the system overview.
