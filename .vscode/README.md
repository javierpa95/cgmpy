# VSCode Workspace

This directory contains shared VSCode configuration for the
project.

## Files

| File | Purpose | In repo? |
|------|---------|----------|
| `settings.json` | Default workspace settings (Python, ruff, format, etc.) | Yes |
| `extensions.json` | Recommended + unwanted extensions | Yes |
| `launch.json` | Debug configurations (pytest, current file, examples) | Yes |
| `tasks.json` | Common task shortcuts (test, lint, build, docs) | Yes |
| `settings.local.json` | Personal overrides (your machine only) | **No** (gitignored) |
| `launch.local.json` | Personal debug overrides | **No** (gitignored) |

## Personal overrides

To customize settings **without affecting other contributors**, create
`.vscode/settings.local.json`. It is gitignored and overrides the
shared `settings.json`.

Example `.vscode/settings.local.json`:

```json
{
  "python.defaultInterpreterPath": "C:/Python311/python.exe",
  "editor.rulers": [80, 100, 120]
}
```

## Devcontainer

For a reproducible environment, open the project in a container:

1. Install the **Dev Containers** extension
   (`ms-vscode-remote.remote-containers`).
2. Press `F1` → **Dev Containers: Reopen in Container**.
3. The container builds, then `scripts/container-init.sh` runs:
   - creates `.venv` with `uv`,
   - installs the package with `[dev,docs]` extras,
   - installs pre-commit hooks.

See `.devcontainer/devcontainer.json` for the full configuration.

## Tasks

Open the command palette and start typing "Tasks: Run Task":

- `test (fast)` — quick unit tests
- `test (full)` — full suite including slow/clinical/agata
- `test (coverage)` — with HTML coverage report
- `lint (check)`, `lint (fix)`, `format`
- `typecheck` — mypy
- `security` — bandit
- `docs: serve` — mkdocs at <http://localhost:8000>
- `docs: build` — strict mkdocs build
- `build: sdist + wheel` — `make build`
- `pre-commit (all files)` — `make pre-commit-all`
- `clean` — remove build artifacts

## Debug

Switch to the **Run and Debug** panel (Ctrl+Shift+D) and pick a
configuration:

- **Python: Current File** — debug the file in the active editor
- **Python: pytest (current file)** — debug tests in the active file
- **Python: pytest (full suite)** — debug the whole suite
- **Python: pytest (fast)** — skip slow/clinical/agata
- **Python: example 01 quickstart** — debug an example script
- **Python: Attach (debugpy)** — attach to a running debugpy server

## See also

- [`.devcontainer/`](../../.devcontainer/) — reproducible dev environment
- [`Makefile`](../../Makefile) — most task definitions
- [`docs/development/setup.md`](../../docs/development/setup.md) — full setup guide
