# DevContainer

This directory contains the [Development Container](https://containers.dev/)
configuration for CGMPy. It gives every contributor a reproducible
environment with:

- Python 3.11 (slim, Debian Bullseye)
- `uv` for fast dependency management
- `git` and common utilities
- A pre-configured `.venv` inside the project
- The package installed in editable mode (`-e .[dev,docs]`)
- Pre-commit hooks installed

## Usage

### With VSCode (recommended)

1. Install the **Dev Containers** extension:
   `ms-vscode-remote.remote-containers`
2. Open the project in VSCode.
3. Run **Dev Containers: Reopen in Container** (F1 → start typing).
4. Wait for the postCreateCommand to finish.

### With Docker + CLI

```bash
docker build -f .devcontainer/Dockerfile -t cgmpy-dev .
docker run --rm -it -v "${PWD}:/workspaces/cgmpy" cgmpy-dev
```

### With GitHub Codespaces

Click the **Code** button on GitHub → **Codespaces** →
**New codespace**. The same configuration applies.

## Files

| File | Purpose |
|------|---------|
| `devcontainer.json` | Container config (image, features, extensions, mounts) |
| `Dockerfile` | Optional custom image (only needed for non-standard deps) |

The `Dockerfile` is **not required** when using the
`mcr.microsoft.com/devcontainers/python:3.11-bullseye` image
referenced by `devcontainer.json`. Create it only if you need extra
system libraries.

## Mounts

The config declares two named volumes for faster rebuilds:

- `cgmpy-venv` — the Python virtualenv (re-used across rebuilds).
- `cgmpy-uv-cache` — the `uv` cache.

To reset the environment, remove the volumes:

```bash
docker volume rm cgmpy-venv cgmpy-uv-cache
```

## Port

Port `8000` is forwarded for `mkdocs serve`. Once the devcontainer
is running, run `make docs-serve` and open
<http://localhost:8000>.

## See also

- [`.vscode/`](../vscode/) — VSCode workspace settings
- [`scripts/container-init.sh`](../scripts/container-init.sh) — runs on creation
- [`docs/development/setup.md`](../docs/development/setup.md)
