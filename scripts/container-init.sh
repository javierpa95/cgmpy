#!/usr/bin/env bash
# postCreateCommand for the CGMPy dev container.
#
# Runs once when the container is first created. It:
#   1. Creates the .venv with uv
#   2. Installs the package with dev + docs extras
#   3. Installs pre-commit
#   4. Prints a welcome message

set -euo pipefail

cd /workspaces/cgmpy

echo "==> Creating .venv with uv"
# Use the Python version from the devcontainer base image
uv venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing cgmpy with [dev,docs] extras"
uv pip install -e ".[dev,docs]"

echo "==> Installing pre-commit hooks"
pre-commit install --install-hooks

echo
echo "==================================================="
echo "  CGMPy dev environment is ready."
echo "==================================================="
echo
echo "  Common commands (use the VSCode terminal):"
echo "    make test-fast         # quick test suite"
echo "    make lint              # ruff"
echo "    make format            # ruff format"
echo "    make typecheck         # mypy"
echo "    make docs-serve        # mkdocs at http://localhost:8000"
echo "    make build             # sdist + wheel"
echo
echo "  The package is installed in editable mode, so"
echo "  changes to cgmpy/ take effect without reinstall."
echo
