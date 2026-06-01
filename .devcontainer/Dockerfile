# ==========================================
# CGMPy — Optional custom devcontainer image
# ==========================================
#
# This Dockerfile is OPTIONAL. The default devcontainer.json uses
# mcr.microsoft.com/devcontainers/python:3.11-bullseye directly.
#
# Build this only if you need system libraries that are not in the
# base image (e.g., libgomp for OpenMP, mysql-client, etc.).
#
# Usage:
#   docker build -f .devcontainer/Dockerfile -t cgmpy-dev .
#
# Then update devcontainer.json to use:
#   "build": { "dockerfile": "Dockerfile" }

ARG PYTHON_VERSION=3.11
FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION}-bullseye

# System packages (add as needed)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create non-root user (vscode uid=1000 is the default in devcontainers)
USER vscode
WORKDIR /workspaces/cgmpy

# Pre-create the venv so it lives in the volume mount
RUN uv venv .venv --python ${PYTHON_VERSION}

# Copy only pyproject.toml first for better Docker layer caching
COPY --chown=vscode:vscode pyproject.toml ./

# Install runtime + dev + docs dependencies
RUN . .venv/bin/activate && uv pip install -e ".[dev,docs]"

# Copy the rest of the source
COPY --chown=vscode:vscode . .

# Re-install in editable mode to pick up the source
RUN . .venv/bin/activate && uv pip install -e ".[dev,docs]"

# Install pre-commit hooks
RUN . .venv/bin/activate && pre-commit install --install-hooks || true

CMD ["/bin/bash"]
