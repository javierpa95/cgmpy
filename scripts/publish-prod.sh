#!/usr/bin/env bash
# Upload built distributions to production PyPI (https://pypi.org/).
#
# Usage:
#   scripts/publish-prod.sh
#
# WARNING: this is IRREVERSIBLE. Always publish to Test PyPI first.
#
# Requires: twine (via the `dev` extra), and a PyPI API token in the
# env var PYPI_TOKEN. On CI, prefer Trusted Publishing (OIDC); see
# .github/workflows/publish-pypi.yml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# -------------------------------------------------------------- sanity --
if [[ ! -d dist/ ]] || [[ -z "$(ls -A dist/ 2>/dev/null)" ]]; then
  echo "Error: dist/ is empty. Run scripts/build-dist.sh first." >&2
  exit 1
fi

if [[ -z "${PYPI_TOKEN:-}" ]] && [[ -z "${TWINE_USERNAME:-}" ]]; then
  echo "Error: set PYPI_TOKEN in the environment (or use TWINE_USERNAME +"
  echo "       TWINE_PASSWORD). For CI, prefer Trusted Publishing (OIDC)." >&2
  exit 1
fi

# ------------------------------------------------------------ confirm --
VERSION="$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
echo "About to upload cgmpy $VERSION to PRODUCTION PyPI."
echo
ls -lh dist/
echo
read -r -p "Type 'publish $VERSION' to confirm: " ans
if [[ "$ans" != "publish $VERSION" ]]; then
  echo "Aborted." >&2
  exit 1
fi

# --------------------------------------------------------- credentials --
if [[ -n "${PYPI_TOKEN:-}" ]]; then
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="${PYPI_TOKEN}"
  REPO="pypi"
else
  REPO="${TWINE_REPOSITORY:-pypi}"
fi

# ------------------------------------------------------------- upload --
echo "==> Uploading to PRODUCTION PyPI (repository: $REPO)"
twine upload --repository "$REPO" dist/*

echo
echo "==> Done. Check https://pypi.org/project/cgmpy/ to verify."
echo "    Install with:  pip install cgmpy"
