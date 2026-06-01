#!/usr/bin/env bash
# Upload built distributions to Test PyPI (https://test.pypi.org/).
#
# Usage:
#   scripts/publish-test.sh
#
# Requires: twine (via the `dev` extra), and a Test PyPI API token in
# the env var TEST_PYPI_TOKEN. Alternatively, you can use Trusted
# Publishing (OIDC) on CI; this script is intended for local verification.
#
# The script refuses to run if it detects that the dist/ directory is
# empty or stale. Run scripts/build-dist.sh first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# -------------------------------------------------------------- sanity --
if [[ ! -d dist/ ]] || [[ -z "$(ls -A dist/ 2>/dev/null)" ]]; then
  echo "Error: dist/ is empty. Run scripts/build-dist.sh first." >&2
  exit 1
fi

DIST_AGE_LIMIT=3600
if [[ -n "$(find dist/ -maxdepth 1 -mmin +$((DIST_AGE_LIMIT / 60)) -print -quit 2>/dev/null)" ]]; then
  echo "Warning: dist/ files are older than $((DIST_AGE_LIMIT / 60)) minutes."
  echo "         Re-run scripts/build-dist.sh if you have made changes."
  read -r -p "Continue anyway? [y/N] " ans
  if [[ ! "$ans" =~ ^[Yy]$ ]]; then
    exit 0
  fi
fi

# --------------------------------------------------------- credentials --
if [[ -z "${TEST_PYPI_TOKEN:-}" ]] && [[ -z "${TWINE_USERNAME:-}" ]]; then
  echo "Error: set TEST_PYPI_TOKEN in the environment (or use TWINE_USERNAME +"
  echo "       TWINE_PASSWORD). For local dev, you can also use a keyring."
  echo ""
  echo "       Trusted Publishing (OIDC) is the recommended way on CI;"
  echo "       see .github/workflows/publish-pypi.yml." >&2
  exit 1
fi

if [[ -n "${TEST_PYPI_TOKEN:-}" ]]; then
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="${TEST_PYPI_TOKEN}"
  REPO="testpypi"
else
  REPO="${TWINE_REPOSITORY:-testpypi}"
fi

# ------------------------------------------------------------- upload --
echo "==> Uploading to Test PyPI (repository: $REPO)"
twine upload --repository "$REPO" dist/*

echo
echo "==> Done. Check https://test.pypi.org/project/cgmpy/ to verify."
echo "    Install with:  pip install --index-url https://test.pypi.org/simple/ cgmpy"
