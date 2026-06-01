#!/usr/bin/env bash
# Build source and wheel distributions for CGMPy.
#
# Usage:
#   scripts/build-dist.sh            # clean build (sdist + wheel)
#   scripts/build-dist.sh --no-clean # skip the clean step
#   scripts/build-dist.sh --sdist    # only sdist
#   scripts/build-dist.sh --wheel    # only wheel
#
# Requires: build, twine (installed via the `dev` extra).
#
# Output: dist/cgmpy-VERSION.tar.gz and dist/cgmpy-VERSION-py3-none-any.whl

set -euo pipefail

# -------------------------------------------------------------- argparse --
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLEAN=1
BUILD_SDIST=1
BUILD_WHEEL=1

for arg in "$@"; do
  case "$arg" in
    --no-clean) CLEAN=0 ;;
    --sdist)    BUILD_WHEEL=0 ;;
    --wheel)    BUILD_SDIST=0 ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

# --------------------------------------------------------------- checks --
cd "$REPO_ROOT"

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python is not on PATH." >&2
  exit 1
fi

PY_VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
  echo "Error: Python 3.10+ is required (found $PY_VERSION)." >&2
  exit 1
fi

if ! python -c "import build" >/dev/null 2>&1; then
  echo "Error: 'build' is not installed. Run: pip install -e '.[dev]'" >&2
  exit 1
fi

if ! python -c "import twine" >/dev/null 2>&1; then
  echo "Error: 'twine' is not installed. Run: pip install -e '.[dev]'" >&2
  exit 1
fi

# ---------------------------------------------------------------- clean --
if [[ "$CLEAN" -eq 1 ]]; then
  echo "==> Cleaning previous build artifacts"
  rm -rf build/ dist/ *.egg-info cgmpy.egg-info src/*.egg-info
fi

# --------------------------------------------------------------- build ---
echo "==> Building distributions (Python $PY_VERSION)"

ARGS=()
if [[ "$BUILD_SDIST" -eq 1 ]]; then ARGS+=(--sdist); fi
if [[ "$BUILD_WHEEL" -eq 1 ]]; then ARGS+=(--wheel); fi

python -m build "${ARGS[@]}"

# -------------------------------------------------------------- verify ---
echo "==> Verifying distributions with twine"
twine check dist/*

# -------------------------------------------------------------- report ---
echo
echo "==> Build complete. Artifacts:"
ls -lh dist/
echo
echo "Next steps:"
echo "  Test PyPI:    scripts/publish-test.sh"
echo "  Production:   scripts/publish-prod.sh    (then confirm with twine upload)"
