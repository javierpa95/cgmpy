# ==========================================
# CGMPy - Makefile
# ==========================================
# Cross-platform Makefile. On Windows, use Git Bash or WSL.
# Alternatively, run the python scripts directly (scripts/*.py).
#
# Most targets delegate to scripts/ so they can also be invoked
# from CI / OpenCode agents without needing GNU make installed.

.PHONY: help install install-dev install-docs install-agata \
        dev test test-fast test-unit test-integration test-clinical test-agata \
        lint lint-fix format format-check typecheck security secrets \
        docs-serve docs-build docs-deploy \
        pre-commit pre-commit-all clean build dist-check publish-test \
        release-dry-run hooks

# Default Python interpreter (override with `make PYTHON=python3.11 test`)
PYTHON ?= python

# ==========================================
# 🆘 Help
# ==========================================
help:
	@echo "CGMPy - available make targets:"
	@echo " "
	@echo "  === Install ==="
	@echo "  install           Install runtime dependencies"
	@echo "  install-dev       Install with dev tools (ruff, pytest, bandit, ...)"
	@echo "  install-docs      Install with docs tools (mkdocs)"
	@echo "  install-agata     Install with agata (optional, parity tests)"
	@echo "  install-all       Install everything"
	@echo " "
	@echo "  === Development ==="
	@echo "  test              Run full test suite"
	@echo "  test-fast         Skip slow + clinical + agata"
	@echo "  test-unit         Only unit tests"
	@echo "  test-integration  Only integration tests"
	@echo "  test-clinical     Only clinical regression tests"
	@echo "  test-agata        Only AGATA parity tests"
	@echo "  test-coverage     Run with coverage report"
	@echo " "
	@echo "  === Quality ==="
	@echo "  lint              ruff check"
	@echo "  lint-fix          ruff check --fix"
	@echo "  format            ruff format (apply)"
	@echo "  format-check      ruff format --check (CI)"
	@echo "  typecheck         mypy cgmpy/"
	@echo "  security          bandit + detect-secrets"
	@echo "  secrets           Detect hardcoded secrets"
	@echo " "
	@echo "  === Docs ==="
	@echo "  docs-serve        mkdocs serve (live preview)"
	@echo "  docs-build        mkdocs build --strict"
	@echo "  docs-deploy       mike deploy (versioned docs)"
	@echo " "
	@echo "  === Pre-commit ==="
	@echo "  pre-commit        Install pre-commit hooks"
	@echo "  pre-commit-all    Run pre-commit on all files"
	@echo " "
	@echo "  === Build & Release ==="
	@echo "  clean             Remove build artifacts"
	@echo "  build             Build sdist + wheel"
	@echo "  dist-check        twine check dist/*"
	@echo "  publish-test      Upload to test.pypi.org"
	@echo "  release-dry-run   Show what release-please would do"

# ==========================================
# 📦 Install
# ==========================================
install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-docs:
	$(PYTHON) -m pip install -e ".[docs]"

install-agata:
	$(PYTHON) -m pip install -e ".[agata]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,docs,agata]"

# ==========================================
# 🧪 Tests
# ==========================================
test:
	$(PYTHON) -m pytest -v

test-fast:
	$(PYTHON) -m pytest -v -m "not slow and not clinical and not agata"

test-unit:
	$(PYTHON) -m pytest tests/unit -v

test-integration:
	$(PYTHON) -m pytest tests/integration -v

test-clinical:
	$(PYTHON) -m pytest tests/clinical -v

test-agata:
	$(PYTHON) -m pytest -v -m agata

test-coverage:
	$(PYTHON) -m pytest --cov=cgmpy --cov-report=term-missing --cov-report=html

# ==========================================
# 🔍 Lint
# ==========================================
lint:
	$(PYTHON) -m ruff check .

lint-fix:
	$(PYTHON) -m ruff check --fix .

# ==========================================
# ✨ Format
# ==========================================
format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

# ==========================================
# 🔎 Type Check
# ==========================================
typecheck:
	$(PYTHON) -m mypy cgmpy/

# ==========================================
# 🔒 Security
# ==========================================
security:
	$(PYTHON) -m bandit -r cgmpy/ -ll

secrets:
	@echo "🔒 Searching for hardcoded secrets in staged files..."
	@git grep -iE "password|secret|api[_-]key|token|credential" --cached || echo "✅ No obvious secrets found"

# ==========================================
# 📚 Docs
# ==========================================
docs-serve:
	$(PYTHON) -m mkdocs serve

docs-build:
	$(PYTHON) -m mkdocs build --strict

docs-deploy:
	$(PYTHON) -m mike deploy --push --update-aliases latest

# ==========================================
# 🪝 Pre-commit
# ==========================================
pre-commit:
	$(PYTHON) -m pip install pre-commit
	$(PYTHON) -m pre_commit install
	@echo "✅ pre-commit hooks installed. Run 'make pre-commit-all' to test."

pre-commit-all:
	$(PYTHON) -m pre_commit run --all-files

# ==========================================
# 🧹 Clean
# ==========================================
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info cgmpy.egg-info/
	@rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/ htmlcov/ .coverage
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned."

# ==========================================
# 📦 Build & Publish
# ==========================================
build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

dist-check:
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine check dist/*

publish-test: build dist-check
	@echo "🚀 Uploading to https://test.pypi.org ..."
	$(PYTHON) -m twine upload --repository testpypi dist/*

# ==========================================
# 🚀 Release dry-run
# ==========================================
release-dry-run:
	@echo "🔍 Checking what release-please would do..."
	@if command -v release-please >/dev/null 2>&1; then \
		release-please manifest-pr --dry-run; \
	else \
		echo "⚠️  release-please not installed. Run: npm i -g release-please"; \
	fi
