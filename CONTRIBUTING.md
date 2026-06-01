# Contributing to CGMPy

Thank you for your interest in contributing to **CGMPy** — a modular Python library for Continuous Glucose Monitoring (CGM) data analysis. 🎉

This document explains how to set up a development environment, report bugs, propose features, and submit pull requests.

---

## 📜 Code of Conduct

This project adheres to the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Report unacceptable behavior to the maintainer (javierpenatearrieta@gmail.com).

---

## 🚀 Quick Links

- 🐛 [Report a bug](../../issues/new?template=bug_report.md)
- 💡 [Propose a feature](../../issues/new?template=feature_request.md)
- ❓ [Ask a question](../../discussions)
- 🔒 [Report a security vulnerability](SECURITY.md)

---

## 🛠️ Development Setup

### Prerequisites

- **Python ≥ 3.10**
- **Git**
- **uv** (recommended) or **pip**
- Recommended editor: **VS Code** with the [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and Python extensions.

### Clone and install

```bash
git clone https://github.com/javierpa95/cgmpy.git
cd cgmpy

# With uv (recommended)
uv venv
uv pip install -e ".[dev,docs,agata]"

# Or with pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs,agata]"

# Install pre-commit hooks
pre-commit install
```

### Verify the install

```bash
pytest                    # Run the test suite
ruff check .              # Lint
ruff format --check .     # Format check
```

---

## 🧪 Running Tests

```bash
# Full test suite
pytest

# With coverage
pytest --cov=cgmpy --cov-report=html

# Specific test file
pytest tests/unit/test_basic_metrics.py

# By marker
pytest -m "not slow"           # Skip slow tests
pytest -m integration          # Run only integration tests
pytest -m "not agata"          # Skip AGATA-dependent tests
```

### Writing Tests

- Place unit tests under `tests/unit/test_<module>/`.
- Place integration tests under `tests/integration/`.
- Place clinical regression tests (against published reference datasets) under `tests/clinical/`.
- Use the fixtures from `tests/conftest.py` or add new ones there if they are reusable.
- Mark tests with the appropriate marker (`@pytest.mark.slow`, `@pytest.mark.integration`, etc.).
- New features should not be merged without at least one corresponding test.

---

## 📁 Project Structure

```
cgmpy/
├── cgmpy/                # Source code
│   ├── data/             # Loaders, parsers, processors, exporters
│   ├── metrics/          # Clinical metric calculations
│   ├── plotting/         # Visualizations
│   ├── analysis/         # High-level orchestrators
│   ├── agata/            # AGATA integration
│   ├── utils/            # Helpers
│   └── __init__.py       # Public API
├── tests/                # Test suite
├── examples/             # Usage examples
├── docs/                 # Documentation
└── pyproject.toml        # Package metadata and tool config
```

For full conventions, see [docs/development/architecture.md](docs/development/architecture.md) and [`AGENTS.md`](AGENTS.md) (used by AI coding assistants).

---

## 🌿 Branching & Commit Conventions

### Branching

- `main` — production-ready. Protected.
- `feat/<scope>` — new features.
- `fix/<scope>` — bug fixes.
- `docs/<scope>` — documentation only.
- `chore/<scope>` — tooling, refactors with no behavior change.
- `release-please/*` — automated by release-please.

Always create a **new branch** from `main` for your work. Do **not** commit directly to `main`.

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/). Your commit messages will be checked by `commitlint` (see pre-commit hooks).

```
<type>(<scope>): <short description>

<optional body>

<optional footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`.

**Examples**:

```
feat(metrics): add MAGE-2 calculation
fix(data-loader): handle empty CSV without raising
docs(readme): add citation section
test(pregnancy): add regression tests for GDM
chore(deps): bump pandas to >=2.1
```

---

## 🔄 Pull Request Workflow

1. **Fork & branch** from `main`.
2. **Make your changes** in small, focused commits.
3. **Add or update tests** for any behavioral change.
4. **Update documentation** under `docs/` if your change affects the public API, behavior, or configuration. See [`AGENTS.md`](AGENTS.md) → "Documentation Golden Rule".
5. **Update `CHANGELOG.md`** if your change is user-facing (feat, fix).
6. **Run the full check** before opening the PR:
   ```bash
   ruff check .
   ruff format --check .
   pytest
   ```
7. **Open a Pull Request** using the PR template. Fill in:
   - What changed and why
   - How it was tested
   - Any breaking changes
   - Related issue(s)
8. **Wait for CI** — all checks must pass. Address review comments.

---

## 🆕 Adding a New Feature

### Adding a new device loader

1. Create `cgmpy/data/loader_<device>.py` with a class subclassing `DataLoader`.
2. Register the device in `cgmpy/data/specialized.py`.
3. Add tests under `tests/unit/test_data/`.
4. Add a small synthetic CSV to `tests/fixtures/`.
5. Update `docs/user-guide/loading-data.md`.

### Adding a new clinical metric

1. Decide the appropriate module (`metrics/basic.py`, `metrics/time_in_range.py`, `metrics/variability.py`, or a new file).
2. Implement the calculation as a function (and/or method on the relevant class).
3. Add **at least one unit test** with a known-answer dataset.
4. Add a **clinical regression test** under `tests/clinical/` if a published reference exists.
5. Document the metric in `docs/user-guide/computing-metrics.md`, including the formula and the reference paper.
6. Cross-validate against AGATA when possible.

### Adding a new plot

1. Implement the plot in `cgmpy/plotting/<plotter>.py`.
2. Wire it through `GlucoseAnalysis.plot_*()` in `cgmpy/analysis/core.py`.
3. Add a headless test that the plot is generated without raising (use `matplotlib.use("Agg")`).
4. Document the plot in `docs/user-guide/visualization.md`.

---

## 📏 Coding Style

- **PEP 8** + **Ruff** (line length 100). Run `ruff format .` before committing.
- **Type hints** are encouraged for all new public functions.
- **Docstrings** follow Google or NumPy style. At minimum, every public function/class has a one-line summary.
- **English** in code, comments, and documentation.
- **No print statements** in library code; use the `logging` module if needed.
- **No hardcoded paths**; use `pathlib.Path` and project-relative paths.
- **No PHI** (Protected Health Information) in tests, examples, or comments.

---

## 🌐 Internationalization

User-facing strings in error messages and CLI output are in **English**. If you add new error messages, keep them in English and clear.

---

## 🤖 Using AI Assistants

This project is designed to be friendly to AI coding agents while keeping humans in control. See [`AGENTS.md`](AGENTS.md) for the full conventions.

When you use an AI assistant (OpenCode, Claude Code, GitHub Copilot, Cursor, etc.) to contribute:

- **Read [`AGENTS.md`](AGENTS.md) first** — many conventions are documented there.
- **Always review the AI's output** before committing.
- **AI agents must not commit secrets, PHI, or hardcoded credentials.** This is enforced by pre-commit hooks and CI.
- **AI agents must update docs** when they change code (see the "Documentation Golden Rule").

---

## 📚 Resources

- [Project Roadmap](ROADMAP.md)
- [Architecture overview](docs/development/architecture.md)
- [Git workflow](docs/development/git-workflow.md)
- [Release process](docs/development/release-process.md)
- [Technical debt log](docs/development/technical-debt.md)

---

## 💬 Getting Help

- Open a [Discussion](../../discussions) for general questions.
- Open an [Issue](../../issues) for bugs and feature requests.
- Email the maintainer at **javierpenatearrieta@gmail.com** for private matters.

---

Thank you for helping make CGMPy better! 💙
