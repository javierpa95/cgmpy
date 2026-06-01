# post-coding-check Skill

> **When to use:** after every coding session, before commit, before PR.

## Quick Check (< 2 minutes)

```bash
# Lint
ruff check .

# Format check
ruff format --check .

# Tests
pytest -q
```

## Full Check (3-5 minutes)

```bash
# Lint with auto-fix
ruff check --fix .

# Format
ruff format .

# Type check
mypy cgmpy/

# Tests with coverage
pytest --cov=cgmpy --cov-report=term-missing

# Security
bandit -r cgmpy/ -ll

# Doc coverage
interrogate -vv cgmpy/
```

## Pre-Commit Checklist

Run these in order. **All must pass** before `git commit`:

- [ ] **No secrets in staged files**:
  ```bash
  git grep -i "password\|secret\|api[_-]key" --cached
  ```
- [ ] **No PHI identifiers**:
  ```bash
  git grep -i "patient_id\|patient_name\|subject_id" --cached
  ```
- [ ] **No print() in library code**:
  ```bash
  git diff --cached --name-only | grep "^cgmpy/" | xargs grep -n "print(" 2>/dev/null
  ```
- [ ] **No new files that should be gitignored**:
  ```bash
  git status --porcelain | grep -E "^\?\? "
  ```
- [ ] **If `cgmpy/` modified: docs updated** (per Documentation Golden Rule).
- [ ] **If user-facing: `CHANGELOG.md` updated**.
- [ ] **`ruff check .`** passes (0 errors, warnings OK).
- [ ] **`ruff format --check .`** passes.
- [ ] **`mypy cgmpy/`** passes (0 errors).
- [ ] **`pytest`** passes (0 failures).
- [ ] **`bandit -r cgmpy/`** reports 0 high-severity issues.
- [ ] On a `feat/`, `fix/`, `docs/`, or `chore/` branch — **never on `main`**.
- [ ] Commit message follows Conventional Commits.

## What to Do If Checks Fail

| Failure               | Action                                                          |
|-----------------------|-----------------------------------------------------------------|
| `ruff` errors         | Run `ruff check --fix .`, then manual fix remaining.            |
| `ruff format` diff    | Run `ruff format .`.                                            |
| `mypy` errors         | Fix types before anything else.                                 |
| `pytest` failures     | Fix failing tests before committing.                            |
| `bandit` HIGH         | Block commit, fix or document with justification comment.       |
| `git grep` finds PHI  | Remove the PHI, regenerate the fixture if needed.               |
| `git grep` finds secret | Rotate the secret immediately, remove from history.          |

## Cross-Platform Notes

- On Windows, use **Git Bash** or **WSL** for `xargs` and `grep` commands.
- `pre-commit run --all-files` works cross-platform.
- `bandit` and `mypy` may need separate installation: `pip install cgmpy[dev]`.

## Success Criteria

- [ ] `ruff check .` exits 0.
- [ ] `ruff format --check .` exits 0.
- [ ] `mypy cgmpy/` exits 0.
- [ ] `pytest` exits 0.
- [ ] `bandit -r cgmpy/` reports no HIGH/CRITICAL.
- [ ] No secrets, no PHI, no `print()` in staged files.
- [ ] On a non-`main` branch.
- [ ] Conventional Commits message.
