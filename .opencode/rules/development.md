# Development Rules

> Per-domain rules for the `cgmpy-architect` and execution agents.

## Mandatory Planning

Before touching any file, the agent MUST:

1. Read `AGENTS.md` and `ROADMAP.md` to understand the current state.
2. Read the relevant `docs/development/*.md` for the area of work.
3. Present a brief plan (max 5 bullet points) with:
   - Which files will be modified.
   - The risks of the change.
   - How it will be verified (tests, lint, manual).
4. Wait for explicit human approval (for changes > 50 lines or > 3 files).

## Change Budget

- **Maximum 3 files per session** without an explicit approved plan.
- **Maximum 50 lines modified** without an intermediate review.
- **One change at a time**: do not mix a refactor with a feature.
- **If you break an API**: update **all** consumers before declaring "done".

## Testing and Verification

Before declaring a task complete:

1. `ruff check .` → 0 errors.
2. `ruff format --check .` → 0 diffs.
3. `pytest` → all tests pass.
4. `bandit -r cgmpy/` → 0 high-severity issues.
5. `git grep -i "password\|secret\|api[_-]key" --cached` → 0 matches.
6. `git status` is clean (or only the expected files are changed).
7. The branch is **not** `main`.

## Commits

- **Conventional Commits** (enforced by `commitlint` in pre-commit, Phase 4).
- Subject ≤ 72 chars, imperative mood, no trailing period.
- No `WIP`, `temp`, `test` in commit messages.
- No empty commits.
- Squash commits of trial-and-error before merging.

## Dependencies

- **No global installs** (`pip install -g`, `npm install -g`).
- Document new dependencies in `pyproject.toml` and the corresponding
  optional-dependency group (`dev`, `docs`, `agata`).
- Use `pip` or `uv`; do not introduce a different package manager without consensus.
- Pin via `>=` and `~=` in `pyproject.toml`. `uv.lock` is the source of truth.

## Branching

- `main` is protected. **Never** push to it directly.
- Branch from `main`: `feat/*`, `fix/*`, `docs/*`, `chore/*`, `ci/*`.
- One concern per branch.
- Delete branches after merge.

## Things You Must Not Do

1. ❌ Create a top-level folder without approval.
2. ❌ Install dependencies globally.
3. ❌ Change the minimum Python version without approval.
4. ❌ Commit real patient data (PHI), even in WIP branches.
5. ❌ Disable a security check "temporarily".
6. ❌ Force-push to `main`.
7. ❌ Amend a commit that has already been pushed.
8. ❌ Add a `print()` and forget to remove it.
