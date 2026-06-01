# Git Workflow

CGMPy uses a **simplified Git Flow** with **Conventional Commits** and
**squash-merge** to `main`.

## Branches

| Pattern           | Purpose                                       |
|-------------------|-----------------------------------------------|
| `main`            | Production-ready. Protected.                  |
| `feat/<scope>`    | New features.                                 |
| `fix/<scope>`     | Bug fixes.                                    |
| `docs/<scope>`    | Documentation-only changes.                   |
| `chore/<scope>`   | Tooling, refactors with no behavior change.   |
| `ci/<scope>`      | CI/CD-only changes.                           |
| `release-please/*`| Automated by release-please.                  |

> **Never commit directly to `main`.** Always create a branch and open a PR.

## Commit messages — Conventional Commits

```
<type>(<scope>): <subject — imperative, lower-case, ≤ 72 chars>

<body — explain WHY, not WHAT. Wrap at 100.>

<footer — BREAKING CHANGE:, Closes #N, Co-authored-by:>
```

### Allowed types

`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`.

### Allowed scopes (enumerated in `commitlint.config.cjs`)

`data`, `metrics`, `plotting`, `analysis`, `agata`, `utils`, `docs`, `ci`, `deps`, `release`, `agents`, `examples`, `tests`, `harness`, `tooling`.

### Examples

```
feat(metrics): add MAGE-2 calculation

MAGE-2 is the second-generation Mean Amplitude of Glycemic Excursions,
which is more robust to outliers than the original MAGE.

Closes #42
```

```
fix(data-loader): handle empty CSV without raising

Empty files were crashing load_from_csv with a generic Pandas error.
Now we raise GlucoseDataError with a clear, actionable message.
```

## Pull requests

1. **Create a branch** from `main`: `git checkout -b feat/<scope>`.
2. **Make small, focused commits** following Conventional Commits.
3. **Run the full check** before pushing:
   ```bash
   make pre-commit-all
   make test
   ```
4. **Push and open a PR**:
   ```bash
   git push -u origin feat/<scope>
   gh pr create --fill
   ```
5. **Wait for CI** — all checks must pass.
6. **Address review comments** — push more commits (do not amend pushed commits).
7. **Squash-merge** when approved.

The `pr-standards` workflow:

- Validates the PR title follows Conventional Commits (enumerated types and scopes).
- Reminds to update `CHANGELOG.md` if `cgmpy/` changed.
- Warns on large PRs (> 20 files or > 500 lines).
- Auto-labels by area.

## Pre-commit hooks

Pre-commit runs on every `git commit`:

- trailing-whitespace, end-of-file-fixer, mixed-line-ending (force LF).
- check-yaml, check-toml, check-json.
- check-added-large-files (5 MB max).
- check-merge-conflict, detect-private-key.
- ruff check + format.
- interrogate (docstring coverage ≥ 70 %).
- commitlint (Conventional Commits).
- A local hook that warns if `cgmpy/` changed but `docs/` did not.

## Releases

See [Release process](release-process.md).

## Common pitfalls

- ❌ `git commit --amend` on a pushed commit (rewrites history).
- ❌ `git push --force` to `main`.
- ❌ Mixing `feat:` and `fix:` in the same commit (release-please gets confused).
- ❌ Squash-merging a feature without a final squash-merge commit.
- ❌ Committing a `print()` and forgetting it.

## See also

- [`AGENTS.md`](https://github.com/javierpa95/cgmpy/blob/main/AGENTS.md) § 5.
- [Release process](release-process.md).
- `.opencode/skills/git-advisor/SKILL.md` — agent-side guide.
