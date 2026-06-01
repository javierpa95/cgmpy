# git-advisor Skill

> **When to use:** whenever the user mentions git, commits, branches, push, or PRs.

## Branching Model

CGMPy uses a simplified Git Flow:

| Branch pattern         | Purpose                                       |
|------------------------|-----------------------------------------------|
| `main`                 | Production-ready. Protected.                  |
| `feat/<scope>`         | New features.                                 |
| `fix/<scope>`          | Bug fixes.                                    |
| `docs/<scope>`         | Documentation-only changes.                   |
| `chore/<scope>`        | Tooling, refactors with no behavior change.   |
| `release-please/*`     | Automated by release-please.                  |

**Never commit directly to `main`.** Always create a branch and open a PR.

## Commit Messages (Conventional Commits)

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

**Allowed types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`.

**Subject**:

- ≤ 72 characters.
- Imperative mood ("add", not "added" or "adds").
- Lowercase after the colon.
- No trailing period.

**Body** (optional):

- Explain the *why*, not the *what*.
- Wrap at 100 characters.

**Footer** (optional):

- `BREAKING CHANGE: <description>` for MAJOR bumps.
- `Closes #N` or `Fixes #N` to auto-close issues.
- `Co-authored-by: Name <email>` for co-authors.

## Examples

```
feat(metrics): add MAGE-2 calculation

MAGE-2 is the second-generation Mean Amplitude of Glycemic Excursions,
which is more robust to outliers than the original MAGE.

Closes #42
```

```
fix(data-loader): handle empty CSV without raising

Empty files were crashing load_from_csv with a generic Pandas error.
Now we raise GlucoseDataError with a clear message.
```

## Commands

```bash
# Create a feature branch
git checkout -b feat/<scope>

# Stage and commit
git add <files>
git commit -m "feat(<scope>): <description>"

# Push and open a PR
git push -u origin feat/<scope>
gh pr create --fill
```

## Pre-Push Checklist

- [ ] `ruff check .` passes.
- [ ] `ruff format --check .` passes.
- [ ] `pytest` passes locally.
- [ ] `git status` is clean.
- [ ] On a `feat/`, `fix/`, `docs/`, or `chore/` branch — never on `main`.
- [ ] No `print()`, no hardcoded secrets, no `__pycache__/`.

## Common Pitfalls

- ❌ `git commit --amend` on a pushed commit (rewrites history).
- ❌ `git push --force` to `main`.
- ❌ Squash-merging a feature with `feat:` and `fix:` mixed (release-please gets confused).
- ❌ Committing merge commits with `--no-ff` (we use squash-merge by default).
