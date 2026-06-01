# Release Process

CGMPy follows **Semantic Versioning** and uses
[release-please](https://github.com/googleapis/release-please) to automate
version bumps and `CHANGELOG.md` updates. PyPI publication is **manual**
per the project's security posture.

## Versioning

Given a version `MAJOR.MINOR.PATCH`:

- **MAJOR** — backward-incompatible API changes. **Rare in 0.x.**
- **MINOR** — new features, backward-compatible.
- **PATCH** — bug fixes, backward-compatible.

While CGMPy is in **0.y.z** (pre-1.0), MINOR bumps MAY include small
backward-incompatible changes. Document them clearly in `CHANGELOG.md`.

## Pre-release checklist

- [ ] All CI checks are green on `main`.
- [ ] `CHANGELOG.md` `[Unreleased]` is complete.
- [ ] `ROADMAP.md` reflects shipped work.
- [ ] No `# TODO` markers in shipped code.
- [ ] No `# type: ignore` without a justifying comment.
- [ ] `interrogate` (docstring coverage) is at the configured threshold.
- [ ] Local build works: `make build && twine check dist/*`.

## Step 1 — Let release-please open a PR

release-please runs on every push to `main` (see
`.github/workflows/release-please.yml`). When it finds commits that
warrant a version bump, it opens (or updates) a PR titled:

```
chore(main): release X.Y.Z
```

That PR:

- Bumps the version in `pyproject.toml`.
- Moves `[Unreleased]` entries into a new versioned section in `CHANGELOG.md`.
- Updates the release date.

> **If you need a manual release** (e.g., hotfix), trigger the workflow
> from the Actions tab: `release-please` → `Run workflow`.

## Step 2 — Review and merge the release PR

The maintainer reviews the generated PR. The commit messages and the
resulting `CHANGELOG.md` should be accurate. If anything is wrong, fix
it in the PR (do not bypass release-please).

When approved, **squash-merge** the PR. A git tag is created
automatically on the merge commit (this is configured in the
release-please workflow).

## Step 3 — Publish to PyPI (manual)

> The `publish-pypi.yml` workflow has `workflow_dispatch` trigger only.
> No automatic PyPI push.

```bash
# 1. Verify the tag exists
git fetch --tags
git tag --list | tail -3
# e.g., v0.4.0

# 2. Optionally build locally to sanity-check
make build
twine check dist/*

# 3. Trigger the publish workflow
gh workflow run publish-pypi.yml -f tag=vX.Y.Z
```

In the GitHub UI:

1. Go to **Actions** → **Publish to PyPI** → **Run workflow**.
2. Enter the tag (e.g., `v0.4.0`).
3. Optionally choose `testpypi` for a dry-run.

## Step 4 — Verify

- Visit <https://pypi.org/project/cgmpy/#history> to confirm the new
  version is listed.
- Run `pip install --upgrade cgmpy` in a clean venv to verify
  installation.

## Step 5 — Tag the release on GitHub

If release-please did not auto-create a GitHub release, do it manually:

```bash
gh release create vX.Y.Z \
  --title "vX.Y.Z" \
  --notes "$(sed -n '/^## \[X.Y.Z\]/,/^## \[/p' CHANGELOG.md)"
```

## Hotfix flow

For an urgent PATCH release (security fix, critical bug):

1. Branch from the latest release tag: `git checkout -b fix/hotfix vX.Y.Z`.
2. Cherry-pick or write the fix.
3. Open a PR targeting `main`.
4. After merge, trigger release-please manually with `release` type
   override (or just push the merge commit — release-please will detect it).
5. After the release-please PR merges, run the manual publish step.

## Pre-1.0 caveats

While CGMPy is in **0.y.z**:

- The public API may shift between MINOR versions. Always read the
  `CHANGELOG.md` for the section "Changed" or "Removed".
- Pin dependencies loosely (`cgmpy>=0.4,<0.5`) rather than exact.

## See also

- [`AGENTS.md` § 7](https://github.com/javierpa95/cgmpy/blob/main/AGENTS.md) — agent role for releases.
- `.opencode/agents/release-manager.md` — agent instructions.
- `.opencode/commands/release.md` — `/release` command.
