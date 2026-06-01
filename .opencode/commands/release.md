# /release — Prepare a Release

## Description

Prepares a release of CGMPy following the Semantic Versioning and release-please
process documented in `docs/development/release-process.md` and
`.opencode/agents/release-manager.md`.

## What it does

1. Confirms all CI checks are green on `main`.
2. Confirms `CHANGELOG.md` `[Unreleased]` is complete.
3. Confirms `ROADMAP.md` reflects shipped work.
4. Triggers the `release-please.yml` workflow (or, if unavailable, asks the
   user to do so manually).
5. Waits for the release-please PR to be opened, reviewed, and merged.
6. After merge, prompts the user to run `publish-pypi.yml` manually:
   ```bash
   gh workflow run publish-pypi.yml -f tag=vX.Y.Z
   ```

## Usage

```
/release                # Auto-detect version bump from CHANGELOG entries
/release patch          # Force PATCH bump
/release minor          # Force MINOR bump
/release major          # Force MAJOR bump (rare in 0.x)
/release dry-run        # Show what would be done, but do nothing
```

## Pre-release checklist

The agent will verify:

- [ ] On `main`, no uncommitted changes.
- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `pytest` passes.
- [ ] `CHANGELOG.md` has at least one entry under `[Unreleased]`.
- [ ] `pyproject.toml` version is consistent with the previous release.
- [ ] No `# TODO` or `# FIXME` in shipped code (warnings, not blockers).

## Manual PyPI publish

After release-please merges, the user runs the manual publish:

```bash
# Build the distribution locally to verify
python -m build
twine check dist/*

# Trigger the workflow
gh workflow run publish-pypi.yml -f tag=vX.Y.Z

# Verify on PyPI
open https://pypi.org/project/cgmpy/#history
```

## When to use

- When `[Unreleased]` has accumulated enough entries for a release.
- After a hotfix needs to ship.
- At regular cadence (e.g., monthly minor release).
