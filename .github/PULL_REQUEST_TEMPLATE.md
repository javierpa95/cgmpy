---
name: Pull Request
about: Submit changes to CGMPy
title: "[<type>(<scope>)]: <description>"
---

## 📋 Summary

<!-- Replace this line with a one-paragraph summary of what this PR does. -->

## 🎯 Motivation

<!-- Why is this change needed? What problem does it solve? -->

## 🔧 Changes

<!-- Bullet list of the main changes. -->

- ...

## 📸 Screenshots / Outputs

<!-- If the change affects plots, reports, or UI, include a screenshot or paste the relevant output. -->

## 🧪 Testing

- [ ] I have added or updated unit tests.
- [ ] I have added a clinical regression test (if a published reference exists).
- [ ] I have checked AGATA parity (if applicable).
- [ ] I have run `make lint` and `make test` locally — all pass.

## 📚 Documentation

Per the [Documentation Golden Rule](../AGENTS.md) §3:

- [ ] `docs/user-guide/...` updated (if user-facing).
- [ ] `docs/api/...` regenerated (auto, on next build).
- [ ] `CHANGELOG.md` `[Unreleased]` updated (if user-facing).
- [ ] Docstrings added / updated for new public symbols.

## 🔒 Security

- [ ] No PHI in code, tests, examples, or fixtures.
- [ ] No hardcoded credentials or secrets.
- [ ] No new dependencies without justification.

## ⚠️ Breaking Changes

<!-- If this PR introduces breaking changes, describe the migration path. -->

- [ ] No breaking changes.
- [ ] Breaking change described above, with migration notes.

## 📎 Related Issues

<!-- Use keywords to auto-close: Closes #N, Fixes #N, Relates to #N -->

Closes #

## ✅ Pre-Submit Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md).
- [ ] My branch is up to date with `main` (`git fetch && git rebase origin/main`).
- [ ] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
- [ ] The PR title follows Conventional Commits (verified by CI).
- [ ] I have run `make pre-commit-all` — all hooks pass.
- [ ] I have updated relevant tests and documentation.
