# docs-maintainer Skill

> **When to use:** after modifying code in `cgmpy/`, or when the user asks to update documentation.

## The Documentation Golden Rule

> **If you modify code in `cgmpy/`, you MUST verify and update the corresponding documentation before declaring the task complete.**

The mapping is in `AGENTS.md` §3. This skill enforces it.

## Workflow

1. **Identify** the files you changed: `git diff --name-only HEAD~1`.
2. **Map** them to documentation files using the table in `AGENTS.md` §3.
3. **For each affected doc file**:
   - Read the current content.
   - Identify the section that needs updating.
   - Update it. Be concise — prefer tables and bullet lists.
4. **Update CHANGELOG.md** if the change is user-facing.
5. **Update docstrings** in the source code for the auto-generated API reference.
6. **Re-run the docs build** (Phase 7):
   ```bash
   mkdocs build --strict
   mkdocs serve  # for local preview
   ```

## Doc Style

- **English**, Markdown, with `trim_trailing_whitespace = false`.
- **Relative links** for inter-doc navigation: `[Setup](setup.md)`.
- **Runnable code blocks** — copy-paste should work in a fresh venv.
- **Citations** for clinical metrics: paper, year, DOI.
- **"See also"** section at the end of each user-guide page.

## What Goes Where

| Content type                           | Location                              |
|----------------------------------------|---------------------------------------|
| Installation, quickstart, formats      | `docs/getting-started/`               |
| How to use features (load, compute...)  | `docs/user-guide/`                    |
| Function / class reference             | `docs/api/` (auto-generated)          |
| Architecture, design decisions         | `docs/architecture/`, `docs/development/architecture.md` |
| Privacy, GDPR                          | `docs/legal/`                         |
| ADRs                                   | `docs/architecture/decisions/`        |
| Changelog                              | `CHANGELOG.md` (root) + `docs/CHANGELOG.md` (mirror) |
| Roadmap                                | `ROADMAP.md` (root) + `docs/ROADMAP.md` (mirror) |

## CHANGELOG Conventions

Under `## [Unreleased]`:

```markdown
### Added
- New feature description

### Changed
- Behavior change

### Fixed
- Bug fix

### Removed
- Deprecated feature

### Security
- Security fix (anonymized, do not describe the exact vulnerability)
```

## Pre-Commit Hook (Doc)

A pre-commit hook can be added in Phase 4 to detect changes in `cgmpy/` that
have no corresponding changes in `docs/` or `CHANGELOG.md`:

```yaml
- repo: local
  hooks:
    - id: cgmpy-docs-sync
      name: cgmpy-docs-sync
      entry: bash -c 'git diff --cached --name-only | grep -q "^cgmpy/" && ! git diff --cached --name-only | grep -qE "^(docs/|CHANGELOG.md|AGENTS.md)" && echo "WARNING: cgmpy/ changed but docs/ and CHANGELOG.md did not. Update docs!" && exit 1 || exit 0'
      language: system
      pass_filenames: false
```
