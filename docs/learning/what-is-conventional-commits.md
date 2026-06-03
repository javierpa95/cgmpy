# What are Conventional Commits?

## What is it?

A convention for writing git commit messages. Instead of "fixed stuff" or "changes", you write:

```
fix(metrics): handle empty glucose series without raising ZeroDivisionError
```

Conventional Commits give every commit a **type**, an optional **scope**, and a **description** that explains what changed and why.

## The format

```
<type>(<scope>): <description>

<body -- why this change matters>

<footer -- references, breaking changes>
```

The subject line (first line) must be 72 characters or fewer. The body explains the reasoning behind the change. The footer can reference issues or mark breaking changes.

## Types used in CGMPy

| Type | When to use | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(metrics): add MAGE-2 calculation` |
| `fix` | Bug fix | `fix(data): handle empty CSV gracefully` |
| `docs` | Documentation | `docs: add pregnancy analysis guide` |
| `refactor` | Code change, no behaviour change | `refactor(metrics): extract pure functions from mixins` |
| `test` | Adding or updating tests | `test: add clinical reference for MAGE` |
| `chore` | Tooling, config, dependencies | `chore: update ruff to v0.15` |
| `ci` | CI or CD changes | `ci: add Python 3.12 to test matrix` |

## Real CGMPy example

Here is a commit from the project's history:

```
feat(metrics): implement MAGE_Baghurst algorithm

Adds the Baghurst smoothing approach for MAGE calculation,
with three configurable methods (smoothing, direct elimination,
simplified). Includes guard clauses for datasets with fewer
than 9 readings or zero standard deviation.

Closes #42
```

The subject tells you what was added. The body tells you why and how. The footer links to the original issue.

## Why it matters

Six months later, you can run `git log --oneline` and understand exactly what changed and why. The changelog (see `CHANGELOG.md`) is auto-generated from these commit messages -- so `docs:` commits automatically appear in the docs section of the changelog, `fix:` commits appear in the bug fix section, and so on.

## How to write a good commit

- Subject line: 72 characters or fewer, imperative mood ("add feature", not "added feature"), no trailing period.
- Body: explain **why**, not **what**. Git already shows you what changed -- the body should tell you why the change was necessary.
- Reference issues: use `Closes #42` or `Fixes #42` to auto-close issues when the PR merges.

## Why it is good for learners

It forces you to think about what you did before you commit. What type of change is this? Is it a `feat` or a `refactor`? Which scope does it affect? Answering these questions makes your commits a readable, well-organised history of the project -- and that history is invaluable when someone (including future you) tries to understand why a change was made.
