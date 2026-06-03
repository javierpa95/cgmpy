# What is Pre-commit?

## What is it?

Imagine a checklist that runs automatically before every git commit. It
checks: trailing whitespace? Valid YAML? Code formatted? Commit message
follows the right convention? If any check fails, the commit is blocked
until you fix it.

That checklist is pre-commit. It is a framework that runs a series of
hooks — small programs that inspect your code — every time you type
`git commit`. If a hook finds something wrong, it either fixes it
automatically or tells you what to fix.

## Why we use it

Ten people writing Python code ten different ways is chaos. One person
forgets to run the linter, another uses tabs instead of spaces, a third
writes a commit message that says only "wip". Over time the codebase
becomes inconsistent and harder to maintain.

Pre-commit enforces consistency without anyone having to remember to do
it. The checks happen automatically, every single time. No willpower
required.

## The hooks we use

CGMPy uses the following pre-commit hooks. You can find the full
configuration in `.pre-commit-config.yaml` at the root of the project.

| Hook | What it checks | What happens if it fails |
|---|---|---|
| `trailing-whitespace` | Removes spaces at the end of lines | Auto-fixes |
| `end-of-file-fixer` | Ensures every file ends with a newline | Auto-fixes |
| `check-yaml` | Validates YAML syntax | Blocks commit |
| `check-toml` | Validates TOML syntax | Blocks commit |
| `check-json` | Validates JSON syntax | Blocks commit |
| `check-added-large-files` | Blocks files over 5 MB | Blocks commit |
| `check-merge-conflict` | Detects leftover merge markers | Blocks commit |
| `check-case-conflict` | Blocks filenames differing only in case | Blocks commit |
| `mixed-line-ending` | Forces LF line endings | Auto-fixes |
| `detect-private-key` | Blocks accidental SSH key commits | Blocks commit |
| `ruff` (lint) | Finds code style violations and bugs | Shows errors, some auto-fix |
| `ruff-format` | Ensures code formatting is consistent | Auto-fixes |
| `interrogate` | Checks docstring coverage (minimum 70%) | Blocks commit, shows missing coverage |
| `commitlint` | Validates commit message follows Conventional Commits | You must rewrite the message |
| `cgmpy-docs-sync` | Verifies documentation is in sync with code | Blocks commit with instructions |

## How to use it

```bash
# Run checks on all staged files (fastest, most common)
pre-commit run

# Run checks on every file in the repository
pre-commit run --all-files

# Install the hooks so they run automatically on git commit
pre-commit install

# Emergency bypass — use sparingly!
git commit --no-verify
```

The `--no-verify` flag is like a clinical protocol waiver: it exists for
genuine emergencies, but if you use it every day, something is wrong.

## Real CGMPy example

You add a new metric function — say, a custom glucose variability index.
You write the code, test it, and it works. You stage the file and run
`git commit`. Pre-commit runs interrogate, which checks that every public
function has a docstring. Your new function does not. The commit is
blocked, and interrogate tells you exactly which function is missing
documentation. You add the docstring, stage again, and this time the
commit goes through.

Without pre-commit, that missing docstring would have made it into the
repository and stayed there until someone happened to notice. With
pre-commit, it was caught before it ever reached the codebase.

## Why it's good for learners

You do not need to memorise every style rule or convention. Pre-commit
teaches you as you go. Every time it blocks a commit, you learn something:
"Ah, I need a docstring here", or "Right, the commit message needs a type
prefix". The tool is also the teacher.
