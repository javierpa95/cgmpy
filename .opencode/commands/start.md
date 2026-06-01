# /start — Load Session Context

## Description

Loads the full project context at the beginning of a session. Run this **first**
when you open OpenCode to work on CGMPy.

## What it does

1. Reads `AGENTS.md` and the active agent's role file.
2. Reads `ROADMAP.md` to understand the current phase.
3. Reads `CHANGELOG.md` `[Unreleased]` to see pending work.
4. Reads `session-log.md` (if it exists) to see the most recent session.
5. Reads `agent_memory.md` (if it exists) for cross-session learnings.
6. Runs `git status` and `git log --oneline -10` to see the current state.
7. Reports a one-paragraph summary: where we are, what's open, what to do.

## When to use

- First command of every session.
- After a long break.
- When switching branches.

## Example output

```
=== CGMPy session context ===

Branch: feat/v0.3-baseline-refactor (3 commits ahead of main)
Last commit: refactor(data,metrics): complete v0.3 modular refactor

ROADMAP phase: Phase 1 — Open Source Readiness

Pending [Unreleased] entries:
- (none)

session-log.md: last session 2 days ago, closed feat/opencode-harness.

Ready. What do you want to do?
```
