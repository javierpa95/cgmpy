# /end — Persist Session Learnings

## Description

Persists the learnings from the current session to `session-log.md` and
`agent_memory.md`. Run this **last**, before closing OpenCode.

## What it does

1. Summarizes what was done in this session (commits, files changed).
2. Captures decisions made and why.
3. Captures pitfalls encountered and how they were resolved.
4. Updates `agent_memory.md` with cross-session findings.
5. Appends a new entry to `session-log.md` with date, branch, and summary.
6. If the working tree is dirty, warns the user to commit or stash.

## When to use

- Last command of every session.
- Before switching branches.
- Before a long break.

## session-log.md format

```markdown
## 2026-06-01 — feat/v0.3-baseline-refactor

**Author:** @opencode (cgmpy-architect)
**Branch:** feat/v0.3-baseline-refactor
**Commits:**
- c48cefa refactor(data,metrics): complete v0.3 modular refactor
- v0.3.0 tag

**Summary:**
- Committed the working tree refactor as v0.3.0 baseline.

**Decisions:**
- Chose Conventional Commits with full body explaining motivation.

**Pitfalls:**
- None.

**Next steps:**
- Phase 1: tooling base.
```

## agent_memory.md format

```markdown
## Findings

### 2026-06-01
- `cgmpy/data/README.md` had 240 lines of internal documentation that would
  be better suited in `docs/`. Will move in Phase 6 / 7.
```
