# ADR 0004 — Asymmetric agent harness for AI-assisted development

* Status: Accepted
* Date: 2026-06-02
* Deciders: @javierpa95

## Context and Problem Statement

CGMPy is developed by a solo maintainer who uses AI coding assistants
(OpenCode, Cursor, Copilot) extensively. Early sessions showed a
recurring problem: the AI would make changes outside its area of
competence (e.g., modifying security-sensitive code while working on
a plotting feature), or would fail to update documentation after
changing code.

We need a multi-agent system where each agent has bounded authority
and clear domain knowledge, but the solo maintainer must remain the
sole decision-maker and code-committer.

## Considered Options

* **Symmetric agents** — every agent can read and edit any file. Trust
  the AI to stay on task.
* **Asymmetric agents (chosen)** — read-only specialists for domain
  knowledge, a limited set of execution agents for edits, one
  orchestrator to coordinate.
* **Monolithic single-agent** — one agent prompt that covers the entire
  codebase.

## Decision Outcome

Chosen option: **Asymmetric agents**, because:

1. The solo maintainer must remain the single point of commit.
   Execution agents have edit capability but never commit — they
   produce diffs for human review.
2. Read-only specialists (the Guardians: `@data-guardian`,
   `@metrics-guardian`, `@plotting-guardian`, `@agata-integrator`,
   `@security-guard`) are consulted for domain advice but cannot
   modify files.
3. Execution agents (`@test-engineer`, `@docs-maintainer`,
   `@release-manager`) can edit files in their domain, but only
   when delegated by the orchestrator.
4. The orchestrator (`cgmpy-architect`) loads full context and
   delegates to specialists. It is the only agent authorized to
   decide PASS/FAIL on the final review.

### Consequences

* Good, because domain experts don't accidentally step on each other.
* Good, because security-sensitive decisions are always reviewed by
  `@security-guard` before execution.
* Good, because documentation updates are baked into the workflow
  (docs-maintainer is always looped in after code changes).
* Bad, because the orchestration overhead adds latency (multiple
  agent calls per task). Acceptable for a solo maintainer working
  asynchronously.

## Pros and Cons of the Options

### Asymmetric agents (chosen)

* Good, clear separation of concerns.
* Good, security by design (editors don't touch sensitive areas
  without consulting the security guard).
* Good, documentation is never forgotten (docs-maintainer is
  mandatory in the workflow).
* Bad, more files to maintain than a monolithic approach.

### Symmetric agents

* Good, simpler architecture (one agent type, fewer config files).
* Bad, harder to enforce domain boundaries — a plotting agent could
  accidentally modify security policy.
* Bad, no built-in documentation update enforcement.

### Monolithic single-agent

* Good, single prompt to maintain.
* Bad, prompt grows unboundedly and becomes ineffective.
* Bad, no specialization — the agent must know everything about
  everything.
