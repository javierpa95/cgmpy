# ADR 0001 — Record architecture decisions

* Status: Accepted
* Date: 2026-06-01
* Deciders: @javierpa95

## Context and Problem Statement

CGMPy is a clinical-data library with a non-trivial design space
(modular vs. monolithic, NumPy-first vs. pure Python, AGATA parity vs.
innovation, etc.). Without a written record, the rationale for past
decisions is lost and we risk making inconsistent choices over time.

We want lightweight, in-repo, version-controlled decision records.

## Considered Options

* **MADR (Markdown Any Decision Records)** — small, well-known, easy to
  author in plain Markdown.
* **Lightweight ADR by Jeff Tyree and Alexander Zaks** — even shorter,
  but lacks "considered options" and "consequences" sections.
* **No ADRs** — rely on commit messages and `AGENTS.md`.

## Decision Outcome

Chosen option: **MADR**, because it is the de-facto standard for
Python and Web projects, supports both lightweight and detailed
records, and is plain Markdown (no special tooling required).

### Consequences

* Good, because every significant decision is documented in a single
  place (`docs/architecture/decisions/`).
* Good, because contributors can read past ADRs to understand context.
* Bad, because writing good ADRs takes time. We will only use them
  for significant decisions, not for every small refactor.

## Pros and Cons of the Options

### MADR

* Good, plain Markdown.
* Good, structured (Context, Options, Decision, Consequences).
* Good, Git-friendly.
* Bad, slightly more verbose than minimal ADRs.

### Lightweight ADR (Tyree & Zaks)

* Good, very short.
* Bad, no "considered options" section — easy to skip alternatives.
* Bad, no "consequences" section — easy to miss downsides.

### No ADRs

* Good, no overhead.
* Bad, rationale is lost.
* Bad, hard to onboard new contributors to past decisions.

## More Information

* Inspired by the [Digital Twin Project](https://github.com/) (where
  ADRs were a missing piece that hurt onboarding).
* See [MADR](https://adr.github.io/madr/) for the template.
