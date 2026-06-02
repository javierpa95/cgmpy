# Architecture Decision Records (ADRs)

This directory contains the Architecture Decision Records (ADRs) for CGMPy.

An ADR captures **one significant decision** in a short, immutable
document. The collection tells the story of how and why the project
evolved.

## Format

Each ADR follows the [MADR](https://adr.github.io/madr/) template:

```markdown
# ADR NNNN — Short title of the decision

* Status: [Proposed | Accepted | Deprecated | Superseded by NNNN]
* Date: YYYY-MM-DD
* Deciders: @maintainer1, @maintainer2

## Context and Problem Statement

[Describe the context and the problem in 2-3 sentences.]

## Considered Options

* [Option 1]
* [Option 2]
* [Option 3]

## Decision Outcome

Chosen option: "[option 1]", because [justification].

### Consequences

* Good, because [positive consequence].
* Bad, because [negative consequence].

## Pros and Cons of the Options

### Option 1

[...]

### Option 2

[...]
```

## Index

| #   | Title                                                                       | Status   |
|-----|-----------------------------------------------------------------------------|----------|
| 0001 | [Record architecture decisions](0001-record-architecture-decisions.md) | Accepted |

## How to add a new ADR

1. Copy `0001-record-architecture-decisions.md` to a new file:
   ```bash
   cp docs/architecture/decisions/0001-record-architecture-decisions.md \
      docs/architecture/decisions/0002-my-decision.md
   ```
2. Update the title, status, date, and content.
3. Add a row to the **Index** table above.
4. Open a PR with the `architecture` label.

## See also

- [MADR](https://adr.github.io/madr/) — the ADR template we follow.
- [System overview](../system-overview.md).
