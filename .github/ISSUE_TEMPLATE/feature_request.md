---
name: Feature request
about: Suggest a new feature for CGMPy
title: "[Feature]: "
labels: ["enhancement", "needs-triage"]
assignees: []
---

## 🚀 Feature Description

A clear and concise description of the feature you want.

## 💡 Motivation

What problem does this solve? Why is it useful for clinical or research workflows?

## 📐 Proposed API

If you have an idea of the API, sketch it out:

```python
# Example usage
from cgmpy.metrics import ...

result = ...
```

## 🔗 References

- Clinical / academic reference (paper, DOI, URL)
- Similar feature in another library (e.g., R's `iglu`, Python's `agata`)
- Related issue or discussion

## 🎯 Acceptance Criteria

- [ ] The metric / loader / plot is implemented.
- [ ] Unit tests are added (with a known-answer dataset).
- [ ] Clinical regression test is added (if a published reference exists).
- [ ] Documentation is updated (`docs/user-guide/...`).
- [ ] `CHANGELOG.md` `[Unreleased]` is updated.
- [ ] AGATA parity is checked (if applicable).

## 📊 Priority

How important is this for your work?

- [ ] Blocking my work / study.
- [ ] Would significantly improve my workflow.
- [ ] Nice to have.

## ✅ Checklist

- [ ] I have searched the [existing issues](../../issues) for a similar request.
- [ ] I am willing to submit a PR for this feature (or a draft of it).
