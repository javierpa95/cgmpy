---
name: Bug report
about: Report a bug to help CGMPy improve
title: "[Bug]: "
labels: ["bug", "needs-triage"]
assignees: []
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 📋 Reproduction

Minimal, complete example to reproduce the issue:

```python
# Paste your code here
```

## 📊 Data (anonymized!)

If the bug requires specific data:

- **Is the data synthetic or anonymized?** (required — see [SECURITY.md](../SECURITY.md))
- **What format is it in?** (CSV, Parquet, DataFrame)
- **How many rows / time span?**
- **What device does it come from?** (Dexcom, Libre, ...)

> ⚠️ **DO NOT paste real CGM data, even if you think it's anonymous.** Use `python scripts/anonymize_cgm.py` first.

## ✅ Expected Behavior

What you expected to happen.

## ❌ Actual Behavior

What actually happened. Include the full traceback:

```
Paste the full traceback here
```

## 🖥️ Environment

- **OS**: (e.g., Windows 11, macOS 14, Ubuntu 22.04)
- **Python version**: (output of `python --version`)
- **CGMPy version**: (output of `pip show cgmpy`)
- **Install method**: (pip, uv, from source)
- **Optional dependencies installed**: (e.g., `agata`, `pyarrow` version)

## 📝 Additional Context

Add any other context, screenshots, or links here.

## ✅ Checklist

- [ ] I have searched the [existing issues](../../issues) for this bug.
- [ ] I have read [SECURITY.md](../SECURITY.md) and will **not** include real patient data.
- [ ] I am using the latest released version of CGMPy (or the latest `main`).
