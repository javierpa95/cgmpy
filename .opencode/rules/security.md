# Security Rules

> Per-domain rules for the `security-guard` agent and for all contributors.

## Non-Negotiable Prohibitions

CGMPy processes medical data. The following are **absolute prohibitions** in
**any committed file** (code, tests, examples, docs, fixtures, comments):

1. ❌ **PHI** — patient identifiers, names, emails, MRN, device serial numbers.
2. ❌ **Real CGM exports** — even if you think the patient is anonymous.
3. ❌ **Hardcoded credentials** — passwords, API keys, tokens, production URLs.
4. ❌ **`pickle.loads()`** on untrusted input.
5. ❌ **`eval()` / `exec()`** on user-provided data.
6. ❌ **Disabled auth / rate limiting** "temporarily for testing".

If a request requires any of the above, **stop** and report to
`javierpenatearrieta@gmail.com` per `SECURITY.md`.

## Required Practices

### Logging

- Use the `logging` module, not `print()`.
- Never log full glucose arrays or patient IDs.
- Sanitize exception messages: `logger.error("Failed to load %s", path)` not
  `logger.error("Failed to load %s: %s", path, df.head())`.

### Input Validation

- Validate CSV columns before parsing.
- Validate glucose values are within physiological ranges (40–400 mg/dL by
  default; configurable in `GlucoseTargets`).
- Use `pd.to_datetime(..., errors="coerce")` not `format=...` for robustness.

### Dependencies

- Pin via `>=` and `~=` in `pyproject.toml`.
- Run `pip-audit` in CI (Phase 5).
- Avoid packages with known CVEs.

### CI/CD

- Use `secrets.X` in workflows, not literal values.
- Mask secrets in logs: `echo "::add-mask::$SECRET"`.
- Limit `pull_request_target` triggers to read-only operations.

## Anonymization Helper

If real data needs to be shared (e.g., to reproduce a bug), require the
contributor to run:

```bash
python scripts/anonymize_cgm.py --in raw.csv --out anonymized.csv
```

The script:

1. Shifts timestamps by a random offset (30 ± 7 days).
2. Replaces patient IDs with `PATIENT_NNN`.
3. Drops PII-suspect columns (`*name*`, `*email*`, `*id*`, `*dob*`, `*mrn*`,
   `*phone*`).
4. Keeps the structure intact so the bug is reproducible.

## Reporting a Vulnerability

See `SECURITY.md`. The summary is: **email `javierpenatearrieta@gmail.com`**,
do not open a public issue. Expect a response within 72 hours.

## Risk Levels for the Agent

When auditing, use:

| Level     | Trigger                                                   | Action            |
|-----------|-----------------------------------------------------------|-------------------|
| 🔴 CRIT   | PHI leak, hardcoded credentials, RCE                     | Block the commit  |
| 🟠 HIGH   | Logged identifiers, weak secret handling, CVE in dep      | Block the PR      |
| 🟡 MED    | Missing input validation, broad exception catching        | File follow-up    |
| 🟢 LOW    | Best-practice deviation, no immediate risk                | Optional          |
