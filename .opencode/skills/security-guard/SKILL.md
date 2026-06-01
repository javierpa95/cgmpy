# security-guard Skill

> **When to use:** before every commit, before opening a PR, and whenever you touch `cgmpy/`, `tests/`, `examples/`, `pyproject.toml`, or `.github/workflows/`.

## Pre-Commit Audit

Run all of these **before** `git commit`:

```bash
# 1. Look for hardcoded secrets
git grep -i "password\|secret\|api[_-]key\|token\|credential" --cached

# 2. Look for patient identifiers in staged files
git diff --cached --name-only | xargs -I{} grep -l -i "patient_id\|patient_name\|subject_id\|mrn" "{}" 2>/dev/null

# 3. Check for untracked CSVs that look like real data
git status --porcelain | grep -E "\?\? .*\.csv$"

# 4. Check for print statements in library code (not in tests / examples)
git diff --cached --name-only | grep "^cgmpy/" | xargs grep -n "print(" 2>/dev/null

# 5. Bandit (security linter)
bandit -r cgmpy/ -ll

# 6. Pip-audit (dependency vulnerabilities)
pip-audit

# 7. Detect-secrets (baseline scan)
detect-secrets scan --baseline .secrets.baseline
```

## Risk Severity

- 🔴 **CRITICAL** — PHI leak, hardcoded production credentials, RCE vector. **Block the commit. Open a security advisory.**
- 🟠 **HIGH** — logged identifiers, weak secret handling, deprecated dependency with CVE. **Block the PR.**
- 🟡 **MEDIUM** — missing input validation, broad exception handling. **File an issue, fix in follow-up PR.**
- 🟢 **LOW** — best-practice deviation, no immediate risk. **Optional fix.**

## Data Anonymization

If a contributor submits real data in a bug report, ask them to run:

```bash
python scripts/anonymize_cgm.py --in raw.csv --out anonymized.csv
```

The script:

1. Shifts timestamps by a random offset (e.g., 30 ± 7 days).
2. Replaces patient IDs with `PATIENT_NNN`.
3. Drops any column whose name matches `*name*`, `*email*`, `*id*`, `*dob*`, `*mrn*`, `*phone*`.
4. Keeps the column structure intact so the bug is still reproducible.

## Secrets in CI

CI workflows must **never** echo secrets. Verify `.github/workflows/*.yml`:

- Use `secrets.X` references, not literal values.
- Use `${{ github.event.pull_request.head.repo.full_name }}` only for
  read-only operations.
- Mask secrets in logs: `echo "::add-mask::$SECRET"`.

## Reporting a Vulnerability

If you find a vulnerability in CGMPy or in a dependency, follow `SECURITY.md`:

- Email `javierpenatearrieta@gmail.com`.
- Do **not** open a public GitHub issue.
- Coordinate disclosure with the maintainer.

## Common Pitfalls

- ❌ Adding `print(df)` for debugging and committing it.
- ❌ Loading a CSV with `pd.read_csv(path)` and logging `df.head()`.
- ❌ Pinning a dependency to a version with a known CVE.
- ❌ Reading user input with `eval()` (RCE).
- ❌ Using `pickle` on user-provided files.
