#!/usr/bin/env python
"""Check that changes in cgmpy/ are accompanied by doc/CHANGELOG updates.

Run by pre-commit on every commit. Warns (does not block) when
cgmpy/ files are staged without corresponding docs/ or CHANGELOG.md
changes.

Exit codes:
    0 — no issues
    0 — only warnings (pre-commit shows them but the commit succeeds)
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    # Files staged for this commit
    diff = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    )
    staged = [line.strip() for line in diff.stdout.splitlines() if line.strip()]

    cgmpy_changed = any(f.startswith("cgmpy/") for f in staged)
    docs_changed = any(
        f.startswith("docs/") or f in {"CHANGELOG.md", "AGENTS.md", "README.md"}
        for f in staged
    )

    if cgmpy_changed and not docs_changed:
        print(
            "WARNING: cgmpy/ changed but docs/, CHANGELOG.md, AGENTS.md, or "
            "README.md did not.",
            file=sys.stderr,
        )
        print(
            "       Per the Documentation Golden Rule (AGENTS.md §3), "
            "consider updating:",
            file=sys.stderr,
        )
        print("         - docs/user-guide/ — user-facing behavior", file=sys.stderr)
        print("         - docs/api/ — auto-generated, docstring-only", file=sys.stderr)
        print("         - CHANGELOG.md — if user-facing (feat/fix)", file=sys.stderr)
        # Do not block; just warn. Set returncode=0 to commit succeeds.

    return 0


if __name__ == "__main__":
    sys.exit(main())
