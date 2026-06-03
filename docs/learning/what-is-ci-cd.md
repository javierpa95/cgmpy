# What is CI/CD? (GitHub Actions)

## What is it?

CI/CD means: every time you push code to GitHub, a robot checks it. Runs tests, checks formatting, scans for security issues. If anything fails, the robot tells you before the code reaches your users.

- **CI** stands for Continuous Integration. Every code change is automatically integrated, built, and tested. If someone pushes broken code, you know within minutes.
- **CD** stands for Continuous Deployment (or Delivery). Once tests pass, the code is automatically packaged and deployed — or in CGMPy's case, published to PyPI.

## Why we use it

Manual testing is slow and error-prone. CI/CD makes quality automatic. Push code, get results in 5 minutes.

Before CI/CD, the workflow was: write code, run tests locally (if you remembered), cross your fingers, push to production. With CI/CD, every push runs the same checks on a clean machine. No "it works on my machine" surprises.

## CGMPy's CI pipeline

This is what happens when you push to main:

```
1. Lint -> ruff checks code style
2. Type check -> mypy checks types
3. Security -> bandit scans for vulnerabilities
4. Test (3 OS x 3 Python versions) -> pytest on Linux, Windows, macOS
5. Coverage -> computes test coverage, uploads to Codecov
6. Docs -> builds and deploys mkdocs site
7. CodeQL -> GitHub's security analysis
```

Each step is a gate. If any gate fails, the pipeline stops and the PR gets a red cross. The reviewer (or the author) knows exactly what went wrong without running anything locally.

### What each step does

1. **Lint (ruff):** Enforces consistent code style. No tabs where spaces should be, no unused imports. It's like a spelling checker for code.
2. **Type check (mypy):** Catches type errors as described in the mypy article. Ensures that `mean()` never receives a string.
3. **Security (bandit):** Scans for hardcoded passwords, unsafe `eval()` calls, SQL injection patterns. In a medical data library, this is non-negotiable.
4. **Test (3 OS x 3 Python versions):** Runs the full test suite on Linux, Windows, and macOS, with Python 3.10, 3.11, and 3.12. A bug on Windows is caught before it reaches a user.
5. **Coverage:** Measures what percentage of the codebase is exercised by tests.
6. **Docs:** Builds the mkdocs site and deploys it. If the documentation has a broken link or invalid syntax, you find out here.
7. **CodeQL:** GitHub's own security analysis. It looks for patterns that could lead to vulnerabilities.

## What is a workflow?

A YAML file in `.github/workflows/` that tells GitHub what to do. Each workflow is a set of jobs that run on specific triggers (push, PR, schedule).

Here is a simplified version of a CI workflow:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest
      - run: mypy cgmpy
```

What this does:

- **`on:`** — when this workflow runs. Here, on pushes and PRs to main.
- **`jobs:`** — what to run. We define a `test` job.
- **`strategy.matrix:`** — run the same job with different parameters. Here, three Python versions.
- **`steps:`** — the actual commands. Check out the code, install Python, install dependencies, run tests, run mypy.

GitHub Actions provides the virtual machine (runs-on). You don't need to set up anything locally. The workflow file is the entire configuration, committed to the repository.

## Security scanning

Bandit checks for:

- Hardcoded passwords or API keys
- SQL injection vulnerabilities
- Unsafe use of `eval()`, `exec()`, or `pickle.loads()`
- Use of `assert` in production code (it can be disabled with `-O`)
- Known vulnerable function calls

In a medical data library, running bandit on every push is non-negotiable. If someone accidentally commits a secret or introduces an unsafe pattern, the pipeline catches it.

## Why it's good for learners

You don't need to remember to run 10 commands. CI/CD runs them all for you.

When you open a pull request, the CI results are right there in the GitHub interface. Green checkmarks mean everything is fine. Red crosses point you directly to the failure. You don't need to set up a local environment with multiple Python versions to verify your changes work everywhere.

CI/CD turns quality from a manual checklist into an automated process. Push code, get feedback, iterate. It is the closest thing to having a robot assistant that checks your work on every commit.
