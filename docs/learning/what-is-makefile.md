# What is the Makefile?

## What is it?

A Makefile is a list of shortcuts. Instead of typing:

```
python -m pytest tests/ -x --tb=short -q --cov=cgmpy --cov-report=term --cov-report=xml
```

you type:

```
make test-coverage
```

A Makefile lets you define named commands (called "targets") so you don't have to remember long, complex shell commands.

## Why we use it

You should not need to remember 30 different commands and their flags. The Makefile remembers them for you. If you want to run the tests, you type `make test`. If you want to check code style, you type `make lint`. One command, one job.

It also ensures everyone on the team runs the **same** command. No more "works on my machine" because someone used different pytest flags.

## Most useful targets

| Command | What it does |
|---------|-------------|
| `make test` | Run all tests |
| `make test-fast` | Quick smoke test (skip slow tests) |
| `make test-coverage` | Tests with coverage report |
| `make lint` | Check code style with ruff |
| `make lint-fix` | Auto-fix style issues |
| `make typecheck` | Run mypy type checker |
| `make security` | Run bandit security scanner |
| `make docs-serve` | Preview documentation locally |
| `make build` | Build distribution package |

## How to read a Makefile

A target looks like this:

```makefile
.PHONY: test-fast
test-fast:          # target name
    pytest -m "not slow" -q  # command (indented with tab)
```

- `.PHONY` tells Make this is not a real file — always run it.
- `test-fast:` is the name you type after `make`.
- The line below is the command to run. It must be indented with a **tab**, not spaces.

## Why it is good for learners

You can contribute to CGMPy without knowing every tool in the stack. Want to check if your code is correctly formatted? Run `make lint`. Want to make sure you have not broken anything? Run `make test`. The Makefile is your safety net — it runs the exact same checks that CI will run, so you catch problems before they reach a pull request.
