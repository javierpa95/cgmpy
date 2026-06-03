# What is Ruff? (Lint + Format)

## What is it?

Ruff is two tools in one: a **linter** (it finds problems in your code)
and a **formatter** (it fixes how your code looks).

Imagine a clinical report reviewer who says: "Put values in tables, not
paragraphs. Units in parentheses. References at the end." Ruff is that
reviewer for Python code — but it can also fix many of the issues it
finds, automatically.

## Lint vs Format

These two concepts are often mentioned together, but they do different
things.

**Lint** finds bugs and bad patterns before they cause problems.
- An unused variable that suggests dead code.
- A dangerous comparison like `is` instead of `==` for strings.
- An import that does not exist (typo in the module name).
- A wildcard import that pollutes the namespace.

**Format** makes code look consistent, with no semantic impact.
- Consistent indentation (spaces, not tabs).
- Line breaks before or after operators.
- Single vs double quotes (Ruff normalises them).
- Blank lines between functions.

A lint error might indicate a real bug. A format issue is cosmetic, but
cosmetic consistency matters when ten people collaborate on the same file.

## Real CGMPy example

```python
# BAD — ruff would catch this
from cgmpy import *   # wildcard import — bad practice!

# GOOD
from cgmpy import GlucoseData  # explicit import
```

Wildcard imports make it impossible to tell where a name came from. Ruff
flags them with rule F403 and suggests explicit imports instead.

```python
# BAD — ruff would catch this
def calculate_mean(values):
    total = sum(values)
    count = len(values)
    result = total / count
    return result

# GOOD (count is used, but if it were not, ruff would flag F841)
def calculate_mean(values):
    return sum(values) / len(values)
```

## Why we use Ruff instead of Black or Flake8

Historically, Python projects used Flake8 for linting and Black for
formatting, plus isort for import sorting, plus a half-dozen other tools.
That meant:
- Five different configuration files to maintain.
- Slow execution because each tool loaded your code from disk separately.
- Occasional conflicts between what one tool wanted and another enforced.

Ruff replaces all of them. It is 10 to 100 times faster because it is
written in Rust, it has built-in autofix for many rules, and it is
configured in a single place: the `[tool.ruff]` section of `pyproject.toml`.

| Tool | Replaced by Ruff |
|---|---|
| Flake8 | `ruff check` |
| Black | `ruff format` |
| isort | `ruff check --fix` (rule I001) |
| autoflake | `ruff check --fix` (rule F841 and others) |
| pyupgrade | `ruff check --fix` (rule UP... ) |

## How to use it

```bash
# Find all issues in the current directory
ruff check .

# Find issues and fix what can be fixed automatically
ruff check . --fix

# Format all Python files in place
ruff format .

# Check formatting without modifying files (used in CI)
ruff format --check .
```

In CGMPy, you rarely need to run Ruff manually. Pre-commit runs it for you
on every commit. But when you do run it directly, these are the commands.

## Common errors and what they mean

| Code | Meaning | Fix |
|---|---|---|
| F401 | Imported module or name is never used | Remove the import, or use it |
| F841 | Variable assigned but never used | Remove the assignment, or use the variable |
| E501 | Line is too long (over 100 characters) | Break the line into multiple lines |
| I001 | Imports are in the wrong order | `ruff check --fix` sorts them automatically |
| F403 | Wildcard import (`from x import *`) | Replace with explicit imports |
| W291 | Trailing whitespace at end of line | `ruff check --fix` removes it |
| N802 | Function name should be lowercase | Rename the function to snake_case |
| D100 | Missing docstring in public module | Add a module-level docstring |

The rule codes tell you the category: `F` is from pyflakes (logic errors),
`E` and `W` are from pycodestyle (formatting), `I` is from isort (imports),
`N` is from pep8-naming (naming conventions), `D` is from pydocstyle
(docstrings).

## Why it is good for learners

You learn Python style by fixing the issues Ruff finds. It is like a
patient teacher that never gets tired of explaining. Write a long line,
and Ruff tells you to break it up. Use a vague variable name, and Ruff
suggests a better one. Every warning is a lesson.
