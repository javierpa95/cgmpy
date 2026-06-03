# Learning Path: From Clinician to Python Developer

## Introduction

You know CGM data inside out. Time in range, MAGE, the Ambulatory Glucose
Profile — these are second nature. You can look at a glucose trace and spot
the dawn phenomenon before your coffee kicks in.

Now imagine building the tools that compute these numbers. Not just running
them in a spreadsheet, but writing the software that does it automatically,
reproducibly, and at scale. That is what this learning path is for.

Each step links an article that teaches one tool, then sends you to look at
the real configuration file in CGMPy. Theory meets practice, every time.

## The path

1. **Start here: what-is-pytest.md** — Write a test for mean glucose, watch
   it pass, then break something and see it fail. You will never trust a
   metric again without a test to back it up.

2. **Make it pretty: what-is-ruff.md** — Python code can be written in a
   hundred different styles. Ruff picks one and enforces it. Let the tool
   fix your formatting so you can focus on the science.

3. **Catch bugs early: what-is-mypy.md** — Type hints prevent the kind of
   mistake where you pass a patient name instead of a glucose value. MyPy
   catches these before the code runs.

4. **Stop repeating yourself: what-is-pre-commit.md** — Quality checks are
   easy to skip when you are in a hurry. Pre-commit automates them before
   every git commit so you never forget.

5. **Ship it: what-is-ci-cd.md** — Continuous integration means every pull
   request runs your tests automatically on GitHub. No more "it works on my
   machine".

6. **Talk to your future self: what-is-conventional-commits.md** — A commit
   message like "fix stuff" is useless six months later. Conventional
   commits force you to be clear. Your future self will thank you.

7. **One command to rule them all: what-is-makefile.md** — Instead of
   remembering six different commands, a Makefile gives you shortcuts:
   `make test`, `make lint`, `make docs`. One command, one job.

8. **Document or it didn't happen: what-is-mkdocs.md** — This website you
   are reading right now is built with MkDocs. Documentation that lives
   with the code and updates automatically.

9. **Version with intention: what-is-release-please.md** — Release Please
   automates version bumps and changelogs based on your commit history.
   No more manual "oh, did I update the version number?"

## The golden rule

Read the article, then open the project and find the actual config file.
For example: read about Ruff, then open `pyproject.toml` and find the
`[tool.ruff]` section. Read about pre-commit, then open
`.pre-commit-config.yaml`. Every tool in this path has a real file in this
repository. The article explains the *what* and the *why*; the config file
shows the *how* for this specific project.

Theory without practice is just trivia. Practice without theory is cargo
culting. Do both.
