# What is Release Please?

## What is it?

Release Please is a GitHub bot that creates a pull request to release a new version of the library. It reads all the commit messages since the last release, determines whether the new version is a major, minor, or patch bump, updates the version in `pyproject.toml`, generates the changelog entries, and opens a release PR -- all automatically.

## Why we use it

Manual releases are error-prone. Did you remember to update the version number? Did you update the changelog? Did you catch all the breaking changes? Release Please automates the entire process so that a human only needs to review and merge the generated PR.

## How it determines the version bump

Release Please reads your Conventional Commit messages and decides the next version number:

- `fix:` commits -- patch bump (0.5.1 to 0.5.2)
- `feat:` commits -- minor bump (0.5.2 to 0.6.0)
- `BREAKING CHANGE:` in the commit footer -- major bump (0.6.0 to 1.0.0)

If there are both `fix:` and `feat:` commits since the last release, it picks the highest bump (minor wins over patch). If there is a breaking change, it becomes a major release.

## The release workflow

```
1. Developer merges conventional commits to main
2. Release-please bot opens a release pull request
3. Maintainer reviews the generated changelog
4. Maintainer merges the release pull request
5. Release-please creates a GitHub Release and git tag
6. (Future) Automatic PyPI publish
```

The release PR includes the updated version number, the changelog for this release, and a summary of all changes grouped by type.

## Configuration files

- `release-please-config.json` -- tells Release Please about the project structure, which files to update, and how to handle the version bump
- `.release-please-manifest.json` -- tracks the current version of the project so the bot knows what the last release was

## Why it is good for learners

Release Please shows how automation can eliminate repetitive, error-prone tasks. You focus on writing code and writing good commit messages. The bot handles the admin work -- version bumps, changelog generation, and GitHub releases. It is a great example of how investing a little time in tooling saves hours of manual work over the life of a project.
