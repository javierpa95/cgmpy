# What is MkDocs?

## What is it?

MkDocs turns Markdown files into a website. The documentation site you are reading right now was built with MkDocs. You write pages in plain Markdown, and MkDocs compiles them into a clean HTML site with navigation, search, and a beautiful theme.

## Why we use it

GitHub can render `.md` files, but a real documentation website is much more useful -- especially for clinical users who might not be comfortable reading code on GitHub. MkDocs gives us:

- A table of contents / navigation sidebar
- Full-text search
- Auto-generated API reference from Python docstrings
- Versioned documentation (each release keeps its own docs)
- A professional look with the Material theme

## How it works

```
docs/*.md  ──→  MkDocs  ──→  HTML website
(markdown)        (build)      (site/ directory)
```

You edit the Markdown files in `docs/`, run `make docs-serve` to preview locally, and when you are happy, the built site is deployed to GitHub Pages.

## Key features

- **Material theme** -- the clean, readable theme you see on this site
- **mkdocstrings** -- auto-generates API documentation from your Python docstrings (Google style). No need to write API docs by hand.
- **Full-text search** -- users can search across the entire documentation
- **Versioned docs (mike)** -- each release gets its own version of the docs, so users reading an older version see the correct documentation
- **Live preview** -- run `make docs-serve` and a local server updates the site as you edit files

## How to write docs

1. Add a `.md` file to `docs/`
2. Add it to the navigation section in `mkdocs.yml`
3. Run `make docs-serve` to see it live
4. Write your content

The API reference pages are auto-generated from docstrings -- you do not need to write those by hand. Just add a Google-style docstring to your Python function and it will appear in the API docs automatically.

## Why it is good for learners

Documentation is how you share what you have built. MkDocs makes it easy to create professional-looking documentation without knowing HTML, CSS, or JavaScript. You just write Markdown, and the tool handles the rest. For CGMPy, good documentation is especially important because many users are clinicians or researchers who need clear, readable explanations of how each metric works.
