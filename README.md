# PyTorch Learning Hub

Source for `https://pytorch.dazu.xyz/`: an extensible Material for MkDocs knowledge library containing original PyTorch explanations and projects.

The public site contains only courses with useful content. It does not expose course status, progress, completion state, accounts, analytics, or personal learning history.

## Local preview

```bash
python -m pip install -r requirements.txt
python -m mkdocs serve
```

Open `http://127.0.0.1:8000/`.

## Validation

```bash
python -m mkdocs build --strict
python tests/validate_content.py
python tests/smoke_examples.py
```

The example tests use synthetic, deterministic fixtures. Dataset downloads and full training runs are deliberately separate so documentation validation stays fast.

## Add a course

1. Copy `templates/course.md` into `docs/courses/<course-slug>/index.md`.
2. Add original Markdown chapters and local assets.
3. Add the course to `nav` in `mkdocs.yml` only when it contains useful material.
4. Link shared explanations from `docs/concepts/` and `docs/reference/` rather than duplicating them.
5. Add related projects under `docs/projects/`.
6. Run all validation commands before publishing.
7. Add a neutral link from `dazu.xyz/learn/pytorch/`; never add progress or status labels.

## Publishing model

This repository is intended to be connected to its own Netlify site. Netlify builds the Markdown with `mkdocs build --strict` and publishes `site/`. Assign `pytorch.dazu.xyz` after the temporary Netlify URL has been verified.

The main DaZu repository remains separate and manually deployed.

## Content policy

- Publish original explanations, examples, diagrams, and independently structured projects.
- Do not publish Coursera assessment prompts, protected notebooks, quiz answers, or graded solutions.
- Label conceptual charts as illustrative.
- Publish measured results only when the corresponding reproducible run is retained.
- Include a `last_reviewed` value in page metadata.

