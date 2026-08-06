# PyTorch Learning Hub

Source for `https://pytorch.dazu.xyz/`: an extensible Material for MkDocs library of original PyTorch explanations, visual interactions, and runnable projects.

The site helps people learn, review, and debug PyTorch. It does not track courses, progress, completion, accounts, analytics, or personal learning history.

## Content structure

The public navigation stays small:

- `docs/index.md`: orientation and recommended starting point.
- `docs/guides/<guide>/`: exactly two substantial pages per published guide.
- `docs/projects/`: project gallery and one complete page per runnable project.
- `docs/reference/index.md`: reminder, cheatsheet, troubleshooting, and glossary.
- `docs/about/index.md`: purpose, sources, attribution, and results policy.

PyTorch Fundamentals currently uses:

1. `docs/guides/fundamentals/core-workflow.md`
2. `docs/guides/fundamentals/vision-real-data.md`

Do not create future-guide placeholders. Add a guide to `mkdocs.yml` only after both pages contain useful material.

## Update locally

```powershell
Set-Location "C:\Users\Daniel Zurita\OneDrive\Escritorio\DaZu\pytorch-learning-hub"
python -m pip install -r requirements.txt
python -m mkdocs serve
```

Open `http://127.0.0.1:8000/`. The existing helper performs the same preview setup:

```powershell
.\scripts\preview.ps1
```

Write educational content in Markdown, include `last_reviewed` metadata, store local images under `docs/assets/images/`, and keep interactions in the shared JavaScript and stylesheet.

## Validate

Install the project dependencies once when running smoke tests or complete examples:

```powershell
python -m pip install -r requirements-projects.txt
```

Run every check before pushing:

```powershell
python -m mkdocs build --strict
python tests/validate_content.py
python tests/validate_built_site.py
python tests/smoke_examples.py
python -m py_compile scripts/generate_concept_visuals.py examples/regression_demo.py examples/emnist_model.py examples/robust_dataset.py examples/nature_cnn.py
node --check docs/assets/javascripts/interactions.js
```

The GitHub Pages workflow repeats these checks and publishes only when they all pass.

## Publish with GitHub Pages

The public repository is `danielmza-web/pytorch-learning-hub`. Publishing is automatic after a successful push to `main`:

```powershell
git add .
git commit -m "Describe the PyTorch documentation update"
git push origin main
```

`.github/workflows/pages.yml` builds and deploys the `site/` artifact. `docs/CNAME` preserves the custom domain `pytorch.dazu.xyz`.

This documentation repository no longer deploys to Netlify. Detailed-library edits therefore use no Netlify production-deploy credits. DaZu remains a separate build-free repository and is published only when its own quick-guide pages change.

After a successful workflow, verify the changed page, direct anchors, search, mobile navigation, images, code-copy controls, and affected interactions.

## Add a future PyTorch guide

1. Create `docs/guides/<guide-slug>/`.
2. Copy `templates/guide-core.md` and `templates/guide-applied.md` into that folder with meaningful filenames.
3. Replace the prompts with original explanations, examples, visuals, and relevant project links.
4. Add both pages as one guide group under `Guides` in `mkdocs.yml`.
5. Add a short DaZu page at `/learn/pytorch/<guide-slug>/` and list it on `/learn/pytorch/` only when ready.
6. Validate, commit, and push.

OpenCV, YOLO, and other non-PyTorch topics do not belong in this repository.

## Complete projects

```powershell
python examples/regression_demo.py
python examples/emnist_model.py
python examples/robust_dataset.py
python examples/nature_cnn.py
```

Default configurations are small, reproducible, and CPU-first. The image projects download reputable public datasets through TorchVision. Use `--full --device cuda` for the optional longer GPU configuration or `--smoke-test` to validate code paths without a dataset download.

Generated artifacts are stored under `artifacts/`; downloaded data stays under `data/`. Both are ignored by Git.

## Conceptual visuals

```powershell
python scripts/generate_concept_visuals.py
```

The script regenerates diagrams in `docs/assets/images/`. Visuals shared with DaZu also have local copies under `../site/learn/pytorch/assets/` so the two sites work independently. Inspect regenerated files and label them illustrative unless they come from a retained experiment.

## Content policy

- Publish original explanations, examples, diagrams, and independently structured projects.
- Do not publish Coursera assessments, protected notebooks, quizzes, or graded solutions.
- Clearly distinguish illustrative visuals, smoke-test output, and reproduced measurements.
- Publish model-quality results only when the retained run can reproduce them.
