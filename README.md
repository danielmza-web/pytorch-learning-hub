# PyTorch Learning Hub

Source for `https://pytorch.dazu.xyz/`: an extensible Material for MkDocs knowledge library containing original PyTorch explanations and projects.

The public site contains only courses with useful content. It does not expose course status, progress, completion state, accounts, analytics, or personal learning history.

## Manual update workflow

Work from this repository directory:

```powershell
Set-Location "C:\Users\Daniel Zurita\OneDrive\Escritorio\DaZu\pytorch-learning-hub"
```

Write or update original Markdown in `docs/`. Use `templates/course.md` for a
new course, add its chapters and local assets, then add it to `nav` in
`mkdocs.yml` only when the course contains useful public material.

Keep shared explanations in `docs/concepts/` or `docs/reference/`, add
associated work to `docs/projects/`, and include `last_reviewed` in page
metadata. Do not publish Coursera assessment prompts, protected notebooks, quiz
answers, or graded solutions.

## Local preview

```bash
python -m pip install -r requirements.txt
python -m mkdocs serve
```

Open `http://127.0.0.1:8000/`.

Or use the helper:

```powershell
.\scripts\preview.ps1
```

## Validation

```bash
python -m mkdocs build --strict
python tests/validate_content.py
python tests/smoke_examples.py
```

The example tests use synthetic, deterministic fixtures. Dataset downloads and full training runs are deliberately separate so documentation validation stays fast.
The smoke test requires PyTorch in the active Python environment; check it with
`python -c "import torch; print(torch.__version__)"` before publishing.

## Run the complete projects

Install the project dependencies once. This is intentionally separate from the
lightweight MkDocs build requirements:

```bash
python -m pip install -r requirements-projects.txt
```

Every project uses a small deterministic CPU-first default and downloads its
public data through TorchVision when needed. Use `--full --device cuda` for the
longer GPU configuration, or `--smoke-test` when you only want to validate the
code without a dataset download.

```bash
python examples/regression_demo.py
python examples/emnist_model.py
python examples/robust_dataset.py
python examples/nature_cnn.py
```

Artifacts such as predictions, confusion matrices, curves, metrics, and model
checkpoints are written below `artifacts/` and are deliberately ignored by Git.

## Add a guide collection

1. Copy `templates/course.md` into `docs/courses/<guide-slug>/index.md`.
2. Add original Markdown topics and local assets.
3. Add the guide to `nav` in `mkdocs.yml` only when it contains useful material.
4. Link shared explanations from `docs/concepts/` and `docs/reference/` rather than duplicating them.
5. Add related projects under `docs/projects/`.
6. Run all validation commands before publishing.
7. Add a neutral link from `dazu.xyz/learn/pytorch/`; never add progress or status labels.

## Manual publishing

Run the release helper only after the validation commands pass:

```powershell
.\scripts\publish.ps1 -Message "Describe the update"
```

It rebuilds the site, runs every validation command, and deploys the generated
`site/` folder to the PyTorch Netlify project. The helper always passes the
project ID explicitly (`f58cc486-94eb-4664-ba33-ae59491a53bc`) because this
nested repository can otherwise inherit the DaZu Netlify link.

To run the final deploy command yourself after a successful build:

```powershell
netlify deploy --prod `
  --site f58cc486-94eb-4664-ba33-ae59491a53bc `
  --dir=site `
  --no-build `
  --message "Describe the update"
```

Verify `https://pytorch.dazu.xyz/`, the changed page, mobile navigation,
search, affected images, code-copy controls, and affected interactive elements.

Commit the documentation change separately:

```powershell
git add .
git commit -m "Describe the PyTorch documentation update"
```

If a new course becomes public, add one neutral link to the DaZu quick hub and
release DaZu through its own validation, commit, push, and deployment process.

The main DaZu repository remains separate and manually deployed.

## Content policy

- Publish original explanations, examples, diagrams, and independently structured projects.
- Do not publish Coursera assessment prompts, protected notebooks, quiz answers, or graded solutions.
- Label conceptual charts as illustrative.
- Publish measured results only when the corresponding reproducible run is retained.
- Include a `last_reviewed` value in page metadata.
