# PyTorch Learning Hub

Source for `https://pytorch.dazu.xyz/`: an extensible Material for MkDocs library of original PyTorch explanations, visual interactions, and runnable projects.

The site helps people learn, review, and debug PyTorch. It does not track courses, progress, completion, accounts, analytics, or personal learning history.

Read [CONTEXT.md](CONTEXT.md) before changing the information architecture, routes, project behavior, publishing workflow, or content policy.

Use [CHANGELOG.md](CHANGELOG.md) for verifiable history and [AGENTS.md](AGENTS.md) for Codex continuity. When checked out inside DaZu, the [parent README](../README.md) is the workspace index. That relative link is only available in the combined local workspace; this library can also run as a standalone repository.

## Verified recall publication — 2026-10-01

Hub source `a9b734f` was pushed independently to `main`; [GitHub Pages run 36789441624](https://github.com/danielmza-web/pytorch-learning-hub/actions/runs/36789441624) passed build, checks and deployment. DaZu website source `ef1ac3d` was pushed to `main` and published from `site/` only in [Netlify deploy 6abd966abd624111e6e967df](https://app.netlify.com/projects/dazu/deploys/6abd966abd624111e6e967df). Both reviewed source pushes had zero divergence.

Production verification passed 68 HTTPS route/resource checks, including all 24 learning pages, the nine DaZu canonical routes and six aliases. All 34 compared resources matched their publication source: committed Git blobs for Pages (LF normalization included), working files for Netlify. Browser checks of the 24 learning routes found no page overflow at desktop/default and 390-pixel mobile widths, no broken loaded images or failed diagram labels. Live metrics rejected negative counts; padding at length 8 displayed five PAD positions and eight removed tokens (IDs 9–16); search and direct anchors opened the matching details; copied code matched the selected text. Published training curves were visually inspected. CUDA/AMP, full dataset training, pretrained downloads and physical-device touch remain unexecuted.

Documentation-only follow-ups are synchronized separately. They do not change DaZu static files or require another Netlify deployment; Hub follow-ups trigger the same Pages workflow and their matching run must be checked.

## Current recall features — 2026-10-01

Existing routes now show compact explanations and local diagrams first; long code/variants are expandable. Direct anchors and search results reveal the relevant details. The project gallery uses mobile cards and the reference groups functions by purpose. Start here explains how guides, projects and reference connect.

The three selected Course 2 workflows have retained CPU reports and reproducible new figures under `docs/assets/data/recall-2026-10-01/`. Run `python scripts/render_recall_evidence.py` to render the new evidence charts and original transform illustration; it does not replace historical EMNIST/regression figures. Training benchmarking is optional: `python examples/training_comparison.py --benchmark --device cpu` (CPU default), `--device auto` or `--device cuda`. CUDA additionally attempts AMP and records timing, throughput, peak allocation and available CUDA memory. No CUDA run is claimed for this implementation session.

EMNIST/Nature now reserve 20% of the training pool for validation, restore the minimum-loss checkpoint and evaluate official test at the end. Historical EMNIST results are explicitly retained as the original configuration. The source archive remains read-only. When editing the new shared recall CSS/JS, synchronize their copies in DaZu's `site/learn/pytorch/assets/` and validate both independent repositories.


Local browser verification on 2026-10-01 covered all 24 learning routes (19 Hub and five DaZu) at desktop 1440×900 and mobile 390×844. The four replacement diagrams displayed their real stage labels; new transform/task/metric charts were inspected as rendered images. No page overflow or broken loaded images remained; lazy Fundamentals images were checked after scrolling into view. Metrics zero/absent positives/invalid values, padding masks/truncation, keyboard disclosure activation, direct inner anchors, same-anchor search reopening and code-copy content passed. The Hub now uses the modern clipboard API: the selected folded code and DaZu code were independently compared with clipboard text, then the prior clipboard was restored. Both themes were inspected; the new matrix fits a phone column and keyboard-focus text uses dark ink on the orange background. 73 rendered guide Python excerpts parsed successfully. Formal whole-site WCAG certification and physical-device touch are not claimed.

## Previous verified release — 2026-09-30

Orientation/example source `8670ca5` was pushed to `main`; [Pages run 36777275109](https://github.com/danielmza-web/pytorch-learning-hub/actions/runs/36777275109) completed successfully, including the new offline workflow checks. All nineteen public pages returned HTTPS 200. Live mobile navigation reached the new comparison project; accumulation, noise and padding controls responded without logged errors. A follow-up replaces home tables with paragraphs/cards for phone readability. DaZu quick guides were released separately to Netlify. This verifies publication, not full model training or every optional ecosystem snippet. Documentation-only follow-ups also trigger Pages; consult the current commit/run for their status.

## Recorded hosting state

The previous handoff records the GitHub Pages migration as verified on 7 August 2026. The 2026-09-03 review confirmed local source configuration, not live DNS, HTTPS, account access, workflow runs or rollback availability.

| Item | Recorded configuration |
| --- | --- |
| Repository | `https://github.com/danielmza-web/pytorch-learning-hub` |
| Production branch | `main` |
| Public site | `https://pytorch.dazu.xyz/` |
| DNS | CNAME to `danielmza-web.github.io` |
| Publishing | GitHub Actions workflow `.github/workflows/pages.yml` |
| HTTPS | Certificate active; **Enforce HTTPS** enabled |
| Rollback snapshot | `https://dazu-pytorch.netlify.app/` |

The rollback snapshot is intentionally detached from `pytorch.dazu.xyz`. Do not deploy new versions to it. Historical local `.netlify/` metadata was removed in the 2026-09-03 cleanup; it must not be recreated as a release mechanism.

## Content structure

The public navigation follows a connected route: Start here → Fundamentals → Improve training → Work with vision → Work with text → projects → Quick reference → About. Each of the four guides has two pages. There are 19 maintained Markdown pages and 20 generated HTML pages including the error page.

The maintained source pages are:

- `docs/index.md`: what the Hub contains, guide order, selected Course 2 examples and how to use the library; it does not repeat the Fundamentals lesson.
- `docs/guides/<guide>/`: exactly two substantial pages per published guide.
- `docs/projects/`: project gallery and one complete page per runnable project.
- `docs/reference/index.md`: reminder, cheatsheet, troubleshooting, and glossary.
- `docs/about/index.md`: purpose, sources, attribution, and results policy.

The six training, vision and text pages cover metrics, schedulers, Optuna, model budgets, DataLoader tuning, Lightning, profiling, precision, accumulation, image noise, pretrained weights, transfer learning, tokens, padding, embeddings and text classifiers. Fundamentals adds tensor/storage details, loss contracts, model inspection and safe dataset splits. Each page connects concepts, functions, important parameters, original excerpts, common mistakes and a recall check. The reference has a function finder with direct section links. [Course coverage](notes/course-coverage-2026-09-30.md) records the read-only source review and its limits; protected course material is not included in the public site.

PyTorch Fundamentals currently uses:

1. `docs/guides/fundamentals/core-workflow.md`
2. `docs/guides/fundamentals/vision-real-data.md`

Do not create future-guide placeholders. Add a guide to `mkdocs.yml` only after both pages contain useful material.

`examples/recall_patterns.py` contains the original, tested impulse-noise, masked-pooling, EmbeddingBag-collation and gradient-accumulation excerpts included in the guides. It uses the existing project dependencies and performs no downloads. Optional ecosystem excerpts need `optuna`, `lightning`, `torchmetrics` or `transformers`; these are not required to build the website or run the seven maintained projects. On 2026-09-30, Transformers was available locally, while Optuna, Lightning and TorchMetrics were absent. Nothing was installed for this review, and those optional snippets have not been executed end to end.

## Update locally

Requirements: Python 3.14 to match `.github/workflows/pages.yml`, Git for source synchronization, and a browser. Node.js is needed for the JavaScript syntax check; the workflow uses the runner's available Node version. Documentation dependencies are pinned to MkDocs `1.6.1` and Material `9.7.7`. Project dependencies have version ranges, not a complete lockfile: Matplotlib `>=3.8,<4`, Pillow `>=12,<13`, Torch `>=2.2,<3`, and TorchVision `>=0.17,<1`.

On this PC, Python 3.14.7 and all declared dependencies are installed globally. Use `py -3.14` instead of `python` in the commands below; no installation is needed for the verified environment. The moved, broken `.venv` was removed. To preview from this repository, run `py -3.14 -m mkdocs serve`.

On another PC, clone this repository separately (the parent DaZu clone omits it), then run these commands from this repository root. The Python 3.14 Windows launcher must already be available; elsewhere use a Python 3.14 executable in place of `py -3.14`.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m mkdocs serve
```

Open `http://127.0.0.1:8000/`. Stop with Ctrl+C. Recreate `.venv/` locally; do not copy the old environment, whose configuration contains machine-specific paths. The commands below use `python` to mean this environment's executable: use `.\.venv\Scripts\python.exe` on Windows without changing execution policy, or `.venv/bin/python` on Unix.

The helper below uses `.venv/Scripts/python.exe` when present, otherwise `py -3.14` on Windows or `python` on other setups. It starts preview without installing or upgrading dependencies:

```powershell
.\scripts\preview.ps1
```

Write educational content in Markdown, include `last_reviewed` metadata, store local images under `docs/assets/images/`, and keep interactions in the shared JavaScript and stylesheet.

A local compatibility check on 2026-09-03 used Python 3.14.7, Pillow 12.3.0, Torch 2.14.0+cu130, TorchVision 0.29.0+cu130 and Material 9.7.7. The strict build, content/link validators and all four existing smoke examples passed on a disposable copy. This does not validate complete dataset training or a GitHub Actions run. The local workflow target is now 3.14. Check the Actions run for the current commit separately; local validation does not establish deployment success.

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
python tests/validate_recall_patterns.py
python tests/validate_selected_examples.py
python tests/validate_learning_contracts.py
node tests/validate_recall_visuals.cjs
python tests/smoke_in_copy.py
python -m py_compile scripts/generate_concept_visuals.py examples/regression_demo.py examples/emnist_model.py examples/robust_dataset.py examples/nature_cnn.py
node --check docs/assets/javascripts/interactions.js
node --check docs/assets/javascripts/recall-visuals.js
```

The GitHub Pages workflow repeats these checks and publishes only when they all pass. `tests/smoke_in_copy.py` runs the smoke suite in a disposable checkout. Calling the underlying `tests/smoke_examples.py` directly still runs regression training, writes `artifacts/regression_metrics.json`, and regenerates the tracked `docs/assets/images/regression-comparison.png`; inspect `git diff` afterward. Building replaces generated `site/`, and Python compilation creates caches. These are not read-only checks.

For a review that must preserve non-documentation files, use `python -B tests/validate_content.py`, `python -B tests/validate_built_site.py` after a fresh strict build, `python -B tests/validate_recall_patterns.py`, and the JavaScript syntax check. The built-site validator rejects absent output and checks every maintained source page. Run the smoke suite on a disposable source copy to preserve checked-in visuals and retained experiment results.

When nested inside DaZu, build here first, then run `python -B tests/validate_pytorch_hub.py` from the parent root to verify all four quick-guide links against the generated library. Content validation compares navigation with the actual source inventory and retains the two-page-per-guide rule. Update the parent guide list deliberately when adding another guide.

## Publish with GitHub Pages

The public repository is `danielmza-web/pytorch-learning-hub`. Publishing is automatic after a successful push to `main`:

```powershell
git status --short
git add <reviewed-files>
git commit -m "Describe the PyTorch documentation update"
git push origin main
```

`.github/workflows/pages.yml` builds and deploys the `site/` artifact. `docs/CNAME` preserves the custom domain `pytorch.dazu.xyz`.

Replace `<reviewed-files>` with the intended file paths. A push to `main`, including a documentation-only push, starts publication; do it only when release is intended. GitHub write access and an enabled Pages/Actions configuration are required; the workflow uses GitHub-managed token/OIDC permissions (`pages: write`, `id-token: write`). No project API keys are needed for local use. Reauthenticate separately on another PC and do not copy credentials or historical `.netlify/` state.

This documentation repository no longer deploys to Netlify. Detailed-library edits therefore use no Netlify production-deploy credits. DaZu remains a separate build-free repository and is published only when its own quick-guide pages change.

After a successful workflow, verify the changed page, direct anchors, search, mobile navigation, images, code-copy controls, and affected interactions. Check the run at `https://github.com/danielmza-web/pytorch-learning-hub/actions`; do not change DNS or use Netlify for an ordinary content release.

## Add a future PyTorch guide

1. Create `docs/guides/<guide-slug>/`.
2. Copy `templates/guide-core.md` and `templates/guide-applied.md` into that folder with meaningful filenames.
3. Replace the prompts with original explanations, examples, visuals, and relevant project links.
4. Add both pages as one guide group under `Guides` in `mkdocs.yml`.
5. Add a short DaZu page at `/learn/pytorch/<guide-slug>/` and list it on `/learn/pytorch/` only when ready.
6. Validate, commit, and push.

OpenCV, YOLO, and other non-PyTorch topics do not belong in this repository.

## Selected Course 2 workflows

Three original CPU-first examples combine mechanisms across the labs. Their default runs are small and offline:

```powershell
python examples/training_comparison.py
python examples/vision_head.py
python examples/text_bags.py
```

Training compares learning rates with shared initial weights/splits, macro F1, plateau scheduling and sample-correct accumulation. Vision uses impulse noise and a replacement ResNet head with fixed backbone parameters and BatchNorm statistics; the default random/synthetic run proves mechanics only. `--data my-images --pretrained` uses matching train/val class folders and may download ImageNet weights. Text builds a training-only vocabulary, collates offsets, pools embeddings and computes class weights. Each writes JSON plus a checkpoint where relevant under its selected `--output` directory. Keep ignored results separately when you need to reproduce a measured claim.

`tests/validate_selected_examples.py` uses temporary outputs and checks metrics, learning, frozen backbone/buffers, real-file loading/class mismatch, saved metadata and pooling invariance without downloads or retained-image changes. It does not validate pretrained transfer quality, CUDA or a large dataset.

## Complete foundation projects

```powershell
python examples/regression_demo.py
python examples/emnist_model.py
python examples/robust_dataset.py
python examples/nature_cnn.py
```

Regression runs on CPU with seeded synthetic data and has no command-line flags. The three image projects use small seeded configurations and default to `--device auto`, selecting CUDA if available; pass `--device cpu` to require CPU. They support `--full --device cuda` for longer runs and `--smoke-test` for checks without dataset downloads. CUDA requires compatible hardware, drivers and a compatible Torch/TorchVision installation; it is optional. Do not assume identical numerical results across devices or dependency versions.

EMNIST downloads EMNIST Letters; the robust pipeline downloads CIFAR-10; Nature CNN downloads CIFAR-100 through TorchVision. Network access and disk space are needed on first use. Outputs go under ignored `artifacts/` and downloads under ignored `data/`, except regression also writes the maintained comparison image under `docs/assets/images/`. Preserve selected metrics/checkpoints separately when transferring reproducibility evidence; they are not included in a clone. Restore caches or allow the examples to download them again.

Local website preview needs no login. Package installation and dataset downloads need external access; Material's font configuration requests Google Fonts. Images, Mermaid and interactions are stored locally. The installed dependency combination was checked locally. Current DNS/HTTPS settings, dataset download availability and full training require separate verification when needed.

## Local data and cleanup

The 2026-09-03 cleanup removed the broken `.venv`, historical `.netlify` metadata, bytecode caches, empty CIFAR-10 download and unused EMNIST variants plus the original archive. The maintained example uses `split="letters"`; its four train/test image/label files remain under `data/emnist/EMNIST/raw/` (about 109 MiB) and load without downloading. Other EMNIST variants are not currently used and would require a download if requested. CIFAR-10/CIFAR-100 examples download their inputs on first complete use.

Keep `artifacts/`: the retained `demo/emnist` run supports the published 24.4% CPU result and its three images match the maintained documentation images. Keep `site/` for immediate local reference and cross-site anchor checks; it is generated and can be rebuilt. Empty retired content directories were removed. Active content, templates, examples and shared images remain intentional inputs, even where images are duplicated in the independent DaZu site.

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
