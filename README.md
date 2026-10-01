# PyTorch Learning Hub

Source for `https://pytorch.dazu.xyz/`: an extensible Material for MkDocs library of original PyTorch explanations, visual interactions, and runnable projects.

The site helps people learn, review, and debug PyTorch. It does not track courses, progress, completion, accounts, analytics, or personal learning history.

Read [CONTEXT.md](CONTEXT.md) before changing the information architecture, routes, project behavior, publishing workflow, or content policy.

Use [CHANGELOG.md](CHANGELOG.md) for verifiable history and [AGENTS.md](AGENTS.md) for Codex continuity. When checked out inside DaZu, the [parent README](../README.md) is the workspace index. That relative link is only available in the combined local workspace; this library can also run as a standalone repository.

## Git and publication policy

Daniel's instruction on 2026-10-01 separates local version control from external publication:

- Reviewed local staging and commits may happen automatically, in each repository separately.
- A Git push sends commits to GitHub. Do not push to any remote unless Daniel explicitly requests that GitHub action; task completion is not authorization.
- DaZu production or draft publication on Netlify requires an explicit request. Production still requires validation, review, a commit on `main` and a successful authorized GitHub push before deploying only `site/` with the explicit DaZu site ID.
- The Hub's existing push-to-`main` workflow publishes GitHub Pages, including documentation-only commits. Treat such a push as publication and perform it only with explicit authorization covering that release. Never deploy the Hub to Netlify.
- Do not enable or change provider auto-deploy settings, hooks, DNS or the Pages trigger as part of ordinary maintenance. No hosting setting was changed by this review.

## Source and publication state — reviewed 2026-10-01

At the start of the earlier documentation-only audit, DaZu was clean at `a45aee7` and the independent Hub was clean at `c470159`, both on `main` with no divergence from their locally cached `origin/main`. No remote fetch or provider check was performed, so this is local tracking evidence.

| Area | Latest publication recorded in CHANGELOG | Current difference |
| --- | --- | --- |
| DaZu | Source `da71922`, [Netlify deploy 6abe99081f0cca60f9947ad8](https://app.netlify.com/projects/dazu/deploys/6abe99081f0cca60f9947ad8), 2026-10-01 | Lens `a45aee7` and quality improvements are published. |
| Hub | Source `db4ea63`, [Pages run 36899971094](https://github.com/danielmza-web/pytorch-learning-hub/actions/runs/36899971094), 2026-10-01 | Application publication passed; later documentation pushes have separate runs. |

DaZu application source `da71922` is now published from `site/` in Netlify deploy `6abe99081f0cca60f9947ad8`, including Lens implementation `a45aee7`. Hub application source `db4ea63` is published through successful Pages run `36899971094`. Both authorized GitHub pushes succeeded. See CHANGELOG for live checks, the resolved Netlify CLI error and verification limits; documentation-only follow-ups are synchronized separately.

## Quality improvements implemented locally — 2026-10-01

- Landing and CV select lossless WebP through CSS `image-set`, with the original PNG fallback retained. Each 1672 × 941 background is 1,519,998 bytes instead of 2,007,954: 487,956 bytes (24.3%) less when WebP is selected. Decoded RGBA pixels and dimensions are identical; gradients, cover positioning and animation code are unchanged. Keeping both formats increases source storage; it does not make a browser fetch both backgrounds.
- Each full-size visible logo is 1,356,611 bytes instead of 1,555,146 (198,535 bytes / 12.8% saved). Compared against the pre-change copy: all decoded pixels, alpha and PNG metadata are identical. Separate transparent 128 px favicon (19,432 bytes) and 180 px touch icon (36,267 bytes) retain the contained logo proportions and colour metadata. Icon downsampling is deliberate; full-size logos remain available.
- Intrinsic dimensions reserve space for the five quick-guide images and all 18 educational Hub images. Responsive proportions and existing lazy-loading behavior remain. The phone photograph, its 6688 × 3762 resolution, zoom/panning implementation, calculator formulas, examples and retained training evidence are unchanged.
- Shared copy handling announces success/failure using a polite accessible status. Denied or missing clipboard access selects the exact code and gives Ctrl+C / Command+C instructions. Original button text/title/accessible label return after 1.8 seconds; rapid retries cannot let an older failure replace a newer success. Hub icon buttons retain their icons.
- Removed the unused 2,571,900-byte Mermaid bundle and its exclusive fence/style configuration after checking active source references. Existing diagrams are local images/stage cards. This reduces generated Hub payload, not current browser execution or download time.

Local acceptance: strict Hub build; source, generated-link/resource/anchor and image-dimension validators; JavaScript syntax, clipboard success/failure/label/race tests; recall numeric, example and learning-contract checks; four smoke examples on a disposable copy. Shared copy/recall JS and CSS match byte for byte. Browser checks visited all 28 routes at 1440 × 900, 820 × 1180 and 390 × 844, plus all 19 Hub routes in dark theme at those sizes (141 visits): no page overflow, broken completed images or captured warnings/errors. Representative baseline/current CV captures at all three sizes and Hub light/dark views were inspected. Lazy-image reserved space was checked before loading. Real keyboard copy success matched code text on both sites and the prior clipboard was restored; dedicated real-browser fixtures checked permission denial/missing API, selection and label restoration. Lens 6 m/min → 100 mm/s retained 1/500; phone 3× retained 69 mm, keyboard panning/recenter and CV keyboard expansion worked; landing WebGL initialized.

Limits of the earlier implementation checks: these were local verification; the authorized release is recorded above and in CHANGELOG. Route visits do not exhaust every control or lazy image; no measured whole-page CLS/performance benchmark, physical touch, formal screen-reader/WCAG audit, live aliases/provider checks, full/pretrained/GPU training or new CV PDF was performed. Byte savings describe individual resources, not measured total page loading time. Frozen V4/V5, the personal PyTorch archive and experiment evidence were preserved. Reproducible checks are in README; ignored `tmp/quality-check/` contains optional captures/results and is not required to run the project.

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

Python 3.14.7 was rechecked on this PC on 2026-10-01; the current build and offline project checks passed using the existing global dependencies. Use `py -3.14` instead of `python` in the commands below; no installation is needed for the verified environment. The moved, broken `.venv` was removed. To preview from this repository, run `py -3.14 -m mkdocs serve`.

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

The `overrides/main.html` template adds the touch icon; MkDocs uses the separate favicon while keeping the full-size visible logo. Declare real width/height on new teaching images, preserve responsive/lazy behavior, and run the presentation validator after building. `copy-code.js`, `recall-visuals.js` and `recall-visuals.css` must match the independent DaZu copies; the parent asset validator checks them when both repositories exist. Icon/logo regeneration is available through the optional parent `scripts/optimize_site_assets.py`; the standalone Hub has all required committed assets.

Write educational content in Markdown, include `last_reviewed` metadata, store local images under `docs/assets/images/`, and keep interactions in the shared JavaScript and stylesheet.

A local compatibility check on 2026-09-03 used Python 3.14.7, Pillow 12.3.0, Torch 2.14.0+cu130, TorchVision 0.29.0+cu130 and Material 9.7.7. The strict build, content/link validators and all four existing smoke examples passed on a disposable copy. This does not validate complete dataset training or a GitHub Actions run. The local workflow target is now 3.14. Check the Actions run for the current commit separately; local validation does not establish deployment success.

## Validate

Install the project dependencies once when running smoke tests or complete examples:

```powershell
python -m pip install -r requirements-projects.txt
```

Before an explicitly authorized publication, run every check below. For documentation-only edits, link/consistency checks and `git diff --check` are sufficient unless code, requirements, configuration or public content also changes:

```powershell
python -m mkdocs build --strict
python tests/validate_content.py
python tests/validate_built_site.py
python tests/validate_presentation.py
node tests/validate_copy_code.cjs
node --check docs/assets/javascripts/copy-code.js
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

The public repository is `danielmza-web/pytorch-learning-hub`. Local commits may happen automatically. A remote push requires Daniel’s explicit instruction; a push to `main` automatically publishes after the checks succeed, including documentation-only changes. Run this sequence only for an explicitly authorized GitHub Pages release:

```powershell
git status --short
git add <reviewed-files>
git commit -m "Describe the PyTorch documentation update"
git push origin main
```

`.github/workflows/pages.yml` builds and deploys the `site/` artifact. `docs/CNAME` preserves the custom domain `pytorch.dazu.xyz`.

Replace `<reviewed-files>` with the intended file paths. A push to `main`, including a documentation-only push, starts publication; do it only when Daniel explicitly authorizes that publication. GitHub write access and an enabled Pages/Actions configuration are required; the workflow uses GitHub-managed token/OIDC permissions (`pages: write`, `id-token: write`). No project API keys are needed for local use. Reauthenticate separately on another PC and do not copy credentials or historical `.netlify/` state.

This documentation repository no longer deploys to Netlify. Detailed-library edits therefore use no Netlify production-deploy credits. DaZu remains a separate build-free repository and is published only when its own quick-guide pages change.

After a successful workflow, verify the changed page, direct anchors, search, mobile navigation, images, code-copy controls, and affected interactions. Check the run at `https://github.com/danielmza-web/pytorch-learning-hub/actions`; do not change DNS or use Netlify for an ordinary content release.

## Add a future PyTorch guide

1. Create `docs/guides/<guide-slug>/`.
2. Copy `templates/guide-core.md` and `templates/guide-applied.md` into that folder with meaningful filenames.
3. Replace the prompts with original explanations, examples, visuals, and relevant project links.
4. Add both pages as one guide group under `Guides` in `mkdocs.yml`.
5. Add a short DaZu page at `/learn/pytorch/<guide-slug>/` and list it on `/learn/pytorch/` only when ready.
6. Validate and commit locally. Push only after an explicit release request.

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

Local website preview needs no login. Package installation and dataset downloads need external access; Material's font configuration requests Google Fonts. Images and interactions are stored locally. The unused Mermaid bundle and exclusive configuration have been removed; no current diagram needs it. The installed dependency combination was checked locally. Current DNS/HTTPS settings, dataset download availability and full training require separate verification when needed.

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
