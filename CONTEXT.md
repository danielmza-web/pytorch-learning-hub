# PyTorch Learning Hub context

Read this before changing the detailed library. It records the current architecture and release decisions without reproducing development history.

## Verified publication — 2026-09-30

Content commit `10cff36` was pushed to `main` after the user explicitly requested GitHub/Netlify release. GitHub Pages run `36770490711` succeeded. All sixteen public pages returned HTTPS 200. The live vision guide displayed the seeded noise comparison and responded to Gaussian/severity controls without logged errors. DaZu website commit `90a8b6b` was independently published with Netlify deploy `6abd6c46b1fd9399018fe764`; its nine routes and three new guide aliases returned 200. No Hub files were deployed to Netlify. Documentation follow-up commits trigger Pages too; verify their own run before claiming the latest head is deployed.

## Course coverage — 2026-09-30

Reviewed the complete sibling `Pytorch` archive as a read-only source: 171 files, eight slide decks (1,656 pages), lab/assignment exports, two notebooks including stored outputs, four support scripts, all 111 stored images/plots, dataset/checkpoint inventory and its README. The user's completion of Course 2 is a supplied fact; the archive README still describes it as pending. [Coverage notes](notes/course-coverage-2026-09-30.md) map every module and teaching export, identify three missing image references and explain missing auxiliary files. Binary datasets/checkpoints were inventoried and hashed, not executed or deserialized. Only representative PDF pages were visually inspected; all slide text was extracted and reviewed.

Added three two-page guides: Training (metrics/search and efficient pipelines), Vision (transforms/noise and pretrained models), Text (tokens/embeddings and classifiers). Expanded the two Fundamentals pages with Course 1 tensor/storage operations, loss contracts, model introspection and separate train/validation transforms. Home, reference function finder and About connect all eight modules. Original tested snippets live in `examples/recall_patterns.py`; interactive illustrations show noise, accumulation and padding. The actual lab noise is salt and pepper; Gaussian/read-noise and photon-noise comparisons are explicitly explanatory extensions. No course assessments or solutions were copied into the public library.

The current source has 16 Markdown pages and produces 17 HTML pages. Content checks derive the inventory from navigation and enforce two pages per guide; built-site checks reject missing output. The Pages workflow also checks the new recall examples. Optuna, Lightning and TorchMetrics are absent from the global environment; optional ecosystem snippets are source-checked explanations, not claimed end-to-end runs. Nothing was installed. The archive and retained experiment images/results remain unchanged. The initial content review did not commit or publish; the later authorized release is recorded above.

Local checks passed the strict build, 16/17 source/output validators, four-guide DaZu links, shared JavaScript syntax, 68 guide Python excerpt syntax checks, original recall-pattern tests and all four existing smoke examples on a disposable copy. Offline tiny random DistilBERT checked module paths, zero-block/head-only freezing, logits and external weighted loss; a random ResNet checked head replacement and frozen BatchNorm mode. No pretrained downloads or accuracy claims result from those checks. CUDA was unavailable to this process, so the AMP excerpt was not run on a GPU. Browser checks at 1440×900 and 390×844 covered new-page layout, noise controls, accumulation/padding including invalid input, search results, mobile guide navigation and code-copy success feedback. No horizontal page overflow was observed. Initial local checks did not verify real-device touch, full training, optional ecosystem runtime, CI/hosting or Netlify aliases. Publication and route checks are recorded above; the training/runtime/touch limits still apply.

## Previous local review — 2026-09-27

A later local learning pass expanded the two Fundamentals guides with shape traces, ReLU's exact behavior, data-split cautions, CNN block reasoning, and short self-checks. All four project pages gained a focused question, and the home page now routes directly to ReLU and the compact Course 2 reference. The reference documents metrics, epoch-based learning-rate scheduling, and a frozen-backbone transfer-learning starting point from the organized labs studied so far. This does not mark Course 2 complete. The separate `Pytorch` archive was not edited. DaZu's two entry pages were revised in the parent repository and require their own release.

The ten maintained pages were revised for clearer orientation, shorter page introductions, and consistent section, table, card, and focus styling. The top tab bar was removed; the sidebar now leads from Start here through two numbered Fundamentals pages, a project chooser, a quick reference, and About. Stable page routes and DaZu-linked anchors remain unchanged. The reference now has a brief decision table connecting second-course topics without publishing an incomplete second guide or course exercises. Project copy distinguishes the small default run from its automatic CPU/CUDA device selection. Two interactive inputs now show errors for invalid numbers instead of silently substituting `1`.

The strict local build, ten-page content validation, eleven-page generated link/anchor validation, JavaScript syntax check, and DaZu cross-site link check passed. The browser tool rejected opening the generated local file under its URL policy. A later live check of the initial Pages deployment confirmed the narrow-screen menu, guide route, project gallery, reference, and invalid tensor-input message. It led to moving the project chooser above explanatory text and removing a repeated home heading. Confirm the final release commit's workflow and live page separately. The separate `Pytorch` course archive was read only and remains unchanged.

## Latest verified point — 2026-09-03

The library is an independent repository on `main`. GitHub authentication and the remote branch were checked before the authorized cleanup. Use current Git status and the matching Actions run for synchronization/deployment state; the parent repository cannot save this library.

This Windows PC has global Python 3.14.7, MkDocs 1.6.1, Material 9.7.7, Matplotlib 3.11.1, Pillow 12.3.0, Torch 2.14.0+cu130 and TorchVision 0.29.0+cu130. All declared requirements are satisfied and `pip check` passed. Torch/TorchVision imports, a CUDA operation on the RTX 5070 Laptop GPU and retained EMNIST Letters train/test data were verified. The workflow now targets Python 3.14. Use `py -3.14` on this PC; README retains isolated-environment setup for another machine.

The broken moved `.venv`, obsolete `.netlify` metadata, bytecode caches, unused EMNIST splits/archive and empty CIFAR-10 archive were removed. Four EMNIST Letters files, all experiment artifacts and generated `site/` output remain. The preview helper selects an existing Windows virtual environment or installed Python and does not install packages. No public Markdown content, example code or maintained images changed during cleanup.

The final cleanup verification passed a fresh strict MkDocs build, content validation (10 Markdown pages), generated-link validation (11 HTML pages), all four smoke examples and the JavaScript syntax check on a disposable source copy. Maintained website images and experiment artifacts were not regenerated in this checkout. Full training and comprehensive browser checks were not part of this maintenance.

## Known gaps and next steps

1. Save and synchronize this repository separately. A push to `main` triggers GitHub Pages, including maintenance changes; confirm the matching Actions run before claiming publication.
2. Use Python 3.14 and the current requirements for future environments. Dependency ranges are not a lockfile; record versions with new experiment evidence.
3. Build before validating links. Content validation compares navigation with every source page; generated-site validation rejects missing output. The current source/output counts are 16/17.
4. Smoke tests regenerate a maintained regression image and local results. Use a disposable copy when preserving existing website assets and evidence is required.
5. Full dataset training, fresh dataset downloads and comprehensive browser/mobile/accessibility checks remain separate tasks. The retained Letters dataset works without the removed EMNIST variants/archive.
6. Preserve `artifacts/`: the retained CPU experiment supports the published result. Before a release, inspect the revised pages in a local browser at desktop and phone widths and exercise the interactive controls.

See README for commands, CHANGELOG for verified milestones and AGENTS for working rules.

## Purpose and boundaries

`pytorch.dazu.xyz` is a practical PyTorch learning and recall library. It should be useful to someone learning a topic for the first time and to someone returning later to remember shapes, training logic, debugging, or project structure. It is not a course interface: do not add progress, completion, status, accounts, analytics, personal history, graded questions, or empty future-guide placeholders.

The content is original. The Coursera course *PyTorch: Fundamentals* may appear as study context in sources, but the library must not reproduce assessments, protected notebooks, quizzes, or graded solutions.

## Source and public structure

Markdown under `docs/` is the source of truth. `mkdocs.yml` owns the explicit public navigation:

- Start here
- Learn the fundamentals (two numbered pages)
- Improve training (metrics/tuning and efficient pipelines)
- Work with vision (transforms/noise and pretrained models)
- Work with text (tokens/embeddings and classifiers)
- Practice with projects (chooser and four examples)
- Quick reference
- About this library

Each published PyTorch guide uses two substantial pages rather than many small pages. Fundamentals currently consists of:

- `/guides/fundamentals/core-workflow/`
- `/guides/fundamentals/vision-real-data/`

The library also has a project gallery, four complete project pages, one combined reference, and one combined about/sources page. Preserve stable anchors when editing long pages because the four DaZu quick guides link directly to them. Each detailed page starts with a return-to-section map; keep explanations moderate and connected, with concrete function and parameter details rather than a transcript of the course.

Future PyTorch guides follow the same two-page pattern and enter navigation only when both pages contain useful material. OpenCV, YOLO, and other non-PyTorch libraries belong in separate future hubs, not here.

## Projects and visuals

The maintained examples are:

- `examples/regression_demo.py`
- `examples/emnist_model.py`
- `examples/robust_dataset.py`
- `examples/nature_cnn.py`

Regression uses seeded synthetic data on CPU and has no CLI flags. The three image projects use small seeded configurations, download public data through TorchVision and default to `--device auto`, which selects CUDA when available. They accept `--device cpu`, `--smoke-test`, and the optional longer `--full --device cuda` mode. Seeding does not guarantee identical results across environments. Project pages explain the practical question, important excerpts, evidence or clearly labelled conceptual visuals, an interaction, run instructions, expected artifacts, and the complete included source.

Do not invent metrics. Reproduced claims require retained results; structural diagrams and interactive curves must say when they are illustrative. Generated artifacts stay under ignored `artifacts/`, downloaded data under ignored `data/`, and maintained local visuals under `docs/assets/images/`.

Regression also regenerates `docs/assets/images/regression-comparison.png`. The smoke suite calls regression, so it can modify that tracked visual. Do not run it as a read-only check.

## Local update workflow

From this repository:

```powershell
py -3.14 -m mkdocs serve
```

The current PC already has the documentation and project dependencies. On another machine, follow README to create an environment and install the appropriate requirement files. Before pushing, run the checks listed in `README.md`; the GitHub workflow repeats them and publishes only after success.

## Publishing and hosting

The public repository is `https://github.com/danielmza-web/pytorch-learning-hub`. A push to `main` triggers `.github/workflows/pages.yml`, which validates and publishes the generated `site/` artifact with GitHub Pages.

Production state recorded as verified in the previous handoff on 7 August 2026; not rechecked during the 2026-09-03 review:

- `pytorch.dazu.xyz` is a CNAME to `danielmza-web.github.io` in Netlify DNS.
- GitHub Pages reports the DNS check as successful.
- HTTPS is active and **Enforce HTTPS** is enabled.
- The former Netlify project is detached from the custom domain.
- `https://dazu-pytorch.netlify.app/` remains a rollback snapshot only.

Never publish this repository with `netlify deploy`. Historical `.netlify/` metadata was removed; do not recreate it as a release mechanism. Ordinary detailed-library updates require only validation, commit, and push and consume no Netlify production-deploy credits.

DaZu is independent. Update `../site/learn/pytorch/` only when the short guide index, quick explanation, copied visual, or link set changes. That parent site has its own explicit, manual Netlify release gate.

## Verification after publication

Confirm the GitHub Actions run succeeded, then check:

- the homepage and both pages of every affected guide;
- all four project pages and complete-source inclusions;
- direct anchors used by DaZu;
- search, mobile navigation, images, and code-copy controls;
- affected keyboard interactions and reduced-motion behavior;
- HTTPS responses for direct routes.

Keep README operational and this file decision-oriented. Replace stale statements instead of appending a transcript.

## Python 3.14 compatibility update (2026-09-03)

The previous Pillow <12 bound applied to runnable examples and was not required by the image APIs they use. The local dependency declarations now allow Pillow 12 and use Python 3.14 for setup and the Pages workflow. Material is pinned to 9.7.7; MkDocs remains 1.6.1. On a disposable copy, Python 3.14.7 with Pillow 12.3.0, Torch 2.14.0+cu130 and TorchVision 0.29.0+cu130 passed the strict documentation build, content/link validators and all four existing smoke examples. Generated images/results were confined to that copy. Complete dataset training, browser rendering and the updated CI job were not tested. That compatibility review did not commit or publish. Earlier recorded checks remain historical evidence; consult the latest Git/Actions state for subsequent releases.
