# PyTorch Learning Hub context

Use README for commands and CHANGELOG for dated history. This file contains current architecture, evidence, decisions and next work.

## Git and publication policy

Daniel's instruction on 2026-10-01 separates local version control from external publication:

- Reviewed local staging and commits may happen automatically, in each repository separately.
- A Git push sends commits to GitHub. Do not push to any remote unless Daniel explicitly requests that GitHub action; task completion is not authorization.
- DaZu production or draft publication on Netlify requires an explicit request. Production still requires validation, review, a commit on `main` and a successful authorized GitHub push before deploying only `site/` with the explicit DaZu site ID.
- The Hub's existing push-to-`main` workflow publishes GitHub Pages, including documentation-only commits. Treat such a push as publication and perform it only with explicit authorization covering that release. Never deploy the Hub to Netlify.
- Do not enable or change provider auto-deploy settings, hooks, DNS or the Pages trigger as part of ordinary maintenance. No hosting setting was changed by this review.

## Visual understanding review — local 2026-10-11

The approved plan is implemented locally: 13 deterministic lab instances across the twelve guides, six focused DaZu counterparts, readable native diagrams, original-size figure links, corrected convolution arithmetic, coherent curve scenarios and strict shape inputs. Routes, numbering and practice connections remain unchanged. Shared learning-labs JS/CSS are copied into each independent site and checked for byte equality. Three guide/project templates now match the learning orientation and evidence rules.

Local strict build, content/navigation, resources/anchors/image dimensions, independent CPU numeric comparisons, existing example/contract/lifecycle/copy/visual tests and disposable smoke passed. Browser review covered all 31 learning routes at four widths, both Hub themes (220 visits), plus 83 control checks across all 19 new lab instances and additional training/curve/shape checks. Retained measured figures and personal course materials were preserved. [Review evidence and limits](notes/visual-review-2026-10-11.md) distinguishes synthetic calculations from measurements. No remote publication was performed; the previous published state remains the release recorded below.

## Current published state — 2026-10-10

The approved navigation/clarity plan is published in the independent Hub after Daniel requested the GitHub push. Application commit `420c3d5` passed [Pages run 38079506200](https://github.com/danielmza-web/pytorch-learning-hub/actions/runs/38079506200). Live HTTPS checks confirmed all 24 routes, their generated IDs, home topic order and gallery P1–P8 order. Keep the six topic numbers 1–6, page numbers 1.1–6.2 and project numbers P1–P8 consistent across sidebar, home, headings, gallery and footer. Projects follow regression, EMNIST, robust pipeline, Nature CNN, controlled training, vision head, text bags and model lifecycle. `tests/validate_navigation.py`, also invoked by content validation, rejects numbering/order drift and missing reciprocal guide/project connections.

All twelve guides and eight project pages have explicit goals, prerequisites and code requirements. Essential distinctions remain visible; longer implementations and variants use disclosures. Architecture practice points to its own visual checks/runnable toys, since the eight projects do not implement transformer or diffusion training. Preserve old home/gallery section IDs when changing visible labels. Home now has one six-topic chooser, a compact reading map and practice spanning the complete route, with no progress tracking.

Local strict build, content/navigation, built links/resources/anchors and presentation checks passed. All 24 routes were visited in desktop/mobile and light/dark (96 visits); sidebar active states and footer ordering matched. Search, repeated disclosure reveal, mobile menu, 18 representative page disclosures and attention/memory controls worked. Source examples, maintained images and retained reports are unchanged. Local review did not repeat complete training; the authorized Pages release subsequently passed the complete workflow, including disposable smoke/example tests. No DaZu source or Netlify publication changed. Dated evidence and limits are in CHANGELOG.

## Previous Course 3 publication — 2026-10-10

The Course 3 expansion and Course 1/2 improvements are published after Daniel explicitly requested GitHub synchronization and Netlify production publication. Source pushes `49186c2` (DaZu) and `ee71ba4` (Hub) completed with zero divergence in each independent repository. No hosting trigger, DNS or provider configuration was changed.

| Area | Verified application publication | Scope |
| --- | --- | --- |
| DaZu | Source `49186c2`, [Netlify deploy 6aca8184f7fe4f6033f3740e](https://app.netlify.com/projects/dazu/deploys/6aca8184f7fe4f6033f3740e), 2026-10-10 | Only DaZu `site/`; six quick guides and connected navigation. |
| Hub | Source `ee71ba4`, [Pages run 38075215577](https://github.com/danielmza-web/pytorch-learning-hub/actions/runs/38075215577), 2026-10-10 | Six two-page guides, 24 source pages and eight projects through GitHub Pages. |

Live verification passed 61 DaZu HTTPS checks and 56 Hub checks, plus 70 desktop/mobile canonical-route visits. New memory/attention controls initialized correctly; no page overflow, broken completed images or captured warnings/errors were observed. See CHANGELOG for exact scope and limits. Documentation follow-ups have separate commits; Hub documentation pushes start additional Pages runs and their outcome must be verified separately.

Daniel confirmed completion of the whole certificate on 2026-10-10. Preserve that confirmation as private study context; do not invent a credential issue date, change the CV or add public progress/status UI. The archive README was corrected, but its Course 3 materials remain unchanged. [Coverage notes](notes/course-coverage-2026-10-10.md) record the primary sources, syllabus/material mapping, current API caveats and original local verification limits.

## Quality improvements implemented locally — 2026-10-01

- Landing and CV select lossless WebP through CSS `image-set`, with the original PNG fallback retained. Each 1672 × 941 background is 1,519,998 bytes instead of 2,007,954: 487,956 bytes (24.3%) less when WebP is selected. Decoded RGBA pixels and dimensions are identical; gradients, cover positioning and animation code are unchanged. Keeping both formats increases source storage; it does not make a browser fetch both backgrounds.
- Each full-size visible logo is 1,356,611 bytes instead of 1,555,146 (198,535 bytes / 12.8% saved). Compared against the pre-change copy: all decoded pixels, alpha and PNG metadata are identical. Separate transparent 128 px favicon (19,432 bytes) and 180 px touch icon (36,267 bytes) retain the contained logo proportions and colour metadata. Icon downsampling is deliberate; full-size logos remain available.
- Intrinsic dimensions reserve space for the five quick-guide images and all 18 educational Hub images. Responsive proportions and existing lazy-loading behavior remain. The phone photograph, its 6688 × 3762 resolution, zoom/panning implementation, calculator formulas, examples and retained training evidence are unchanged.
- Shared copy handling announces success/failure using a polite accessible status. Denied or missing clipboard access selects the exact code and gives Ctrl+C / Command+C instructions. Original button text/title/accessible label return after 1.8 seconds; rapid retries cannot let an older failure replace a newer success. Hub icon buttons retain their icons.
- Removed the unused 2,571,900-byte Mermaid bundle and its exclusive fence/style configuration after checking active source references. Existing diagrams are local images/stage cards. This reduces generated Hub payload, not current browser execution or download time.

Local acceptance: strict Hub build; source, generated-link/resource/anchor and image-dimension validators; JavaScript syntax, clipboard success/failure/label/race tests; recall numeric, example and learning-contract checks; four smoke examples on a disposable copy. Shared copy/recall JS and CSS match byte for byte. Browser checks visited all 28 routes at 1440 × 900, 820 × 1180 and 390 × 844, plus all 19 Hub routes in dark theme at those sizes (141 visits): no page overflow, broken completed images or captured warnings/errors. Representative baseline/current CV captures at all three sizes and Hub light/dark views were inspected. Lazy-image reserved space was checked before loading. Real keyboard copy success matched code text on both sites and the prior clipboard was restored; dedicated real-browser fixtures checked permission denial/missing API, selection and label restoration. Lens 6 m/min → 100 mm/s retained 1/500; phone 3× retained 69 mm, keyboard panning/recenter and CV keyboard expansion worked; landing WebGL initialized.

Limits of the earlier implementation checks: these were local verification; the authorized release is recorded above and in CHANGELOG. Route visits do not exhaust every control or lazy image; no measured whole-page CLS/performance benchmark, physical touch, formal screen-reader/WCAG audit, live aliases/provider checks, full/pretrained/GPU training or new CV PDF was performed. Byte savings describe individual resources, not measured total page loading time. Frozen V4/V5, the personal PyTorch archive and experiment evidence were preserved. Reproducible checks are in README; ignored `tmp/quality-check/` contains optional captures/results and is not required to run the project.

## Current implementation and evidence

The source contains 24 public Markdown pages and builds 25 HTML pages including 404: six two-page guides, home, project chooser/eight complete projects, reference and About. Keep English content, original explanations, stable routes/anchors, sidebar/footer navigation and the existing learning order. Long implementations/variants use disclosures; search/direct anchors must reveal the destination, including a repeated link after closing its section. The project gallery uses mobile cards, and the function finder groups by purpose.

Four failed Mermaid diagrams were replaced with local responsive stage cards. The unused Mermaid bundle and exclusive fence/style configuration were removed; current diagrams use local images/stage cards. Shared `recall-visuals.js`/`.css` are byte-identical to the DaZu copies; update and validate both repositories if they change. Preserve invalid input/zero-denominator metrics, padding/masks/truncation and code-copy feedback in both themes. New visuals distinguish conceptual illustrations, tiny synthetic runs and retained measurements.

Retained Course 2 CPU JSON reports live in `docs/assets/data/recall-2026-10-01/`; `scripts/render_recall_evidence.py` regenerates the corresponding new figures and original transform illustration without replacing historical EMNIST/regression visuals. Training compares shared starts/splits with sample-correct accumulation; optional `--benchmark --device cpu|auto|cuda` records timing/throughput and, for CUDA, AMP/memory metadata. No GPU benchmark was repeated in this audit. Vision checks frozen backbone parameters/BatchNorm buffers and head changes; the random offline default is mechanism evidence, while `--data ... --pretrained` needs own train/val data and may download weights. Text retains vocabulary/IDs/offsets/embedding/prediction contracts on original toy phrases, not a language benchmark.

EMNIST and Nature split the training pool with a fixed seed, reserve 20% for deterministic validation, restore minimum-validation-loss weights and measure official test at the end. Checkpoints retain best epoch, classes, normalization and disjoint source indices. Preserve the historical EMNIST 24.4% CPU result/images as evidence from the earlier test-during-training configuration. The archive's user-supplied certificate completion is study context, not a model-quality result; its README was corrected on 2026-10-10. Coverage notes in `notes/course-coverage-2026-09-30.md` retain the original read-only archive review, gaps and limits.

MkDocs 1.6.1 and Material 9.7.7 remain pinned; Pages targets Python 3.14. Project dependencies use ranges, not a lockfile. Optional Optuna/Lightning/TorchMetrics/Transformers snippets are not required for the eight maintained examples or website build. Prior installed-package/GPU/hosting snapshots do not establish current access or optional runtime support; recheck when that task needs them. No environment or package changes accompanied this audit.

## Audit evidence — 2026-10-01

The inventory covered all 136 tracked files: 54 in DaZu and 82 in the Hub. Readable text/configuration, Python/JavaScript syntax, local HTML references/IDs, JSON and SVG parsing, and raster-image integrity were checked. This includes the frozen V4/V5 files without changing them. Across both repositories: 21 Python sources, eight standalone JavaScript files, six inline scripts, 11 source HTML files, six JSON reports, seven SVGs and 38 raster images passed their applicable structural checks. Image integrity is not a visual judgment of every asset.

A fresh Hub strict build passed. Source validation checked 19 Markdown pages; generated validation checked 20 HTML pages including 404 and their links/resources/anchors. Recall patterns, selected examples, split/checkpoint/evidence contracts, JavaScript edge cases and four foundation smoke examples passed; smoke outputs stayed in a disposable copy. DaZu's five quick pages and detailed-Hub anchors passed against that fresh build. Shared recall CSS/JS copies are byte-identical.

Current local browser checks visited all nine DaZu routes and all 19 Hub routes at 1280×720 and 390×844: no page-level horizontal overflow, broken completed images or captured warning/error logs were observed. Loaded-image checks do not prove lazy images below the viewport rendered. The Lens default and `6 m/min → 100 mm/s → 0.1 m/s` conversion kept the `1/500` camera shutter. Metrics invalid-count/preset checks and disclosure keyboard behavior were also exercised. This is a scoped browser audit, not a repetition of every earlier interaction or a full accessibility certification.

During that earlier documentation-only audit, no packages, website source, maintained assets, course-archive files or hosting settings were changed. Generated ignored Hub `site/` was rebuilt. Full dataset/pretrained/GPU runs, fresh downloads, physical-device touch, exhaustive visual/offline testing, optional ecosystem runtimes, CV PDF generation/layout, external URLs, live DNS/redirects/account access and provider workflow status were not checked. Use the earlier CHANGELOG entries for historical verification only.

## Improvements identified and next work

| Priority | Evidence / affected files | Proposed improvement and acceptance check |
| --- | --- | --- |
| Medium | CI tests syntax/numeric contracts but does not exercise a real browser. | Add focused desktop/mobile checks for disclosure reveal through search/direct anchors, mobile navigation, code copy and metrics/padding controls. Avoid assuming syntax or a strict build proves interactions. |
| Medium | Shared recall files are byte-identical today; project dependency versions use broad ranges. | Cross-repository drift verification is implemented in the parent asset validator; maintain a recorded compatible experiment environment. Test upgrades in isolation; do not replace the current global environment by default. |
| Low | Current source has eight projects; preserve inventory-derived counts when extending the library. | Keep inventory-derived counts and one current state section; move dated results into CHANGELOG rather than appending competing snapshots. Corrected by this documentation review. |
| Low | Optional Optuna/Lightning/TorchMetrics/Transformers explanations exceed the tested default projects. | Keep optional dependencies and evidence limits explicit. Execute representative optional snippets only when that work is requested; do not claim benchmark quality from tiny synthetic runs. |

The existing automatic Pages trigger was inspected and left unchanged. Explicit publication authorization is enforced by withholding remote pushes until requested.

## Purpose and boundaries

`pytorch.dazu.xyz` is a practical PyTorch learning and recall library. It should be useful to someone learning a topic for the first time and to someone returning later to remember shapes, training logic, debugging, or project structure. It is not a course interface: do not add progress, completion, status, accounts, analytics, personal history, graded questions, or empty future-guide placeholders.

The content is original. The three-course *PyTorch for Deep Learning Professional Certificate* may appear as study context in sources, but the library must not reproduce assessments, protected notebooks, quizzes, or graded solutions.

## Source and public structure

Markdown under `docs/` is the source of truth. `mkdocs.yml` owns the explicit public navigation:

- Start here
- 1 · Learn the fundamentals (1.1–1.2)
- 2 · Improve training (2.1–2.2)
- 3 · Work with vision (3.1–3.2)
- 4 · Work with text (4.1–4.2)
- 5 · Explore advanced architectures (5.1–5.2)
- 6 · Save and deploy models (6.1–6.2)
- Practice with projects (chooser, P1–P4 foundations, P5–P7 training/vision/text, P8 lifecycle)
- Quick reference
- About this library

Each published PyTorch guide uses two substantial pages rather than many small pages. Fundamentals currently consists of:

- `/guides/fundamentals/core-workflow/`
- `/guides/fundamentals/vision-real-data/`

The library also has a project gallery, eight complete project pages, one combined reference, and one combined about/sources page. Preserve stable anchors when editing long pages because the six DaZu quick guides link directly to them. Use the sidebar and page contents rather than repeated return-to-section maps; keep explanations moderate and connected, with concrete function and parameter details rather than a transcript of the course.

Future PyTorch guides follow the same two-page pattern and enter navigation only when both pages contain useful material. OpenCV, YOLO, and other non-PyTorch libraries belong in separate future hubs, not here.

## Projects and visuals

The maintained examples are:

- `examples/regression_demo.py`
- `examples/emnist_model.py`
- `examples/robust_dataset.py`
- `examples/nature_cnn.py`
- `examples/training_comparison.py`
- `examples/vision_head.py`
- `examples/text_bags.py`
- `examples/model_lifecycle.py`

Regression uses seeded synthetic data on CPU and has no CLI flags. EMNIST and Nature use small seeded configurations, download public data through TorchVision and default to `--device auto`, which selects CUDA when available. They accept `--device cpu`, `--smoke-test`, and the optional longer `--full --device cuda` mode. Seeding does not guarantee identical results across environments. Project pages explain the practical question, important excerpts, evidence or clearly labelled conceptual visuals, an interaction, run instructions, expected artifacts, and the complete included source.

Do not invent metrics. Reproduced claims require retained results; structural diagrams and interactive curves must say when they are illustrative. Generated artifacts stay under ignored `artifacts/`, downloaded data under ignored `data/`, and maintained local visuals under `docs/assets/images/`.

Regression also regenerates `docs/assets/images/regression-comparison.png`. The smoke suite calls regression, so it can modify that tracked visual. Do not run it as a read-only check.

## Local update workflow

From this repository:

```powershell
py -3.14 -m mkdocs serve
```

The current build and offline tests passed on this PC with Python 3.14.7 and existing dependencies on 2026-10-10. On another machine, follow README to create an environment and install the appropriate requirement files. Before pushing, run the checks listed in `README.md`; the GitHub workflow repeats them and publishes only after success.

## Publishing and hosting

The public repository is `https://github.com/danielmza-web/pytorch-learning-hub`. A push to `main` triggers `.github/workflows/pages.yml`, which validates and publishes the generated `site/` artifact with GitHub Pages.

Production state recorded as verified in the previous handoff on 7 August 2026; not rechecked during the 2026-09-03 review:

- `pytorch.dazu.xyz` is a CNAME to `danielmza-web.github.io` in Netlify DNS.
- GitHub Pages reports the DNS check as successful.
- HTTPS is active and **Enforce HTTPS** is enabled.
- The former Netlify project is detached from the custom domain.
- `https://dazu-pytorch.netlify.app/` remains a rollback snapshot only.

Never publish this repository with `netlify deploy`. Historical `.netlify/` metadata was removed; do not recreate it as a release mechanism. Ordinary detailed-library updates require validation and a local commit. Remote pushes require explicit authorization and activate Pages on `main`; they consume no Netlify production-deploy credits.

DaZu is independent. Update `../site/learn/pytorch/` only when the short guide index, quick explanation, copied visual, or link set changes. That parent site has its own explicit, manual Netlify release gate.

## Verification after publication

Confirm the GitHub Actions run succeeded, then check:

- the homepage and both pages of every affected guide;
- the project chooser, all eight project pages and complete-source inclusions;
- direct anchors used by DaZu;
- search, mobile navigation, images, and code-copy controls;
- affected keyboard interactions and reduced-motion behavior;
- HTTPS responses for direct routes.

Keep README operational and this file decision-oriented. Replace stale statements instead of appending a transcript.
