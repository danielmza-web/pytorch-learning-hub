# Changelog

Selected changes supported by local checks or this repository's Git history. Local review dates are not publication dates. DaZu has a separate history and deployment process.

## 2026-09-30 — Hub orientation and selected Course 2 examples

- Replaced Start here with an orientation page describing the library contents, guide order, example selection and lookup workflow. Removed its repeated Fundamentals lesson and duplicate route lists.
- Added three original runnable projects: controlled learning-rate comparison with metrics/scheduling/accumulation; image augmentation and frozen-backbone head training; variable-length text classification with vocabulary/offsets/class weights. Added source excerpts, complete scripts, output contracts and interactive checks; reorganized the gallery into Course 2 and foundation workflows.
- Added offline checks and Pages integration for metrics, learning, frozen parameters/BatchNorm, real-file/class-map handling, saved contracts and pooling invariance. Strict build, 19-page content/20-page generated checks, recall checks and the four foundation smokes passed. Existing smokes ran on a disposable copy; retained visuals/results and the course archive were unchanged.
- Desktop/phone browser checks covered orientation, project layout, accumulation invalid input, noise modes, padding/truncation, diagram rendering and full-source expansion, with no page overflow or logged errors. Default toy runs demonstrate mechanisms; pretrained quality, CUDA, full training and real-device touch were not tested. Released source `8670ca5` through successful Pages run `36777275109`. All nineteen public pages returned 200; live mobile project navigation and accumulation/noise/padding controls worked without errors. The home layout was then simplified into readable paragraphs/topic cards for phone widths. Home-layout/documentation follow-ups use the same Pages workflow; verify their own current-head run separately.

## 2026-09-30 — Course 1/2 recall library

- Reviewed all 171 files in the sibling course archive, including text from eight slide decks (1,656 pages), lab/assignment exports, notebooks and support interfaces. Added internal source coverage notes; preserved the archive, course solutions and retained experiment evidence outside the public content. Recorded three missing source images and auxiliary/runtime limits.
- Added six original pages in three connected two-page guides: metrics/tuning and efficient pipelines; image transforms/noise and pretrained models; tokens/embeddings and classifiers. Filled Fundamentals gaps in tensor/storage operations, losses, model inspection and split transforms. Added direct function lookup and module/source mapping to reference/About.
- Added original noise, masked-pooling, bag-collation and accumulation excerpts plus meaningful invariant checks. Added seeded visual noise comparison and interactive padding/accumulation arithmetic with invalid-input messages. Adapted navigation and validators for 16 Markdown / 17 generated HTML pages; Pages now validates recall excerpts too.
- Strict build, source/generated links, JavaScript syntax, 68 guide excerpt syntax checks, original recall tests, DaZu cross-site anchors and four existing smoke examples passed. Smoke outputs stayed on a disposable copy. Offline random DistilBERT/ResNet checks confirmed selected contracts without downloads. CUDA AMP runtime was skipped because CUDA was unavailable; Optuna/Lightning/TorchMetrics and full training were not executed.
- Local browser checks covered new guides at 1440×900 and 390×844, working noise/padding/accumulation controls, invalid input, search, mobile navigation and code-copy feedback; no page overflow or logged errors observed. After explicit release authorization, content commit `10cff36` was pushed and Pages run `36770490711` succeeded. All sixteen HTTPS routes returned 200; live noise controls worked without logged browser errors. The DaZu quick guides were published separately in Netlify deploy `6abd6c46b1fd9399018fe764`.

## 2026-09-27 — learning-detail pass (local)

- Added runnable microexamples and explicit learning checks for tensor flattening, ReLU and nonlinear capacity, CNN shapes, data leakage, and validation. Added one check to each of the four project pages and clearer study paths on Home.
- Expanded the compact Course 2 reference with metrics, a scheduler placement example, and a frozen-backbone transfer-learning example based on the labs reviewed so far. Kept the two-page Fundamentals structure and existing routes.
- Local strict build, content validation (10 Markdown pages), generated-link/anchor validation (11 HTML pages), shared JavaScript syntax, DaZu cross-site link validation, and tensor/ReLU/CNN microexamples passed. No new full training or smoke run was needed for documentation-only edits. Check the matching GitHub Actions run for publication status.

## 2026-09-27 — local hub review

- Revised all ten public pages for clearer learning paths, consistent introductions, and easier project selection. Simplified the MkDocs navigation by removing the top tab layer and naming the two Fundamentals pages in reading order. Kept existing routes and DaZu-linked anchors.
- Corrected default-device descriptions on image projects and two scripts included in the public pages, clarified classifier logits versus regression output, made sample-count loss averaging robust to skipped batches, and added a compact second-course decision reminder to the reference. Invalid numeric inputs in the batch and tensor tools now show a prompt instead of silently becoming `1`.
- Local strict build, content validation (10 Markdown pages), generated-link validation (11 HTML pages), JavaScript syntax, and DaZu cross-site link check passed. All four smoke examples passed on a disposable copy. Browser visual inspection of the generated local file was blocked by the browser tool's URL policy; release and live verification must be checked separately.
- Live review after the initial Pages deployment confirmed the narrow-screen menu and guide routes. It also showed that the project chooser appeared below a long instruction list and that the home page repeated "Start here" as a section heading; the chooser now appears first and the workflow heading is distinct.

## 2026-09-03 — workspace maintenance

- 2026-09-03: fresh strict build, content/generated-link validation and all four project smoke tests passed on a disposable copy; JavaScript syntax passed. Maintained website assets and retained experiment results were preserved.

- 2026-09-03: removed the broken moved virtual environment, old Netlify metadata, bytecode caches, unused EMNIST variants/archive and empty CIFAR-10 archive. Preserved Letters, retained experiments and generated reference pages. Updated context and global-Python instructions; preview no longer installs packages and selects an available interpreter.

- 2026-09-03: updated local setup and Pages workflow to Python 3.14, allowed Pillow 12 for demos, and pinned Material 9.7.7. Verified the strict build, content/link checks and four smoke examples on a disposable Python 3.14.7 copy; no generated production assets, commit, push or deployment.

- Reviewed documentation on 2026-09-03: preserved the earlier local hosting handoff; added portable setup, dependency/access notes, verification limits and next steps.
- Added Codex continuation instructions and this changelog. Corrected CPU/automatic-device behavior and documented regression/smoke-test image regeneration and generated-site validation limits. No code, assets or hosting settings changed; no commit, push or deployment was performed.

## 2026-08-06

- `dd8a612`: consolidated Fundamentals into two guide pages and combined reference/about pages; added strict generated-link validation, the custom-domain CNAME and GitHub Pages workflow; removed the old Netlify configuration and publish helper.
- `509c5dc`: added runnable reference collections and projects; `28be0fe` documented them.
- `fa5346d`: clarified the visual start path; `bfbaf43` documented the visual asset workflow.

## 2026-08-05

- `dfdca70`: initial extensible learning-hub implementation.
- `5a3e4ab`: added the then-current manual documentation release workflow, later replaced by GitHub Pages in `dd8a612`.
- `d73a8c4`: reframed the hub as reference guides.
