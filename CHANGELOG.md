# Changelog

Selected changes supported by local checks or this repository's Git history. Local review dates are not publication dates. DaZu has a separate history and deployment process.


## 2026-10-01 — lossless assets, stable images and resilient copying (local)

- Landing and CV select lossless WebP through CSS `image-set`, with the original PNG fallback retained. Each 1672 × 941 background is 1,519,998 bytes instead of 2,007,954: 487,956 bytes (24.3%) less when WebP is selected. Decoded RGBA pixels and dimensions are identical; gradients, cover positioning and animation code are unchanged. Keeping both formats increases source storage; it does not make a browser fetch both backgrounds.
- Each full-size visible logo is 1,356,611 bytes instead of 1,555,146 (198,535 bytes / 12.8% saved). Compared against the pre-change copy: all decoded pixels, alpha and PNG metadata are identical. Separate transparent 128 px favicon (19,432 bytes) and 180 px touch icon (36,267 bytes) retain the contained logo proportions and colour metadata. Icon downsampling is deliberate; full-size logos remain available.
- Intrinsic dimensions reserve space for the five quick-guide images and all 18 educational Hub images. Responsive proportions and existing lazy-loading behavior remain. The phone photograph, its 6688 × 3762 resolution, zoom/panning implementation, calculator formulas, examples and retained training evidence are unchanged.
- Shared copy handling announces success/failure using a polite accessible status. Denied or missing clipboard access selects the exact code and gives Ctrl+C / Command+C instructions. Original button text/title/accessible label return after 1.8 seconds; rapid retries cannot let an older failure replace a newer success. Hub icon buttons retain their icons.
- Removed the unused 2,571,900-byte Mermaid bundle and its exclusive fence/style configuration after checking active source references. Existing diagrams are local images/stage cards. This reduces generated Hub payload, not current browser execution or download time.

Local acceptance: strict Hub build; source, generated-link/resource/anchor and image-dimension validators; JavaScript syntax, clipboard success/failure/label/race tests; recall numeric, example and learning-contract checks; four smoke examples on a disposable copy. Shared copy/recall JS and CSS match byte for byte. Browser checks visited all 28 routes at 1440 × 900, 820 × 1180 and 390 × 844, plus all 19 Hub routes in dark theme at those sizes (141 visits): no page overflow, broken completed images or captured warnings/errors. Representative baseline/current CV captures at all three sizes and Hub light/dark views were inspected. Lazy-image reserved space was checked before loading. Real keyboard copy success matched code text on both sites and the prior clipboard was restored; dedicated real-browser fixtures checked permission denial/missing API, selection and label restoration. Lens 6 m/min → 100 mm/s retained 1/500; phone 3× retained 69 mm, keyboard panning/recenter and CV keyboard expansion worked; landing WebGL initialized.

Limits: this is local verification, not a release. Route visits do not exhaust every control or lazy image; no measured whole-page CLS/performance benchmark, physical touch, formal screen-reader/WCAG audit, live aliases/provider checks, full/pretrained/GPU training or new CV PDF was performed. Byte savings describe individual resources, not measured total page loading time. Frozen V4/V5, the personal PyTorch archive and experiment evidence were preserved. Reproducible checks are in README; ignored `tmp/quality-check/` contains optional captures/results and is not required to run the project.

## 2026-10-01 — full-source audit and current handoff

- Reviewed all 82 tracked Hub files as part of the 136-file workspace inventory: Markdown/navigation/includes, examples/helpers/tests, workflow/dependencies/templates, JSON evidence and local image/script/style assets. Structural integrity/syntax/resource checks passed; image-file validation is distinct from reviewing every image visually.
- Replaced accumulated dated CONTEXT snapshots with current 19-page/four-guide/seven-project architecture, source/evidence rules, reproduction limits, publication state and prioritized next work. Kept operational setup/example commands in README and important dated verification in this CHANGELOG. Corrected the old four-project publication checklist and obsolete claim that Mermaid is loaded.
- Updated README/CONTEXT/AGENTS to allow reviewed local commits and require explicit authorization before remote pushes. Documentation pushes to `main` also activate Pages. No workflow, provider setting or Netlify target was changed.
- Strict build, 19-source/20-generated content/link/resource/anchor validation, recall patterns, selected examples, training split/best-state/retained-report contracts and JavaScript edge cases passed. All four foundation smokes passed in a disposable copy, preserving maintained images/results. Parent anchors and identical shared recall files also passed.
- Browser checks visited all 19 public routes at 1280×720 and 390×844 without page overflow, broken completed images or captured warnings/errors; the workspace's selected metric/disclosure controls were exercised. Proposed optimizations include the 1.55 MB logo/favicon, unused 2.57 MB copied Mermaid bundle, browser regression checks, shared-file drift and recorded dependency combinations. No bundle deletion or code optimization was performed.
- Rebuilt ignored `site/` only. No packages, course archive, maintained assets or experiment evidence were changed. No remote push/publication occurred; the recorded `a9b734f` release remains historical and the `c470159` follow-up run was not queried. Full/pretrained/GPU runs, optional ecosystems, downloads, physical touch and current provider state remain unverified in this audit.

## 2026-10-01 — verified recall publication

Hub source `a9b734f` was pushed independently to `main`; [GitHub Pages run 36789441624](https://github.com/danielmza-web/pytorch-learning-hub/actions/runs/36789441624) passed build, checks and deployment. DaZu website source `ef1ac3d` was pushed to `main` and published from `site/` only in [Netlify deploy 6abd966abd624111e6e967df](https://app.netlify.com/projects/dazu/deploys/6abd966abd624111e6e967df). Both reviewed source pushes had zero divergence.

Production verification passed 68 HTTPS route/resource checks, including all 24 learning pages, the nine DaZu canonical routes and six aliases. All 34 compared resources matched their publication source: committed Git blobs for Pages (LF normalization included), working files for Netlify. Browser checks of the 24 learning routes found no page overflow at desktop/default and 390-pixel mobile widths, no broken loaded images or failed diagram labels. Live metrics rejected negative counts; padding at length 8 displayed five PAD positions and eight removed tokens (IDs 9–16); search and direct anchors opened the matching details; copied code matched the selected text. Published training curves were visually inspected. CUDA/AMP, full dataset training, pretrained downloads and physical-device touch remain unexecuted.

Documentation-only follow-ups are synchronized separately. They do not change DaZu static files or require another Netlify deployment; Hub follow-ups trigger the same Pages workflow and their matching run must be checked.

## 2026-10-01 — visual recall and reproducible learning contracts

- Replaced four failed Mermaid project diagrams with responsive local stage cards. Added concrete batches/gradients, tensor operations, schedules, microbatch updates, image-transform/task/freeze visuals and token/offset/mask rows. Added editable exact confusion metrics with undefined-denominator handling. All new visuals distinguish illustrations, toy executions and retained measurements.
- Reduced visible Fundamentals/reference content using disclosures; preserved previous headings/anchors and added automatic reveal for direct links and search, including same-anchor reopening. Kept the four two-page guides, original home orientation and sidebar/footer navigation. Converted project tables to mobile cards and grouped the function finder by purpose.
- Retained actual Course 2 CPU training curves/confusion matrix, physical/accumulated benchmark reports, frozen-backbone checks and text vocabulary/IDs/offsets/embedding/prediction evidence. Added optional CPU/auto/CUDA benchmarking with CUDA AMP/memory metadata; CUDA execution is not claimed. EMNIST/Nature select and restore validation checkpoints before final test; original images/results remain historical evidence.
- Added real offline integration checks for disjoint source splits, worse-final-epoch restoration and single final test evaluation, retained reports and incomplete accumulation. Added JavaScript edge-case tests, local-resource checking and disposable-copy smoke execution to Pages. Local strict/content/resource/contract/control/smoke checks passed. Browser/publication evidence is recorded in the verified recall publication above; full training, pretrained downloads, CUDA and real-device touch remain optional/unverified.


Local browser verification on 2026-10-01 covered all 24 learning routes (19 Hub and five DaZu) at desktop 1440×900 and mobile 390×844. The four replacement diagrams displayed their real stage labels; new transform/task/metric charts were inspected as rendered images. No page overflow or broken loaded images remained; lazy Fundamentals images were checked after scrolling into view. Metrics zero/absent positives/invalid values, padding masks/truncation, keyboard disclosure activation, direct inner anchors, same-anchor search reopening and code-copy content passed. The Hub now uses the modern clipboard API: the selected folded code and DaZu code were independently compared with clipboard text, then the prior clipboard was restored. Both themes were inspected; the new matrix fits a phone column and keyboard-focus text uses dark ink on the orange background. 73 rendered guide Python excerpts parsed successfully. Formal whole-site WCAG certification and physical-device touch are not claimed.

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
