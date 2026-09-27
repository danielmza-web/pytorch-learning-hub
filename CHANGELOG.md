# Changelog

Selected changes supported by local checks or this repository's Git history. Local review dates are not publication dates. DaZu has a separate history and deployment process.

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
