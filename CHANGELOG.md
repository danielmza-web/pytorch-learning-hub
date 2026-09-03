# Changelog

Selected changes supported by this repository's local Git history. Dates are commit dates, not inferred production dates. DaZu has a separate history and deployment process.

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
