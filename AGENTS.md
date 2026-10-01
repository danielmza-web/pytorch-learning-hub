# Working in PyTorch Learning Hub

- Read `README.md`, `CONTEXT.md` and `CHANGELOG.md` before changes; inspect Git status and preserve existing edits. This is an independent Git repository, even when nested inside DaZu. Do not rely on the parent's commit to save this work.
- `docs/`, `examples/`, `mkdocs.yml` and local assets are the maintained sources. `site/` is generated. Follow CONTEXT's two-page guide structure, stable anchors, original-content policy and evidence labels; do not publish protected assessments or invented metrics.
- Follow README's scoped validation. The smoke suite regenerates a tracked regression image; do not use it for a read-only task. A generated-site check with zero pages is not verification. Check affected browser behavior after meaningful UI changes.
- Reviewed local commits may happen automatically; remote pushes require Daniel’s explicit request covering publication. Pushes to `main` trigger GitHub Pages, including documentation-only changes. Follow the release instructions; never publish this library to Netlify. Coordinate copied visuals and quick-guide anchors with DaZu only when affected.
- Maintain README for operation/setup, CONTEXT for decisions/current verification/next steps, and CHANGELOG for significant evidenced changes. Distinguish local checks from live publication; keep dates factual and preserve useful prior decisions.
- Recreate environments on each PC and preserve necessary ignored experiment evidence separately. Never put credentials in files or transfer machine-specific environments as setup instructions.
