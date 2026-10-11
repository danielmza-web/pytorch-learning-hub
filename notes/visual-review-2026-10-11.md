# PyTorch visual review — 2026-10-11

Implementation of the approved understanding/comparison plan. The checks below record local implementation evidence. Subsequent authorized publication has separate live evidence:

Daniel explicitly requested GitHub pushes and the Netlify update. Application commits `4a50387` (DaZu) and `31e85f2` (Hub) were pushed separately to `main`, with zero divergence. DaZu published only its `site/` through [Netlify deploy 6acb706cabc97a5f665167d6](https://app.netlify.com/projects/dazu/deploys/6acb706cabc97a5f665167d6); the independent Hub published through successful [Pages run 38135003428](https://github.com/danielmza-web/pytorch-learning-hub/actions/runs/38135003428). No hosting configuration changed.

Live HTTPS verification passed 86 DaZu checks (11 canonical routes, 52 committed route/resource matches and all 34 configured aliases) and 58 Hub checks (24 routes and 34 committed resource matches). All locally generated page IDs and all 19 lab slots matched live HTML. Text comparisons normalize CRLF; generated Hub HTML is not compared byte for byte across build environments. Browser review covered all 35 canonical routes at 1440/390 px (70 visits), with no observed overflow, broken completed images or captured warnings/errors after initialization. All 19 labs initialized; reset and available select changes worked, and the SGD update/reset was checked. This supplements the preceding local arithmetic and exhaustive option/invalid-input checks; physical touch, formal screen-reader certification and full/pretrained/GPU training were not repeated.

## Review scope

The initial review covered all 99 versioned Hub files and 31 related DaZu files: the seven learning pages, their assets, cross-site validators, workspace handoff documents and publication configuration. Markdown/HTML explanations, example implementations, templates, scripts, tests and settings were read; Python structure, JSON/SVG content, image integrity/dimensions and generated routes were checked. The maintained source inventory remains 24 Hub pages (12 guides, eight projects, gallery, home, reference and about) plus seven DaZu learning pages. Frozen portfolio references, CV claims, personal course materials, downloads and retained model-quality evidence were preserved.

## Implemented comparisons

| Guide | New visual calculation | Decision it makes visible |
| --- | --- | --- |
| 1.1 | Exact two-sample SGD update; reshape/transpose/cat/stack | Backward changes gradients; step changes weights. Equal output shapes can contain different values. |
| 1.2 | Sliding 3×3 filter with stride, zero padding and selected products | An output cell is a sum of products, not a hand-written feature-map guess. |
| 2.1 | Fixed four scores and adjustable threshold | Decisions and confusion counts change while model scores stay fixed. No alarms gives undefined precision. |
| 2.2 | Full and final microbatch groups | The last 37 samples in a 133-sample epoch need their actual denominator. |
| 3.1 | Shared flip or crop/resize of pixels, box and mask | Geometry must stay aligned; class masks are not image-normalized. |
| 3.2 | Head-only, partial and full parameter ownership | Frozen weights still run forward; BatchNorm/Dropout mode is a separate policy. |
| 4.1 | Fixed lookup table plus separate positional signal | A token ID selects a row; position and contextual mixing are different mechanisms. |
| 4.2 | Mean/sum/max with PAD exclusion and reversed words | Pooling loses order; zero padding can corrupt max over negative coordinates. |
| 5.1 | Residual addition, dense concatenation, depthwise + pointwise | Aligned shapes, growing channels and smaller weight counts are different changes. |
| 5.2 | Fixed Q/K/V, scaled scores, allowed pairs, softmax and output | Masks eliminate key contributions before normalization; a PAD query is not automatically zero. |
| 6.1 | Construct/train/save/close/reconstruct/restore | RAM, persistent files and compatible architecture are distinct; inference weights do not restore optimizer moments. |
| 6.2 | Dense zeros, smaller shape and INT8 payload; 4-bit numeric toy | Zero entries do not remove dense slots. Rounding, clipping and metadata matter. |

The Hub has 13 lab instances and DaZu has six focused counterparts. Native controls have labels, text results, reset actions and keyboard support. Calculations are deterministic and run without training, downloads or additional dependencies. New JS/CSS copies are byte-identical across the independently hosted sites, enforced by the parent asset validator.

## Corrections and readable figures

- The convolution generator now computes the actual valid cross-correlation. Its rows are `[0.1,2.9,2.9,0.1]` three times, then `[0,3,3,0]`. Cells show their values. The interactive calculation and raster are independently checked against `conv2d`.
- Decode and transforms are shown inside Dataset access before DataLoader collation. Raster labels no longer sit behind decorative pixels or run through box edges.
- Native diagrams stack vertically on mobile. Small-text conceptual rasters are folded reference figures; retained measurement images have original-size links. Four conceptual PNGs were regenerated; retained regression, EMNIST and training result images were not regenerated in the maintained checkout.
- A triplet branches through one shared encoder. Grad-CAM follows one chosen class score. Diffusion training and sampling have separate diagrams.
- Synthetic curves select coherent overfitting/balanced/underfitting regimes, have labeled axes, text descriptions and reset. Canvas resolution follows its displayed width, including after disclosure reveal, so mobile labels are not scaled down from a desktop bitmap.
- Empty, fractional, negative or over-pooled shape inputs produce a correction message. Unsafe tensor products are rejected. Decimal matrices may scroll within their figure on narrow screens, retaining readable numbers and the selected result in text.
- Guide/project templates now require numbering, orientation, prerequisites, reciprocal practice links, code/data requirements and evidence labels.

## Local verification

- Strict build; content/navigation; links/resources/anchors across 25 generated HTML pages including 404; all 18 teaching-image occurrences retain correct intrinsic dimensions; parent quick-guide/asset checks.
- Independent CPU PyTorch comparisons: SGD at three learning rates, 16 convolution settings, four attention masks, six pooling variants, 16 quantization cases, paired geometry and the raster's computed feature map. Additional boundary tests cover invalid numbers, exact threshold equality, absent alarms, all-PAD pooling, incomplete groups, blocked attention, ties-to-even rounding and curve regimes.
- Existing recall, selected-example, learning-contract, lifecycle, copy, visual-control and disposable-copy smoke checks. Smoke training ran in a disposable checkout to preserve maintained result images. Windows test temporary outputs stayed in ignored workspace directories.
- All 31 public learning pages visited at 1440, 820, 390 and 320 px; both Hub themes: 220 route visits. No observed page overflow, missing completed images, uninitialized labs or captured warnings/errors. All 19 lab instances passed 83 control checks across options, blank number fields, arrow keys and reset. Training step boundaries and the existing curve/shape tools received additional checks.

Limits: these are local browser and CPU checks, not live publication, physical touch, a formal screen-reader audit, full/pretrained/GPU training or device latency/memory/accuracy benchmarks. Lazy images were covered structurally; the route sweep only classified completed image loads. Numeric displays round to four decimals; calculations use full JavaScript precision. Quantization's numeric demonstration is explicitly signed 4-bit, while its separate payload comparison states FP32/INT8.

## Maintenance

Keep `docs/assets/javascripts/learning-labs.js` and `docs/assets/stylesheets/learning-labs.css` as the shared implementation. Copy them to DaZu's learning assets when changed, then run both repositories' checks. The new numeric tests run in the existing Pages build job; the publication trigger and destination are unchanged. Commit the two repositories separately. Remote publication still requires Daniel's explicit request.
