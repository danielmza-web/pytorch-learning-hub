# Visual-first review — 2026-10-11

Daniel approved simplifying the existing visuals: the complete flow or the important differences should be visible before any controls. Public content remains English. The initial implementation was local; Daniel subsequently authorized the GitHub/Netlify release recorded below.

## Presentation decisions

The 31 reviewed pages contain 35 committed HTML/SVG overview instances. Text is selectable; SVG supplies geometry, pixels or curves rather than tiny embedded labels. Main explanations are outside disclosures. DaZu uses shorter counterparts; the Hub retains exact examples and optional calculations.

| Hub page | Visible explanation | Optional exploration |
| --- | --- | --- |
| 1.1 | Five training stages; backward changes gradients, step changes weights. Reshape/transpose/cat/stack on fixed data; compatible/incompatible broadcasting. | Image dimensions, broadcasting and batch arithmetic. |
| 1.2 | Input/filter/output; one output-cell control. Three conceptual loss situations together. | Nine products, filter, stride, padding and shape tracing. |
| 2.1 | Two 99%-accuracy systems with 100% versus 0% defect recall. Fixed scores and labels with one threshold slider. | Editable confusion counts; reset restores the explained 100-case example. |
| 2.2 | Updates 48, 48, 37; last group 16 + 16 + 5 uses divisor 37. | Three numeric group-size fields, as agreed. |
| 3.1 | Original/flip/crop with aligned object, box and mask; clean/impulse/Gaussian views. | Noise type and amount. |
| 3.2 | Head, last block + head, whole model: 10/30/50 trainable toy values. Separate freezing versus eval-mode comparison. | Code details; no strategy selector. |
| 4.1 | Fixed lookup ID, separate positional signal; lengths 2/3 under dynamic padding, fixed length and truncation. | Other lengths and maximum. |
| 4.2 | Mean/sum/max on the same vectors; PAD inclusion and order reversal; explicit empty-input policy. | Code details; no pooling selectors. |
| 5.1 | Residual 16, concatenated 48, depthwise/pointwise 32 output channels; 656 versus 4,608 weights. | Code details; no connection/size controls. |
| 5.2 | Bidirectional/causal masks together. | One access-mode choice; row 1 weights/output. PAD, other query rows and full matrices are further details. |
| 6.1 | Construct/train/save/close/reconstruct/restore across RAM and disk; three artifact routes; FP32/FP16/INT8 bars. | Custom weight payload and implementation details. |
| 6.2 | Dense zeros 128 bytes, smaller layer 64, hypothetical INT8 32. Separate scale-0.25 rounding explanation. | One value to quantize; clearly labeled signed four-bit toy range. |

| Project | Change |
| --- | --- |
| P1 | Removed the parameter-count text slider; retained the measured figure and explained nonlinearity directly. |
| P2 | Visible 28 → 14 → 7; optional custom tracer. |
| P3 | Valid, corrupt and unsupported-extension decisions visible together. |
| P4 | Three conceptual learning-curve situations together, explicitly separate from measured results. |
| P5 | Reused accumulation example; optional size calculator. |
| P6 | Reused clean/impulse/Gaussian comparison; optional noise tool. |
| P7 | Reused dynamic/fixed/truncated padding comparison; optional lengths calculator. |
| P8 | Reused the three artifact comparisons alongside the corresponding code. |

Home, project chooser, reference and About retain their organization. Reading cues, removed-control references and the three authoring templates were updated. DaZu's home and six guides retain navigation; training, transfer, pooling, attention and lifecycle diagrams are compact. Fundamentals calculators share a disclosure, curves and memory budgets are static, and threshold is the only main interactive control.

## Implementation and checks

- `scripts/render_learning_overviews.cjs` authors committed overview slots, using the verified numeric exports where applicable. It generates detailed Hub and compact DaZu variants, supports standalone Hub checkouts and offers a read-only `--check` mode. Numerical functions remain available; unused UI renderers and text-only project handlers were removed.
- Strict MkDocs build passed. Content/navigation, built links/resources/anchors, presentation and the new visual-first contract passed: 24 Markdown routes, 25 built HTML pages including 404, 18 retained teaching images and 35 visible source overviews across both sites. The new contract is in the existing Pages workflow.
- Independent CPU PyTorch comparisons passed for SGD, 16 convolution cases, four attention masks, six pooling cases, 16 rounding cases, paired geometry and raster response. Existing Node checks passed for learning labs, recall controls, Course 3 arithmetic and copy behavior. JavaScript syntax passed.
- Parent PyTorch links/assets and clipboard tests passed; shared JS/CSS copies are byte-identical. Static denominators, padding counts, recall, channel/weight counts and byte budgets agree with the retained functions and checked tools.
- All 5,335 previous generated Hub IDs and 45 DaZu learning-page IDs were preserved, including `#interactive-check` on P1–P4 and the former `#regularization` target. Compared with the starting Git heads, 34 parent and 36 Hub maintained image/example/data files were unchanged; text comparisons normalize line endings.
- Browser review covered all 31 pages at 1440 × 900, 820 × 1180, 390 × 844 and 320 × 844, with both Hub themes: 220 route visits. No page overflow, broken completed images, uninitialized main labs or captured warnings/errors were observed. After shortening lifecycle text and compacting DaZu, its seven pages were rechecked at all four widths (28 visits), and the Hub lifecycle at four widths/two themes (eight visits).
- Representative controls passed actual browser checks: all four convolution filters, stride/padding, keyboard output-cell selection/reset; threshold endpoints/reset on both sites; metric preset/reset/blank count; attention modes/query/PAD/reset; quantization blank/clipping/reset; complete/incomplete accumulation and invalid microbatch; tensor dimensions, incompatible broadcasting, drop-last batches, noise, weight budget, P2 shapes and P5 accumulation. Legacy optional calculators initialized on their pages. Native disclosure keyboard activation, direct and repeated hidden-code anchors, search-result reveal and mobile navigation to P8 worked.
- Two temporary script-free browser fixtures checked the main Hub lifecycle and DaZu Fundamentals, including a native disclosure. Six lifecycle stages remained visible in the document and neither fixture overflowed. The fixtures were removed by the final strict rebuild; all main static overviews are also verified structurally outside details with no controls/canvas dependency.

## Limits and preservation

The implementation checks above were local; live-provider evidence is recorded separately below. No dependency installation, fresh training, GPU/pretrained benchmark, personal course-archive edit, CV change or frozen-reference change was needed. Existing runnable examples, measured images and retained reports were preserved, so the full training/smoke suites were not repeated during the local implementation. The authorized Pages workflow subsequently passed its existing disposable smoke/example checks. Browser route checks do not exhaust every lazy image or every combination of optional calculator fields. Physical touch, native zoom, formal screen-reader certification and whole-page performance measurements were not performed. Temporary browser captures remain under ignored parent `tmp/visual-first/` and are optional evidence, not build inputs.

## Authorized production release

Daniel explicitly authorized GitHub pushes and Netlify publication. Application commits `359a659` (DaZu) and `8ed6514` (Hub) were pushed separately to `main`, with zero divergence afterward. DaZu published only `site/` through [Netlify deploy 6acb7f0d22b00e3e8fbe265e](https://app.netlify.com/projects/dazu/deploys/6acb7f0d22b00e3e8fbe265e), confirmed ready; the Hub passed its complete validation workflow and published through successful [Pages run 38138866427](https://github.com/danielmza-web/pytorch-learning-hub/actions/runs/38138866427). No hosting configuration changed.

Live HTTPS verification passed 86 DaZu checks (11 canonical routes, 52 committed route/resource matches and all 34 aliases) and 58 Hub checks (24 routes and 34 committed resources). All generated IDs, six remaining lab slots and 35 static overview slots matched live HTML. Text comparisons normalize CRLF; generated Hub HTML is checked for routes, IDs and visual/control markers rather than exact build-byte equality. Browser review covered all 31 learning routes at 1440/390 px (62 visits), with no observed overflow, broken completed images, uninitialized labs or captured warnings/errors. Representative threshold, convolution, attention, accumulation, quantization and reset checks passed in production. Physical touch and a formal accessibility audit were not repeated.

Documentation follow-ups retain the application publication references and are pushed separately. Their Hub head needs its own successful Pages run; no additional DaZu deployment is needed when only these non-published documentation files change.
