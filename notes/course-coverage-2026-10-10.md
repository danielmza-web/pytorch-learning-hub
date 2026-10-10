# Certificate coverage review — 2026-10-10

This is an internal coverage and verification note, not a public course-completion interface. Daniel confirmed completion of the entire certificate on this date. It does not establish an independently verified credential issue date. The public library remains original English teaching material, organized as six two-page guides and eight practical projects.

## Sources and inventory

Reviewed the current [Coursera certificate syllabus](https://www.coursera.org/professional-certificates/pytorch-for-deep-learning) and its three course descriptions: Fundamentals; Techniques and Ecosystem Tools; Advanced Architectures and Deployment. Course 3's [official description](https://www.coursera.org/learn/pytorch-advanced-architectures-and-deployment) connects advanced vision architectures, transformer building blocks, transformer applications and model deployment. These provider descriptions establish topic coverage, not the user's completion or reproduction of training results.

The local `Pytorch/COURSE 3` archive contains 104 files: four PDF decks (231, 201, 283 and 233 pages; 948 total), 17 Markdown exports (13 labs and four assignments), and 83 supporting images. Deck text and exports were reviewed locally. Course 1/2 source coverage and retained-run limits are recorded in [the earlier coverage note](course-coverage-2026-09-30.md). The archive README was corrected separately to reflect Daniel's completion confirmation and actual module links. No decks, exports, supporting images, graded questions, solutions or protected excerpts were copied into this repository. The 104 Course 3 files matched their pre-review SHA-256 hashes after the work.

## Coverage map

| Course/module | Connected library explanation | Boundaries |
| --- | --- | --- |
| Course 1 foundations | Fundamentals: tensor/storage semantics, modules, autograd, losses, training loop, shapes, CNNs and dataset splits | Synthetic examples and preserved historical project evidence; no fresh full training claim |
| Course 2 training | Training: metrics/thresholds, tuning, scheduling, profiling, precision, accumulation, DataLoader and optional ecosystem tools | Trial pruning differs from weight pruning; validation/search reuse can overfit |
| Course 2 vision | Vision: noise/transforms, pretrained weights, frozen backbones and task-specific evaluation | Paired transforms preserve boxes/masks; random offline backbones demonstrate mechanisms |
| Course 2 text | Text: vocabulary, token IDs, padding, embeddings, classifiers and loss normalization | Mask conventions and denominators are explicit; bridge from pooling to attention |
| Course 3 module 1 | Architectures/vision-generative: modular/shared encoders, multiple inputs/outputs, Siamese/triplet retrieval and ResNet/DenseNet/MobileNet/EfficientNet | Structural comparisons and original toys; no downloaded pretrained models or benchmark claims |
| Course 3 module 2 | Architectures/vision-generative: hooks, saliency/Grad-CAM, receptive fields and diffusion | Original gradient/heatmap toys; no generated-image quality measurements |
| Course 3 module 3 | Architectures/transformers: positions, Q/K/V shapes, scaled/multi-head attention, residuals/norms, encoder/decoder families, masks, shifted targets, teacher forcing, autoregressive generation, caches and cross-attention | Original CPU attention/causal toys; no language-model pretraining or quality measurements |
| Course 3 module 4 | Deployment/model-lifecycle: live objects versus files, parameter/buffer state, component reuse, snapshots/checkpoints, resume, restricted loading, mmap/meta/assign and MLflow | `weights_only=True` restricts deserialization; it does not extract only weights from a checkpoint |
| Course 3 module 4 and practical extension | Deployment/optimization-serving: pruning, quantization modes/calibration, distillation, ONNX, input contracts, target measurements, monitoring and rollback | Distillation is identified as an extension; zero weights alone do not shrink dense shapes or guarantee acceleration |

The new project `examples/model_lifecycle.py` uses an original 58-parameter CPU network and synthetic inputs. It trains one SGD-momentum step, writes/restores weights and a resumable checkpoint, saves/reuses an encoder, checks snapshot independence and demonstrates pruning serialization. The exact FP32 weight payload is 232 bytes; filesystem size and runtime/training memory are separate quantities.

## Primary technical references and API decisions

- [PyTorch saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html) and [efficient module loading](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html): reconstruct architecture, restore persistent buffers, set inference mode separately, and create the optimizer after assigned/meta loading. Shape mismatches are not bypassed by `strict=False`.
- [Pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html): masks/reparameterization and `prune.remove` preserve dense shapes. Structural/sparse execution and real target measurements are required for speed claims.
- [PyTorch quantization](https://docs.pytorch.org/docs/stable/quantization.html) and [torchao](https://docs.pytorch.org/ao/stable/): current recommended APIs move toward torchao; eager course examples are labeled legacy rather than presented as new defaults. Calibration uses representative training-derived data, reserving final test evaluation.
- [ONNX export](https://docs.pytorch.org/docs/stable/onnx.html): the original optional toy uses the current dynamo exporter and bounded dynamic batch dimensions. TorchScript is identified as deprecated in current PyTorch documentation.
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) and [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/): export, tracking and serving are distinct steps; acceptable quality/latency must be measured on the intended target.
- [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) and [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html): boolean masks use opposite conventions. Causal attention includes the current token. Key-padding masking does not itself zero padded query outputs.

Architecture pages link original papers and primary framework references alongside the related explanation. Heatmaps are sensitivity tools, not proof of reasoning; seeds and additional diffusion steps do not guarantee output quality. Temperature changes sampling probabilities but does not change greedy argmax ordering.

## Verification and limits

Local strict MkDocs build: 24 Markdown source pages, 25 generated HTML pages including 404, six two-page guides and eight projects. Content, local links/resources/anchors, teaching-image dimensions, recall patterns, selected examples, learning contracts, copy controls, model-lifecycle integration tests and numerical widget tests passed. All 123 Python Markdown excerpts parsed; 80 local handoff/archive links resolved. All four foundation smokes passed in a disposable copy, preserving maintained figures. Original architecture toys verified registration, triplet arithmetic, gradient/heatmap shapes, attention agreement and causal invariance. The optional ONNX CPU toy matched PyTorch for batch sizes 1, 3 and 16. No packages were installed; Python 3.14.7 and Torch 2.14.0+cu130 were already available. torchao was absent.

Local browser checks visited all 24 Hub routes at 1440 and 390 px, and all seven DaZu learning routes at 1440, 390 and 320 px. No page overflow, broken completed images or captured warnings/errors were observed. Representative light/dark and desktop/mobile views were inspected. Attention/memory controls were exercised with valid and invalid inputs, including an 8-position causal table fitting a 320 px phone and exact byte labels for tiny models. Mobile menu opening, search results containing the new lifecycle pages and exact DaZu code-copy text were checked; the prior clipboard was restored. Route visits do not exhaust every interaction or below-fold lazy image.

No full/pretrained/GPU training, physical-device inference/quantization benchmark, torchao execution, MLflow end-to-end run, formal screen-reader audit or production redirect check was performed. No remote fetch/push, Netlify deployment or Pages publication occurred. Generated output and temporary evidence remain ignored; the two source repositories are committed independently.
