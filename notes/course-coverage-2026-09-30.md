# Course 1 and Course 2 source coverage

Local review: 2026-09-30. This is an internal provenance and maintenance note, outside the public MkDocs navigation.

## Scope and review method

The sibling `../Pytorch/` archive was read only. Every folder and all 171 files were inventoried; all SHA-256 hashes matched the baseline after the review. The archive contains 30 Markdown files (README plus teaching exports), two notebooks, four support scripts, eight PDFs, 111 images, and 16 dataset/checkpoint/compressed assets. Course 1 supplies seven labs and four assignments; Course 2 supplies fifteen labs and four assignments.

All Markdown teaching text/code and the complete extracted PDF text were reviewed. Incremental duplicate slide text was condensed for reading. The eight decks contain 1,656 PDF pages: Course 1 modules 1–4 have 188, 217, 185 and 144 pages; Course 2 has 254, 252, 245 and 171. Representative first/middle/last slides and the custom-noise slide were rendered for visual inspection. This is not a claim that all 1,656 pages were inspected as images. Notebook code and stored outputs were reviewed, including the additional MNIST export output. Support-script interfaces were inspected. All 111 stored images/plots were inspected through labelled contact sheets and image references were checked. Binary datasets, pickles and checkpoints were inventoried/hashed, not deserialized or executed.

Read the archive README and both repositories' AGENTS/README/CONTEXT/CHANGELOG files. Existing library guides, all four project pages, navigation, interactions and validators were inspected. Course assessments provided a topic inventory only; their questions, notebooks and solutions are not copied into public pages. Public snippets are original and independently structured.

## Connected topic map

| Source module | Covered mechanisms | Main destination |
| --- | --- | --- |
| Course 1, module 1 | Tensors, layers, activations and autograd | [Guide](../docs/guides/fundamentals/core-workflow.md) |
| Course 1, module 2 | Classification, losses and validation | [Guide](../docs/guides/fundamentals/core-workflow.md) |
| Course 1, module 3 | Datasets, transforms, labels and robust loading | [Guide](../docs/guides/fundamentals/vision-real-data.md) |
| Course 1, module 4 | CNNs, regularization, inspection and debugging | [Guide](../docs/guides/fundamentals/vision-real-data.md) |
| Course 2, module 1 | Metrics, schedules, search and efficiency budgets | [Guide](../docs/guides/training/training-quality.md) |
| Course 2, module 2 | TorchVision data, utilities, pretrained models and transfer | [Guide](../docs/guides/vision/augmentation.md) |
| Course 2, module 3 | Tokens, embeddings, pooling and fine-tuning | [Guide](../docs/guides/text/tokens-embeddings.md) |
| Course 2, module 4 | DataLoader tuning, Lightning, profiling, precision and accumulation | [Guide](../docs/guides/training/efficient-training.md) |

Vision continues in [pretrained models](../docs/guides/vision/pretrained-models.md); text continues in [classifiers](../docs/guides/text/text-classifiers.md). The [function finder](../docs/reference/index.md#function-finder) provides direct API lookup. The four existing projects retain their own evidence and complete source.

## Teaching-file inventory

Paths below are relative to the sibling `Pytorch` archive. Duplicate `.ipynb`/`.md` views were compared; they are not counted as separate topics.

| Source file | Main destination |
| --- | --- |
| `COURSE 1/C1_M1_Lab_1_simple_nn/C1_M1_Lab_1_simple_nn.md` | [Course 1 module 1](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1_M1_Lab_2_activation_functions/C1_M1_Lab_2_activation_functions.md` | [Course 1 module 1](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1_M1_Lab_3_tensors.md` | [Course 1 module 1](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1_M2_Lab_1_mnist_classifier/C1_M2_Lab_1_mnist_classifier.ipynb` | [Course 1 module 2](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1_M2_Lab_1_mnist_classifier/C1_M2_Lab_1_mnist_classifier.md` | [Course 1 module 2](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1_M2_Lab_1_mnist_classifier/helper_utils.py` | [Course 1 module 2](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1_M3_Lab_data_management/C1_M3_Lab_data_management.md` | [Course 1 module 3](../docs/guides/fundamentals/vision-real-data.md) |
| `COURSE 1/C1_M4_Lab_1_cnn_nature_classifier/C1_M4_Lab_1_cnn_nature_classifier.md` | [Course 1 module 4](../docs/guides/fundamentals/vision-real-data.md) |
| `COURSE 1/C1_M4_Lab_2_debugging/C1_M4_Lab_2_debugging.md` | [Course 1 module 4](../docs/guides/fundamentals/vision-real-data.md) |
| `COURSE 1/C1M1_Assignment/C1M1_Assignment.md` | [Course 1 module 1](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1M2_Assignment/C1M2_Assignment.ipynb` | [Course 1 module 2](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1M2_Assignment/helper_utils.py` | [Course 1 module 2](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1M2_Assignment/unittests.py` | [Course 1 module 2](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1M2_Assignment/unittests_utils.py` | [Course 1 module 2](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/C1M3_Assignment/C1M3_Assignment.md` | [Course 1 module 3](../docs/guides/fundamentals/vision-real-data.md) |
| `COURSE 1/C1M4_Assignment/C1M4_Assignment.md` | [Course 1 module 4](../docs/guides/fundamentals/vision-real-data.md) |
| `COURSE 1/PyTorch_C1_M1.pdf` | [Course 1 module 1](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/PyTorch_C1_M2.pdf` | [Course 1 module 2](../docs/guides/fundamentals/core-workflow.md) |
| `COURSE 1/PyTorch_C1_M3.pdf` | [Course 1 module 3](../docs/guides/fundamentals/vision-real-data.md) |
| `COURSE 1/PyTorch_C1_M4.pdf` | [Course 1 module 4](../docs/guides/fundamentals/vision-real-data.md) |
| `COURSE 2/C2_M1_Lab_1_tuning_and_metrics/C2_M1_Lab_1_tuning_and_metrics.md` | [Course 2 module 1](../docs/guides/training/training-quality.md) |
| `COURSE 2/C2_M1_Lab_2_Schedulers/C2_M1_Lab_2_Schedulers.md` | [Course 2 module 1](../docs/guides/training/training-quality.md) |
| `COURSE 2/C2_M1_Lab_3_Optuna/C2_M1_Lab_3_Optuna.md` | [Course 2 module 1](../docs/guides/training/training-quality.md) |
| `COURSE 2/C2_M1_Lab_4_Efficiency/C2_M1_Lab_4_Efficiency.md` | [Course 2 module 1](../docs/guides/training/training-quality.md) |
| `COURSE 2/C2_M2_Lab_1_torchvision_1/C2_M2_Lab_1_torchvision_1.md` | [Course 2 module 2](../docs/guides/vision/augmentation.md) |
| `COURSE 2/C2_M2_Lab_2_torchvision_2/C2_M2_Lab_2_torchvision_2.md` | [Course 2 module 2](../docs/guides/vision/augmentation.md) |
| `COURSE 2/C2_M2_Lab_3_torchvision_3/C2_M2_Lab_3_torchvision_3.md` | [Course 2 module 2](../docs/guides/vision/augmentation.md) |
| `COURSE 2/C2_M2_Lab_4_transfer_learning/C2_M2_Lab_4_transfer_learning.md` | [Course 2 module 2](../docs/guides/vision/augmentation.md) |
| `COURSE 2/C2_M3_Lab_1_basic_tokenization.md` | [Course 2 module 3](../docs/guides/text/tokens-embeddings.md) |
| `COURSE 2/C2_M3_Lab_2_embeddings/C2_M3_Lab_2_embeddings.md` | [Course 2 module 3](../docs/guides/text/tokens-embeddings.md) |
| `COURSE 2/C2_M3_Lab_3_build_text_classifier/C2_M3_Lab_3_build_text_classifier.md` | [Course 2 module 3](../docs/guides/text/tokens-embeddings.md) |
| `COURSE 2/C2_M3_Lab_4_finetuned_text_classifier.md` | [Course 2 module 3](../docs/guides/text/tokens-embeddings.md) |
| `COURSE 2/C2_M4_Lab_1_optimizing_dataloaders/C2_M4_Lab_1_optimizing_dataloaders.md` | [Course 2 module 4](../docs/guides/training/efficient-training.md) |
| `COURSE 2/C2_M4_Lab_2_profiling.md` | [Course 2 module 4](../docs/guides/training/efficient-training.md) |
| `COURSE 2/C2_M4_Lab_3_optimization/C2_M4_Lab_3_optimization.md` | [Course 2 module 4](../docs/guides/training/efficient-training.md) |
| `COURSE 2/C2M1_Assignment/C2M1_Assignment.md` | [Course 2 module 1](../docs/guides/training/training-quality.md) |
| `COURSE 2/C2M2_Assignment/C2M2_Assignment.md` | [Course 2 module 2](../docs/guides/vision/augmentation.md) |
| `COURSE 2/C2M3_Assignment/C2M3_Assignment.md` | [Course 2 module 3](../docs/guides/text/tokens-embeddings.md) |
| `COURSE 2/C2M4_Assignment/C2M4_Assignment.md` | [Course 2 module 4](../docs/guides/training/efficient-training.md) |
| `COURSE 2/PyTorch_C2_M1.pdf` | [Course 2 module 1](../docs/guides/training/training-quality.md) |
| `COURSE 2/PyTorch_C2_M2.pdf` | [Course 2 module 2](../docs/guides/vision/augmentation.md) |
| `COURSE 2/PyTorch_C2_M3.pdf` | [Course 2 module 3](../docs/guides/text/tokens-embeddings.md) |
| `COURSE 2/PyTorch_C2_M4.pdf` | [Course 2 module 4](../docs/guides/training/efficient-training.md) |

## Course 1 gaps filled

- Tensor construction/storage (`tensor`, `as_tensor`, `from_numpy`), indexing, concatenation/stacking, reduction, reshape/view/flatten and dtype/device assignment.
- Explicit MSE broadcasting and batch-size-one squeeze pitfalls; MSE/L1/SmoothL1, cross-entropy, BCE-with-logits and NLL output/target contracts.
- Derivatives, graph recreation, registered model layers, named modules/parameters/buffers, activation inspection and parameter counting.
- Separate transforms on shared train/validation split indices, label mappings and best-state/checkpoint details.

## Interesting Course 2 mechanisms retained

- Macro/micro/weighted metrics, confusion matrices and metric state; scheduling step placement; Optuna trials/search spaces and fresh model creation.
- Memory/latency constraints versus validation quality; trainable counts versus stored bytes versus peak training memory; synchronized CUDA timing.
- PIL/tensor/range contracts, transform order, train-only normalization, salt-and-pepper corruption and realistic augmentation. Gaussian and shot-noise comparisons are explicitly additional explanation, not reproduced lab experiments.
- Weight-specific preprocessing and class metadata; classification versus segmentation/detection output structures; feature extraction, partial and full fine-tuning; frozen BatchNorm/dropout behavior.
- Word/subword tokenization, unknown/special tokens, padding/masks/collation, static/contextual embeddings, similarity limits, EmbeddingBag offsets and masked pooling.
- Training-only class weights, external weighted transformer loss, partial DistilBERT unfreezing and save/reload contracts.
- Workers/prefetch/pinning, Windows process safeguards, Lightning hooks/callbacks, profiling, mixed precision and actual-sample accumulation.

## Archive limitations

Three referenced images are absent:

- `COURSE 1/C1M3_Assignment/exp_out_1.png`
- `COURSE 1/C1M4_Assignment/nb_image/lab_1_training_plot.png`
- `COURSE 2/C2_M1_Lab_1_tuning_and_metrics/nb_image/cifar10.png`

Many teaching exports import course-supplied `helper_utils`, `model_architectures` or `unittests`, or require datasets/weights not bundled beside the exports. They are a learning archive, not a verified portable runnable environment. The archive README still says Course 2 is pending download, while its files and the user's completion report are present. Its package inventory is also historical: Optuna, Lightning and TorchMetrics were not installed in the checked global environment. Transformers is present. The source archive was preserved and its README was not edited.

## Verification and limits

Strict MkDocs build, 16-page content validation, 17-page generated link/anchor validation, four-guide DaZu cross-site validation and shared JavaScript syntax passed. The original recall-pattern tests passed: impulse-noise boundaries/input preservation, masked pooling/padding invariance, EmbeddingBag collation and SGD accumulation equivalence with uneven final samples. All four existing smoke examples passed on a disposable copy, preserving the maintained regression image and prior experiment artifacts.

Browser verification covers affected routes at 1440×900 and 390×844, valid and invalid accumulation/padding inputs, both noise modes and zero/noisy levels, mobile guide navigation, search results and code-copy success feedback. No horizontal page overflow was observed; clipboard contents were not independently read back by the browser tool. Offline random tiny DistilBERT/ResNet checks confirmed selected freezing, head, logits and weighted-loss contracts without downloads. CUDA was unavailable to the process, so AMP GPU execution was skipped. Full dataset training, pretrained downloads, model-quality reproduction, optional Optuna/Lightning/TorchMetrics runtime execution, real-device touch, live deployment and redirect behavior were not validated. Nothing was installed, committed or published during the original content-review phase. The subsequent explicit release request pushed Hub content commit `10cff36`; Pages run `36770490711` succeeded and all sixteen public routes returned HTTPS 200. DaZu quick guides were independently released from website commit `90a8b6b` in Netlify deploy `6abd6c46b1fd9399018fe764`.
