---
title: Start here
tags: [home]
last_reviewed: 2026-10-10
---

# PyTorch Learning Hub

A connected library of explanations, visual checks and runnable examples. Begin with tensors and training, then move through vision, text, architectures and deployment. Jump directly to a topic when reviewing.
{ .page-lead }

## What you will find here

**Guides:** six topics, two pages each. Understand the idea, follow shapes and operations, then check your reasoning.

[**Projects**](projects/index.md): eight original scripts, ordered P1–P8. Read the prerequisites, run a small workflow and inspect its output.

[**Quick reference**](reference/index.md): find a function, shape reminder or common error when debugging.

## Choose a guide

Follow the numbered route once; use the same numbers in the sidebar to return to a specific page. Each topic builds on ideas introduced earlier.

<ol class="learning-map" aria-label="Recommended learning route" markdown="1">
<li><a href="guides/fundamentals/core-workflow/"><span>1</span>Fundamentals</a></li>
<li><a href="guides/training/training-quality/"><span>2</span>Training</a></li>
<li><a href="guides/vision/augmentation/"><span>3</span>Vision</a></li>
<li><a href="guides/text/tokens-embeddings/"><span>4</span>Text</a></li>
<li><a href="guides/architectures/vision-generative/"><span>5</span>Architectures</a></li>
<li><a href="guides/deployment/model-lifecycle/"><span>6</span>Deployment</a></li>
</ol>

<div class="topic-grid" markdown="1">

<div class="topic-card" data-topic="1" markdown="1">

**1 · Learn the fundamentals**

Tensors, batches, logits, gradients and image classification.

[1.1 · Build and train](guides/fundamentals/core-workflow.md)

[1.2 · Images and CNNs](guides/fundamentals/vision-real-data.md)

</div>

<div class="topic-card" data-topic="2" markdown="1">

**2 · Improve training**

Validation, fair comparisons, data loading and memory use.

[2.1 · Metrics and tuning](guides/training/training-quality.md)

[2.2 · Efficient pipelines](guides/training/efficient-training.md)

</div>

<div class="topic-card" data-topic="3" markdown="1">

**3 · Work with vision**

Image variation, preprocessing, task outputs and transfer learning.

[3.1 · Data, transforms and noise](guides/vision/augmentation.md)

[3.2 · Pretrained models](guides/vision/pretrained-models.md)

</div>

<div class="topic-card" data-topic="4" markdown="1">

**4 · Work with text**

Tokens, embeddings, variable lengths and text classification.

[4.1 · Tokens and embeddings](guides/text/tokens-embeddings.md)

[4.2 · Classifiers and fine-tuning](guides/text/text-classifiers.md)

</div>

<div class="topic-card" data-topic="5" id="continue-into-course-3-topics" markdown="1">

**5 · Explore advanced architectures**

Shared modules, similarity, efficient backbones, generation and attention.

[5.1 · Vision, similarity and generation](guides/architectures/vision-generative.md)

[5.2 · Attention and transformers](guides/architectures/transformers.md)

</div>

<div class="topic-card" data-topic="6" markdown="1">

**6 · Save and deploy models**

RAM, saved state, component reuse, compression and target-device evaluation.

[6.1 · Memory, checkpoints and tracking](guides/deployment/model-lifecycle.md)

[6.2 · Compression, export and deployment](guides/deployment/optimization-serving.md)

</div>

</div>

## Practice as you learn {#selected-course-2-examples}

Choose a project after its related guide. The order below matches the project gallery and sidebar; projects have their own **P** numbers so they are easy to distinguish from guide pages.

| After studying | Apply it | Inspect the result |
| --- | --- | --- |
| 1.1 · Core workflow | [P1 · Nonlinear regression](projects/regression.md) | Activation, curve fit and validation loss |
| 1.2 · Images and CNNs | [P2 · EMNIST letters](projects/emnist.md), [P3 · Robust image pipeline](projects/robust-image-pipeline.md), [P4 · Nature CNN](projects/nature-cnn.md) | Shapes, readable files, mistakes and generalization |
| 2.1–2.2 · Training | [P5 · Controlled training comparison](projects/training-comparison.md) | Matched experiments and validation metrics |
| 3.1–3.2 · Vision | [P6 · Image augmentation and head training](projects/vision-head.md) | Transform order, frozen features and a changing head |
| 4.1–4.2 · Text | [P7 · Variable-length text classifier](projects/text-bags.md) | Vocabulary, offsets and pooling limits |
| 5.1–5.2 · Architectures | Visual checks and runnable toys within the guides | Shared weights, attention and allowed information flow |
| 6.1–6.2 · Deployment | [P8 · Model lifecycle and pruning](projects/model-lifecycle.md) | Restored state, component reuse and unchanged dense shapes |

For a first offline run, start with P1. P5–P8 also have small CPU defaults without downloads. P2–P4 use public image datasets for full runs and offer offline smoke checks. [Choose a project](projects/index.md) explains the data and outputs before you run it.

## How to use a page

1. Read its learning goal and prerequisites.
2. Read the complete diagram, predict its key shape or decision, then open optional calculations if you want to explore.
3. Expand code and variants when you need implementation. Excerpts name the objects they require; complete runnable scripts live in Projects.
4. Run the related project and inspect the stated output. Use **Continue reading** for the next guide or the sidebar to change topics.

Search for an exact function such as `EmbeddingBag` or `ReduceLROnPlateau`. Search results and direct section links reveal their matching details. Keep [Quick reference](reference/index.md) nearby for shape and error reminders.

Illustrations explain mechanisms; synthetic runs test small examples; retained measurements describe their specific execution. [About and sources](about/index.md) explains the original-content and evidence policy.
