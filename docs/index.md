---
title: PyTorch Learning Hub
tags:
  - pytorch
  - deep-learning
last_reviewed: 2026-08-05
---

<section class="hub-hero" markdown>

<span class="eyebrow">DANIEL ZURITA · LEARNING REFERENCE</span>

# Understand the workflow.<br>Return when you need it.

A visual, practical PyTorch reference built from original explanations, executable examples, and projects. Use it to understand a mechanism, refresh an idea you have already seen, or connect code to a concrete decision.

[Open the Fundamentals guide](courses/fundamentals/index.md){ .md-button .md-button--primary }
[Open the quick reference](courses/fundamentals/quick-review.md){ .md-button }

</section>

## The complete mental model

```mermaid
flowchart LR
    A["Data"] --> B["Dataset"]
    B --> C["DataLoader"]
    C --> D["Model"]
    D --> E["Logits"]
    E --> F["Loss"]
    F --> G["Gradients"]
    G --> H["Optimizer"]
    H --> D
```

Every training project is a variation of this loop. The model changes, the data changes, and the evaluation becomes more sophisticated—but the core flow remains recognizable.

<div class="feature-grid" markdown>

<article class="feature-card" markdown>

### Guides

Browse connected topics when you want the larger mental model, from tensor shapes to reliable model evaluation.

[Open the Fundamentals guide →](courses/fundamentals/index.md)

</article>

<article class="feature-card" markdown>

### Concepts

Return to one focused explanation: tensor shapes, the training loop, convolution, or generalization.

[Browse concepts →](concepts/index.md)

</article>

<article class="feature-card" markdown>

### Projects

See how the pieces connect in regression, handwriting recognition, robust data loading, and nature classification.

[Explore projects →](projects/index.md)

</article>

<article class="feature-card" markdown>

### Reference

Use the cheatsheet, error guide, and glossary while writing or debugging code.

[Open reference →](reference/index.md)

</article>

</div>

!!! info "A reference, not a course platform"
    This site stores no progress, completion state, accounts, analytics, or personal learning history. Open the topic that answers your question; there is no required sequence.

## Featured original projects

| Project | Core question | What it demonstrates |
| --- | --- | --- |
| [Nonlinear regression](projects/regression.md) | When does a linear model stop being enough? | Tensors, autograd, loss, optimization |
| [EMNIST letter classifier](projects/emnist.md) | Why does image structure matter? | Dense baseline, CNN, distribution shift |
| [Robust image pipeline](projects/robust-image-pipeline.md) | How do we stop data problems from breaking training? | Custom Dataset, transforms, validation |
| [Nature CNN](projects/nature-cnn.md) | How do we recognize and reduce overfitting? | CNN blocks, regularization, diagnostics |

## How to use this hub

1. Read the [quick reference](courses/fundamentals/quick-review.md) before an exercise or interview.
2. Browse the [Fundamentals topic map](courses/fundamentals/index.md) when you want the whole connection.
3. Open a [concept](concepts/index.md) when one mechanism is unclear.
4. Use the [common errors guide](reference/common-errors.md) when code fails.
5. Study a [project](projects/index.md) to connect the individual pieces.
