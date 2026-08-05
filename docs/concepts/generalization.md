---
title: Generalization
tags:
  - generalization
  - validation
last_reviewed: 2026-08-05
---

# Generalization

Generalization asks whether patterns learned from training examples remain useful on unseen data from the intended environment.

## Four gaps to inspect

1. **Train–validation gap:** has the model memorized training-specific detail?
2. **Validation–deployment gap:** does the validation set represent real inputs?
3. **Aggregate–class gap:** does one headline metric hide weak classes?
4. **Accuracy–cost gap:** is the improvement worth memory, latency, and complexity?

## Distribution shift

A model can improve on its held-out test set and still get worse on an external sample. That does not necessarily contradict the test result: the two data sources may represent different distributions.

For handwriting recognition:

```text
training/test: standardized EMNIST letters
external use:  one person's distinctive handwriting
```

Treat external examples as a separate evaluation domain. Report both results rather than choosing the metric that looks best.

## Better experiments

- State one hypothesis per run.
- Freeze the data split and seed.
- Record the exact preprocessing.
- Compare class-level behavior.
- Retain the selected checkpoint.
- Explain regressions as well as improvements.

