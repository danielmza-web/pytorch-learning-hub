---
title: Advanced vision — architectures, similarity and generation
tags: [architectures, siamese, resnet, densenet, interpretability, diffusion]
last_reviewed: 2026-10-10
---

# Advanced vision — architectures, similarity and generation

A classifier chooses a label. An embedding model compares inputs. A generative model creates a new input. Understanding these different outputs makes architecture, loss and evaluation choices easier to connect.
{ .page-lead }

Read [pretrained models](../vision/pretrained-models.md) first for normalization, freezing and task heads. This page develops the Course 3 vision themes through original explanations and examples; [transformers](transformers.md) continues the architecture story for sequences.

**Code key:** “Runnable toy” needs only installed PyTorch and uses synthetic inputs. An “excerpt” needs the objects named in its section. The diagrams illustrate mechanisms; they are not training results or copied course figures.

## Build the computation you actually need

`nn.Sequential` fits a single chain. A custom `nn.Module.forward()` can accept several inputs, return several outputs and select a branch. For example, combine an image embedding `[N,32]` and numeric sensor features `[N,4]` with `torch.cat(..., dim=1)` to create `[N,36]`. Separate heads can predict a defect class and a severity value. Both encoders can learn jointly if gradients flow from the combined loss: separate modules do not imply independent training.

For two tasks, an objective such as `classification_loss + weight * regression_loss` balances different units and scales. Validate both tasks; a lower combined number can hide a worse individual task. Fit feature normalization on training data and reject metadata that reveals the target or is unavailable at inference.

Modules assigned as attributes are registered. Variable-depth components belong in `nn.ModuleList`; named alternatives belong in `nn.ModuleDict`. A plain Python list still executes its contained layers, but those layers are absent from the parent's parameter iterator, device moves and `state_dict`. Consequently the usual parent optimizer will miss them. See [module registration](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html).

**Runnable toy — depth is configuration, weights are shared by reuse:**

```python
import torch
from torch import nn

class PanelEncoder(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.input_layer = nn.Linear(6, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(depth)])
        self.output_layer = nn.Linear(8, 3)

    def forward(self, x):
        x = self.input_layer(x).relu()
        for block in self.blocks:
            x = (x + block(x).relu())  # same shape on the residual paths
        return self.output_layer(x)

encoder = PanelEncoder(depth=2)
left, right = torch.randn(4, 6), torch.randn(4, 6)
z_left, z_right = encoder(left), encoder(right)
assert z_left.shape == z_right.shape == (4, 3)
assert "blocks.1.weight" in encoder.state_dict()
print(sum(p.numel() for p in encoder.parameters()))  # 227
```

Calling `encoder` twice reuses one parameter set; constructing two encoders makes two sets. Gradients from both calls accumulate into the shared parameters before one optimizer update. With Dropout or BatchNorm, shared weights still have training-time randomness or changing running statistics; use the correct evaluation mode when comparing stored embeddings.

A Python condition in `forward()` builds the eager graph for the path actually executed. The unused branch usually has no gradient on that step. Create and register reusable trainable layers in `__init__`, then choose among them in `forward`; recreating layers inside every forward introduces fresh weights and optimizer problems. Conditional routing also affects batching and export compatibility. The [model lifecycle guide](../deployment/model-lifecycle.md) connects this live object to saved tensors and deployment.

## Learn similarity instead of a fixed class list

Suppose an inspection system retrieves panels with similar surface defects. A classifier returns class logits `[N,C]`; an encoder returns feature vectors `[N,D]`. The latter lets a query be compared with a gallery without changing the output head whenever a new gallery item arrives. Useful similarity for unseen items must still be demonstrated on held-out identities or acquisition groups.

<div class="recall-flow" role="group" aria-label="Shared encoder and similarity learning">
<div><b>Anchor</b><code>panel A, view 1</code><small>reference image</small></div>
<div><b>Positive</b><code>panel A, view 2</code><small>different capture, same intended identity</small></div>
<div><b>Negative</b><code>panel B</code><small>different identity under the task definition</small></div>
<div><b>One encoder</b><code>f(A), f(P), f(N)</code><small>all three use the same parameters</small></div>
<div><b>Compare</b><code>d(A,P) + margin ≤ d(A,N)</code><small>desired relative ordering</small></div>
</div>
<p class="visual-caption">Conceptual triplet. Identity, similarity and negative sampling must be defined for the actual application.</p>

### Contrastive pairs and triplet margins

For a **pair**, define `same=1` for similar and `same=0` for different. One contrastive objective is `same * d² + (1-same) * max(margin-d, 0)²`. Similar pairs are pulled together; different pairs already beyond the margin contribute zero. Other implementations reverse the label convention, so inspect the equation before reusing labels. Class weights address imbalance only when calculated from the training partition and applied to the intended terms.

For a **triplet**, PyTorch's `nn.TripletMarginLoss(margin=m, p=2)` uses `max(d(A,P) - d(A,N) + m, 0)` with a Euclidean norm, including its small numerical epsilon. It does **not** square that distance. A squared-distance variant is a different objective, available through a suitable custom distance in `TripletMarginWithDistanceLoss`. [PyTorch loss definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html).

**Runnable toy — a bad negative earns a penalty:**

```python
import torch
from torch import nn

anchor = torch.tensor([[0.0, 0.0]])
positive = torch.tensor([[0.2, 0.0]])
far_negative = torch.tensor([[1.2, 0.0]])
close_negative = torch.tensor([[0.3, 0.0]])
objective = nn.TripletMarginLoss(margin=0.5, p=2)
good = objective(anchor, positive, far_negative)
bad = objective(anchor, positive, close_negative)
assert good.item() == 0.0 and bad.item() > 0.0
print(round(bad.item(), 3))  # approximately 0.4
```

The training **margin** shapes relative distances. A deployment **threshold** decides whether a pair is accepted. They are different quantities; there is no universal rule that one must exceed the other. Choose the threshold on validation pairs using the relevant false accept/false reject trade-off, then hold it fixed on test data. Report both errors and class prevalence, rather than only accuracy.

### Make retrieval and change detection reliable

Normalize embeddings with `F.normalize(z, dim=-1)` if cosine similarity or a fixed norm is part of your design. For unit vectors, squared Euclidean distance equals `2 - 2*cosine`; it orders candidates the same way, but its threshold has different units. Store gallery embeddings with the encoder version and preprocessing contract. Updating the encoder requires rebuilding the gallery.

Split underlying images or identities **before** constructing pairs and triplets. Splitting virtual indices of a randomly sampling dataset does not prove disjoint source images. Fix validation pairs for reproducible comparisons, sample distinct anchor/positive captures, and inspect mislabeled or unreasonably hard negatives. For satellite or time-separated pairs, split by location/group to prevent nearly identical scenes crossing partitions. Apply synchronized geometry when alignment matters.

A symmetric distance says **how different** two embeddings are; swapping before/after leaves it unchanged. It cannot by itself distinguish “vegetation increased” from “vegetation decreased.” A directional change head needs ordered features such as `z_after - z_before`, suitable labels and validation. Color-based proxy labels also need checking against lighting, season and acquisition effects. Retrieval uses ranking measures such as recall at a chosen rank; verification uses calibrated acceptance errors. These are distinct evaluation contracts.

## Residual, dense and efficient backbones

The encoder can be custom or pretrained. Its connection pattern changes feature reuse, gradient paths and resource demands.

<div class="recall-flow" role="group" aria-label="Three architectural strategies">
<div><b>Residual block</b><code>x + F(x)</code><small>add a correction to an aligned shortcut</small></div>
<div><b>Dense block</b><code>cat(x, new features)</code><small>keep old channels and append new ones</small></div>
<div><b>Mobile convolution</b><code>depthwise → pointwise</code><small>separate spatial filtering and channel mixing</small></div>
</div>
<p class="visual-caption">Connection-pattern illustration. Smaller parameter counts or operation estimates do not establish faster execution on a particular CPU.</p>

### Residual addition needs matching shapes

In ResNet, a block learns a correction `F(x)` and adds a shortcut. The direct path makes optimization of deep networks easier; it does not guarantee perfect gradients or eliminate overfitting. If the main path changes `[N,16,32,32]` into `[N,32,16,16]`, the shortcut needs a projection, commonly a `1×1` convolution with stride 2, before addition. Broadcasting is not an acceptable substitute for an intentional shortcut. [Residual learning paper](https://arxiv.org/abs/1512.03385).

### Dense concatenation grows channels

In a DenseNet block each layer receives earlier features and contributes `k` new channels, where `k` is the **growth rate**. Starting with 12 channels and adding four layers of growth 6 produces `12 + 4×6 = 36` channels at unchanged height/width. A `1×1` bottleneck limits the inputs to a costlier `3×3` convolution; transition layers compress channels and downsample between blocks. Unlike residual addition, concatenation preserves channels separately. It can reduce parameters through reuse while increasing activation storage and memory traffic. [DenseNet paper](https://arxiv.org/abs/1608.06993).

### Efficient models start with efficient operators

A dense `3×3` convolution with 16 inputs and 32 outputs has `9×16×32 = 4,608` weights without bias. Depthwise filtering followed by `1×1` mixing has `9×16 + 16×32 = 656`. In PyTorch, `nn.Conv2d(16,16,3,groups=16)` supplies one spatial filter per input channel; a following `nn.Conv2d(16,32,1)` mixes channels. MobileNet builds on this idea; later variants also use inverted residuals and other refinements. [MobileNets paper](https://arxiv.org/abs/1704.04861).

Depth, width and image resolution are resource choices. EfficientNet balances them through compound scaling. [EfficientNet paper](https://arxiv.org/abs/1905.11946). TorchVision provides ResNet, DenseNet, MobileNet and EfficientNet families with associated pretrained weights and preprocessing; choose the matching weight transform rather than guessing normalization. [TorchVision model inventory](https://docs.pytorch.org/vision/stable/models.html).

Compare architectures on the same held-out task and actual target device: preprocessing, batch size, input size, peak memory, latency distribution and accuracy matter. Quantization or pruning is a later transformation of a model, discussed in [optimization and serving](../deployment/optimization-serving.md); it is not implied by choosing a small backbone.

## Look inside: features, receptive fields and hooks

Convolutional feature maps have shape `[N,C,H,W]`. A filter produces one output channel; that channel is an activation map, not a class probability. Early layers often capture local patterns and later layers combine broader context, but individual channels are not guaranteed to correspond to named objects.

For ordinary convolution/pooling layers, track the theoretical receptive-field size `r` and the input-pixel spacing `j` between output centers. Start with `r=1, j=1`; for kernel `k`, dilation `d` and stride `s`, update `r ← r + (k-1)*d*j`, then `j ← j*s`. A `3×3` stride-1 convolution gives `r=3,j=1`; a `2×2` stride-2 pool gives `r=4,j=2`; another `3×3` convolution gives `r=8,j=2`. Padding affects boundary alignment; actual influence inside this theoretical region depends on weights, input and nonlinearities. Global pooling can then aggregate across the spatial map.

A forward hook observes a registered module's output without rewriting `forward()`. Save detached copies for visualization, bound how much you retain, and remove the hook afterward. For reusable feature extraction, TorchVision's graph-based extractor is another option; node names depend on the architecture. [TorchVision feature extraction](https://docs.pytorch.org/vision/stable/feature_extraction.html).

**Excerpt — one capture from an existing `model` and input `images`:**

```python
captured = {}
def save_map(module, args, output):
    captured["stage"] = output.detach().cpu()

handle = model.features.register_forward_hook(save_map)  # choose a real layer
try:
    model.eval()
    with torch.inference_mode():
        predictions = model(images)
finally:
    handle.remove()
```

`model.features` exists on some architectures, not every model. Inspect `named_modules()` to select a layer. Detaching avoids retaining its backward graph; it also prevents that captured copy from participating in later gradient calculations.

## Saliency, CAM and Grad-CAM answer different questions

| Method | Inputs to the explanation | What the map describes |
| --- | --- | --- |
| Input saliency | gradient of one class score with respect to input pixels | local sensitivity of that score |
| Original CAM | convolutional maps plus linear classifier weights after global average pooling | weighted spatial evidence under this architecture |
| Grad-CAM | convolutional maps plus gradients of the chosen score with respect to those maps | coarse class-specific localization without CAM's exact head restriction |

Original CAM combines maps with the selected class's classifier coefficients. Grad-CAM computes channel coefficients by spatially averaging feature gradients, forms their weighted activation sum and commonly applies ReLU. Interpolating a small map to image size makes an overlay easier to see; it does not add spatial evidence. [CAM paper](https://arxiv.org/abs/1512.04150), [Grad-CAM paper](https://arxiv.org/abs/1610.02391).

**Runnable toy — capture features and differentiate the actual score:**

```python
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
features = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
head = nn.Linear(4, 2)
cache = {}
def keep_features(module, args, output):
    cache["activation"] = output  # retain graph for this one explanation

handle = features.register_forward_hook(keep_features)
try:
    image = torch.rand(1, 1, 8, 8, requires_grad=True)
    activation = features(image)
    logits = head(activation.mean(dim=(2, 3)))
    pixel_grad, feature_grad = torch.autograd.grad(
        logits[0, 1], (image, cache["activation"])
    )
    saliency = pixel_grad.abs().amax(dim=1)
    channel_weight = feature_grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((channel_weight * activation.detach()).sum(dim=1))
    assert saliency.shape == cam.shape == (1, 8, 8)
    assert torch.isfinite(saliency).all() and torch.isfinite(cam).all()
finally:
    handle.remove()
```

This random toy demonstrates gradient contracts, not a useful explanation of a trained vision system. For a real model, use its normal input preprocessing and `model.eval()`, then enable gradients for the explanation. Evaluation mode disables training behavior; it does not disable autograd. `no_grad()` and `inference_mode()` would prevent the needed gradient computation. `autograd.grad` obtains these derivatives without accumulating ordinary parameter `.grad` buffers. Normalize displays with a guard for a constant map and record the chosen class score.

Saliency is a local derivative, not a causal proof. A convincing heatmap can still reflect spurious backgrounds or be unstable. Compare classes and counterexamples, test controlled input changes, and evaluate performance by acquisition condition or relevant groups. Never interpret a heatmap as an accuracy score or a complete fairness check.

## Diffusion connects noise prediction to image generation

An embedding compresses an image for comparison. A diffusion model instead learns a denoising task that supports sampling new images. During training, a clean image (or latent) is mixed with Gaussian noise at a sampled timestep:

`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon`.

Here `alpha_bar_t` comes from the noise schedule and `epsilon` is sampled noise. A common objective trains a network to predict `epsilon` from `x_t`, the timestep and optional conditioning. Other checkpoints predict velocity or another parameterization, so match the scheduler's configuration. The learned reverse process is probabilistic; it does not recover a uniquely determined original image from arbitrary noise. [DDPM paper](https://arxiv.org/abs/2006.11239).

<div class="recall-flow" role="group" aria-label="Training and sampling of latent diffusion">
<div><b>Training example</b><code>clean latent + known noise</code><small>choose a timestep and target</small></div>
<div><b>Denoiser learns</b><code>noisy latent, t, condition</code><small>predict the configured target</small></div>
<div><b>Sampling begins</b><code>random latent</code><small>there is no original image to recover</small></div>
<div><b>Scheduler repeats</b><code>prediction → next latent</code><small>controlled reverse updates</small></div>
<div><b>Decode</b><code>latent → visible image</code><small>VAE decoder in latent pipelines</small></div>
</div>
<p class="visual-caption">Conceptual latent-diffusion pipeline. It shows separate training and sampling processes, without claiming generated outputs or quality measurements.</p>

Stable Diffusion commonly combines a text encoder, denoiser, scheduler and variational autoencoder. Operating in a smaller latent grid reduces denoising costs; model weights and intermediate activations still require substantial memory. `DiffusionPipeline.from_pretrained(...)` selects a pipeline from its checkpoint configuration. The correct pipeline class, dtype, access and supported arguments depend on the model, so an older `StableDiffusionPipeline` example should not be assumed to work for every newer checkpoint. [Diffusers text-to-image documentation](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation).

`num_inference_steps` controls reverse updates and computation; more steps do not guarantee better quality. `guidance_scale` changes the conditioning trade-off in pipelines that support classifier-free guidance. Negative prompts influence supported pipelines but cannot guarantee removal of an object or artifact. A seeded `torch.Generator` supports controlled comparisons; consuming its state advances it, and seeds do not ensure identical images across hardware/software versions. Recreate generators for matched comparisons and record checkpoint, scheduler, prompts and parameters. [Diffusers reproducibility](https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds).

Validate prompt adherence, artifacts and representative failure cases over multiple inputs/seeds, including any sensitive deployment context. Check model/data licensing and privacy before collecting or releasing outputs. Synthetic augmentation is an experiment: label correctness and improvement on untouched real test data must be established. Keep generated-data provenance and prevent synthetic relatives of test images leaking into training.

## Remember and apply

**Check yourself:** if two encoders have the same class definition, are their weights shared? Only if you reuse the same module object or deliberately tie parameters. Can a distance tell whether a change was an increase? A symmetric distance gives magnitude, so directional prediction needs more information. Is a sharper heatmap necessarily more faithful? No; display resolution and explanation validity are different.

Continue with [attention and transformer architectures](transformers.md), then follow [model lifecycle](../deployment/model-lifecycle.md) to preserve the learned state and [optimization and serving](../deployment/optimization-serving.md) to test resource constraints.
