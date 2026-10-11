(function () {
  "use strict";

  const by = (root, selector) => root.querySelector(selector);

  function parseShape(value) {
    if (!value.trim()) return [];
    const shape = value.split(",").map((part) => Number(part.trim()));
    return shape.every((dimension) => Number.isInteger(dimension) && dimension > 0) ? shape : null;
  }

  function broadcastResult(a, b) {
    const width = Math.max(a.length, b.length);
    const left = Array(width - a.length).fill(1).concat(a);
    const right = Array(width - b.length).fill(1).concat(b);
    const result = [];
    for (let index = 0; index < width; index += 1) {
      if (left[index] !== right[index] && left[index] !== 1 && right[index] !== 1) return null;
      result.push(Math.max(left[index], right[index]));
    }
    return result;
  }

  function initBroadcastLab(root) {
    const first = by(root, "[data-shape-a]");
    const second = by(root, "[data-shape-b]");
    const output = by(root, "[data-broadcast-output]");
    const update = () => {
      const a = parseShape(first.value);
      const b = parseShape(second.value);
      if (!a || !b) {
        output.textContent = "Use positive integers separated by commas.";
        return;
      }
      const result = broadcastResult(a, b);
      output.textContent = result
        ? `[${a}] and [${b}] → [${result}]`
        : `[${a}] and [${b}] are not broadcast-compatible`;
    };
    by(root, "[data-check-broadcast]").addEventListener("click", update);
    first.addEventListener("input", update);
    second.addEventListener("input", update);
    update();
  }

  function initBatchLab(root) {
    const samples = by(root, "[data-samples]");
    const size = by(root, "[data-batch-size]");
    const drop = by(root, "[data-drop-last]");
    const output = by(root, "[data-batch-output]");
    const update = () => {
      const total = Number(samples.value);
      const batch = Number(size.value);
      if (!Number.isSafeInteger(total) || total < 1 || !Number.isSafeInteger(batch) || batch < 1) {
        output.textContent = "Enter positive whole numbers for samples and batch size.";
        return;
      }
      const full = Math.floor(total / batch);
      const remainder = total % batch;
      const batches = drop.checked ? full : Math.ceil(total / batch);
      const ending = remainder && !drop.checked ? `; final batch: ${remainder}` : "";
      const dropped = remainder && drop.checked ? `; dropped samples: ${remainder}` : "";
      output.textContent = `${batches} batches${ending}${dropped}`;
    };
    [samples, size, drop].forEach((control) => control.addEventListener("input", update));
    update();
  }

  function initShapeTracer(root) {
    const spatial = by(root, "[data-spatial-size]");
    const blocks = by(root, "[data-pool-blocks]");
    const output = by(root, "[data-shape-output]");
    const update = () => {
      let size = spatial.value.trim() ? Number(spatial.value) : NaN;
      const count = blocks.value.trim() ? Number(blocks.value) : NaN;
      if (!Number.isSafeInteger(size) || size < 1 || !Number.isInteger(count) || count < 0 || count > 6 || Math.floor(size / 2 ** count) < 1) {
        output.textContent = "Use a positive whole size large enough for 0–6 pooling blocks.";
        return;
      }
      const trace = [`${size}×${size}`];
      for (let index = 0; index < count; index += 1) {
        size = Math.floor(size / 2);
        trace.push(`${size}×${size}`);
      }
      output.textContent = trace.join(" → ") + (size < 1 ? " · too many pooling blocks" : "");
    };
    by(root, "[data-trace-shape]").addEventListener("click", update);
    spatial.addEventListener("input", update);
    blocks.addEventListener("input", update);
    update();
  }

  function initTensorLab(root) {
    const inputs = [...root.querySelectorAll("[data-dim]")];
    const output = by(root, "[data-tensor-output]");
    const update = () => {
      const values = Object.fromEntries(inputs.map((input) => [input.dataset.dim, Number(input.value)]));
      if (Object.values(values).some((value) => !Number.isSafeInteger(value) || value < 1)) {
        output.textContent = "Enter positive whole numbers for all four dimensions.";
        return;
      }
      const elements = values.batch * values.channels * values.height * values.width;
      if (!Number.isSafeInteger(elements)) {
        output.textContent = "The element count exceeds safe whole-number arithmetic. Use smaller dimensions.";
        return;
      }
      const megabytes = elements * 4 / (1024 * 1024);
      output.textContent = `[${values.batch}, ${values.channels}, ${values.height}, ${values.width}] · ${elements.toLocaleString()} float32 values · ${megabytes.toFixed(2)} MiB`;
    };
    inputs.forEach((input) => input.addEventListener("input", update));
    update();
  }


  function initAccumulationLab(root) {
    const micro = by(root, "[data-microbatch]");
    const steps = by(root, "[data-accumulation]");
    const samples = by(root, "[data-epoch-samples]");
    const output = by(root, "[data-accumulation-output]");
    const update = () => {
      const values = [micro, steps, samples].map((input) => Number(input.value));
      const [size, count, total] = values;
      if (values.some((v) => !Number.isSafeInteger(v) || v < 1) || !Number.isSafeInteger(size * count)) {
        output.textContent = "Enter positive whole numbers within the safe integer range.";
        return;
      }
      const effective = size * count;
      const updates = Math.ceil(total / effective);
      const last = total % effective || Math.min(effective, total);
      output.textContent = `${size} × ${count} = ${effective} samples per full update. ${updates} updates; final update: ${last} samples. BatchNorm still sees individual microbatches.`;
    };
    [micro, steps, samples].forEach((input) => input.addEventListener("input", update));
    update();
  }

  function initPaddingLab(root) {
    const lengths = by(root, "[data-sequence-lengths]");
    const maximum = by(root, "[data-fixed-length]");
    const output = by(root, "[data-padding-output]");
    const update = () => {
      const items = lengths.value.split(",").map((value) => Number(value.trim()));
      const fixed = Number(maximum.value);
      if (!items.length || items.length > 64 || items.some((v) => !Number.isInteger(v) || v < 1 || v > 4096) || !Number.isInteger(fixed) || fixed < 1 || fixed > 4096) {
        output.textContent = "Use up to 64 positive sequence lengths and a maximum from 1 to 4096.";
        return;
      }
      const tokens = items.reduce((sum, v) => sum + v, 0);
      const dynamic = Math.max(...items) * items.length;
      const kept = items.reduce((sum, v) => sum + Math.min(v, fixed), 0);
      const slots = items.length * fixed;
      output.textContent = `Dynamic: ${dynamic} positions, ${dynamic - tokens} padding. Fixed length ${fixed}: ${slots} positions, ${slots - kept} padding; ${tokens - kept} tokens truncated. A mask marks padding but does not eliminate all its computation.`;
    };
    [lengths, maximum].forEach((input) => input.addEventListener("input", update));
    update();
  }

  function initNoiseLab(root) {
    const clean = by(root, "[data-noise-clean]");
    const noisy = by(root, "[data-noise-changed]");
    const kind = by(root, "[data-noise-kind]");
    const amount = by(root, "[data-noise-amount]");
    const output = by(root, "[data-noise-output]");
    const cleanContext = clean.getContext("2d");
    const noisyContext = noisy.getContext("2d");
    if (!cleanContext || !noisyContext) {
      output.textContent = "Canvas unavailable. Impulse noise selects pixels to turn black or white; Gaussian noise adds fluctuating values.";
      return;
    }
    const original = cleanContext.createImageData(clean.width, clean.height);
    for (let y = 0; y < clean.height; y += 1) {
      for (let x = 0; x < clean.width; x += 1) {
        const index = (y * clean.width + x) * 4;
        let value = 45 + Math.round(45 * x / clean.width);
        if (x > 25 && x < 214 && y > 25 && y < 136) value = 135 + Math.round(60 * y / clean.height);
        if (x > 55 && x < 190 && y > 45 && y < 111 && x + y > 160) value = 220;
        if (x > 150 && x < 156 && y > 65 && y < 96) value = 65;
        if ((x - 82) ** 2 + (y - 75) ** 2 < 100) value = 95;
        original.data.set([value, value, value, 255], index);
      }
    }
    cleanContext.putImageData(original, 0, 0);
    const update = () => {
      let seed = 731;
      const random = () => {
        seed = (1664525 * seed + 1013904223) >>> 0;
        return (seed + 1) / 4294967297;
      };
      const level = Number(amount.value) / 100;
      const result = new ImageData(new Uint8ClampedArray(original.data), clean.width, clean.height);
      let selected = 0;
      for (let i = 0; i < result.data.length; i += 4) {
        if (kind.value === "impulse") {
          const draw = random();
          if (draw < level) {
            selected += 1;
            const value = draw < level / 2 ? 255 : 0;
            result.data[i] = value; result.data[i + 1] = value; result.data[i + 2] = value;
          }
        } else {
          for (let channel = 0; channel < 3; channel += 1) {
            const gaussian = Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
            result.data[i + channel] += 255 * level * gaussian;
          }
        }
      }
      noisyContext.putImageData(result, 0, 0);
      output.textContent = kind.value === "impulse"
        ? `Salt and pepper: ${(level * 100).toFixed(0)}% selection probability; ${selected} of ${clean.width * clean.height} spatial pixels selected in this seeded illustration. Half bright / half dark in expectation.`
        : `Gaussian comparison: σ = ${level.toFixed(2)} on a 0–1 pixel scale, then clipped. This is a simplified additive model, not a measured camera simulation.`;
    };
    kind.addEventListener("change", update);
    amount.addEventListener("input", update);
    update();
  }

  function initAll() {
    document.querySelectorAll("[data-broadcast-lab]").forEach(initBroadcastLab);
    document.querySelectorAll("[data-batch-lab]").forEach(initBatchLab);
    document.querySelectorAll("[data-shape-tracer]").forEach(initShapeTracer);
    document.querySelectorAll("[data-tensor-lab]").forEach(initTensorLab);

    document.querySelectorAll("[data-accumulation-lab]").forEach(initAccumulationLab);
    document.querySelectorAll("[data-padding-lab]").forEach(initPaddingLab);
    document.querySelectorAll("[data-noise-lab]").forEach(initNoiseLab);

  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initAll);
  else initAll();
})();
