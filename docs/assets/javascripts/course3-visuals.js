/* Original, deterministic teaching tools. No model runs in this page. */
(function () {
  'use strict';
  function weightBudget(parameters, bits) {
    if (!Number.isSafeInteger(parameters) || parameters < 1 || parameters > 1e12 || ![8, 16, 32].includes(bits)) return null;
    return { bytes: parameters * bits / 8, fp32: parameters * 4 };
  }
  function attentionMask(length, kind) {
    if (!Number.isInteger(length) || length < 1 || length > 8 || !['encoder', 'causal'].includes(kind)) return null;
    return Array.from({length}, (_, query) => Array.from({length}, (_, key) => kind === 'encoder' || key <= query));
  }
  function payloadLabel(bytes) {
    if (bytes < 1024) return `${bytes.toLocaleString('en')} ${bytes === 1 ? 'byte' : 'bytes'}`;
    if (bytes < 1e6) return `${bytes.toLocaleString('en')} bytes (${(bytes / 1024).toLocaleString('en', {maximumFractionDigits: 3})} KiB)`;
    return `${(bytes / 1e6).toLocaleString('en', {maximumFractionDigits: 3})} MB (${(bytes / 1048576).toLocaleString('en', {maximumFractionDigits: 3})} MiB)`;
  }
  if (typeof module !== 'undefined') module.exports = {weightBudget, attentionMask, payloadLabel};
  if (typeof document === 'undefined') return;
  const units = payloadLabel;
  function initialize() {
    document.querySelectorAll('[data-weight-budget]').forEach(panel => {
      if (panel.dataset.ready) return;
      panel.dataset.ready = 'true';
      const count = panel.querySelector('[data-parameters]');
      const bits = panel.querySelector('[data-weight-bits]');
      const output = panel.querySelector('[data-budget-output]');
      const bars = panel.querySelector('[data-budget-bars]');
      function update() {
        const value = weightBudget(count.value.trim() ? Number(count.value) : NaN, Number(bits.value));
        bars.replaceChildren();
        if (!value) { output.textContent = 'Enter a whole parameter count from 1 to 1,000,000,000,000.'; return; }
        for (const [label, bytes] of [['FP32 baseline', value.fp32], [`${bits.value}-bit payload`, value.bytes]]) {
          const row = document.createElement('div');
          const name = document.createElement('span');
          name.textContent = `${label}: ${units(bytes)}`;
          const bar = document.createElement('span');
          bar.className = 'budget-bar';
          bar.style.width = `${100 * bytes / value.fp32}%`;
          bar.setAttribute('aria-hidden', 'true');
          row.append(name, bar); bars.append(row);
        }
        output.textContent = `${units(value.bytes)} of weight payload; ${100 * value.bytes / value.fp32}% of the FP32 payload. Actual model and runtime memory need separate measurement.`;
      }
      count.addEventListener('input', update); bits.addEventListener('change', update); update();
    });
    document.querySelectorAll('[data-attention-mask]').forEach(panel => {
      if (panel.dataset.ready) return;
      panel.dataset.ready = 'true';
      const count = panel.querySelector('[data-token-count]');
      const kind = panel.querySelector('[data-attention-kind]');
      const grid = panel.querySelector('[data-mask-grid]');
      const output = panel.querySelector('[data-mask-output]');
      function update() {
        const mask = attentionMask(count.value.trim() ? Number(count.value) : NaN, kind.value);
        grid.replaceChildren();
        if (!mask) { output.textContent = 'Enter a whole sequence length from 1 to 8.'; return; }
        const table = document.createElement('table');
        const caption = document.createElement('caption');
        caption.textContent = 'Rows: query position. Columns: key position. ✓ allowed; × blocked.';
        table.append(caption);
        const head = document.createElement('thead'); const heading = document.createElement('tr');
        for (const label of ['Q / K', ...mask.map((_, i) => String(i))]) {
          const cell = document.createElement('th'); cell.textContent = label; cell.scope = 'col'; heading.append(cell);
        }
        head.append(heading); table.append(head);
        const body = document.createElement('tbody');
        mask.forEach((row, q) => {
          const tr = document.createElement('tr'); const label = document.createElement('th');
          label.textContent = String(q); label.scope = 'row'; tr.append(label);
          row.forEach(allowed => {
            const cell = document.createElement('td'); cell.textContent = allowed ? '✓' : '×';
            cell.className = allowed ? 'mask-allowed' : 'mask-blocked';
            cell.setAttribute('aria-label', allowed ? 'allowed' : 'blocked'); tr.append(cell);
          }); body.append(tr);
        });
        table.append(body); grid.append(table);
        output.textContent = kind.value === 'causal' ? 'Causal attention includes the current position and earlier positions. Future keys are blocked.' : 'Encoder attention can read every non-padding position. This illustration has no padding.';
      }
      count.addEventListener('input', update); kind.addEventListener('change', update); update();
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize); else initialize();
})();
