(function () {
  "use strict";
  function binaryMetrics(tp, fp, fn, tn) {
    const counts = [tp, fp, fn, tn];
    if (counts.some(v => !Number.isSafeInteger(v) || v < 0) || !Number.isSafeInteger(counts.reduce((a, b) => a + b, 0))) return null;
    const ratio = (n, d) => d ? n / d : null;
    return {accuracy: ratio(tp + tn, tp + fp + fn + tn), precision: ratio(tp, tp + fp), recall: ratio(tp, tp + fn), f1: ratio(2 * tp, 2 * tp + fp + fn)};
  }
  function paddingRows(lengths, maximum) {
    if (!lengths.length || lengths.length > 64 || lengths.some(v => !Number.isInteger(v) || v < 1 || v > 4096) || !Number.isInteger(maximum) || maximum < 1 || maximum > 4096) return null;
    return lengths.slice(0, 8).map(length => ({kept: Math.min(length, maximum), padding: Math.max(0, maximum - length), truncated: Math.max(0, length - maximum)}));
  }
  if (typeof module !== "undefined") module.exports = {binaryMetrics, paddingRows};
  if (typeof document === "undefined") return;

  function initMetrics(root) {
    const inputs = ["tp", "fp", "fn", "tn"].map(name => root.querySelector(`[data-${name}]`));
    const output = root.querySelector("[data-metric-output]");
    const grid = root.querySelector("[data-metric-matrix]");
    const update = () => {
      const counts = inputs.map(input => input.value.trim() === "" ? NaN : Number(input.value));
      const result = binaryMetrics(...counts);
      if (!result) { grid.replaceChildren(); output.textContent = "Enter non-negative whole counts within the safe integer range."; return; }
      const [tp, fp, fn, tn] = counts;
      grid.innerHTML = `<table><caption>Confusion matrix · positive = defect</caption><thead><tr><th>Actual / predicted</th><th>Defect</th><th>Good</th></tr></thead><tbody><tr><th>Defect</th><td>TP ${tp}</td><td>FN ${fn}</td></tr><tr><th>Good</th><td>FP ${fp}</td><td>TN ${tn}</td></tr></tbody></table>`;
      output.textContent = Object.entries(result).map(([name, value]) => `${name}: ${value === null ? "N/A (zero denominator)" : (value * 100).toFixed(1) + "%"}`).join(" · ");
    };
    inputs.forEach(input => input.addEventListener("input", update));
    root.querySelector("[data-metric-preset]").addEventListener("click", () => { [0, 0, 1, 99].forEach((value, index) => { inputs[index].value = value; }); update(); });
    update();
  }
  function initPadding(root) {
    const lengths = root.querySelector("[data-sequence-lengths]");
    const maximum = root.querySelector("[data-fixed-length]");
    const visual = document.createElement("div");
    visual.className = "token-rows";
    visual.setAttribute("aria-label", "Fixed-length token and mask rows");
    root.append(visual);
    const update = () => {
      const items = lengths.value.split(",").map(v => Number(v.trim()));
      const fixed = Number(maximum.value);
      const rows = paddingRows(items, fixed);
      visual.replaceChildren();
      if (!rows) return;
      const legend = document.createElement("p");
      legend.textContent = "Toy IDs 1, 2, …: token / mask 1. PAD 0: mask 0. Removed tokens cannot be recovered by masking.";
      visual.append(legend);
      rows.forEach((row, index) => {
        const line = document.createElement("div");
        const label = document.createElement("b");
        label.textContent = `Sequence ${index + 1} (${items[index]} tokens)`;
        line.append(label);
        const cells = document.createElement("div");
        cells.className = "token-cells";
        for (let i = 0; i < Math.min(fixed, 32); i += 1) {
          const cell = document.createElement("span");
          const pad = i >= row.kept;
          cell.className = pad ? "is-pad" : "is-token";
          cell.textContent = pad ? "0 / 0" : `${i + 1} / 1`;
          cells.append(cell);
        }
        const summary = document.createElement("small");
        summary.textContent = `${row.kept} kept · ${row.padding} PAD · ${row.truncated} truncated${fixed > 32 ? " · first 32 positions shown" : ""}`;
        if (row.truncated) {
          const removed = Array.from({length: Math.min(row.truncated, 12)}, (_, i) => fixed + i + 1);
          summary.textContent += ` · removed toy IDs: ${removed.join(", ")}${row.truncated > 12 ? ", …" : ""}`;
        }
        line.append(cells, summary);
        visual.append(line);
      });
      if (items.length > 8) { const note = document.createElement("p"); note.textContent = "First 8 sequences shown; totals above include all sequences."; visual.append(note); }
    };
    [lengths, maximum].forEach(input => input.addEventListener("input", update));
    update();
  }
  function revealDestination() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (_) { return; }
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    let parent = target.parentElement, opened = false;
    while (parent) { if (parent.tagName === "DETAILS" && !parent.open) { parent.open = true; opened = true; } parent = parent.parentElement; }
    // Heading anchors remain outside disclosures. Reveal their relevant explanation too.
    if (/^H[23]$/.test(target.tagName)) {
      let sibling = target.nextElementSibling;
      const terms = (new URLSearchParams(location.search).get("h") || "").toLowerCase().split(/\s+/).filter(Boolean);
      while (sibling && !/^H[123]$/.test(sibling.tagName)) {
        if (sibling.tagName === "DETAILS" && (!terms.length || terms.some(term => sibling.textContent.toLowerCase().includes(term)))) sibling.open = true;
        sibling = sibling.nextElementSibling;
      }
    }
    if (opened) requestAnimationFrame(() => target.scrollIntoView({block: "start", behavior: "instant"}));
  }
  function init() {
    document.querySelectorAll("[data-metrics-lab]").forEach(initMetrics);
    document.querySelectorAll("[data-padding-lab]").forEach(initPadding);
    revealDestination();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
  window.addEventListener("hashchange", revealDestination);
  window.addEventListener("load", revealDestination);
  document.addEventListener("click", event => {
    const anchor = event.target.closest("a[href]");
    if (!anchor) return;
    const destination = new URL(anchor.href, location.href);
    if (destination.origin === location.origin && destination.pathname === location.pathname && destination.hash) requestAnimationFrame(revealDestination);
  });
})();
