const assert = require('node:assert/strict');
const {weightBudget, attentionMask, payloadLabel} = require('../docs/assets/javascripts/course3-visuals.js');
assert.equal(payloadLabel(1), '1 byte');
assert.equal(payloadLabel(4), '4 bytes');
assert.equal(payloadLabel(232), '232 bytes');
assert.equal(payloadLabel(1024), '1,024 bytes (1 KiB)');
assert.equal(payloadLabel(1e7), '10 MB (9.537 MiB)');
assert.deepEqual(weightBudget(1e7, 8), {bytes:1e7, fp32:4e7});
assert.equal(weightBudget(1e7, 16).bytes, 2e7);
for (const count of [0, -1, NaN, 1.5, 1e12 + 1]) assert.equal(weightBudget(count, 32), null);
assert.equal(weightBudget(10, 4), null);
for (let n = 1; n <= 8; n++) {
  const mask = attentionMask(n, 'causal');
  assert.equal(mask.flat().filter(Boolean).length, n*(n+1)/2);
  mask.forEach((row, q) => { assert.equal(row[q], true); row.slice(q+1).forEach(v => assert.equal(v, false)); });
  assert.equal(attentionMask(n, 'encoder').flat().filter(Boolean).length, n*n);
}
for (const n of [0, 9, 1.5, NaN]) assert.equal(attentionMask(n, 'causal'), null);
assert.equal(attentionMask(4, 'unknown'), null);
console.log('Course 3 tools passed: payload units/precision, invalid input and causal visibility');
