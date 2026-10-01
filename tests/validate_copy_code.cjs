/* Exercise actual shared handler, including denial, absent API and retry races. */
const assert = require('node:assert/strict');
const {copyCode} = require('../docs/assets/javascripts/copy-code.js');
const originalTimeout = global.setTimeout, originalClear = global.clearTimeout;
const timers = new Map(); let nextTimer = 0;
global.setTimeout = callback => { timers.set(++nextTimer, callback); return nextTimer; };
global.clearTimeout = id => timers.delete(id);
function restore() { const pending = [...timers.values()]; timers.clear(); pending.forEach(fn => fn()); }

function fixture(clipboard, plain = true) {
  let selected = null, written = null;
  const nodes = [], attrs = new Map([['title', 'Original title'], ['aria-label', 'Original label']]);
  const doc = {createElement: () => ({setAttribute() {}, textContent: ''}),
    createRange: () => ({selectNodeContents(node) { selected = node; }}),
    defaultView: {navigator: {clipboard}, getSelection: () => ({removeAllRanges() {}, addRange() {}})}};
  const code = {ownerDocument: doc, textContent: '  x = 1\n# exact spaces\n'};
  const button = {parentElement: {append(node) { nodes.push(node); }}, textContent: plain ? 'Copy code' : '',
    classList: {contains: () => plain}, getAttribute: key => attrs.get(key) ?? null,
    setAttribute: (key, value) => attrs.set(key, value), removeAttribute: key => attrs.delete(key)};
  Object.defineProperty(button, 'title', {get: () => attrs.get('title'), set: value => attrs.set('title', value)});
  return {button, code, attrs, nodes, selected: () => selected};
}

(async () => {
  let text;
  let f = fixture({writeText: async value => { text = value; }});
  await copyCode(f.button, f.code);
  assert.equal(text, f.code.textContent);
  assert.equal(f.button.textContent, 'Copied');
  assert.equal(f.nodes[0].textContent, 'Copied to clipboard.');
  restore(); assert.equal(f.button.textContent, 'Copy code');
  assert.equal(f.attrs.get('aria-label'), 'Original label');
  assert.equal(f.attrs.get('title'), 'Original title');

  for (const clipboard of [undefined, {writeText: async () => { throw new Error('Permission denied'); }}]) {
    f = fixture(clipboard, false);
    await copyCode(f.button, f.code);
    assert.equal(f.selected(), f.code);
    assert.match(f.nodes[0].textContent, /Code selected/);
    assert.match(f.attrs.get('aria-label'), /Ctrl\+C or Command\+C/);
    restore(); assert.equal(f.attrs.get('aria-label'), 'Original label');
    assert.equal(f.attrs.get('title'), 'Original title');
  }
  f = fixture(undefined);
  f.attrs.delete('aria-label'); f.attrs.delete('title');
  await copyCode(f.button, f.code); restore();
  assert.equal(f.attrs.has('aria-label'), false); assert.equal(f.attrs.has('title'), false);

  let rejectFirst;
  let calls = 0;
  f = fixture({writeText: () => ++calls === 1 ? new Promise((_, reject) => { rejectFirst = reject; }) : Promise.resolve()});
  const first = copyCode(f.button, f.code);
  await copyCode(f.button, f.code);
  rejectFirst(new Error('Old request rejected')); await first;
  assert.equal(f.selected(), null); assert.equal(f.button.textContent, 'Copied');
  assert.equal(f.nodes.length, 1);
  restore(); assert.equal(f.attrs.get('aria-label'), 'Original label');
  console.log('Copy handler passed: exact text, denial, absent API, selection, label restoration and retry race');
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => {
  global.setTimeout = originalTimeout; global.clearTimeout = originalClear;
});
