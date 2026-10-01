(function () {
  'use strict';
  const states = new WeakMap();

  async function copyCode(button, code) {
    const doc = code.ownerDocument;
    let state = states.get(button);
    if (!state) {
      const status = doc.createElement('span');
      status.className = 'copy-feedback';
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      status.setAttribute('aria-atomic', 'true');
      button.parentElement.append(status);
      state = {status, title: button.getAttribute('title'), label: button.getAttribute('aria-label'),
        text: button.textContent, plain: button.classList.contains('copy-code'), sequence: 0};
      states.set(button, state);
    }
    const sequence = ++state.sequence;
    clearTimeout(state.timer);
    state.status.textContent = '';
    let message, short;
    try {
      if (!doc.defaultView.navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await doc.defaultView.navigator.clipboard.writeText(code.textContent);
      message = 'Copied to clipboard.';
      short = 'Copied';
    } catch (_) {
      if (sequence !== state.sequence) return;
      const selection = doc.defaultView.getSelection();
      if (selection) {
        const range = doc.createRange();
        range.selectNodeContents(code);
        selection.removeAllRanges();
        selection.addRange(range);
        message = 'Copy unavailable. Code selected; press Ctrl+C or Command+C to copy manually.';
      } else {
        message = 'Copy unavailable. Select the code and copy manually.';
      }
      short = 'Copy manually';
    }
    if (sequence !== state.sequence) return;
    state.status.textContent = message;
    button.title = message;
    button.setAttribute('aria-label', message);
    if (state.plain) button.textContent = short;
    state.timer = setTimeout(() => {
      for (const [name, value] of [['title', state.title], ['aria-label', state.label]]) {
        if (value === null) button.removeAttribute(name);
        else button.setAttribute(name, value);
      }
      if (state.plain) button.textContent = state.text;
    }, 1800);
  }

  if (typeof module !== 'undefined') module.exports = {copyCode};
  if (typeof document === 'undefined') return;
  // Capture prevents the theme's clipboard handler from reporting a false success.
  document.addEventListener('click', event => {
    const button = event.target.closest('button.copy-code, button[data-clipboard-target]');
    if (!button) return;
    const selector = button.getAttribute('data-clipboard-target');
    const code = selector ? document.querySelector(selector) : button.closest('.code-panel')?.querySelector('code');
    if (!code) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    copyCode(button, code);
  }, true);
})();
