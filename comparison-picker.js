'use strict';
window.artigentMountCasePicker = (container, selected, prefix, onSelect) => {
  container.setAttribute('role', 'group');
  container.setAttribute('aria-label', 'Choose an interactive object');
  for (const item of window.artigentComparisonCases) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.comparisonCase = item.id;
    button.setAttribute('aria-pressed', String(item.id === selected));
    const img = document.createElement('img');
    img.src = prefix + item.thumbnail;
    img.alt = ''; img.width = 160; img.height = 100;
    img.loading = 'lazy';
    const label = document.createElement('span');
    label.textContent = item.label;
    button.append(img, label);
    button.addEventListener('click', () => {
      if (button.getAttribute('aria-pressed') === 'true') return;
      container.querySelectorAll('button').forEach(node => node.setAttribute('aria-pressed', String(node === button)));
      onSelect(item);
    });
    container.append(button);
  }
};
const comparisonPicker = document.querySelector('#comparison-case-picker');
if (comparisonPicker) {
  const frame = document.querySelector('#method-comparison');
  const fullScreen = document.querySelector('#comparison-fullscreen');
  const current = new URLSearchParams(location.search).get('case');
  const selected = window.artigentComparisonCases.find(item => item.id === current) || window.artigentComparisonCases[0];
  const selectCase = item => {
    const url = new URL(frame.getAttribute('src'), location.href);
    url.searchParams.set('case', item.id);
    // Replacing the frame disposes the previous scene, physics loop, and model data.
    frame.src = url.href;
    frame.title = `${item.label}: six named methods with interactive MuJoCo comparison`;
    fullScreen.href = url.href;
    document.querySelector('#comparison-case-name').textContent = item.label;
  };
  window.artigentMountCasePicker(comparisonPicker, selected.id, '', selectCase);
  if (selected.id !== window.artigentComparisonCases[0].id) selectCase(selected);
}
