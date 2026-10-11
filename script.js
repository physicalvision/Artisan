'use strict';
// Pause the embedded showcase when readers leave its section or hide the tab.
const comparisonFrame = document.querySelector('#method-comparison');
window.addEventListener('message', event => {
  if (!comparisonFrame || event.source !== comparisonFrame.contentWindow ||
      event.origin !== window.location.origin || event.data?.type !== 'artigent-comparison-height') return;
  const height = event.data.height;
  if (Number.isFinite(height) && height > 0 && height <= 12000) {
    comparisonFrame.style.setProperty('--comparison-height', `${Math.ceil(height) + 2}px`);
  }
});
let comparisonVisible = false;
function updateComparisonVisibility() {
  comparisonFrame?.contentWindow?.postMessage({ type: 'artigent-comparison-visibility',
    visible: comparisonVisible && !document.hidden }, window.location.origin === 'null' ? '*' : window.location.origin);
}
if (comparisonFrame && 'IntersectionObserver' in window) {
  new IntersectionObserver(entries => {
    comparisonVisible = entries[0].isIntersecting;
    updateComparisonVisibility();
  }).observe(comparisonFrame);
  comparisonFrame.addEventListener('load', updateComparisonVisibility);
  document.addEventListener('visibilitychange', updateComparisonVisibility);
}
// Keep dimensions separate: ranks are per metric and include every reported row.
let activeBenchmark = 'physical';
let activeMetric = benchmarkData.physical.default;
const metricSelect = document.querySelector('#metric-select');
const arrow = metric => metric.direction === 'up' ? '↑' : '↓';
const formatScore = (value, metric) => value == null ? '—' : value.toFixed(metric.decimals) + metric.unit;

function metricOrder(group, metric) {
  const direction = metric.direction === 'up' ? -1 : 1;
  return [...group.rows].sort((a, b) => {
    const av = a.values[metric.key], bv = b.values[metric.key];
    if (av == null) return bv == null ? 0 : 1;
    if (bv == null) return -1;
    return direction * (av - bv);
  });
}

function renderRanking(announce = true) {
  const group = benchmarkData[activeBenchmark];
  const metric = group.metrics.find(m => m.key === activeMetric);
  const related = group.metrics.filter(m => m.family === metric.family && m.key !== metric.key);
  const columns = [metric, ...related];
  document.querySelector('.ranking-table').style.setProperty('--related-count', related.length);
  const sorted = metricOrder(group, metric);
  const evaluated = sorted.filter(row => row.values[metric.key] != null);
  const counts = new Map();
  evaluated.forEach(row => {
    const value = row.values[metric.key];
    counts.set(value, (counts.get(value) || 0) + 1);
  });
  const bestValues = new Map(columns.map(column => [column.key,
    [...new Set(metricOrder(group, column).map(row => row.values[column.key]).filter(v => v != null))].slice(0, 2)
  ]));
  const head = document.createElement('tr');
  for (const text of ['Rank', activeBenchmark === 'robotics' ? 'Training assets' : 'Method / backbone']) {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = text; head.append(th);
  }
  columns.forEach(column => {
    const th = document.createElement('th'); th.scope = 'col';
    if (column.key === metric.key) th.setAttribute('aria-sort', metric.direction === 'up' ? 'descending' : 'ascending');
    const button = document.createElement('button'); button.type = 'button';
    button.textContent = column.label + ' ' + arrow(column);
    button.setAttribute('aria-label', 'Rank by ' + column.label + ', ' + (column.direction === 'up' ? 'higher' : 'lower') + ' is better');
    button.addEventListener('click', () => { activeMetric = column.key; metricSelect.value = activeMetric; renderRanking(); metricSelect.focus({ preventScroll: true }); });
    th.append(button); head.append(th);
  });
  document.querySelector('#ranking-head').replaceChildren(head);
  const body = document.createDocumentFragment();
  let rank = 0;
  sorted.forEach((row, i) => {
    const value = row.values[metric.key];
    if (value != null && (i === 0 || value !== sorted[i - 1].values[metric.key])) rank = i + 1;
    const tr = document.createElement('tr');
    if (row.method.includes('Artisan')) tr.classList.add('ours-row');
    const rankCell = document.createElement('td'); rankCell.className = 'rank';
    rankCell.textContent = value == null ? '—' : (counts.get(value) > 1 ? '=' : '') + rank;
    if (value != null && rank === 1) rankCell.classList.add('rank-first');
    const methodCell = document.createElement('th'); methodCell.scope = 'row'; methodCell.textContent = row.method;
    if (row.model) { const model = document.createElement('small'); model.textContent = row.model; methodCell.append(model); }
    tr.append(rankCell, methodCell);
    columns.forEach(column => {
      const cell = document.createElement('td'), score = row.values[column.key];
      cell.textContent = formatScore(score, column);
      if (column.key === metric.key) cell.classList.add('selected-score');
      const top = bestValues.get(column.key);
      if (score != null && score === top[0]) cell.classList.add('best-score');
      else if (score != null && score === top[1]) cell.classList.add('second-score');
      tr.append(cell);
    });
    body.append(tr);
  });
  document.querySelector('#ranking-body').replaceChildren(body);
  document.querySelector('#ranking-caption').textContent = metric.label + ' · ' + evaluated.length + ' evaluated ' + (activeBenchmark === 'robotics' ? 'training compositions' : 'configurations');
  document.querySelector('#ranking-direction').textContent = (metric.direction === 'up' ? 'Higher' : 'Lower') + ' is better ' + arrow(metric);
  document.querySelector('#ranking-dataset').textContent = metric.dataset;
  document.querySelector('#ranking-note').textContent = group.note;
  const source = document.querySelector('#leaderboard-source'); source.textContent = group.source + ' ↗'; source.href = 'Artisan.pdf?v=20261011#page=' + group.page;
  if (announce) {
    const leaders = evaluated.filter(row => row.values[metric.key] === evaluated[0].values[metric.key]);
    document.querySelector('#ranking-announcement').textContent = group.label + ', ranked by ' + metric.label + '. ' + leaders.map(row => row.method + (row.model ? ' (' + row.model + ')' : '')).join(' and ') + (leaders.length > 1 ? ' share first place with ' : ' ranks first with ') + formatScore(evaluated[0].values[metric.key], metric) + '.';
  }
}

function selectBenchmark(key, announce = true) {
  activeBenchmark = key; activeMetric = benchmarkData[key].default;
  document.querySelectorAll('[data-benchmark]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.benchmark === key)));
  metricSelect.replaceChildren();
  let family = '', optgroup;
  benchmarkData[key].metrics.forEach(metric => {
    if (family !== metric.family) {
      family = metric.family; optgroup = document.createElement('optgroup'); optgroup.label = family; metricSelect.append(optgroup);
    }
    const option = document.createElement('option'); option.value = metric.key; option.textContent = metric.label + ' ' + arrow(metric); optgroup.append(option);
  });
  metricSelect.value = activeMetric;
  renderRanking(announce);
}
metricSelect.addEventListener('change', () => { activeMetric = metricSelect.value; renderRanking(); });
document.querySelectorAll('[data-benchmark]').forEach(button => button.addEventListener('click', () => selectBenchmark(button.dataset.benchmark)));
selectBenchmark('physical', false);

// Compact overview charts use the same paper values as the detailed ranking.
const detailedMetrics = document.querySelector('#leaderboard');
function revealLinkedMetrics() {
  if (location.hash === '#leaderboard') detailedMetrics.open = true;
}
window.addEventListener('hashchange', revealLinkedMetrics);
revealLinkedMetrics();
const overviewMethods = ['Artisan', 'Articraft', 'Codex', 'Claude', 'Agy', 'Instruct-Particulate'];
const miniCharts = {
  spatial: { keys: ['iou', 'rest_piou', 'open_piou'], max: 1 },
  physical: { keys: ['single', 'multiple', 'rest', 'open'], max: 100 },
  robotics: { keys: ['twin', 'casa'], max: 100 },
  human: { keys: ['top1', 'top3'], max: 100 }
};
document.querySelectorAll('[data-mini-chart]').forEach(chart => {
  const key = chart.dataset.miniChart, config = miniCharts[key], group = benchmarkData[key];
  const select = chart.querySelector('select');
  config.keys.forEach(name => {
    const metric = group.metrics.find(m => m.key === name);
    const option = document.createElement('option'); option.value = name; option.textContent = metric.label + ' ' + arrow(metric); select.append(option);
  });
  function render() {
    const metric = group.metrics.find(m => m.key === select.value);
    const rows = metricOrder(group, metric).filter(row => key === 'robotics' || key === 'human' || (overviewMethods.includes(row.method) && (!['Artisan', 'Articraft', 'Codex'].includes(row.method) || row.model === 'GPT-6 Astra')));
    const bars = document.createDocumentFragment();
    for (const row of rows) {
      const value = row.values[metric.key];
      const bar = document.createElement('div'); bar.className = 'mini-bar' + (row.method.includes('Artisan') ? ' mini-ours' : '');
      const label = document.createElement('div');
      const name = document.createElement('span'); name.textContent = row.method;
      const score = document.createElement('strong'); score.textContent = formatScore(value, metric);
      label.append(name, score);
      const track = document.createElement('div'); track.className = 'mini-track'; track.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('span'); fill.style.width = 100 * value / config.max + '%'; track.append(fill);
      bar.append(label, track); bars.append(bar);
    }
    chart.querySelector('.mini-bars').replaceChildren(bars);
    chart.querySelector('.mini-scale').textContent = (metric.direction === 'up' ? 'Higher' : 'Lower') + ' is better · scale 0–' + config.max + (metric.unit === '%' ? '%' : '');
  }
  select.addEventListener('change', render);
  render();
});

// The showcase runs in an iframe so its offline request adapter stays isolated.
const studyLink = document.querySelector('#public-study-link');
studyLink.href = window.artigentStudy.publicStudyUrl;
const copyButton = document.querySelector('#copy-citation');
let copyReset;
copyButton.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex').textContent;
  let copied = false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(citation);
      copied = true;
    }
  } catch { /* Fall back for local files and browsers without clipboard access. */ }
  if (!copied) {
    const field = document.createElement('textarea');
    field.value = citation;
    field.style.cssText = 'position:fixed;left:-9999px;top:0';
    document.body.append(field);
    field.select();
    try { copied = document.execCommand('copy'); } catch { copied = false; }
    field.remove();
    copyButton.focus({ preventScroll: true });
  }
  clearTimeout(copyReset);
  if (copied) {
    copyButton.textContent = 'Copied ✓';
    document.querySelector('#copy-status').textContent = 'BibTeX copied to clipboard.';
  } else {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = 'Select & copy';
    document.querySelector('#copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
  }
  copyReset = setTimeout(() => { copyButton.textContent = 'Copy BibTeX ⧉'; }, 2500);
});
