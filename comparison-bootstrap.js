'use strict';
const comparisonEmbedded = window.parent !== window;
document.documentElement.classList.toggle('comparison-embedded', comparisonEmbedded);
const comparisonData = window.__US_OFFLINE_DEMO__;
const requestedCase = new URLSearchParams(location.search).get('case');
const comparisonCase = window.artigentComparisonCases.find(item => item.id === requestedCase) || window.artigentComparisonCases[0];
if (comparisonCase.unit) comparisonData.api.unit = comparisonCase.unit;
if (!comparisonEmbedded) {
  const picker = document.createElement('div');
  picker.className = 'comparison-case-picker';
  document.getElementById('root').before(picker);
  window.artigentMountCasePicker(picker, comparisonCase.id, '../../', item => {
    const url = new URL(location.href);
    url.searchParams.set('case', item.id);
    location.assign(url.href);
  });
}
const comparisonMethods = window.artigentStudy.methods;
for (const candidate of comparisonData.api.unit.candidates) {
  const method = comparisonMethods[candidate.label];
  if (!method) throw new Error('Missing verified method mapping: ' + candidate.label);
  candidate.method_label = method.name;
}
// Display Artisan first, keeping each token tied to its original asset.
const comparisonOrder = ['B', 'C', 'F', 'E', 'A', 'D'];
comparisonData.api.unit.candidates.sort((a, b) => comparisonOrder.indexOf(a.label) - comparisonOrder.indexOf(b.label));
comparisonData.api.session.config.admin = true;
comparisonData.api.sessionDone.config.admin = true;
// Keep the initial 3D poster lightweight. The original model payload and WASM
// are requested only by the explicit start action, including in full-screen view.
const loadedScripts = new Map();
function loadComparisonScript(src) {
  if (!loadedScripts.has(src)) {
    loadedScripts.set(src, new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = () => {
        loadedScripts.delete(src);
        script.remove();
        reject(new Error('Unable to load the simulation. Please try again.'));
      };
      document.head.append(script);
    }));
  }
  return loadedScripts.get(src);
}
let physicsConfigured = false;
window.artigentPhysicsGate = (React, Simulation, configurePhysics) => function PhysicsGate(props) {
  const [ready, setReady] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');
  async function start() {
    if (loading) return;
    setLoading(true); setError('');
    try {
      await Promise.all([
        loadComparisonScript('../../' + comparisonCase.payload),
        loadComparisonScript('data/mujoco-wasm.js')
      ]);
      // The offline request adapter holds this original object by reference.
      const physics = comparisonCase.unit ? window.__ARTISAN_CASE_PHYSICS__ : window.__US_OFFLINE_DEMO__;
      comparisonData.files = physics.files;
      comparisonData.types = physics.types;
      window.__US_OFFLINE_DEMO__ = comparisonData;
      if (!physicsConfigured) { configurePhysics(); physicsConfigured = true; }
      window.artigentSimulationStarted = true;
      setReady(true);
    } catch (cause) {
      setError(cause.message || 'Unable to load the simulation. Please try again.');
    } finally { setLoading(false); }
  }
  const help = React.createElement('p', { className: 'comparison-help' },
    'Drag a part or choose ', React.createElement('strong', null, 'Run physics'),
    ' to simulate; drag empty space to orbit. Use ⊕ to compare two methods. WebGL2 required; desktop recommended.');
  const content = ready ? React.createElement(Simulation, props) : React.createElement('div', { className: 'comparison-start', 'aria-busy': loading },
    React.createElement('div', { className: 'comparison-start-scene' },
      React.createElement('picture', null,
        React.createElement('source', { media: '(max-width: 750px)', srcSet: '../../' + comparisonCase.mobilePoster + '?v=20261008-cases-1' }),
        React.createElement('img', { src: '../../' + comparisonCase.poster + '?v=20261008-cases-1', width: 2180, height: 1094,
          alt: `Preview of the 3D interaction interface: Artisan ${comparisonCase.label.toLowerCase()} and six method selectors. Start simulation to interact.`, decoding: 'async' })),
      React.createElement('div', { className: 'comparison-start-overlay' },
        React.createElement('button', { type: 'button', onClick: start, disabled: loading,
          'data-start-simulation': '', 'aria-describedby': 'simulation-load-note' },
          !loading && React.createElement('span', { 'aria-hidden': true }, '▶'),
          loading ? 'Loading simulation…' : error ? 'Retry simulation' : 'Start simulation'))),
    React.createElement('p', { id: 'simulation-load-note' }, 'Simulation may slow down this page. Models and physics load only after you press Start simulation.'),
    React.createElement('p', { role: error ? 'alert' : 'status' }, error || (loading ? 'Loading models and physics engine…' : '')));
  return React.createElement(React.Fragment, null, help, content);
};
// Report the natural content height, including image loads, tab changes and
// expanded controls. Embedded layout uses width-based canvas sizing to avoid
// a feedback loop between the iframe height and viewport-relative children.
let comparisonHeightFrame = 0, previousComparisonHeight = 0;
function queueComparisonHeight() {
  if (!comparisonEmbedded || comparisonHeightFrame) return;
  comparisonHeightFrame = requestAnimationFrame(() => {
    comparisonHeightFrame = 0;
    const workbench = document.querySelector('.rs-workbench');
    if (!workbench) return;
    const height = Math.ceil(workbench.getBoundingClientRect().height);
    if (height > 0 && height !== previousComparisonHeight) {
      previousComparisonHeight = height;
      window.parent.postMessage({ type: 'artigent-comparison-height', height },
        window.location.origin === 'null' ? '*' : window.location.origin);
    }
  });
}
if (comparisonEmbedded) {
  new ResizeObserver(queueComparisonHeight).observe(document.getElementById('root'));
  window.addEventListener('resize', queueComparisonHeight);
}
// Mark the hidden ranking controls inaccessible as well as visually hidden.
const cleanComparison = () => {
  document.title = 'Artisan — Interactive Method Comparison';
  document.querySelectorAll('.rs-judgment-pane, .rs-topbar, .rs-progress, .rs-interaction-status').forEach(node => {
    node.hidden = true;
    node.setAttribute('inert', '');
  });
  queueComparisonHeight();
};
const comparisonObserver = new MutationObserver(cleanComparison);
comparisonObserver.observe(document.getElementById('root'), { childList: true, subtree: true });
