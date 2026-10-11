/* Scheduling for the public showcase only. The anonymous study is untouched. */
window.artigentCreateSimulationLoop = function (Viewer, timestep) {
  'use strict';
  const states = new Map();
  const create = Viewer.create;
  const dispose = Viewer.prototype.dispose;
  const updateButton = (viewer, state) => {
    state.button.textContent = viewer.paused ? 'Run physics' : 'Pause physics';
    state.button.setAttribute('aria-pressed', String(!viewer.paused));
  };
  const wake = (viewer, physics = false) => {
    const state = states.get(viewer);
    if (!state) return;
    state.dirty = true;
    updateButton(viewer, state);
    if (physics) { state.sleeping = false; state.quietSince = null; state.wakeTime = performance.now(); }
    state.loop?.schedule();
  };
  Viewer.create = async function (...args) {
    const viewer = await create.apply(this, args);
    // The viewer is mounted only after the reader explicitly starts simulation.
    viewer.paused = !window.artigentSimulationStarted;
    const state = { visible: false, dirty: true, sleeping: false, quietSince: null,
      wakeTime: performance.now(), renders: 0, steps: 0, stepMs: 0, renderMs: 0, loop: null };
    states.set(viewer, state);
    state.button = document.createElement('button');
    state.button.type = 'button';
    state.button.dataset.physicsToggle = '';
    state.button.addEventListener('click', () => { viewer.paused = !viewer.paused; wake(viewer, !viewer.paused); });
    viewer.canvas.parentElement.querySelector('.rs-canvas-label').append(state.button);
    updateButton(viewer, state);
    state.intersection = new IntersectionObserver(entries => {
      state.visible = entries[0].isIntersecting;
      state.dirty = true;
      state.loop?.schedule();
    });
    state.intersection.observe(viewer.canvas);
    state.resize = new ResizeObserver(() => wake(viewer));
    state.resize.observe(viewer.canvas);
    state.change = () => wake(viewer);
    state.input = () => { wake(viewer, true); queueMicrotask(() => updateButton(viewer, state)); };
    viewer.controls.addEventListener('change', state.change);
    // The viewer stops propagation on successful picking, so wake from its parent.
    state.inputTarget = viewer.canvas.parentElement;
    state.inputTarget.addEventListener('pointerdown', state.input, true);
    return viewer;
  };
  for (const method of ['reset', 'restore', 'setPartColors', 'highlightMovablePart']) {
    const original = Viewer.prototype[method];
    Viewer.prototype[method] = function (...args) {
      const result = original.apply(this, args);
      wake(this, method === 'reset' || method === 'restore');
      return result;
    };
  }
  Viewer.prototype.dispose = function () {
    const state = states.get(this);
    if (state) {
      state.intersection.disconnect(); state.resize.disconnect();
      this.controls.removeEventListener('change', state.change);
      state.inputTarget.removeEventListener('pointerdown', state.input, true);
      state.button.remove();
      states.delete(this);
    }
    return dispose.call(this);
  };

  let parentVisible = true;
  function resumeVisible() {
    for (const [viewer, state] of states) {
      state.loop?.resetClock();
      wake(viewer);
    }
  }
  document.addEventListener('visibilitychange', resumeVisible);
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.origin !== window.location.origin || event.data?.type !== 'artigent-comparison-visibility') return;
    parentVisible = event.data.visible === true;
    resumeVisible();
  });
  // Read-only counters make performance checks reproducible without changing the physics.
  window.artigentComparisonStats = () => [...states].map(([viewer, state]) => ({
    visible: state.visible && parentVisible, sleeping: state.sleeping, paused: viewer.paused,
    renders: state.renders, steps: state.steps, stepMs: Math.round(state.stepMs), renderMs: Math.round(state.renderMs),
    drawCalls: viewer.renderer.info.render.calls, triangles: viewer.renderer.info.render.triangles,
    canvas: [viewer.canvas.width, viewer.canvas.height], simulationTime: viewer.data.time,
    faults: viewer.simulationFaultCount, dragging: Boolean(viewer.drag)
  }));

  return class ShowcaseLoop {
    constructor(viewers) {
      this.viewers = viewers;
      this.running = false;
      this.frameId = 0;
      this.resetClock();
      for (const viewer of viewers) states.get(viewer).loop = this;
    }
    resetClock() { this.previous = null; this.accumulator = 0; }
    start() { this.running = true; this.resetClock(); this.schedule(); }
    stop() { this.running = false; cancelAnimationFrame(this.frameId); this.frameId = 0; this.resetClock(); }
    schedule() {
      if (this.running && !this.frameId && parentVisible && !document.hidden) this.frameId = requestAnimationFrame(this.frame);
    }
    frame = now => {
      this.frameId = 0;
      if (!this.running || document.hidden || !parentVisible) { this.resetClock(); return; }
      const visible = this.viewers.filter(viewer => !viewer.disposed && states.get(viewer)?.visible);
      if (!visible.length) { this.resetClock(); return; }
      // A 30 Hz display avoids spending the whole main thread on continuous rendering.
      if (this.previous !== null && now - this.previous < 32) { this.schedule(); return; }
      const elapsed = this.previous === null ? 1 / 30 : Math.min((now - this.previous) / 1000, .05);
      this.previous = now;
      const active = visible.filter(viewer => {
        const state = states.get(viewer);
        if (viewer.drag) { state.sleeping = false; state.quietSince = null; }
        return !viewer.paused && !viewer.faulted && !state.sleeping;
      });
      // Keep the supplied 2 ms MuJoCo timestep and contact model. Bound catch-up work.
      this.accumulator = active.length ? Math.min(this.accumulator + elapsed, .05) : 0;
      const budgetStart = performance.now();
      while (active.length && this.accumulator + Number.EPSILON >= timestep) {
        for (const viewer of active) {
          const start = performance.now(); viewer.step();
          const state = states.get(viewer); state.steps++; state.stepMs += performance.now() - start; state.dirty = true;
        }
        this.accumulator -= timestep;
        if (performance.now() - budgetStart >= 6) { this.accumulator = 0; break; }
      }
      for (const viewer of visible) {
        const state = states.get(viewer);
        if (active.includes(viewer)) {
          let stationary = !viewer.drag && now - state.wakeTime > 3000;
          for (const velocity of viewer.data.qvel) if (Math.abs(velocity) > .0001) { stationary = false; break; }
          if (!stationary) state.quietSince = null;
          else if (state.quietSince === null) state.quietSince = now;
          else if (now - state.quietSince > 1500) state.sleeping = true;
          viewer.advanceActive(elapsed * 1000);
        }
        if (state.dirty || viewer.drag) {
          state.dirty = false;
          const start = performance.now(); viewer.render(now);
          state.renders++; state.renderMs += performance.now() - start;
        }
      }
      if (visible.some(viewer => {
        const state = states.get(viewer);
        return state.dirty || (!viewer.paused && !viewer.faulted && !state.sleeping);
      })) this.schedule();
      else this.resetClock();
    };
  };
};
