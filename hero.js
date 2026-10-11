/* Match the reference's pinned video and scroll/first-loop metadata reveal. */
(() => {
  'use strict';
  const hero = document.querySelector('.hero');
  const info = hero.querySelector('#hero-paper-info');
  const video = hero.querySelector('#scene-video');
  const play = hero.querySelector('#hero-play');
  const view = hero.querySelector('#hero-view');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let completed = false, previousTime = 0, manual = null, progress = 0, frame = 0;
  let compact = false, lastScroll = window.scrollY;

  function update() {
    frame = 0;
    const viewport = window.innerHeight || 1;
    const distance = -hero.getBoundingClientRect().top;
    const scrollProgress = Math.min(Math.max((distance - viewport * .08) / (viewport * .82), 0), 1);
    const accessibleDefault = reduced.matches || compact || navigator.connection?.saveData;
    hero.classList.toggle('is-static', reduced.matches || Boolean(navigator.connection?.saveData));
    progress = manual ?? (accessibleDefault || completed ? 1 : scrollProgress);
    hero.style.setProperty('--hero-progress', progress.toFixed(4));
    hero.classList.toggle('is-info-visible', progress > .02);
    hero.classList.toggle('is-watching', manual === 0);
    info.inert = progress < .98;
    info.setAttribute('aria-hidden', String(progress <= .02));
    view.textContent = progress > .5 ? 'Watch scene' : 'Paper details ↓';
    view.setAttribute('aria-expanded', String(progress > .5));
  }
  function requestUpdate() {
    if (!frame) frame = requestAnimationFrame(update);
  }
  function fit() {
    const style = getComputedStyle(info);
    const contentHeight = [...info.children].reduce((sum, child) => {
      const css = getComputedStyle(child);
      return sum + child.getBoundingClientRect().height + parseFloat(css.marginTop) + parseFloat(css.marginBottom);
    }, 0);
    const controlsHeight = hero.querySelector('.hero-controls').offsetHeight;
    hero.style.setProperty('--hero-controls-height', `${controlsHeight}px`);
    const required = contentHeight + parseFloat(style.paddingTop) + controlsHeight + 24;
    compact = required > window.innerHeight;
    hero.classList.toggle('is-compact', compact);
    requestUpdate();
  }
  function showPaper() { manual = 1; update(); }
  hero.classList.add('is-enhanced');
  fit(); update();
  window.addEventListener('scroll', () => {
    // Explicit viewing mode lasts until the reader scrolls again.
    if (Math.abs(window.scrollY - lastScroll) > 2) manual = null;
    lastScroll = window.scrollY;
    requestUpdate();
  }, { passive: true });
  window.addEventListener('resize', fit);
  document.fonts.ready.then(fit);
  reduced.addEventListener('change', requestUpdate);
  video.addEventListener('loadedmetadata', () => { previousTime = 0; });
  video.addEventListener('timeupdate', () => {
    if (!completed && Number.isFinite(video.duration) && video.duration > 0 &&
        (video.currentTime >= video.duration - .25 || previousTime > video.currentTime + .5)) {
      completed = true;
      requestUpdate();
    }
    previousTime = video.currentTime;
  });
  video.addEventListener('error', showPaper);
  // Autoplay denial must not leave the paper metadata undiscoverable.
  video.addEventListener('loadeddata', () => {
    if (video.paused) showPaper();
  }, { once: true });
  function updatePlayback() { play.textContent = video.paused ? 'Play video' : 'Pause video'; }
  video.addEventListener('play', updatePlayback);
  video.addEventListener('pause', updatePlayback);
  updatePlayback();
  play.addEventListener('click', () => {
    if (video.paused) video.play().catch(() => { updatePlayback(); showPaper(); });
    else video.pause();
  });
  view.addEventListener('click', () => { manual = progress > .5 ? 0 : 1; update(); });
  document.querySelector('[data-show-paper]').addEventListener('click', event => {
    event.preventDefault();
    // Reveal after scrolling so scroll events cannot discard the explicit choice.
    hero.scrollIntoView({ behavior: 'instant', block: 'start' });
    lastScroll = window.scrollY;
    showPaper();
    info.querySelector('h1').focus({ preventScroll: true });
  });
  function revealHash() {
    if (['#paper-title', '#hero-paper-info'].includes(location.hash)) showPaper();
  }
  window.addEventListener('hashchange', revealHash);
  revealHash();

  const header = document.querySelector('.site-nav');
  const toggle = header.querySelector('.nav-toggle');
  const menu = header.querySelector('nav');
  const mobile = window.matchMedia('(max-width: 1000px)');
  function setMenu(open) {
    const expanded = mobile.matches && open;
    header.classList.toggle('nav-open', expanded);
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.textContent = expanded ? 'Close' : 'Menu';
    menu.inert = mobile.matches && !expanded;
  }
  header.classList.add('is-enhanced'); setMenu(false);
  toggle.addEventListener('click', () => setMenu(!header.classList.contains('nav-open')));
  menu.addEventListener('click', event => { if (event.target.closest('a')) setMenu(false); });
  document.addEventListener('click', event => { if (!header.contains(event.target)) setMenu(false); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && header.classList.contains('nav-open')) { setMenu(false); toggle.focus(); }
  });
  mobile.addEventListener('change', () => setMenu(false));
})();
