/* Recorded comparisons: no simulation, offscreen playback or eager video loading. */
(() => {
  'use strict';
  const data = window.artisanPhysicsData;
  if (!data) return;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const saveData = navigator.connection?.saveData;

  document.querySelectorAll('[data-physics-gallery]').forEach(gallery => {
    const category = gallery.dataset.physicsGallery;
    const cases = data[category];
    const picker = gallery.querySelector('[data-physics-picker]');
    const grid = gallery.querySelector('[data-physics-grid]');
    const playButton = gallery.querySelector('[data-physics-play]');
    const replayButton = gallery.querySelector('[data-physics-replay]');
    const selection = gallery.querySelector('[data-physics-selection]');
    let playing = !reducedMotion.matches && !saveData;
    let active = null;
    let videos = [];
    const visible = new Set();

    function updateButton() {
      playButton.textContent = playing ? 'Pause videos' : 'Play videos';
      playButton.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${category} videos`);
    }
    function sync(video) {
      if (!playing || document.hidden || !visible.has(video)) {
        video.pause();
        return;
      }
      if (!video.hasAttribute('src')) {
        video.src = video.dataset.src;
        video.load();
      }
      video.play().catch(() => {
        // A source switch can abort a pending play; a genuine policy rejection
        // leaves the explicit Play control available on the next click.
        if (video.isConnected && visible.has(video) && video.paused && video.readyState >= 2) {
          playButton.textContent = 'Play videos';
          playButton.setAttribute('aria-label', `Play ${category} videos`);
        }
      });
    }
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) visible.add(entry.target);
        else visible.delete(entry.target);
        sync(entry.target);
      }
    }, { threshold: 0.15 });

    function select(item) {
      if (active === item.id) return;
      active = item.id;
      observer.disconnect();
      visible.clear();
      for (const video of videos) {
        video.pause();
        video.removeAttribute('src');
        video.load();
      }
      videos = [];
      grid.replaceChildren();
      picker.querySelectorAll('button').forEach(button => {
        button.setAttribute('aria-pressed', String(button.dataset.case === item.id));
      });
      selection.textContent = item.label;
      const context = gallery.querySelector('[data-physics-context]');
      if (context) context.textContent = `${item.dataset} · π0.5 policy · Compared by training data`;
      for (const recording of item.videos) {
        const figure = document.createElement('figure');
        if (recording.method.includes('Artisan')) figure.className = 'physics-ours';
        const caption = document.createElement('figcaption');
        caption.textContent = recording.method;
        if (recording.unavailable) {
          const note = document.createElement('p');
          note.className = 'physics-unavailable';
          note.textContent = recording.unavailable;
          figure.append(caption, note);
        } else {
          const video = document.createElement('video');
          video.muted = video.defaultMuted = true;
          video.loop = video.playsInline = true;
          video.preload = 'none';
          video.poster = recording.poster;
          video.dataset.src = recording.src;
          video.setAttribute('aria-label', `${recording.method}: ${item.label}`);
          video.addEventListener('error', () => {
            if (!video.hasAttribute('src')) return;
            const message = document.createElement('p');
            message.className = 'physics-unavailable';
            message.textContent = 'Video unavailable. Select another example.';
            observer.unobserve(video);
            visible.delete(video);
            video.replaceWith(message);
          }, { once: true });
          figure.append(caption, video);
          videos.push(video);
        }
        grid.append(figure);
      }
      videos.forEach(video => observer.observe(video));
      updateButton();
    }
    for (const item of cases) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.case = item.id;
      button.setAttribute('aria-pressed', 'false');
      // Reference each object's Artisan recording; robot tasks use Native + Artisan.
      const preview = item.videos[category === 'robotics' ? 2 : 0];
      const img = document.createElement('img');
      img.src = preview.poster;
      img.alt = '';
      img.loading = 'lazy';
      img.width = 48;
      img.height = 42;
      const label = document.createElement('span');
      label.textContent = item.label;
      button.append(img, label);
      button.addEventListener('click', () => select(item));
      picker.append(button);
    }
    playButton.addEventListener('click', () => {
      // Also allow retrying autoplay when a browser blocks it.
      playing = !playing || playButton.textContent === 'Play videos';
      updateButton();
      videos.forEach(sync);
    });
    replayButton.addEventListener('click', () => {
      playing = true;
      videos.forEach(video => { if (video.hasAttribute('src')) video.currentTime = 0; });
      updateButton();
      videos.forEach(sync);
    });
    document.addEventListener('visibilitychange', () => videos.forEach(sync));
    reducedMotion.addEventListener('change', () => {
      if (reducedMotion.matches) {
        playing = false;
        updateButton();
        videos.forEach(sync);
      }
    });
    select(cases[0]);
  });
})();
