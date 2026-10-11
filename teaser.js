/* Projected scene regions and physical attributes come from the supplied Blender assets. */
(() => {
  'use strict';
  const data = window.ARTIGENT_TEASER;
  const root = document.querySelector('#teaser-interactive');
  if (!root || !data?.objects?.length) return;
  // Number each room independently, in the same order as its visible object buttons.
  const objects = new Map(data.rooms.flatMap(room => data.objects
    .filter(object => object.room === room.id)
    .map((object, index) => [object.id, { ...object, number: String(index + 1).padStart(2, '0') }])));
  const scenes = root.querySelector('.teaser-scenes');
  const video = root.querySelector('#teaser-object-video');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let selection, attributeMode = 'mass', mediaVisible = false, resumeMedia = false;
  let previewPulse;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function objectButton(object, className, text) {
    const button = element('button', className, text);
    button.type = 'button'; button.dataset.teaserObject = object.id;
    button.setAttribute('aria-label', `${object.number} · Explore ${object.name.toLowerCase()}: motion and physical attributes`);
    button.setAttribute('aria-controls', 'teaser-inspector');
    button.setAttribute('aria-pressed', 'false');
    return button;
  }
  const primaryPins = { office_chair: [540, 632], lamp: [510, 477], lab_stool: [311, 576],
    centrifuge: [716, 525], fume_hood: [734, 308], refrigerator: [847, 471] };
  const numberedObjects = new Set([
    'office_chair', 'lamp', 'office-laptop', 'office-printer',
    'office-pullout_shelf_cabinet', 'office-office_window_cabinet_v3',
    'lab_stool', 'centrifuge', 'fume_hood', 'biolab-microscope', 'biolab-freezer', 'biolab-robot',
    'refrigerator', 'kitchen-bar_stool', 'kitchen-bin', 'kitchen-microwave',
    'kitchen-side_cabinet', 'kitchen-wall_cabinet', 'kitchen-sink_counter_cabinet'
  ]);
  function polygonArea(points) {
    return Math.abs(points.reduce((sum, p, i) => { const q = points[(i + 1) % points.length]; return sum + p[0] * q[1] - q[0] * p[1]; }, 0)) / 2;
  }
  function icon(paths, className) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('aria-hidden', 'true');
    svg.classList.add(className);
    for (const d of paths) {
      const path = document.createElementNS(svg.namespaceURI, 'path'); path.setAttribute('d', d); svg.append(path);
    }
    return svg;
  }
  data.rooms.forEach(room => {
    const card = element('div', 'teaser-scene'); card.dataset.scene = room.id;
    const canvas = element('div', 'teaser-scene-canvas');
    const image = element('img'); image.src = room.image;
    image.alt = `Artisan-generated ${room.name.toLowerCase()}. Select articulated objects to watch motion and inspect physical attributes.`;
    image.width = 1600; image.height = 1360;
    const overlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    overlay.setAttribute('viewBox', '0 0 1000 850'); overlay.setAttribute('aria-hidden', 'true');
    overlay.classList.add('teaser-outlines'); canvas.append(image, overlay);
    const roomObjects = [...objects.values()].filter(object => object.room === room.id);
    // A short approach-and-tap gesture teaches clicking, without implying the scene is draggable.
    const hintObject = roomObjects.find(object => object.primary);
    const hintPin = primaryPins[hintObject.id];
    const hint = element('div', 'teaser-click-hint'); hint.setAttribute('aria-hidden', 'true');
    hint.style.left = `${hintPin[0] / 10}%`; hint.style.top = `${hintPin[1] / 8.5}%`;
    hint.append(element('span', 'teaser-tap-ring'), icon(['M8 12V4a2 2 0 0 1 4 0v6l1-1a2 2 0 0 1 3 1 2 2 0 0 1 3 1 2 2 0 0 1 3 1v4c0 3-2 6-5 6h-3c-2 0-3-1-4-2l-5-6a2 2 0 0 1 3-2l3 3'], 'teaser-hand'), element('span', 'teaser-click-label', 'Click an object'));
    canvas.append(hint);
    // Smaller foreground objects stay clickable over desks, cabinets, and benches.
    const regions = roomObjects.flatMap(object => object.regions.map(region => ({ ...region, object })));
    regions.sort((a, b) => polygonArea(b.points) - polygonArea(a.points));
    for (const region of regions) {
      const polygon = document.createElementNS(overlay.namespaceURI, 'polygon');
      polygon.setAttribute('points', region.points.map(point => point.join(',')).join(' '));
      polygon.setAttribute('vector-effect', 'non-scaling-stroke');
      polygon.dataset.teaserObject = region.object.id;
      polygon.dataset.teaserInstance = region.instance;
      const title = document.createElementNS(overlay.namespaceURI, 'title'); title.textContent = `${region.object.number} · ${region.object.name}`;
      polygon.append(title); overlay.append(polygon);
    }
    const choices = element('div', 'teaser-object-choices');
    choices.setAttribute('role', 'group'); choices.setAttribute('aria-label', `All ${roomObjects.length} objects in the ${room.name.toLowerCase()}`);
    for (const object of roomObjects) {
      const points = [...object.regions].sort((a, b) => polygonArea(b.points) - polygonArea(a.points))[0].points;
      const pin = primaryPins[object.id] || [points.reduce((sum, p) => sum + p[0], 0) / points.length, points.reduce((sum, p) => sum + p[1], 0) / points.length];
      const numbered = numberedObjects.has(object.id);
      const marker = objectButton(object, 'teaser-marker' + (numbered ? '' : ' teaser-marker-secondary'), object.number);
      marker.title = `${object.number} · ${object.name}`; marker.style.left = `${pin[0] / 10}%`; marker.style.top = `${pin[1] / 8.5}%`;
      // Every type also has a visible, keyboard-accessible numbered button below the scene.
      if (!numbered) marker.tabIndex = -1;
      canvas.append(marker);
      const choice = objectButton(object, 'teaser-object-choice');
      choice.title = `${object.number} · ${object.name}${object.instanceCount > 1 ? ` · ${object.instanceCount} instances` : ''}`;
      choice.append(element('span', 'teaser-choice-number', object.number), document.createTextNode(object.name));
      choices.append(choice);
    }
    card.append(canvas, element('h3', 'teaser-room-name', room.name), choices); scenes.append(card);
  });
  const hintObserver = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { root.classList.add('teaser-hint-visible'); hintObserver.disconnect(); }
  }, { threshold: .15 });
  hintObserver.observe(scenes);

  function playVisibleMedia() {
    if (mediaVisible && !document.hidden && resumeMedia) video.play().catch(() => { /* Native play control remains available. */ });
  }
  function suspendMedia() {
    if (!video.paused) resumeMedia = true;
    video.pause();
  }
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) { resumeMedia = !video.paused || resumeMedia; mediaVisible = false; suspendMedia(); }
    else { mediaVisible = true; playVisibleMedia(); }
  }, { threshold: .1 }).observe(video);
  document.addEventListener('visibilitychange', () => { if (document.hidden) suspendMedia(); else playVisibleMedia(); });
  video.addEventListener('pause', () => { if (mediaVisible && !document.hidden && video.readyState > 0) resumeMedia = false; });
  video.addEventListener('play', () => { resumeMedia = true; });

  function formatRange(joint) {
    if (joint.type === 'continuous') return 'Continuous';
    const lower = joint.limit?.lower, upper = joint.limit?.upper;
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) return 'Not specified';
    const factor = joint.type === 'prismatic' ? 100 : 180 / Math.PI;
    const number = value => String(Number((value * factor).toFixed(1)));
    return `${number(lower)} – ${number(upper)} ${joint.type === 'prismatic' ? 'cm' : '°'}`;
  }
  function formatParameter(value, unit) {
    return Number.isFinite(value) ? `${Number(value.toPrecision(5))} ${unit}` : '—';
  }
  function renderAttributes(mode) {
    attributeMode = mode;
    const mass = mode === 'mass', dynamics = mode === 'dynamics', inertia = mode === 'inertia';
    root.querySelectorAll('[data-teaser-attribute]').forEach(button => {
      button.hidden = button.dataset.teaserAttribute === 'mass' && !selection.parts.length;
      button.setAttribute('aria-pressed', String(button.dataset.teaserAttribute === mode));
    });
    const table = root.querySelector('.teaser-attributes table');
    table.classList.toggle('teaser-joint-table', !mass);
    table.querySelector('caption').textContent = mass ? 'Part-level mass and materials' : inertia ? 'Center of mass (m) and inertia (kg·m²)' : dynamics ? 'Joint limits, damping and friction' : 'Articulation and joint limits';
    const header = element('tr');
    const columns = mass ? ['Part', 'Mass', 'Materials'] : inertia ? ['Part', 'Center of mass', 'Inertia tensor'] : dynamics ? ['Joint', 'Limits', 'Dynamics'] : ['Joint', 'Motion', 'Range'];
    for (const text of columns) {
      const th = element('th', '', text); th.scope = 'col'; header.append(th);
    }
    table.querySelector('thead').replaceChildren(header);
    const rows = (mass || inertia ? selection.parts : selection.joints).map(item => {
      const row = element('tr');
      const th = element('th', '', mass || inertia ? item.name : item.name.replaceAll('_', ' ')); th.scope = 'row';
      if (!mass && item.axis) th.title = `Local joint axis: ${item.axis.join(', ')}`;
      if (inertia) {
        const center = element('td');
        ['x', 'y', 'z'].forEach((axis, index) => center.append(element('div', '', `${axis}: ${formatParameter(item.centerOfMass?.[index], '')}`)));
        const tensor = element('td');
        ['ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'].forEach(key => tensor.append(element('div', '', `${key}: ${formatParameter(item.inertia?.[key], '')}`)));
        row.append(th, center, tensor);
      } else if (dynamics) {
        const sliding = item.type === 'prismatic';
        const limits = element('td');
        limits.append(element('div', '', `${sliding ? 'Force' : 'Torque'}: ${formatParameter(item.limit?.effort, sliding ? 'N' : 'N·m')}`),
          element('div', '', `Speed: ${formatParameter(item.limit?.velocity, sliding ? 'm/s' : 'rad/s')}`));
        const parameters = element('td');
        parameters.append(element('div', '', `Damping: ${formatParameter(item.dynamics?.damping, sliding ? 'N·s/m' : 'N·m·s/rad')}`),
          element('div', '', `Friction: ${formatParameter(item.dynamics?.friction, sliding ? 'N' : 'N·m')}`));
        row.append(th, limits, parameters);
      } else {
        row.append(th, element('td', 'teaser-part-mass', mass ? item.mass : ({ prismatic: 'Slide', revolute: 'Rotate', continuous: 'Rotate' }[item.type] || item.type)),
          element('td', '', mass ? item.materials.join(' · ') : formatRange(item)));
      }
      return row;
    });
    root.querySelector('#teaser-parts').replaceChildren(...rows);
    root.querySelector('.teaser-attribute-scroll').scrollTop = 0;
    root.querySelector('.teaser-data-note').textContent = mass ? 'Source asset masses and material labels. Total mass is for one instance.' : inertia ? 'Center of mass is in the part frame; inertia is in the source inertial frame.' : dynamics ? 'Source joint parameters. — means not specified.' : 'Source joint limits: rotation in degrees; travel in cm. Continuous joints rotate freely.';
  }
  function selectObject(id, announce = true, instance = null) {
    const object = objects.get(id); if (!object) return;
    const changed = selection?.id !== id; selection = object;
    root.querySelectorAll('[data-teaser-object]').forEach(node => {
      const selected = node.dataset.teaserObject === id;
      node.classList.toggle('is-selected', selected && (!instance || !node.dataset.teaserInstance || node.dataset.teaserInstance === instance));
      if (node.tagName === 'BUTTON') node.setAttribute('aria-pressed', String(selected));
    });
    if (changed) {
      video.pause(); video.poster = object.poster; video.src = object.video;
      video.setAttribute('aria-label', `${object.name}: ${object.videoKind.toLowerCase()}`);
      resumeMedia = !reducedMotion.matches; playVisibleMedia();
    }
    root.querySelector('#teaser-motion-caption').textContent = `${object.videoKind} · ${object.duration} sec`;
    root.querySelector('#teaser-preview-number').textContent = object.number;
    root.querySelector('#teaser-preview-label').textContent = `${object.name} · ${object.videoKind === 'Scene walkthrough' ? 'in this scene' : 'motion preview'}`;
    root.querySelector('#teaser-object-context').textContent = data.rooms.find(room => room.id === object.room).name;
    root.querySelector('#teaser-object-name').textContent = object.name;
    const hasMass = object.totalMassKg !== null;
    const useGrams = hasMass && object.totalMassKg < 1;
    root.querySelector('#teaser-object-mass').textContent = String(hasMass ? Number((object.totalMassKg * (useGrams ? 1000 : 1)).toPrecision(6)) : object.joints.length);
    root.querySelector('.teaser-mass>span').textContent = hasMass ? (useGrams ? 'g' : 'kg') : (object.joints.length === 1 ? 'joint' : 'joints');
    root.querySelector('.teaser-total-label').textContent = hasMass ? 'Total mass per instance' : 'Articulated joints per asset';
    root.querySelector('#teaser-material-summary').textContent = `${object.parts.length} parts · ${object.joints.length} joints · ${object.instanceCount} ${object.instanceCount === 1 ? 'instance' : 'instances'}`;
    renderAttributes(changed ? (object.parts.length ? 'mass' : 'joints') : attributeMode);
    if (announce) {
      root.querySelector('#teaser-status').textContent = `${object.name} selected. The labeled preview shows the corresponding ${object.videoKind.toLowerCase()}. Physical attributes are available beside it.`;
      if (!reducedMotion.matches) {
        previewPulse?.cancel();
        previewPulse = root.querySelector('.teaser-detail-video').animate([
          { boxShadow: '0 0 0 3px rgba(40, 40, 40, .55)' },
          { boxShadow: '0 0 0 8px rgba(40, 40, 40, 0)' }
        ], { duration: 800, easing: 'ease-out' });
      }
    }
  }
  function selectRoom(id, announce = true) {
    if (!data.rooms.some(room => room.id === id)) return;
    root.dataset.room = id;
    root.querySelectorAll('[data-teaser-room]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.teaserRoom === id)));
    scenes.querySelectorAll('[data-scene]').forEach(card => { card.hidden = card.dataset.scene !== id; });
    if (selection?.room !== id) selectObject(data.objects.find(object => object.room === id).id, announce);
  }
  function activateObject(id, instance) {
    root.classList.add('teaser-hint-dismissed');
    selectObject(id, true, instance);
  }
  root.addEventListener('click', event => {
    const object = event.target.closest('[data-teaser-object]');
    if (object) activateObject(object.dataset.teaserObject, object.dataset.teaserInstance);
    const room = event.target.closest('[data-teaser-room]'); if (room) selectRoom(room.dataset.teaserRoom);
    const attribute = event.target.closest('[data-teaser-attribute]'); if (attribute) renderAttributes(attribute.dataset.teaserAttribute);
  });
  selectObject('office_chair', false);
  selectRoom('office', false);
  root.hidden = false; document.querySelector('#teaser-fallback').hidden = true;
})();
