// SiteTwin's 3D preview of the build sequence: the model, one shape per build step, standing on
// the satellite picture of its site. Play runs through the steps; the step going up is shown in
// amber, the steps before it as built, the ones after it not yet there.
import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';

// How each kind of work looks once built.
const LOOK = {
  'Foundations': 0x8d8a83, 'Vertical structure': 0xc9c5bb, 'Horizontal structure': 0xb9b4a8,
  'Floors': 0xd6d2c8, 'Stairs and ramps': 0xbdb8ad, 'Roof': 0x8b8f96, 'Envelope': 0x7fa6c4,
  'Railings and finishes': 0x9a9fa6, 'Services': 0x6f8f7a, 'Other': 0xb5b0a6,
};
const NOW = 0xf2a516;

function tileXY(lat, lon, z) {
  const n = 2 ** z;
  const x = (lon + 180) / 360 * n;
  const y = (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * n;
  return [x, y];
}

// The site's satellite picture, 5 × 5 tiles around it, as a ground texture. Esri's tiles allow
// being drawn into a canvas; if they cannot be fetched the ground stays plain.
async function satellite(lat, lon) {
  const z = 18, R = 2;
  const [fx, fy] = tileXY(lat, lon, z);
  const cx = Math.floor(fx), cy = Math.floor(fy);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256 * (2 * R + 1);
  const ctx = canvas.getContext('2d');
  const jobs = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    jobs.push(new Promise((ok) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => { ctx.drawImage(img, (dx + R) * 256, (dy + R) * 256); ok(true); };
      img.onerror = () => ok(false);
      img.src = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${cy + dy}/${cx + dx}`;
    }));
  }
  const got = await Promise.all(jobs);
  if (!got.some(Boolean)) return null;
  const metresPerTile = 40075016.686 * Math.cos(lat * Math.PI / 180) / 2 ** z;
  // Where the site is inside the picture, in metres from its centre (east, south).
  const offset = [(fx - cx - 0.5) * metresPerTile, (fy - cy - 0.5) * metresPerTile];
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return { texture, size: metresPerTile * (2 * R + 1), offset };
}

export function open(el, ui) {
  const msg = el.querySelector('[data-viewer-msg]');
  const label = el.querySelector('[data-viewer-label]');
  const say = (html) => { msg.innerHTML = html; msg.classList.remove('hidden'); };
  say('<span>Opening the model…</span><div class="bar"><div class="bar-fill" style="width:2%"></div></div>');
  const bar = () => msg.querySelector('.bar-fill');

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  el.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 20000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  scene.add(new THREE.HemisphereLight(0xdfeaf5, 0x6b6250, 1.6));
  const sun = new THREE.DirectionalLight(0xfff3df, 2.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun, sun.target);

  const size = () => {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(size).observe(el);
  size();
  let dirty = true;
  controls.addEventListener('change', () => { dirty = true; });
  (function loop() {
    requestAnimationFrame(loop);
    controls.update();
    if (dirty || controls.enableDamping) renderer.render(scene, camera);
    dirty = false;
  })();

  const meshes = new Map();          // step -> mesh
  let steps = [];
  let shown = 0;
  let groundMesh = null;

  function setStep(n) {
    shown = n;
    for (const [step, mesh] of meshes) {
      if (step === 0) { mesh.visible = true; continue; }       // not in any step: always there
      mesh.visible = step <= n;
      const phase = (steps[step - 1] || {}).phase || 'Other';
      mesh.material.color.setHex(step === n && n < steps.length + 1 ? NOW : (LOOK[phase] || LOOK.Other));
      mesh.material.emissive.setHex(step === n ? 0x3a2400 : 0x000000);
    }
    if (n === steps.length) for (const [step, mesh] of meshes) if (step) mesh.material.color.setHex(LOOK[(steps[step - 1] || {}).phase] || LOOK.Other), mesh.material.emissive.setHex(0);
    if (groundMesh) {
      const below = steps[n - 1] && steps[n - 1].phase === 'Foundations' && n < steps.length;
      groundMesh.material.opacity = below ? 0.35 : 1;
      groundMesh.material.depthWrite = !below;
    }
    ui.scrub.value = n;
    ui.word.textContent = `${n} / ${steps.length}`;
    const s = steps[n - 1];
    if (s && n < steps.length) { label.textContent = `Step ${n} of ${steps.length} · ${s.storey} · ${s.phase} (${s.elements})`; label.classList.remove('hidden'); }
    else if (n === steps.length) { label.textContent = 'Built'; label.classList.remove('hidden'); }
    else { label.textContent = 'The empty site'; label.classList.remove('hidden'); }
    for (const li of ui.list.querySelectorAll('li')) {
      const k = Number(li.dataset.n);
      li.className = k === n && n < steps.length ? 'now' : (k < n ? 'built' : '');
    }
    const now = ui.list.querySelector('li.now');
    if (now && playing) now.scrollIntoView({ block: 'nearest' });
    dirty = true;
  }

  let playing = null;
  function play() {
    if (playing) { clearInterval(playing); playing = null; ui.play.textContent = '▶ Play'; return; }
    if (shown >= steps.length) setStep(0);
    ui.play.textContent = '❚❚ Pause';
    playing = setInterval(() => {
      if (shown >= steps.length) { clearInterval(playing); playing = null; ui.play.textContent = '▶ Play again'; return; }
      setStep(shown + 1);
    }, 900);
  }

  fetch(el.dataset.steps, { credentials: 'same-origin' }).then((r) => r.json()).then((data) => {
    steps = data.steps;
    const byId = {};
    for (const [step, ids] of Object.entries(data.ids)) for (const id of ids) byId[id] = Number(step);
    const worker = new Worker(new URL('sitetwin-ifc-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = async (ev) => {
      const m = ev.data;
      if (m.type === 'progress') {
        if (m.stage === 'download') { bar().style.width = (2 + 60 * m.loaded / m.total) + '%'; msg.querySelector('span').textContent = `Downloading the model… ${(m.loaded / 1048576).toFixed(1)} MB`; }
        else if (m.stage === 'open') msg.querySelector('span').textContent = 'Reading its shapes…';
        else if (m.stage === 'shapes') { bar().style.width = (62 + 36 * Math.min(1, m.done / m.total)) + '%'; }
      } else if (m.type === 'part') {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(m.positions, 3));
        g.setAttribute('normal', new THREE.BufferAttribute(m.normals, 3));
        const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: LOOK.Other, roughness: 0.85, metalness: 0.02, side: THREE.DoubleSide }));
        mesh.castShadow = mesh.receiveShadow = true;
        meshes.set(m.step, mesh);
        scene.add(mesh);
      } else if (m.type === 'done') {
        worker.terminate();
        await frame();
      } else if (m.type === 'error') {
        say(`<span>The 3D preview could not read this model: ${m.message}.</span><span class="small">The film is made from it all the same.</span>`);
      }
    };
    worker.postMessage({ url: el.dataset.model, base: el.dataset.webIfc, steps: byId });
  }).catch((err) => say(`<span>The 3D preview could not open: ${err.message}</span>`));

  async function frame() {
    const box = new THREE.Box3();
    for (const mesh of meshes.values()) box.expandByObject(mesh);
    if (box.isEmpty()) { say('<span>The model has no shapes to show.</span>'); return; }
    const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
    const r = Math.max(s.x, s.y, s.z);
    camera.position.set(c.x + r * 1.3, box.min.y + r * 0.9, c.z + r * 1.5);
    controls.target.set(c.x, box.min.y + s.y * 0.35, c.z);
    camera.near = r / 200; camera.far = r * 60; camera.updateProjectionMatrix();
    sun.position.set(c.x + r, box.max.y + r * 2, c.z + r * 0.6);
    sun.target.position.copy(c);
    const cam = sun.shadow.camera;
    cam.left = cam.bottom = -r; cam.right = cam.top = r; cam.near = 0.1; cam.far = r * 6; cam.updateProjectionMatrix();

    // The ground is the model's zero (the ground floor, as Revit models usually are); piles and
    // basements go below it, and the ground turns see-through while they are being built.
    const groundY = box.min.y < 0 && box.max.y > 0 ? -0.02 : box.min.y - 0.02;
    const lat = parseFloat(el.dataset.lat), lon = parseFloat(el.dataset.lon);
    let ground = null;
    if (isFinite(lat) && isFinite(lon)) ground = await satellite(lat, lon).catch(() => null);
    const extent = ground ? ground.size : Math.max(200, r * 8);
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(extent, extent),
      new THREE.MeshStandardMaterial(Object.assign(ground ? { map: ground.texture, roughness: 1 } : { color: 0xb9b3a4, roughness: 1 }, { transparent: true })));
    groundMesh = plane;
    plane.rotation.x = -Math.PI / 2;
    plane.position.set(ground ? c.x - ground.offset[0] : c.x, groundY, ground ? c.z - ground.offset[1] : c.z);
    plane.receiveShadow = true;
    scene.add(plane);
    msg.classList.add('hidden');
    ui.play.disabled = ui.scrub.disabled = false;
    ui.play.addEventListener('click', play);
    ui.scrub.max = steps.length;
    ui.scrub.addEventListener('input', () => setStep(Number(ui.scrub.value)));
    for (const li of ui.list.querySelectorAll('li')) li.querySelector('button').addEventListener('click', () => setStep(Number(li.dataset.n)));
    setStep(steps.length);
  }
  return { setStep };
}
