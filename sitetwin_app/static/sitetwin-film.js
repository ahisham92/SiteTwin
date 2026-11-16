// SiteTwin's construction film, played and recorded in the browser: free, no GPU, no AI.
//
// The model is read in the browser (web-ifc, one shape per element) and set in its real place:
// the ground from open elevation tiles with the aerial photograph on it, out to the horizon, the
// sea at its level, the sky and the sun. The planner's stages (carried by the model, see
// tools/stage_ifc.py) are played over it in order, each in its share of the time by its days:
// the old deck broken out, piles repaired and driven, fill placed, concrete cast one pour after
// another in its formwork, precast lowered by the crawler crane, furniture fitted. The plant
// works where the work is. The film can be saved as a video: the view and its captions are
// drawn onto one canvas and recorded by the browser (MP4 where it can, WebM otherwise).
//
// Frame: three's x is east, y is up (the model's levels, metres), z is south. The model's
// origin is the site's surveyed point (latitude, longitude), as SiteTwin prepares models.
import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import { Sky } from './vendor/three/Sky.js';

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const INTRO = 5;          // seconds of the site as it is today
const OUTRO = 7;          // seconds of the finished work
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ease = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));
const lerp = (a, b, u) => a + (b - a) * u;

// ---- the real surroundings -----------------------------------------------------------------

const TERRAIN = (z, x, y) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
const IMAGERY = (z, x, y) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`;

function tileXY(lat, lon, z) {
  const n = 2 ** z;
  return [(lon + 180) / 360 * n, (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * n];
}
const metresPerTile = (lat, z) => 40075016.686 * Math.cos(lat * Math.PI / 180) / 2 ** z;

function image(url) {
  return new Promise((ok) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => ok(img);
    img.onerror = () => ok(null);
    img.src = url;
  });
}

// Heights round the site at zoom z, (2R+1)² tiles: a function (east, south) -> metres above sea.
async function heights(lat, lon, z, R) {
  const [fx, fy] = tileXY(lat, lon, z);
  const cx = Math.floor(fx), cy = Math.floor(fy);
  const n = 2 * R + 1;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256 * n;
  const g = canvas.getContext('2d', { willReadFrequently: true });
  const got = await Promise.all(Array.from({ length: n * n }, (_, k) => {
    const dx = (k % n) - R, dy = Math.floor(k / n) - R;
    return image(TERRAIN(z, cx + dx, cy + dy)).then((img) => { if (img) g.drawImage(img, (dx + R) * 256, (dy + R) * 256); return !!img; });
  }));
  if (!got.some(Boolean)) return null;
  const px = g.getImageData(0, 0, 256 * n, 256 * n).data;
  const m = metresPerTile(lat, z);
  const W = 256 * n;
  const at = (i, j) => {
    i = clamp(i, 0, W - 1); j = clamp(j, 0, W - 1);
    const k = (j * W + i) * 4;
    return px[k] * 256 + px[k + 1] + px[k + 2] / 256 - 32768;
  };
  return (east, south) => {
    const u = (fx + east / m - (cx - R)) * 256 - 0.5;
    const v = (fy + south / m - (cy - R)) * 256 - 0.5;
    const i = Math.floor(u), j = Math.floor(v), a = u - i, b = v - j;
    return lerp(lerp(at(i, j), at(i + 1, j), a), lerp(at(i, j + 1), at(i + 1, j + 1), a), b);
  };
}

// One ring of photographed ground: (2R+1)² imagery tiles at zoom z, each a grid lifted to its
// heights. `hole` ({w, e, n, s} in metres east and south) is the area a finer ring covers: the ground
// here goes under it there, so the coarser heights never show through the finer ground.
async function groundRing(scene, lat, lon, z, R, height, lift, hole, seg, onTile) {
  const [fx, fy] = tileXY(lat, lon, z);
  const cx = Math.floor(fx), cy = Math.floor(fy);
  const m = metresPerTile(lat, z);
  const loader = new THREE.TextureLoader();
  loader.setCrossOrigin('anonymous');
  const jobs = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const x0 = (cx + dx - fx) * m, y0 = (cy + dy - fy) * m;          // the tile's north-west corner, east/south
    if (hole && x0 >= hole.w && x0 + m <= hole.e && y0 >= hole.n && y0 + m <= hole.s) continue;
    jobs.push(new Promise((ok) => {
      const made = (tex) => {
        if (tex) { tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8; }
        const geo = new THREE.PlaneGeometry(m, m, seg, seg);
        geo.rotateX(-Math.PI / 2);                                     // lies in x (east), z (south)
        const p = geo.attributes.position;
        // the photograph says where the sea is, better than coarse heights along a coast
        let px = null;
        if (tex && tex.image) {
          try {
            const c = document.createElement('canvas'); c.width = c.height = 64;
            const g2 = c.getContext('2d', { willReadFrequently: true });
            g2.drawImage(tex.image, 0, 0, 64, 64);
            px = g2.getImageData(0, 0, 64, 64).data;
          } catch (err) { px = null; }
        }
        const wet = (e, s) => {
          if (!px) return null;
          const i = clamp(Math.floor((e - x0) / m * 64), 0, 63), j = clamp(Math.floor((s - y0) / m * 64), 0, 63);
          const k = (j * 64 + i) * 4, r = px[k], g = px[k + 1], b = px[k + 2];
          return b > r + 18 && g > r + 12;                            // sea is blue or teal, well clear of red; roads, roofs and quays are grey, land beige or green
        };
        for (let i = 0; i < p.count; i++) {
          const e = p.getX(i) + x0 + m / 2, s = p.getZ(i) + y0 + m / 2;
          const pad = m / seg;
          const under = hole && e > hole.w + pad && e < hole.e - pad && s > hole.n + pad && s < hole.s - pad;
          p.setXYZ(i, e, under ? -60 : lift(height(e, s), wet(e, s)), s);
        }
        geo.computeVertexNormals();
        // without the photograph (the imagery server unreachable), the ground keeps its shape in a dry earth colour
        const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial(tex ? { map: tex, roughness: 0.95, metalness: 0 } : { color: 0xa89c84, roughness: 1 }));
        mesh.receiveShadow = true;
        scene.add(mesh);
        onTile && onTile();
        ok(!!tex);
      };
      loader.load(IMAGERY(z, cx + dx, cy + dy), made, undefined, () => made(null));
    }));
  }
  return Promise.all(jobs);
}

// ---- materials ------------------------------------------------------------------------------

function noiseTexture(base, spread, speck, size = 256, seed = 1) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  let s = seed * 9301 + 49297;
  const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
  // a few octaves of value noise, tileable, for mottling; specks for aggregate
  const oct = [8, 16, 32, 64].map((n) => Array.from({ length: n * n }, rnd));
  const vn = (o, n, x, y) => {
    const fx = x / size * n, fy = y / size * n;
    const i = Math.floor(fx), j = Math.floor(fy), a = ease(fx - i), b = ease(fy - j);
    const q = (u, v) => o[((v % n) + n) % n * n + ((u % n) + n) % n];
    return lerp(lerp(q(i, j), q(i + 1, j), a), lerp(q(i, j + 1), q(i + 1, j + 1), a), b);
  };
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let v = 0;
    [8, 16, 32, 64].forEach((n, k) => { v += (vn(oct[k], n, x, y) - 0.5) / (k + 1); });
    const sp = rnd() < speck ? (rnd() - 0.5) * 0.35 : 0;
    const f = 1 + v * spread + sp;
    const k = (y * size + x) * 4;
    img.data[k] = clamp(base[0] * f, 0, 255); img.data[k + 1] = clamp(base[1] * f, 0, 255); img.data[k + 2] = clamp(base[2] * f, 0, 255); img.data[k + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function plywoodTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#b5763a'; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 90; i++) {
    g.strokeStyle = `rgba(${90 + Math.random() * 40},${50 + Math.random() * 20},20,${0.08 + Math.random() * 0.12})`;
    g.lineWidth = 1 + Math.random() * 3;
    const y = Math.random() * 256;
    g.beginPath(); g.moveTo(0, y); g.bezierCurveTo(80, y + 6, 170, y - 6, 256, y + 2); g.stroke();
  }
  g.strokeStyle = 'rgba(40,25,10,0.5)'; g.lineWidth = 2;
  for (const x of [0, 128]) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, 256); g.stroke(); }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function waterNormals() {
  const size = 256;
  const data = new Uint8Array(size * size * 4);
  const waves = [[3, 1, 0.9], [1, 4, 0.7], [7, 3, 0.35], [5, -6, 0.3], [11, 9, 0.18], [-13, 7, 0.15], [17, -19, 0.08]];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let dx = 0, dy = 0;
    for (const [kx, ky, a] of waves) {
      const p = 2 * Math.PI * (kx * x + ky * y) / size;
      dx += a * kx * Math.cos(p); dy += a * ky * Math.cos(p);
    }
    const n = new THREE.Vector3(-dx * 0.04, -dy * 0.04, 1).normalize();
    const i = (y * size + x) * 4;
    data[i] = (n.x * 0.5 + 0.5) * 255; data[i + 1] = (n.y * 0.5 + 0.5) * 255; data[i + 2] = (n.z * 0.5 + 0.5) * 255; data[i + 3] = 255;
  }
  const t = new THREE.DataTexture(data, size, size);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.needsUpdate = true;
  return t;
}

function materials() {
  const tex = {
    concrete: noiseTexture([172, 168, 160], 0.22, 0.06, 256, 3),
    old: noiseTexture([150, 146, 136], 0.45, 0.1, 256, 5),
    rock: noiseTexture([150, 140, 122], 0.9, 0.25, 256, 7),
    ply: plywoodTexture(),
  };
  const std = (o) => new THREE.MeshStandardMaterial(Object.assign({ roughness: 0.9, metalness: 0 }, o));
  return {
    concrete: () => std({ map: tex.concrete, roughness: 0.88 }),
    old: std({ map: tex.old, roughness: 0.95 }),
    rock: std({ map: tex.rock, roughness: 1, flatShading: true }),
    form: std({ map: tex.ply, roughness: 0.8 }),
    steel: std({ color: 0x5b6067, roughness: 0.5, metalness: 0.6 }),
    yellow: std({ color: 0xe9a90f, roughness: 0.55, metalness: 0.2 }),
    dark: std({ color: 0x24272b, roughness: 0.7 }),
    white: std({ color: 0xeeeeea, roughness: 0.6 }),
    glass: std({ color: 0x24323d, roughness: 0.15, metalness: 0.5 }),
    hull: std({ color: 0x8c2a22, roughness: 0.7 }),
    deck: std({ color: 0x6f7378, roughness: 0.8 }),
    vest: std({ color: 0xff7a00, roughness: 0.8 }),
    skin: std({ color: 0x8a5a3c, roughness: 0.8 }),
    blue: std({ color: 0x25476d, roughness: 0.8 }),
    rebar: std({ color: 0x6b3d24, roughness: 0.7, metalness: 0.4 }),
    none: new THREE.MeshBasicMaterial({ visible: false }),
  };
}

// UVs by projection on each face's main axis, so textures sit at a true size on any shape.
function projectUV(geo, metres) {
  const p = geo.attributes.position, n = geo.attributes.normal;
  const uv = new Float32Array(p.count * 2);
  for (let i = 0; i < p.count; i++) {
    const ax = Math.abs(n.getX(i)), ay = Math.abs(n.getY(i)), az = Math.abs(n.getZ(i));
    let u, v;
    if (ay >= ax && ay >= az) { u = p.getX(i); v = p.getZ(i); } else if (ax >= az) { u = p.getZ(i); v = p.getY(i); } else { u = p.getX(i); v = p.getY(i); }
    uv[i * 2] = u / metres; uv[i * 2 + 1] = v / metres;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
}

// ---- plant and people -----------------------------------------------------------------------

function box(w, h, d, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = m.receiveShadow = true;
  return m;
}

// A lattice boom along +x from its foot: four chords and zig-zag lacing.
function lattice(len, w, mat) {
  const g = new THREE.Group();
  const chord = new THREE.CylinderGeometry(0.09, 0.09, len, 5);
  chord.rotateZ(Math.PI / 2);
  for (const [y, z] of [[-w / 2, -w / 2], [-w / 2, w / 2], [w / 2, -w / 2], [w / 2, w / 2]]) {
    const c = new THREE.Mesh(chord, mat); c.position.set(len / 2, y, z); c.castShadow = true; g.add(c);
  }
  const bays = Math.round(len / w);
  const lace = new THREE.CylinderGeometry(0.04, 0.04, Math.hypot(len / bays, w), 4);
  const pts = [];
  for (let k = 0; k < bays; k++) {
    const x = (k + 0.5) * len / bays;
    const a = Math.atan2(w, len / bays) * (k % 2 ? 1 : -1);
    for (const [y, z, rot] of [[-w / 2, 0, 'y'], [w / 2, 0, 'y'], [0, -w / 2, 'z'], [0, w / 2, 'z']]) {
      const m = new THREE.Mesh(lace, mat);
      m.position.set(x, y, z);
      if (rot === 'y') m.rotation.set(Math.PI / 2, 0, Math.PI / 2 + a); else m.rotation.set(0, 0, Math.PI / 2 + a);
      pts.push(m);
    }
  }
  for (const m of pts) g.add(m);
  return g;
}

function crawlerCrane(M, boomLen) {
  const g = new THREE.Group();
  for (const z of [-2.3, 2.3]) {
    g.add(box(8.5, 1.2, 1.1, M.dark, 0, 0.6, z));
    for (let x = -3.6; x <= 3.6; x += 1.2) g.add(box(0.12, 1.22, 1.14, M.steel, x, 0.6, z));
  }
  g.add(box(3, 0.8, 4.2, M.dark, 0, 1.4, 0));
  const top = new THREE.Group();
  top.position.y = 1.8;
  g.add(top);
  top.add(box(7, 2.2, 3.6, M.yellow, -1.2, 1.1, 0));
  top.add(box(2.4, 2.4, 3.4, M.dark, -4.8, 1.3, 0));                // counterweight
  top.add(box(2.0, 2.0, 1.3, M.yellow, 2.2, 1.6, 1.3));             // cab
  top.add(box(1.6, 1.3, 0.05, M.glass, 3.2, 1.9, 1.96));
  const mastA = lattice(9, 0.6, M.yellow);
  mastA.position.set(-2.5, 2.2, 0); mastA.rotation.z = 1.25;
  top.add(mastA);
  const boom = new THREE.Group();
  boom.position.set(1.8, 1.6, 0);
  top.add(boom);
  const b = lattice(boomLen, 1.4, M.yellow);
  boom.add(b);
  boom.add(box(1.6, 1.2, 1.4, M.yellow, boomLen + 0.4, 0, 0));
  const tip = new THREE.Object3D();
  tip.position.set(boomLen + 0.8, -0.4, 0);
  boom.add(tip);
  return { g, top, boom, tip, boomLen };
}

function excavator(M) {
  const g = new THREE.Group();
  for (const z of [-1.2, 1.2]) g.add(box(4.2, 0.9, 0.6, M.dark, 0, 0.45, z));
  const top = new THREE.Group(); top.position.y = 1; g.add(top);
  top.add(box(3.2, 1.3, 2.5, M.yellow, -0.4, 0.75, 0));
  top.add(box(1.1, 1.3, 1.0, M.glass, 0.9, 1.6, 0.7));
  const arm = new THREE.Group(); arm.position.set(1.1, 1.0, -0.4); top.add(arm);
  const boom = box(4, 0.45, 0.45, M.yellow, 2, 0, 0); arm.add(boom); arm.rotation.z = 0.5;
  const stick = new THREE.Group(); stick.position.set(4, 0, 0); arm.add(stick);
  stick.add(box(3, 0.35, 0.35, M.yellow, 1.5, 0, 0)); stick.rotation.z = -1.6;
  stick.add(box(0.9, 0.3, 0.3, M.steel, 3.2, 0, 0));                 // the breaker
  return { g, top, arm, stick };
}

function pileBarge(M) {
  const g = new THREE.Group();
  g.add(box(24, 1.8, 10, M.deck, 0, 0.2, 0));
  g.add(box(4, 3, 4, M.white, -9, 2.6, -3));
  const mast = box(0.8, 26, 0.8, M.yellow, 6, 14, 0); g.add(mast);
  g.add(box(3, 2, 3, M.yellow, 6, 2.2, 0));
  const hammer = box(1.4, 4, 1.4, M.dark, 6, 20, 0); g.add(hammer);
  for (const [x, z] of [[-11, -4.5], [-11, 4.5], [11, -4.5], [11, 4.5]]) g.add(box(0.6, 14, 0.6, M.steel, x, 4, z));   // spuds
  return { g, hammer, mastX: 6 };
}

function workboat(M) {
  const g = new THREE.Group();
  const hull = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 1.3, 9, 8, 1), M.white);
  hull.rotation.z = Math.PI / 2; hull.scale.set(1, 1, 0.55); hull.position.y = 0.4; hull.castShadow = true;
  g.add(hull);
  g.add(box(2.6, 1.6, 2.2, M.white, -1, 1.6, 0));
  g.add(box(2.5, 0.5, 2.3, M.blue, -1, 2.6, 0));
  return { g };
}

function pumpTruck(M) {
  const g = new THREE.Group();
  g.add(box(9, 1.0, 2.5, M.dark, 0, 1.0, 0));
  g.add(box(2.2, 2.0, 2.5, M.white, 3.8, 2.4, 0));
  g.add(box(5.5, 1.4, 2.3, M.white, -1, 2.2, 0));
  for (const x of [3.4, -0.6, -2.0, -3.4]) for (const z of [-1.2, 1.2]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.4, 12), M.dark); w.rotation.x = Math.PI / 2; w.position.set(x, 0.5, z); g.add(w);
  }
  const boom = [];
  let parent = g;
  let base = new THREE.Vector3(-1, 3.2, 0);
  for (const len of [10, 9, 8]) {
    const j = new THREE.Group(); j.position.copy(base); parent.add(j);
    j.add(box(len, 0.4, 0.4, M.white, len / 2, 0, 0));
    boom.push(j); parent = j; base = new THREE.Vector3(len, 0, 0);
  }
  const hose = box(0.25, 3, 0.25, M.dark, 8, -1.5, 0); boom[2].add(hose);
  return { g, boom };
}

function mixer(M) {
  const g = new THREE.Group();
  g.add(box(8, 1.0, 2.4, M.dark, 0, 1.0, 0));
  g.add(box(2.1, 2.0, 2.4, M.white, 3.0, 2.4, 0));
  const drum = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.3, 4.8, 14), M.white);
  drum.rotation.z = Math.PI / 2 + 0.2; drum.position.set(-1.0, 2.8, 0); drum.castShadow = true;
  g.add(drum);
  for (const x of [2.8, -1.5, -2.9]) for (const z of [-1.15, 1.15]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.4, 12), M.dark); w.rotation.x = Math.PI / 2; w.position.set(x, 0.5, z); g.add(w);
  }
  return { g, drum };
}

function worker(M, rng) {
  const g = new THREE.Group();
  const legs = box(0.34, 0.85, 0.24, M.blue, 0, 0.43, 0);
  const body = box(0.46, 0.65, 0.28, M.vest, 0, 1.18, 0);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 10, 8), M.skin); head.position.y = 1.62;
  const hat = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 6, 0, Math.PI * 2, 0, Math.PI / 2), M.white); hat.position.y = 1.66;
  g.add(legs, body, head, hat);
  g.rotation.y = rng() * Math.PI * 2;
  return g;
}

// ---- the film -------------------------------------------------------------------------------

function mulberry(seed) {
  return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

// How a model without the planner's stages is built: by the kind of work.
const HOW_OF_PHASE = { 'Foundations': 'cast', 'Vertical structure': 'cast', 'Horizontal structure': 'cast', 'Floors': 'cast',
  'Stairs and ramps': 'lift', 'Roof': 'lift', 'Envelope': 'fit', 'Railings and finishes': 'fit', 'Services': 'fit', 'Other': 'fit' };

export function open(el, ui) {
  const msg = el.querySelector('[data-film-msg]');
  const say = (html) => { msg.innerHTML = html; msg.classList.remove('hidden'); };
  say('<span>Opening the model…</span><div class="bar"><div class="bar-fill" style="width:2%"></div></div>');
  const bar = (p) => { const b = msg.querySelector('.bar-fill'); if (b) b.style.width = p + '%'; };
  const word = (t) => { const s = msg.querySelector('span'); if (s) s.textContent = t; };

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.55;
  el.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.5, 60000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.48;
  const M = materials();
  const rng = mulberry(7);
  const hud = el.querySelector('[data-film-hud]');

  const size = () => {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(size).observe(el);
  size();

  const lat = parseFloat(el.dataset.lat), lon = parseFloat(el.dataset.lon);
  const title = el.dataset.title || 'SiteTwin';
  const els = [];                      // {id, mesh, step, how, box, centre, base, top, a, b, ...}
  let film = null;                     // the plan
  let origin = [0, 0, 0];

  fetch(el.dataset.steps, { credentials: 'same-origin' }).then((r) => r.json()).then((data) => {
    const byId = {};
    for (const [step, ids] of Object.entries(data.ids)) for (const id of ids) byId[id] = Number(step);
    const worker = new Worker(new URL('sitetwin-ifc-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = async (ev) => {
      const m = ev.data;
      if (m.type === 'progress') {
        if (m.stage === 'download') { bar(2 + 50 * m.loaded / m.total); word(`Downloading the model… ${(m.loaded / 1048576).toFixed(1)} MB`); }
        else if (m.stage === 'open') word('Reading its shapes…');
        else if (m.stage === 'shapes') bar(52 + 28 * Math.min(1, m.done / m.total));
      } else if (m.type === 'part') {
        if (!m.positions.length) return;
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(m.positions, 3));
        g.setAttribute('normal', new THREE.BufferAttribute(m.normals, 3));
        g.computeBoundingBox();
        els.push({ id: m.id, step: m.step, geo: g });
      } else if (m.type === 'done') {
        worker.terminate();
        origin = m.origin || [0, 0, 0];
        try { await build(data); } catch (err) { say(`<span>The film could not be set up: ${err.message}</span>`); console.error(err); }
      } else if (m.type === 'error') {
        say(`<span>The model could not be read here: ${m.message}.</span>`);
      }
    };
    worker.postMessage({ url: el.dataset.model, base: el.dataset.webIfc, steps: byId, each: true });
  }).catch((err) => say(`<span>The film could not open: ${err.message}</span>`));

  async function build(data) {
    const info = data.film || {};
    const marine = info.setting === 'marine';
    const sea = Number(info.sea_level || 0);
    const steps = data.steps || [];
    // each element's way of coming and going, and its stage
    const stageOf = (e) => steps[e.step - 1] || null;
    for (const e of els) {
      const s = stageOf(e);
      e.how = (info.how && info.how[e.id]) || (s ? (s.stage ? (info.stages || []).find((x) => x.code === s.stage)?.how : HOW_OF_PHASE[s.phase]) : 'exist') || 'fit';
      if (!s) e.how = 'exist';
      e.code = s ? (s.stage || String(e.step)) : '';
      e.box = e.geo.boundingBox.clone();
      e.centre = e.box.getCenter(new THREE.Vector3());
      e.size = e.box.getSize(new THREE.Vector3());
    }
    const built = els.filter((e) => e.how !== 'ground' && e.how !== 'exist');
    const all = new THREE.Box3();
    for (const e of built) all.union(e.box);
    if (all.isEmpty()) { say('<span>The model has no shapes to show.</span>'); return; }
    const C = all.getCenter(new THREE.Vector3()), S = all.getSize(new THREE.Vector3());
    const L = Math.max(S.x, S.z, 30);
    const alongX = S.x >= S.z;
    const along = (p) => (alongX ? p.x : p.z);
    const across = (p) => (alongX ? p.z : p.x);

    // -- the world round it
    word('Fetching the real ground and the aerial photograph…'); bar(82);
    const sunDir = new THREE.Vector3();
    const elev = THREE.MathUtils.degToRad(36), az = THREE.MathUtils.degToRad(215);   // a mid-afternoon sun from the south-west
    sunDir.setFromSphericalCoords(1, Math.PI / 2 - elev, az);
    const sky = new Sky();
    sky.scale.setScalar(50000);
    const su = sky.material.uniforms;
    su.turbidity.value = 2.2; su.rayleigh.value = 1.6; su.mieCoefficient.value = 0.004; su.mieDirectionalG.value = 0.8;
    su.sunPosition.value.copy(sunDir);
    scene.add(sky);
    const pmrem = new THREE.PMREMGenerator(renderer);
    const skyScene = new THREE.Scene(); const sky2 = new Sky(); sky2.scale.setScalar(50000); sky2.material.uniforms.sunPosition.value.copy(sunDir);
    Object.assign(sky2.material.uniforms.turbidity, { value: 2.2 }); sky2.material.uniforms.rayleigh.value = 1.6; skyScene.add(sky2);
    scene.environment = pmrem.fromScene(skyScene).texture;
    scene.fog = new THREE.Fog(0xc6d3dc, L * 12, 26000);

    const sun = new THREE.DirectionalLight(0xfff1dc, 3.2);
    sun.position.copy(C).addScaledVector(sunDir, L * 3);
    sun.target.position.copy(C);
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -L * 0.9; sc.right = sc.top = L * 0.9; sc.near = 1; sc.far = L * 8;
    sun.shadow.bias = -0.0004;
    scene.add(sun, sun.target, new THREE.HemisphereLight(0xcfe0f0, 0x6a5e4a, 0.55));

    // the surveyed point (latitude, longitude) is the model's origin; the worker moved the model by `origin`
    const ox = -origin[0], oz = -origin[2];
    let groundAt = null;
    if (isFinite(lat) && isFinite(lon)) {
      const [hNear, hFar] = await Promise.all([heights(lat, lon, 14, 2), heights(lat, lon, 11, 3)]);
      if (hNear || hFar) {
        const siteH = (hNear || hFar)(C.x - ox, C.z - oz);
        // the land in the model's levels: above sea in metres + mean sea level (marine); on land
        // the ground at the site is the model's zero
        // at sea, ground the photograph shows as water (or, without it, anything below 2.5 m: elevation
        // data along a coast is coarse) goes under the sea; ground it shows as dry stays at least 1.5 m up
        const lift = marine ? (h, wet) => (wet === true || (wet == null && h < 2.5) ? sea - 8 : Math.max(h, 1.5) + sea) : (h) => h - siteH;
        const hn = (e, s) => (hNear ? hNear(e, s) : hFar(e, s));
        groundAt = (x, z) => lift(hn(x - ox, z - oz));
        const place = (g) => { g.position.set(ox, 0, oz); };
        const near = new THREE.Group(); place(near); scene.add(near);
        const far = new THREE.Group(); place(far); scene.add(far);
        let tiles = 0;
        // the area the near ring covers, for the far ring to go under
        const [f17x, f17y] = tileXY(lat, lon, 17), m17 = metresPerTile(lat, 17);
        const nearBox = { w: (Math.floor(f17x) - 4 - f17x) * m17, e: (Math.floor(f17x) + 5 - f17x) * m17, n: (Math.floor(f17y) - 4 - f17y) * m17, s: (Math.floor(f17y) + 5 - f17y) * m17 };
        await Promise.all([
          groundRing(near, lat, lon, 17, 4, hn, lift, 0, 24, () => bar(82 + Math.min(14, ++tiles / 10))),
          groundRing(far, lat, lon, 14, 4, hFar || hn, (h, wet) => lift(h, wet) - 1.2, nearBox, 32),
        ]);
      }
    }
    if (marine) {
      const ripples = waterNormals();
      ripples.repeat.set(900, 900);
      const water = new THREE.Mesh(new THREE.PlaneGeometry(60000, 60000), new THREE.MeshPhysicalMaterial({
        color: 0x0b3448, roughness: 0.16, metalness: 0, normalMap: ripples, normalScale: new THREE.Vector2(0.35, 0.35),
        envMapIntensity: 0.55, clearcoat: 0.3, clearcoatRoughness: 0.2,
      }));
      water.rotation.x = -Math.PI / 2;
      water.position.set(C.x, sea, C.z);
      water.receiveShadow = true;
      scene.add(water);
      film_water = water;
    }

    // -- the model
    const mats = {};
    for (const e of els) {
      projectUV(e.geo, 2.5);
      let mat;
      if (e.how === 'ground') mat = M.rock;
      else if (e.how === 'remove' || e.how === 'repair' || e.how === 'exist') mat = M.old;
      else if (e.how === 'fill') mat = M.rock;
      else mat = M.concrete();
      e.mat = mat;
      e.mesh = new THREE.Mesh(e.geo, mat);
      e.mesh.castShadow = e.mesh.receiveShadow = true;
      e.baseY = e.box.min.y;
      scene.add(e.mesh);
    }

    // -- the plan: stages in order, each its share by its days, the work along from the root
    const stages = [];
    const codes = [];
    for (const s of steps) if (s && !codes.includes(s.stage || String(s.step))) codes.push(s.stage || String(s.step));
    const meta = Object.fromEntries((info.stages || []).map((s) => [s.code, s]));
    for (const code of codes) {
      const mine = els.filter((e) => e.code === code && e.how !== 'ground' && e.how !== 'exist');
      if (!mine.length) continue;
      const st = steps.find((s) => (s.stage || String(s.step)) === code);
      stages.push({ code, name: (meta[code] && meta[code].name) || (st && st.phase) || code, days: Number((meta[code] && meta[code].days) || 0), els: mine });
    }
    const allDays = stages.every((s) => s.days > 0);
    for (const s of stages) if (!allDays) s.days = Math.max(5, s.els.length);
    const totalDays = stages.reduce((n, s) => n + s.days, 0);
    // the root: the end of the long axis towards the land (higher ground), else where the fill is
    const ends = [all.min, all.max].map((p) => (alongX ? new THREE.Vector3(p.x, 0, C.z) : new THREE.Vector3(C.x, 0, p.z)));
    let rootEnd = 0;
    if (groundAt) rootEnd = groundAt(ends[1].x, ends[1].z) > groundAt(ends[0].x, ends[0].z) ? 1 : 0;
    const fills = els.filter((e) => e.how === 'fill');
    const fillBox = new THREE.Box3(); for (const e of fills) fillBox.union(e.box);
    if (!groundAt && fills.length) { const fc = fillBox.getCenter(new THREE.Vector3()); rootEnd = Math.abs(along(fc) - along(all.max)) < Math.abs(along(fc) - along(all.min)) ? 1 : 0; }
    const fromRoot = (p) => (rootEnd ? along(all.max) - along(p) : along(p) - along(all.min));
    // the land side, across the long axis: where the fill is, else the higher ground beside it
    let landSide = 1;
    if (fills.length) landSide = across(fillBox.getCenter(new THREE.Vector3())) < across(C) ? -1 : 1;
    else if (groundAt) {
      const o = (k) => (alongX ? groundAt(C.x, C.z + k * (S.z / 2 + 30)) : groundAt(C.x + k * (S.x / 2 + 30), C.z));
      landSide = o(-1) > o(1) ? -1 : 1;
    }

    film = { stages, totalDays, seconds: 120, rootEnd, landSide, along, across, fromRoot, all, C, S, L, alongX, sea, marine, fillBox, fills };
    film.schedule = (seconds) => {
      film.seconds = seconds;
      const span = seconds - INTRO - OUTRO;
      let t = INTRO;
      for (const s of stages) {
        s.a = t; s.b = t + span * s.days / totalDays; t = s.b;
        const order = s.els.slice().sort((p, q) => (Math.floor(fromRoot(p.centre) / 3) - Math.floor(fromRoot(q.centre) / 3)) || (across(p.centre) - across(q.centre)) || (p.centre.y - q.centre.y));
        const n = order.length, w = s.b - s.a;
        const how = order[0].how;
        const k = how === 'cast' ? 1.8 : how === 'lift' ? 3 : how === 'remove' ? 2.5 : 2;
        const dur = Math.min(w, Math.max(how === 'fill' ? w * 0.6 : 0.6, k * w / n));
        order.forEach((e, i) => { e.a = s.a + (n > 1 ? i * (w - dur) / (n - 1) : 0); e.b = e.a + dur; });
        s.order = order;
      }
      film.end = seconds;
    };
    film.schedule(Number(ui.length.value) || 150);

    // -- formwork, jackets and the plant
    for (const e of els) {
      if (e.how === 'cast') {
        const pad = 0.06;
        const f = new THREE.Mesh(new THREE.BoxGeometry(e.size.x + pad * 2, e.size.y, e.size.z + pad * 2), [M.form, M.form, M.none, M.form, M.form, M.form]);
        f.position.copy(e.centre); f.castShadow = f.receiveShadow = true; f.visible = false;
        scene.add(f); e.form = f;
      } else if (e.how === 'repair') {
        const top = e.box.max.y - 0.15, bot = Math.max(e.box.min.y, sea - 1.5);
        if (top - bot > 0.5) {
          const j = box(e.size.x + 0.3, top - bot, e.size.z + 0.3, M.form, e.centre.x, (top + bot) / 2, e.centre.z);
          j.visible = false; scene.add(j); e.jacket = j;
        }
      }
    }
    const landX = (a) => (alongX ? a : film.craneAcross);
    const landZ = (a) => (alongX ? film.craneAcross : a);
    // the crane works from the land side: on the fill, or on the ground beside the root
    film.craneAcross = fills.length ? across(fillBox.getCenter(new THREE.Vector3())) : across(C) + landSide * (alongX ? S.z : S.x) * 0.5;
    film.craneY = fills.length ? fillBox.max.y : (groundAt ? groundAt(C.x, C.z) : 0);
    const alongMin = fills.length ? along(fillBox.min) + 8 : along(all.min), alongMax = fills.length ? along(fillBox.max) - 8 : along(all.max);
    const crane = crawlerCrane(M, Math.min(60, Math.max(30, L * 0.25)));
    scene.add(crane.g);
    // beyond the crane's reach from the land, it works from a crane barge on the sea side
    const craneBarge = box(32, 2.2, 16, M.deck); craneBarge.visible = false; scene.add(craneBarge);
    const halfWidth = (alongX ? S.z : S.x) / 2;
    const waterAcross = across(C) + seaSide0() * (halfWidth + 11);
    function seaSide0() { return -landSide; }
    const hook = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: 0x222222 }));
    scene.add(hook);
    const hookBlock = box(0.7, 0.9, 0.7, M.yellow); scene.add(hookBlock);
    const exc = excavator(M); scene.add(exc.g);
    const barge = pileBarge(M); scene.add(barge.g);
    const boat = workboat(M); scene.add(boat.g);
    const pump = pumpTruck(M); scene.add(pump.g);
    const mix = mixer(M); scene.add(mix.g);
    const crew = Array.from({ length: 8 }, () => { const w = worker(M, rng); scene.add(w); return w; });
    const craneAt = { a: rootEnd ? alongMax : alongMin, slew: 0 };
    const yOn = (x, z, fallback) => {
      // the top of what is built under (x, z) at this moment, else the ground
      let best = -Infinity;
      for (const e of els) {
        if (!e.mesh.visible || e.how === 'remove' && e.gone) continue;
        const b = e.box;
        if (x >= b.min.x && x <= b.max.x && z >= b.min.z && z <= b.max.z && b.max.y > best && b.max.y < film.all.max.y + 1) best = b.max.y;
      }
      return best > -Infinity ? best : (groundAt ? Math.max(groundAt(x, z), sea) : fallback);
    };

    // -- the camera: on the water (or the open side) off the long face, looking back across it
    const seaSide = -landSide;
    const camHome = () => {
      const p = C.clone();
      const toHead = rootEnd ? -1 : 1;
      if (alongX) { p.z += seaSide * L * 0.85; p.x += toHead * L * 0.95; } else { p.x += seaSide * L * 0.85; p.z += toHead * L * 0.95; }
      p.y = all.max.y + L * 0.42;
      return p;
    };
    camera.position.copy(camHome());
    controls.target.copy(C);
    camera.far = 60000; camera.updateProjectionMatrix();

    // -- one moment of the film
    const startDate = (() => { const d = new Date(); d.setDate(d.getDate() + ((8 - d.getDay()) % 7 || 7)); d.setHours(0, 0, 0, 0); return d; })();
    let lastFront = C.clone();
    function stageAt(t) { return stages.find((s) => t >= s.a && t < s.b) || null; }
    function dayAt(t) {
      let d = 0;
      for (const s of stages) { if (t >= s.b) d += s.days; else if (t > s.a) d += s.days * (t - s.a) / (s.b - s.a); }
      return d;
    }
    function show(t) {
      const s = stageAt(t);
      const active = [];
      for (const e of els) {
        const m = e.mesh;
        m.position.set(0, 0, 0); m.scale.set(1, 1, 1);
        if (e.form) e.form.visible = false;
        if (e.jacket) e.jacket.visible = false;
        if (e.how === 'ground' || e.how === 'exist') { m.visible = true; continue; }
        const u = (t - e.a) / (e.b - e.a);
        if (u > 0 && u < 1) active.push(e);
        if (e.how === 'remove') {
          e.gone = u >= 1;
          m.visible = u < 1;
          if (u > 0 && u < 1) { const k = 1 - ease(u); m.scale.y = Math.max(0.02, k); m.position.y = e.baseY * (1 - m.scale.y); }
        } else if (e.how === 'repair') {
          m.visible = true;
          if (e.jacket && u > 0) { e.jacket.visible = true; e.jacket.material = u < 0.6 ? M.form : (e.jMat || (e.jMat = M.concrete())); }
        } else if (u <= 0) {
          m.visible = false;
        } else if (e.how === 'drive') {
          m.visible = true;
          m.position.y = (1 - ease(Math.min(1, u * 1.1))) * e.size.y * 0.7;
        } else if (e.how === 'fill') {
          m.visible = true; const k = Math.max(0.02, ease(u)); m.scale.y = k; m.position.y = e.baseY * (1 - k);
        } else if (e.how === 'cast') {
          if (e.form) {
            e.form.visible = u < 0.9;
            const fk = clamp(u / 0.25, 0.02, 1);
            e.form.scale.y = fk; e.form.position.y = e.box.min.y + e.size.y * fk / 2;
          }
          const k = clamp((u - 0.3) / 0.45, 0, 1);
          m.visible = k > 0;
          if (k > 0 && k < 1) { m.scale.y = Math.max(0.02, k); m.position.y = e.baseY * (1 - k); }
          // fresh concrete is dark and wet, and dries lighter
          const dry = clamp((t - (e.a + (e.b - e.a) * 0.75)) / 4, 0, 1);
          m.material.color.setScalar(lerp(0.62, 1, dry));
        } else if (e.how === 'lift') {
          m.visible = true;
          m.position.y = (1 - ease(Math.min(1, u * 1.15))) * 7;
        } else {
          m.visible = true; const k = Math.max(0.02, ease(u)); m.scale.set(k, k, k); m.position.set(e.centre.x * (1 - k), e.centre.y * (1 - k), e.centre.z * (1 - k));
        }
      }
      // where the work is now
      const front = active.length ? active.reduce((p, e) => p.add(e.centre), new THREE.Vector3()).divideScalar(active.length) : lastFront.clone();
      // smooth while playing; a jump in time (seeking) puts everything straight where it belongs
      const jump = Math.abs(t - (film.lastT ?? -99)) > 0.5;
      film.lastT = t;
      if (jump) lastFront.copy(front); else lastFront.lerp(front, 0.15);
      const how = s ? s.order[0].how : null;
      const latest = active.length ? active.reduce((p, e) => (e.a > p.a ? e : p)) : null;   // the piece started last

      // the crawler crane follows the work along the land side, and lifts the latest piece
      const want = clamp(along(lastFront), Math.min(alongMin, alongMax), Math.max(alongMin, alongMax));
      const workAlong = along(lastFront);
      const fromLand = Math.abs(workAlong - clamp(workAlong, Math.min(alongMin, alongMax), Math.max(alongMin, alongMax)));
      const afloat = marine && fromLand > crane.boomLen * 0.55;
      if (afloat !== craneAt.afloat || jump) { craneAt.afloat = afloat; craneAt.a = afloat ? workAlong : want; }
      if (afloat) {
        craneAt.a += clamp(workAlong - craneAt.a, -0.8, 0.8);
        const bx = alongX ? craneAt.a : waterAcross, bz = alongX ? waterAcross : craneAt.a;
        craneBarge.visible = true;
        craneBarge.position.set(bx, sea + 0.3 + Math.sin(t * 0.9) * 0.05, bz);
        craneBarge.rotation.y = alongX ? 0 : Math.PI / 2;
        crane.g.position.set(bx, sea + 1.4, bz);
      } else {
        craneBarge.visible = false;
        craneAt.a += clamp(want - craneAt.a, -0.6, 0.6);
        crane.g.position.set(landX(craneAt.a), film.craneY, landZ(craneAt.a));
      }
      const aim = (how === 'lift' || how === 'drive') && latest ? latest.centre : lastFront;
      const dx = aim.x - crane.g.position.x, dz = aim.z - crane.g.position.z;
      const slew = Math.atan2(-dz, dx);
      crane.g.rotation.y = 0;
      craneAt.slew += Math.atan2(Math.sin(slew - craneAt.slew), Math.cos(slew - craneAt.slew)) * (jump ? 1 : 0.12);
      crane.top.rotation.y = craneAt.slew;
      const reach = clamp(Math.hypot(dx, dz) - 1.8, crane.boomLen * 0.42, crane.boomLen * 0.92);
      crane.boom.rotation.z = Math.acos(reach / crane.boomLen);
      crane.g.updateMatrixWorld(true);
      const tip = crane.tip.getWorldPosition(new THREE.Vector3());
      const hookY = how === 'lift' && latest ? latest.box.max.y + latest.mesh.position.y + 0.6 : tip.y - 8;
      const hookP = how === 'lift' && latest ? new THREE.Vector3(latest.centre.x, hookY, latest.centre.z) : new THREE.Vector3(tip.x, hookY, tip.z);
      hook.geometry.setFromPoints([tip, hookP]);
      hookBlock.position.copy(hookP).add(new THREE.Vector3(0, 0.5, 0));

      // the breaker on the deck being broken out
      exc.g.visible = how === 'remove';
      if (exc.g.visible) {
        const ax = along(lastFront) + (rootEnd ? 6 : -6);
        exc.g.position.set(landX(ax), film.craneY, landZ(ax));
        exc.top.rotation.y = Math.atan2(-(lastFront.z - exc.g.position.z), lastFront.x - exc.g.position.x);
        exc.arm.rotation.z = 0.45 + Math.sin(t * 6) * 0.08;
      }
      // the piling barge at the pile being driven
      barge.g.visible = how === 'drive' && !!latest;
      if (barge.g.visible) {
        const p = latest.centre;
        const off = seaSide * 6;
        barge.g.position.set(alongX ? p.x - barge.mastX : p.x + off, sea + 0.4, alongX ? p.z + off : p.z - barge.mastX);
        barge.g.rotation.y = alongX ? 0 : Math.PI / 2;
        const pileTop = latest.box.max.y + latest.mesh.position.y;
        barge.hammer.position.y = pileTop - sea + 2;
      }
      // the work boat at the pile being repaired
      boat.g.visible = how === 'repair' && !!latest;
      if (boat.g.visible) {
        const p = latest.centre;
        boat.g.position.set(alongX ? p.x : p.x + seaSide * 5, sea + 0.1 + Math.sin(t * 1.3) * 0.08, alongX ? p.z + seaSide * 5 : p.z);
        boat.g.rotation.y = alongX ? 0 : Math.PI / 2;
      }
      // the pump and a mixer beside the pour
      pump.g.visible = mix.g.visible = how === 'cast' && !!latest;
      if (pump.g.visible) {
        const ax = clamp(along(latest.centre), Math.min(alongMin, alongMax), Math.max(alongMin, alongMax));
        const across0 = film.craneAcross + landSide * 4;
        pump.g.position.set(alongX ? ax : across0, film.craneY, alongX ? across0 : ax);
        pump.g.rotation.y = alongX ? 0 : Math.PI / 2;
        mix.g.position.set(alongX ? ax - 11 : across0, film.craneY, alongX ? across0 : ax - 11);
        mix.g.rotation.y = pump.g.rotation.y;
        mix.drum.rotation.y = t * 3;
        // the boom reaches over the pour
        pump.g.updateMatrixWorld(true);
        const base = pump.boom[0].getWorldPosition(new THREE.Vector3());
        const target = latest.centre.clone(); target.y = latest.box.max.y + 3;
        const local = pump.g.worldToLocal(target.clone());
        const yaw = Math.atan2(-(local.z), local.x + 1);
        pump.boom[0].rotation.set(0, yaw, 0.0);
        const d = Math.hypot(target.x - base.x, target.z - base.z);
        const r = clamp(d, 6, 26);
        pump.boom[0].rotation.z = Math.acos(clamp(r / 27, 0.2, 1)) * 0.9;
        pump.boom[1].rotation.z = -0.5;
        pump.boom[2].rotation.z = -0.6;
      }
      // the crew where the work is, standing on what is there
      crew.forEach((w, i) => {
        w.visible = !!s && how !== 'drive';
        if (!w.visible) return;
        const ang = i * 2.399 + t * 0.05;
        const rad = 3 + (i % 4) * 2.2;
        let x = lastFront.x + Math.cos(ang) * rad, z = lastFront.z + Math.sin(ang) * rad;
        const y = yOn(x, z, film.craneY);
        if (marine && y <= sea + 0.3) { x = crane.g.position.x + Math.cos(ang) * 6; z = crane.g.position.z + Math.sin(ang) * 6; w.position.set(x, film.craneY, z); return; }
        w.position.set(x, y, z);
      });

      // the camera, held on the water, drifting a little with the work
      if (!controls.enabled) {
        const home = camHome();
        const drift = Math.sin(t / film.end * Math.PI * 2) * 0.05;
        const p = home.clone().sub(C).applyAxisAngle(new THREE.Vector3(0, 1, 0), drift).add(C);
        camera.position.copy(p);
        const look = C.clone().lerp(lastFront, 0.08); look.y = sea;
        if (alongX) look.x -= (rootEnd ? -1 : 1) * L * 0.15; else look.z -= (rootEnd ? -1 : 1) * L * 0.15;
        camera.lookAt(look);
      }

      // the captions
      const d = new Date(startDate.getTime() + dayAt(t) * 7 / 5 * 86400000);
      const wk = Math.floor(dayAt(t) / 5) + 1;
      film.caption = {
        title,
        when: t < INTRO ? 'Today' : `Week ${wk} · ${MONTHS[d.getMonth()]} ${d.getFullYear()}`,
        what: t < INTRO ? 'As it is today' : s ? `Stage ${s.code} · ${s.name}` : t >= stages[stages.length - 1].b ? 'Complete' : '',
        progress: clamp(t / film.end, 0, 1),
      };
      hud.innerHTML = `<b>${film.caption.title}</b><span class="when">${film.caption.when}</span><span class="what">${film.caption.what}</span><span class="track"><span style="width:${(film.caption.progress * 100).toFixed(1)}%"></span></span>`;
    }
    film.show = show;

    msg.classList.add('hidden');
    hud.classList.remove('hidden');
    for (const b of [ui.play, ui.record, ui.length, ui.scrub]) b.disabled = false;
    ui.scrub.max = 1000;
    show(0);
    controls.enabled = false;
    show(INTRO + 0.01); show(0);
    ready();
  }

  // ---- playing, scrubbing and recording ----
  let film_water = null;
  let t = 0, playing = false, last = 0, recording = null;
  function ready() {
    ui.play.addEventListener('click', () => (playing ? pause() : play()));
    ui.length.addEventListener('change', () => { film.schedule(Number(ui.length.value)); t = 0; film.show(0); ui.scrub.value = 0; });
    ui.scrub.addEventListener('input', () => { t = ui.scrub.value / 1000 * film.end; controls.enabled = false; film.show(t); });
    ui.record.addEventListener('click', () => (recording ? stopRecording() : startRecording()));
    renderer.domElement.addEventListener('pointerdown', () => { if (!playing) controls.enabled = true; });
    requestAnimationFrame(loop);
  }
  function play() {
    if (t >= film.end) t = 0;
    playing = true; controls.enabled = false; last = performance.now();
    ui.play.textContent = '❚❚ Pause';
  }
  function pause() {
    playing = false; ui.play.textContent = '▶ Play';
    if (recording) stopRecording();
  }
  function loop(now) {
    requestAnimationFrame(loop);
    if (!film) return;
    if (playing) {
      t += Math.min(0.1, (now - last) / 1000);
      last = now;
      if (t >= film.end) { t = film.end; film.show(t); render(); pause(); ui.play.textContent = '▶ Play again'; return; }
      film.show(t);
      ui.scrub.value = Math.round(t / film.end * 1000);
    }
    if (controls.enabled) controls.update();
    if (film_water) film_water.material.normalMap.offset.set(now / 1000 * 0.004, now / 1000 * 0.0025);
    render();
  }
  function render() {
    renderer.render(scene, camera);
    if (recording) {
      const { g, out } = recording;
      g.drawImage(renderer.domElement, 0, 0, out.width, out.height);
      caption(g, out.width, out.height);
    }
  }
  function caption(g, W, H) {
    const c = film.caption; if (!c) return;
    const k = H / 720;
    g.fillStyle = 'rgba(14,22,32,0.72)';
    g.fillRect(0, H - 86 * k, W, 86 * k);
    g.fillStyle = '#fff'; g.font = `600 ${22 * k}px system-ui, sans-serif`;
    g.fillText(c.title, 24 * k, H - 52 * k);
    const tw = g.measureText(c.title).width;
    g.fillStyle = '#f2a516'; g.font = `600 ${19 * k}px system-ui, sans-serif`;
    g.fillText(c.when, 24 * k + tw + 18 * k, H - 52 * k);
    g.fillStyle = '#e8edf2'; g.font = `${18 * k}px system-ui, sans-serif`;
    g.fillText(c.what, 24 * k, H - 24 * k);
    g.fillStyle = 'rgba(255,255,255,0.18)'; g.fillRect(24 * k, H - 10 * k, W - 48 * k, 3 * k);
    g.fillStyle = '#f2a516'; g.fillRect(24 * k, H - 10 * k, (W - 48 * k) * c.progress, 3 * k);
  }
  const TYPES = ['video/mp4;codecs=avc1.42E01E', 'video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
  function startRecording() {
    const type = window.MediaRecorder && TYPES.find((x) => MediaRecorder.isTypeSupported(x));
    if (!type) { ui.note.textContent = 'This browser cannot record video. Chrome, Edge or Safari can.'; return; }
    const out = document.createElement('canvas');
    out.width = 1920; out.height = 1080;
    const g = out.getContext('2d');
    const rec = new MediaRecorder(out.captureStream(30), { mimeType: type, videoBitsPerSecond: 12_000_000 });
    const chunks = [];
    rec.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: type.split(';')[0] });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${(el.dataset.file || 'construction').replace(/[^\w-]+/g, '-')}-construction.${type.startsWith('video/mp4') ? 'mp4' : 'webm'}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 60000);
      ui.note.textContent = type.startsWith('video/mp4') ? 'Saved as MP4 in your downloads.' : 'Saved as WebM in your downloads (this browser does not record MP4).';
    };
    recording = { rec, out, g };
    t = 0; film.show(0);
    rec.start(1000);
    ui.record.textContent = '■ Stop recording';
    ui.note.textContent = 'Recording: keep this tab open and in front until the film ends.';
    play();
  }
  function stopRecording() {
    if (!recording) return;
    if (recording.rec.state !== 'inactive') recording.rec.stop();
    recording = null;
    ui.record.textContent = '● Record video';
  }
  // One frame of the film at `sec`, with its captions, as a JPEG data URL: how a film is made
  // frame by frame without a screen (tools/film_stills.js on a build machine).
  let still = null;
  function frameAt(sec, W = 1280, H = 720) {
    if (!still) { still = document.createElement('canvas'); still.width = W; still.height = H; }
    t = sec; controls.enabled = false; playing = false;
    film.show(t);
    if (film_water) film_water.material.normalMap.offset.set(sec * 0.004, sec * 0.0025);
    renderer.render(scene, camera);
    const g = still.getContext('2d');
    g.drawImage(renderer.domElement, 0, 0, W, H);
    caption(g, W, H);
    return still.toDataURL('image/jpeg', 0.9);
  }
  return { seek: (sec) => { t = sec; controls.enabled = false; film && film.show(t); }, frameAt, get ready() { return !!film; }, get film() { return film; }, get three() { return { THREE, scene, camera, renderer, controls }; } };
}
