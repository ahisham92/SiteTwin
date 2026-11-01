// Opens the IFC for SiteTwin's 3D preview, off the page so the page never freezes.
// Asked {url, base, steps: {expressID: step}}, it reads the model with web-ifc (from `base`)
// and sends back one merged shape per build step: positions (metres, y up, around the model's
// first placement) and normals. An element in no step is sent with step 0.

const say = (msg, transfer) => self.postMessage(msg, transfer || []);

async function download(url) {
  const answer = await fetch(url, { credentials: 'same-origin' });
  if (!answer.ok) throw new Error(`the model could not be downloaded (error ${answer.status})`);
  const total = Number(answer.headers.get('Content-Length')) || 0;
  if (!answer.body || !total) return new Uint8Array(await answer.arrayBuffer());
  const bytes = new Uint8Array(total);
  const reader = answer.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes.set(value, got);
    got += value.length;
    say({ type: 'progress', stage: 'download', loaded: got, total });
  }
  return bytes;
}

// Openings are cut one boolean at a time, which is where web-ifc spends its time on a big model.
// A preview does without the cuts: each IFCRELVOIDSELEMENT( is renamed to a type web-ifc skips.
function dropOpenings(bytes) {
  const find = new TextEncoder().encode('IFCRELVOIDSELEMENT(');
  const put = new TextEncoder().encode('IFCRELVOIDSELEMENX(');
  let at = 0;
  for (;;) {
    at = bytes.indexOf(find[0], at);
    if (at < 0) return;
    let same = true;
    for (let k = 1; k < find.length; k++) if (bytes[at + k] !== find[k]) { same = false; break; }
    if (same) { bytes.set(put, at); at += find.length; } else at++;
  }
}

class Bucket {
  constructor() { this.pos = []; this.nor = []; this.count = 0; }
  add(verts, index, m, origin) {
    // Unindexed triangles, so each step is one plain buffer the page can show or hide.
    const pos = new Float32Array(index.length * 3);
    const nor = new Float32Array(index.length * 3);
    for (let i = 0; i < index.length; i++) {
      const k = index[i] * 6;
      const x = verts[k], y = verts[k + 1], z = verts[k + 2];
      pos[i * 3] = m[0] * x + m[4] * y + m[8] * z + m[12] - origin[0];
      pos[i * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13] - origin[1];
      pos[i * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14] - origin[2];
      const nx = verts[k + 3], ny = verts[k + 4], nz = verts[k + 5];
      nor[i * 3] = m[0] * nx + m[4] * ny + m[8] * nz;
      nor[i * 3 + 1] = m[1] * nx + m[5] * ny + m[9] * nz;
      nor[i * 3 + 2] = m[2] * nx + m[6] * ny + m[10] * nz;
    }
    this.pos.push(pos); this.nor.push(nor); this.count += pos.length;
  }
  take() {
    const join = (parts) => {
      const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
      let at = 0;
      for (const p of parts) { out.set(p, at); at += p.length; }
      return out;
    };
    return { positions: join(this.pos), normals: join(this.nor) };
  }
}

self.onmessage = async (ev) => {
  const { url, base, steps } = ev.data;
  try {
    const started = import(base + 'web-ifc-api.js');
    const bytes = await download(url);
    dropOpenings(bytes);
    say({ type: 'progress', stage: 'open' });
    const WebIFC = await started;
    const api = new WebIFC.IfcAPI();
    api.SetWasmPath(base, true);
    await api.Init(undefined, true);
    const modelID = api.OpenModel(bytes, { COORDINATE_TO_ORIGIN: false });
    if (modelID < 0) throw new Error('web-ifc could not open the file');
    const buckets = new Map();
    let origin = null;
    let done = 0;
    const total = Object.keys(steps).length || 1;
    api.StreamAllMeshes(modelID, (flat) => {
      const step = steps[flat.expressID] || 0;
      if (!buckets.has(step)) buckets.set(step, new Bucket());
      const bucket = buckets.get(step);
      const placed = flat.geometries;
      for (let i = 0; i < placed.size(); i++) {
        const pg = placed.get(i);
        const m = pg.flatTransformation;
        if (!origin) origin = [Math.round(m[12]), 0, Math.round(m[14])];
        const g = api.GetGeometry(modelID, pg.geometryExpressID);
        const verts = api.GetVertexArray(g.GetVertexData(), g.GetVertexDataSize());
        const index = api.GetIndexArray(g.GetIndexData(), g.GetIndexDataSize());
        bucket.add(verts, index, m, origin);
        g.delete();
      }
      done++;
      if (done % 50 === 0) say({ type: 'progress', stage: 'shapes', done, total });
    });
    api.CloseModel(modelID);
    for (const [step, bucket] of buckets) {
      const part = bucket.take();
      say({ type: 'part', step, ...part }, [part.positions.buffer, part.normals.buffer]);
    }
    say({ type: 'done', origin: origin || [0, 0, 0] });
  } catch (err) {
    say({ type: 'error', message: String(err && err.message || err) });
  }
};
