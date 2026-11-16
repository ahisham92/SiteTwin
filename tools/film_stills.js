// Opens a project's construction film (SiteTwin web app, Video tab) in headless Chromium and
// saves frames of it: stills at the given seconds (TIMES), or with FPS set, every frame of the
// whole film, to be joined into a video with ffmpeg. Used by .github/workflows/browser-film.yml.
//   SITE=http://localhost:5055 PROJECT=1 OUT=frames FPS=24 node tools/film_stills.js
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const LOCAL = process.env.WEB_IFC || 'node_modules/web-ifc/';
const OUT = process.env.OUT || 'film-stills';
const FPS = Number(process.env.FPS || 0);
const LENGTH = Number(process.env.LENGTH || 150);
const TIMES = (process.env.TIMES || '2,20,45,70,95,120,140').split(',').map(Number);
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1280, height: 1000 }, deviceScaleFactor: 1 });
  p.setDefaultTimeout(900000);
  if (fs.existsSync(LOCAL)) {          // web-ifc from a local copy when the CDN is not reachable
    await p.route((u) => u.href.includes('web-ifc@'), async (r) => {
      const f = path.join(LOCAL, path.basename(new URL(r.request().url()).pathname));
      if (!fs.existsSync(f)) return r.fulfill({ status: 404 });
      r.fulfill({ status: 200, body: fs.readFileSync(f), headers: { 'content-type': f.endsWith('.wasm') ? 'application/wasm' : 'application/javascript', 'access-control-allow-origin': '*' } });
    });
  }
  p.on('pageerror', (e) => console.log('pageerror:', String(e).slice(0, 500)));
  await p.goto(`${process.env.SITE || 'http://localhost:5055'}/projects/${process.env.PROJECT || 1}#video`, { waitUntil: 'domcontentloaded' });
  await p.waitForFunction(() => window.sitetwinFilm && window.sitetwinFilm.ready, null, { timeout: 1800000, polling: 2000 });
  await p.selectOption('[data-film-length]', String(LENGTH));
  console.log('film ready');
  const save = async (t, name) => {
    const url = await p.evaluate((t) => window.sitetwinFilm.frameAt(t, 1280, 720), t);
    fs.writeFileSync(path.join(OUT, name), Buffer.from(url.split(',')[1], 'base64'));
  };
  if (FPS) {
    const n = Math.round(LENGTH * FPS);
    for (let i = 0; i < n; i++) {
      await save(i / FPS, `frame_${String(i).padStart(5, '0')}.jpg`);
      if (i % 240 === 0) console.log(`frame ${i} of ${n}`);
    }
  } else {
    for (const t of TIMES) { await save(t, `t${String(t).padStart(3, '0')}.jpg`); console.log('still', t); }
  }
  await b.close();
})();
