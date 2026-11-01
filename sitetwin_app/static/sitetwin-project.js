// A SiteTwin project page, kept live: the four stages as tabs, the model uploaded with its
// progress, the location placed on a satellite map and saved as the pin moves, the sequence
// opened in 3D, and the film followed while it is rendered (its latest frame, how far, how long
// left, what it costs) and played the moment it is ready. Nothing on the page needs a refresh.
(function () {
  'use strict';
  var me = document.currentScript;
  var STATE_URL = me.dataset.stateUrl;
  var ESTIMATE_URL = me.dataset.estimateUrl;
  var VIEWER_JS = me.dataset.viewerJs;
  var state = JSON.parse(document.getElementById('state').textContent);
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };
  var toast = window.siteTwinToast || function () {};
  var esc = function (t) { return String(t == null ? '' : t).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };

  function ask(url, opts) {
    opts = opts || {};
    opts.credentials = 'same-origin';
    opts.headers = Object.assign({ Accept: 'application/json', 'X-Requested-With': 'fetch' }, opts.headers || {});
    return fetch(url, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) throw new Error(body.error || ('The server said ' + r.status));
        return body;
      });
    });
  }

  function duration(s) {
    if (s == null || !isFinite(s)) return '';
    s = Math.max(0, Math.round(s));
    if (s < 60) return s + ' s';
    var m = Math.round(s / 60);
    if (m < 60) return m + ' min';
    return Math.floor(m / 60) + ' h ' + (m % 60) + ' min';
  }
  function filmLength(s) {
    s = Math.round(s);
    var m = Math.floor(s / 60), r = s % 60;
    return m ? (m + ' min' + (r ? ' ' + r + ' s' : '')) : (r + ' s');
  }

  // ---- the stages, as tabs ------------------------------------------------------
  var shown = null;
  function show(key, push) {
    shown = key;
    $$('[data-pane]').forEach(function (p) { p.classList.toggle('shown', p.dataset.pane === key); });
    $$('.step').forEach(function (s) {
      s.classList.toggle('current', s.dataset.step === key);
      $('button', s).setAttribute('aria-selected', s.dataset.step === key ? 'true' : 'false');
    });
    if (push) history.replaceState(null, '', '#' + key);
    if (key === 'location') openMap();
    if (key === 'sequence') openViewer();
  }
  $$('.step').forEach(function (s) { $('button', s).addEventListener('click', function () { show(s.dataset.step, true); }); });

  function firstPane() {
    var h = location.hash.slice(1);
    if (['model', 'location', 'sequence', 'video'].indexOf(h) >= 0) return h;
    var st = state.stages;
    if (st.video === 'active' || st.video === 'done' || st.video === 'failed') return 'video';
    if (st.model !== 'done') return 'model';
    if (st.location !== 'done') return 'location';
    return 'sequence';
  }

  // The one thing to do next, as the page's main button.
  function nextAction() {
    var b = $('[data-next-action]');
    var st = state.stages, j = state.job;
    var label, go;
    if (st.model !== 'done') { label = 'Upload the model'; go = function () { show('model', true); $('.upload input[type=file]').click(); }; }
    else if (st.location !== 'done') { label = 'Place it on the map'; go = function () { show('location', true); }; }
    else if (j && (j.status === 'queued' || j.status === 'running')) { label = 'Watch it being made'; go = function () { show('video', true); }; }
    else if (state.done) { label = 'Watch the film'; go = function () { show('video', true); var v = $('[data-video]'); v.scrollIntoView({ behavior: 'smooth', block: 'center' }); v.play().catch(function () {}); }; }
    else if (shown !== 'sequence') { label = 'See the sequence in 3D'; go = function () { show('sequence', true); }; }
    else { label = 'Make the film'; go = function () { show('video', true); }; }
    b.textContent = label;
    b.onclick = go;
  }

  // ---- 1. the model ----------------------------------------------------------
  var upload = $('.upload');
  if (upload) {
    var drop = $('.drop', upload), input = $('input[type=file]', upload);
    ['dragenter', 'dragover'].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.add('over'); }); });
    ['dragleave', 'drop'].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.remove('over'); }); });
    drop.addEventListener('drop', function (ev) { if (ev.dataTransfer.files[0]) send(ev.dataTransfer.files[0]); });
    input.addEventListener('change', function () { if (input.files[0]) send(input.files[0]); });
    upload.addEventListener('submit', function (ev) { ev.preventDefault(); if (input.files[0]) send(input.files[0]); });
  }
  function send(file) {
    if (!/\.ifc$/i.test(file.name)) { toast('SiteTwin reads .ifc files. In Revit: File › Export › IFC.', true); return; }
    var box = $('.upload-state'), bar = $('[data-upload-bar]'), word = $('[data-upload-word]'), num = $('[data-upload-num]');
    box.classList.remove('hidden');
    var mb = function (b) { return (b / 1048576).toFixed(b < 10485760 ? 1 : 0) + ' MB'; };
    var xhr = new XMLHttpRequest();
    xhr.open('POST', upload.action);
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.setRequestHeader('X-Requested-With', 'fetch');
    xhr.upload.onprogress = function (ev) {
      if (!ev.lengthComputable) return;
      var pct = ev.loaded / ev.total * 100;
      bar.style.width = pct + '%';
      word.textContent = 'Uploading ' + file.name;
      num.textContent = mb(ev.loaded) + ' of ' + mb(ev.total);
      if (ev.loaded >= ev.total) { word.innerHTML = '<span class="pulse"></span>Reading the model: its elements, storeys and location…'; num.textContent = ''; }
    };
    xhr.onload = function () {
      var body = {};
      try { body = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
      if (xhr.status >= 400) { box.classList.add('hidden'); toast(body.error || 'The upload failed.', true); return; }
      word.textContent = 'Read. Opening it…';
      var after = body.project && body.project.latitude == null ? 'location' : 'sequence';
      location.hash = after;
      location.reload();
    };
    xhr.onerror = function () { box.classList.add('hidden'); toast('The upload was cut off. Try again.', true); };
    var data = new FormData();
    data.append('model', file);
    xhr.send(data);
  }

  // ---- 2. the location -------------------------------------------------------
  var map = null, pin = null, ring = null;
  var form = $('[data-location]');
  function openMap() {
    if (map || !window.L) { if (map) map.invalidateSize(); return; }
    var el = $('[data-map]'), at = JSON.parse(el.dataset.map);
    var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 20, maxNativeZoom: 19, attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics' });
    var labels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', { maxZoom: 20, maxNativeZoom: 19 });
    var streets = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' });
    var have = at.lat != null;
    map = L.map(el, { layers: [satellite, labels], zoomControl: true }).setView(have ? [at.lat, at.lon] : [28, 42], have ? 17 : 4);
    L.control.layers({ Satellite: satellite, Map: streets }, { Labels: labels }, { position: 'topright' }).addTo(map);
    L.control.scale({ imperial: false }).addTo(map);
    if (have) place(at.lat, at.lon, false);
    map.on('click', function (ev) { place(ev.latlng.lat, ev.latlng.lng, true); });
  }
  function place(lat, lon, save) {
    if (!pin) {
      pin = L.marker([lat, lon], { draggable: true, title: 'The site' }).addTo(map);
      ring = L.circle([lat, lon], { radius: 350, color: '#fab219', weight: 2, fillOpacity: 0.06, interactive: false }).addTo(map);
      pin.on('drag', function () { ring.setLatLng(pin.getLatLng()); });
      pin.on('dragend', function () { var p = pin.getLatLng(); fill(p.lat, p.lng); saveLocation(); });
    } else { pin.setLatLng([lat, lon]); ring.setLatLng([lat, lon]); }
    fill(lat, lon);
    if (save) saveLocation();
  }
  function fill(lat, lon) {
    form.latitude.value = lat.toFixed(6);
    form.longitude.value = lon.toFixed(6);
  }
  function saveLocation() {
    var saved = $('[data-saved]');
    saved.textContent = 'Saving…';
    ask(form.action, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latitude: form.latitude.value, longitude: form.longitude.value, elevation: form.elevation.value }) })
      .then(function (s) { saved.textContent = '✓ Saved'; $('[data-where-from]').textContent = 'Placed by you'; apply(s); })
      .catch(function (e) { saved.textContent = ''; toast(e.message, true); });
  }
  if (form) {
    form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var lat = parseFloat(form.latitude.value), lon = parseFloat(form.longitude.value);
      if (map && isFinite(lat) && isFinite(lon)) { place(lat, lon, false); map.setView([lat, lon], 17); }
      saveLocation();
    });
    var search = $('[data-search]'), results = $('[data-results]'), timer;
    search.addEventListener('input', function () {
      clearTimeout(timer);
      var q = search.value.trim();
      if (q.length < 3) { results.classList.add('hidden'); return; }
      timer = setTimeout(function () {
        fetch('https://nominatim.openstreetmap.org/search?format=json&limit=6&q=' + encodeURIComponent(q))
          .then(function (r) { return r.json(); })
          .then(function (found) {
            results.innerHTML = found.length ? found.map(function (f, i) { return '<li><button type="button" data-i="' + i + '">' + esc(f.display_name) + '</button></li>'; }).join('')
              : '<li class="small muted" style="padding:7px 10px">Nothing found. Try another name.</li>';
            results.classList.remove('hidden');
            $$('button', results).forEach(function (b) {
              b.addEventListener('click', function () {
                var f = found[+b.dataset.i];
                results.classList.add('hidden');
                search.value = f.display_name.split(',')[0];
                map.setView([+f.lat, +f.lon], 17);
                toast('Now click the map, or drag the pin, onto the site itself.');
                if (!pin) place(+f.lat, +f.lon, false);
              });
            });
          }).catch(function () { results.classList.add('hidden'); });
      }, 350);
    });
  }

  // ---- 3. the sequence in 3D ---------------------------------------------------
  var viewer = null;
  function openViewer() {
    var el = $('[data-viewer]');
    if (!el || viewer) return;
    viewer = 'opening';
    import(VIEWER_JS).then(function (mod) { viewer = mod.open(el, { play: $('[data-play]'), scrub: $('[data-scrub]'), word: $('[data-scrub-word]'), list: $('[data-step-list]') }); })
      .catch(function (err) { viewer = null; $('[data-viewer-msg]').textContent = 'The 3D preview could not open: ' + err.message; });
  }

  // ---- 4. the film -------------------------------------------------------------
  var render = $('[data-render]');
  if (render) {
    var length = $('[data-length]'), lengthWord = $('[data-length-word]');
    var update = function () {
      lengthWord.textContent = filmLength(+length.value);
      var q = (render.querySelector('input[name=quality]:checked') || {}).value || 'draft';
      ask(ESTIMATE_URL + '?quality=' + q + '&seconds=' + length.value).then(function (e) {
        $('[data-est-min]').textContent = duration(e.minutes * 60);
        $('[data-est-gpus]').textContent = e.gpus;
        $('[data-est-cost]').textContent = '$' + e.cost.toFixed(2);
        $('[data-est-frames]').textContent = e.frames.toLocaleString() + ' frames';
      }).catch(function () {});
    };
    render.addEventListener('input', update);
    update();
    render.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var b = $('[data-render-button]');
      b.disabled = true;
      ask(render.action, { method: 'POST', body: new FormData(render) })
        .then(function (s) { apply(s); toast('Asked for. A cloud GPU will pick it up shortly.'); $('[data-live-card]').scrollIntoView({ behavior: 'smooth', block: 'start' }); })
        .catch(function (e) { b.disabled = false; toast(e.message, true); });
    });
  }
  var cancel = $('[data-cancel]');
  cancel.addEventListener('submit', function (ev) {
    ev.preventDefault();
    if (!window.confirm('Stop making this film? What has been rendered so far is thrown away.')) return;
    ask(cancel.action, { method: 'POST' }).then(function (s) { apply(s); toast('Stopped.'); }).catch(function (e) { toast(e.message, true); });
  });

  var ORDER = ['queued', 'starting', 'surroundings', 'scene', 'frames', 'joining', 'uploading', 'done'];
  function stageOf(j) {
    if (j.stage && ORDER.indexOf(j.stage) >= 0) return j.stage;
    if (j.status === 'queued') return 'queued';
    return j.percent < 8 ? 'starting' : (j.percent >= 96 ? 'joining' : 'frames');
  }

  function liveCard(j) {
    var card = $('[data-live-card]');
    var active = j && (j.status === 'queued' || j.status === 'running');
    card.classList.toggle('hidden', !active);
    if (!active) return;
    cancel.action = location.pathname.replace(/\/$/, '') + '/jobs/' + j.id + '/cancel';
    var at = stageOf(j), m = state.machine;
    $$('[data-timeline] li').forEach(function (li) {
      var k = ORDER.indexOf(li.dataset.k), now = ORDER.indexOf(at);
      li.className = k < now ? 'done' : (k === now ? 'now' : '');
    });
    var pct = Math.round(j.percent || 0);
    $('[data-pct]').textContent = pct + '%';
    $('[data-pct-bar]').style.width = pct + '%';
    var title, sub = '';
    if (j.status === 'queued') {
      title = 'Waiting for a cloud GPU';
      sub = 'Asked ' + duration(j.waiting) + ' ago. ' + j.quality.charAt(0).toUpperCase() + j.quality.slice(1) + ', ' + filmLength(j.seconds) + '.';
      if (j.waiting > 240 && !(m.seen_ago != null && m.seen_ago < 240)) {
        sub += m.seen_ago == null ? ' The render machine has never checked in with this site: its address or token may not match.'
          : ' The render machine last checked in ' + duration(m.seen_ago) + ' ago. It looks every 2 minutes, so this is late.';
      } else sub += ' A GPU usually starts within 2 minutes.';
    } else {
      title = { starting: 'Reading the model', surroundings: 'Fetching the real surroundings', scene: 'Building the scene', frames: 'Rendering the frames', joining: 'Joining the film', uploading: 'Sending the film here' }[at] || 'Making the film';
      sub = j.message || '';
      if (j.quiet > 600) sub += ' · No word from the render machine for ' + duration(j.quiet) + '.';
    }
    $('[data-live-title]').textContent = title;
    $('[data-live-sub]').textContent = sub;
    var bits = [];
    if (j.frames_total) bits.push((j.frames_done || 0).toLocaleString() + ' of ' + j.frames_total.toLocaleString() + ' frames');
    if (j.gpus) bits.push(j.gpus + ' GPU' + (j.gpus > 1 ? 's' : ''));
    if (j.elapsed) bits.push(duration(j.elapsed) + ' in');
    if (j.left) bits.push('about ' + duration(j.left) + ' left');
    else if (j.status === 'queued' && j.estimate) bits.push('about ' + duration(j.estimate.minutes * 60) + ' once started');
    if (j.cost != null) bits.push('$' + j.cost.toFixed(2) + ' so far');
    else if (j.estimate) bits.push('about $' + j.estimate.cost.toFixed(2) + ' in all');
    $('[data-numbers]').textContent = bits.join(' · ');
    var frame = $('[data-frame]');
    if (j.preview_url) {
      var img = $('img', frame);
      if (!img) { frame.innerHTML = '<img alt="The latest frame rendered"><span class="tag">Latest frame</span>'; img = $('img', frame); }
      if (img.getAttribute('src') !== j.preview_url) img.src = j.preview_url;
    }
  }

  function filmCard(d) {
    var card = $('[data-film-card]');
    card.classList.toggle('hidden', !d);
    if (!d) return;
    var v = $('[data-video]');
    if (v.getAttribute('src') !== d.video_url) {
      v.src = d.video_url;
      $('[data-download]').href = d.video_url;
    }
    $('[data-film-meta]').textContent = d.quality.charAt(0).toUpperCase() + d.quality.slice(1) + ', ' + filmLength(d.seconds) + (d.message ? ' · ' + d.message : '');
  }

  function historyTable(list) {
    $('[data-history-card]').classList.toggle('hidden', !list.length);
    var when = function (t) { return new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); };
    $('[data-history]').innerHTML = list.map(function (j) {
      var result = j.status === 'done' ? '<span class="badge good">● Made in ' + esc(duration(j.elapsed)) + '</span> <span class="small muted">' + esc(j.message || '') + '</span>'
        : j.status === 'failed' ? '<span class="badge critical">■ ' + (j.message === 'Stopped by you.' ? 'Stopped' : 'Failed') + '</span> <span class="small muted">' + esc(j.message || '') + '</span>'
        : '<span class="badge warning"><span class="pulse"></span>' + (j.status === 'queued' ? 'Waiting' : 'Rendering ' + Math.round(j.percent) + '%') + '</span>';
      return '<tr><td class="nowrap">' + when(j.created) + '</td><td>' + esc(j.quality) + '</td><td class="right tabular">' + filmLength(j.seconds) + '</td><td>' + result + '</td><td class="right">'
        + (j.video_url ? '<a href="' + j.video_url + '" download>MP4</a>' : '') + '</td></tr>';
    }).join('');
  }

  function stepWords() {
    var st = state.stages, p = state.project, j = state.job, m = p.model_summary;
    var words = {
      model: st.model === 'done' && m ? Object.values(m.counts).reduce(function (a, b) { return a + b; }, 0) + ' elements' : 'Upload the IFC',
      location: st.location === 'done' ? p.latitude.toFixed(4) + ', ' + p.longitude.toFixed(4) : (st.location === 'next' ? 'Place it on the map' : 'After the model'),
      sequence: st.sequence === 'done' ? m.sequence.length + ' steps, storey by storey' : 'Read from the model',
      video: st.video === 'active' ? (j.status === 'queued' ? 'Waiting for a GPU' : 'Rendering ' + Math.round(j.percent) + '%') : st.video === 'done' ? 'Ready to watch' : st.video === 'failed' ? 'Last try failed' : st.video === 'next' ? 'Make the film' : 'After the location',
    };
    $$('.step').forEach(function (s, i) {
      var k = s.dataset.step;
      s.className = 'step ' + st[k] + (k === shown ? ' current' : '');
      $('.step-mark', s).textContent = st[k] === 'done' ? '✓' : String(i + 1);
      $('[data-step-state]', s).textContent = words[k];
    });
  }

  function apply(s) {
    var wasActive = state.job && (state.job.status === 'queued' || state.job.status === 'running');
    state = s;
    var j = s.job, active = j && (j.status === 'queued' || j.status === 'running');
    stepWords();
    nextAction();
    liveCard(j);
    filmCard(s.done);
    historyTable(s.history || []);
    var b = $('[data-render-button]');
    if (b) b.disabled = !!active;
    $('[data-make-card]').classList.toggle('hidden', !!active);
    if (wasActive && !active && j) {
      if (j.status === 'done') { toast('The film is ready.'); show('video', true); }
      else toast('The render stopped: ' + (j.message || 'no reason given'), true);
    }
    schedule(active);
  }

  var timer = null;
  function schedule(active) {
    clearTimeout(timer);
    timer = setTimeout(poll, active ? 3000 : 30000);
  }
  function poll() {
    if (document.hidden) { schedule(true); return; }
    ask(STATE_URL).then(apply).catch(function () { schedule(true); });
  }
  document.addEventListener('visibilitychange', function () { if (!document.hidden) poll(); });

  window.addEventListener('hashchange', function () { show(firstPane(), false); });
  show(firstPane(), false);
  apply(state);
})();
