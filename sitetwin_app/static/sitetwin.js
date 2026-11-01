// SiteTwin, on every page: the theme switch (shared with the rest of the site), a toast,
// and the projects list kept live while a film is being made.
(function () {
  'use strict';
  var LABELS = { system: 'Auto', light: 'Light', dark: 'Dark' };
  var NEXT = { system: 'light', light: 'dark', dark: 'system' };
  function current() { try { return localStorage.getItem('pm-theme') || 'system'; } catch (e) { return 'system'; } }
  function apply(t) {
    if (t === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
    var b = document.querySelector('[data-theme-toggle]');
    if (b) { b.textContent = LABELS[t]; b.title = 'Theme: ' + t + '. Click to switch.'; }
  }
  document.addEventListener('click', function (ev) {
    if (!ev.target.closest('[data-theme-toggle]')) return;
    var t = NEXT[current()] || 'light';
    try { localStorage.setItem('pm-theme', t); } catch (e) { /* private mode */ }
    apply(t);
  });
  apply(current());

  document.addEventListener('submit', function (ev) {
    var f = ev.target.closest('form[data-confirm]');
    if (f && !window.confirm(f.dataset.confirm)) ev.preventDefault();
  });

  var toastTimer;
  window.siteTwinToast = function (text, bad) {
    var t = document.querySelector('.toast');
    if (!t) return;
    t.textContent = text;
    t.classList.toggle('error', !!bad);
    t.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.add('hidden'); }, bad ? 6000 : 3000);
  };

  // The projects list: a card whose film is being made asks for its state every few seconds.
  var live = document.querySelectorAll('.project[data-live]');
  if (!live.length) return;
  function badge(s) {
    var j = s.job;
    if (j && j.status === 'running') return '<span class="badge warning"><span class="pulse"></span>Rendering ' + Math.round(j.percent) + '%</span>';
    if (j && j.status === 'queued') return '<span class="badge warning"><span class="pulse"></span>Waiting for a GPU</span>';
    if (s.done) return '<span class="badge good">● Film ready</span>';
    if (j && j.status === 'failed') return '<span class="badge critical">■ Render failed</span>';
    return '<span class="badge">Ready to render</span>';
  }
  function tick() {
    live.forEach(function (card) {
      fetch(card.dataset.live, { credentials: 'same-origin', headers: { Accept: 'application/json' } })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (s) {
          if (!s) return;
          card.querySelector('[data-badge]').innerHTML = badge(s);
          card.querySelectorAll('.pip').forEach(function (pip, i) {
            pip.className = 'pip ' + s.stages[['model', 'location', 'sequence', 'video'][i]];
          });
        }).catch(function () {});
    });
  }
  setInterval(tick, 5000);
})();
