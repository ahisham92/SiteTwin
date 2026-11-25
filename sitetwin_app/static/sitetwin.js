// SiteTwin, on every page: the toast (a CDT InfoBar), and on the projects page the new-project
// dialog, the search and filter chips, and the cards kept live while a film is being made.
// The theme, brand, dialogs and menus come from cdt/cdt.js.
(function () {
  'use strict';
  var CDT = window.CDT;
  window.siteTwinToast = function (text, bad) { CDT.toast(text, bad ? 'error' : 'success'); };
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  // ---- a new project, named in a dialog --------------------------------------------------
  var form = $('[data-new-project]');
  function askName() {
    CDT.prompt({ title: 'A new project', message: 'Give it a name you will know it by. The model and its place come next.',
      icon: 'plus', ok: 'Create', field: { label: 'Project name', placeholder: 'e.g. Jiyeh tower', max: 200 } })
      .then(function (name) { if (!name) return; form.elements.name.value = name; form.submit(); });
  }
  if (form) {
    form.classList.add('as-button');
    form.elements.name.required = false;
    form.addEventListener('submit', function (ev) { if (!form.elements.name.value.trim()) { ev.preventDefault(); askName(); } });
    var b = $('[data-new-project-button]');
    if (b) b.addEventListener('click', askName);
  }

  // ---- search and filter chips -------------------------------------------------------------
  var cards = $$('.project[data-key]');
  var search = $('[data-project-search]'), filter = 'all';
  function apply() {
    var q = search ? search.value.trim().toLowerCase() : '', shown = 0;
    cards.forEach(function (c) {
      var ok = (filter === 'all' || c.dataset.key === filter) && (!q || c.dataset.text.indexOf(q) >= 0);
      c.classList.toggle('hidden', !ok);
      if (ok) shown++;
    });
    var none = $('[data-no-match]');
    if (none) none.classList.toggle('hidden', shown > 0 || !cards.length);
  }
  if (search) search.addEventListener('input', apply);
  $$('[data-filter]').forEach(function (chip) {
    chip.addEventListener('click', function () {
      filter = chip.dataset.filter;
      $$('[data-filter]').forEach(function (c) { c.setAttribute('aria-pressed', c === chip ? 'true' : 'false'); });
      apply();
    });
  });
  var clear = $('[data-clear-filter]');
  if (clear) clear.addEventListener('click', function () { if (search) search.value = ''; var all = $('[data-filter=all]'); if (all) all.click(); else apply(); });

  // ---- cards whose film is being made ask for their state every few seconds ------------------
  var live = $$('.project[data-live]');
  if (!live.length) return;
  function outcome(kind, word) {
    var glyph = kind === 'warning' ? '<span class="cdt-dot warn"></span>' : CDT.icon({ success: 'check', danger: 'error', accent: 'film' }[kind] || 'info');
    return '<span class="cdt-outcome ' + kind + '">' + glyph + CDT.esc(word) + '</span>';
  }
  function badge(s) {
    var j = s.job;
    if (j && j.status === 'running') return outcome('warning', 'Rendering ' + Math.round(j.percent) + '%');
    if (j && j.status === 'queued') return outcome('warning', 'Waiting for a GPU');
    if (s.done) return outcome('success', 'Film ready');
    if (j && j.status === 'failed') return outcome('danger', 'Render failed');
    return outcome('accent', 'Ready to render');
  }
  function tick() {
    live.forEach(function (card) {
      fetch(card.dataset.live, { credentials: 'same-origin', headers: { Accept: 'application/json' } })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (s) {
          if (!s) return;
          $('[data-badge]', card).innerHTML = badge(s);
          $$('.pip', card).forEach(function (pip, i) {
            pip.className = 'pip ' + s.stages[['model', 'location', 'sequence', 'video'][i]];
          });
        }).catch(function () {});
    });
  }
  setInterval(tick, 5000);
})();
