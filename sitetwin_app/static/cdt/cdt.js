// CDT.Web: the behaviours of the CDT.SDK controls, for the browser. Pairs with cdt.css.
//   CDT.theme / CDT.brand          ThemeManager.Apply(theme, brand), remembered per browser
//   CDT.toast(text, severity)      an InfoBar with AutoCloseAfter (paused under the mouse)
//   CDT.confirm({...}) -> Promise  cdt:DialogHost.ShowAsync: a dialog in the page, Esc closes
//   CDT.prompt({...})  -> Promise  the same, with one field
//   [data-cdt-number]              cdt:NumberBox: unit, spin buttons, arrows / PageUp, arithmetic, Enter / Esc
//   [data-cdt-dropdown]            cdt:DropDownButton: the button opens the menu after it
//   .cdt-segment[data-cdt-segment] CDT.SegmentedList: one selected, arrow keys move
//   .cdt-slider                    CDT.Slider: the filled part of the track follows the value
//   .cdt-grid[data-cdt-sort]       CDT.DataGrid: a click on a header sorts by it
//   FocusAssist                    Esc or a click on nothing clears the focus from a field
(function () {
  'use strict';
  var CDT = window.CDT = window.CDT || {};
  var doc = document, root = doc.documentElement;
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };
  var esc = CDT.esc = function (t) {
    return String(t == null ? '' : t).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; });
  };
  CDT.icon = function (name, cls) { return '<svg class="cdt-ico ' + (cls || '') + '" aria-hidden="true"><use href="#i-' + name + '"/></svg>'; };

  // ---- theme and brand (ThemeManager) ----------------------------------------------------
  // The theme key is the website's own, so every app on the site switches together.
  var THEME_KEY = 'pm-theme', BRAND_KEY = 'cdt-brand';
  CDT.theme = function (t) {
    if (t === undefined) return store.get(THEME_KEY) || 'system';
    store.set(THEME_KEY, t);
    if (t === 'system') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', t);
    sync();
  };
  CDT.brand = function (b) {
    if (b === undefined) return store.get(BRAND_KEY) || 'dar';
    store.set(BRAND_KEY, b);
    root.setAttribute('data-brand', b);
    sync();
  };
  function sync() {
    var t = CDT.theme(), b = CDT.brand();
    doc.querySelectorAll('[data-cdt-theme]').forEach(function (el) { el.setAttribute('aria-pressed', el.dataset.cdtTheme === t ? 'true' : 'false'); });
    doc.querySelectorAll('[data-cdt-brand]').forEach(function (el) { el.setAttribute('aria-pressed', el.dataset.cdtBrand === b ? 'true' : 'false'); });
    doc.dispatchEvent(new CustomEvent('cdt:theme', { detail: { theme: t, brand: b } }));
  }
  doc.addEventListener('click', function (ev) {
    var t = ev.target.closest('[data-cdt-theme]');
    if (t) CDT.theme(t.dataset.cdtTheme);
    var b = ev.target.closest('[data-cdt-brand]');
    if (b) CDT.brand(b.dataset.cdtBrand);
  });

  // ---- InfoBar toasts ------------------------------------------------------------------
  var GLYPH = { info: 'info', success: 'check-circle', warning: 'warning', error: 'error' };
  CDT.infobar = function (o) {
    var sev = o.severity || 'info';
    var el = doc.createElement('div');
    el.className = 'cdt-infobar ' + sev;
    el.setAttribute('role', sev === 'error' ? 'alert' : 'status');
    el.innerHTML = CDT.icon(GLYPH[sev] || 'info') + '<div class="ib-text">' + (o.title ? '<b>' + esc(o.title) + '</b>' : '') + '<span>' + esc(o.message || '') + '</span></div>'
      + '<div class="ib-actions"><button type="button" class="cdt-btn icon sm" aria-label="Close">' + CDT.icon('close') + '</button></div>';
    el.querySelector('[aria-label=Close]').addEventListener('click', function () { el.remove(); });
    return el;
  };
  CDT.toast = function (text, severity) {
    if (severity === true) severity = 'error';
    var host = doc.querySelector('.cdt-toasts');
    if (!host) { host = doc.createElement('div'); host.className = 'cdt-toasts'; host.setAttribute('aria-live', 'polite'); doc.body.appendChild(host); }
    var el = CDT.infobar({ message: text, severity: severity || 'success' });
    host.appendChild(el);
    while (host.children.length > 3) host.firstChild.remove();
    var left = severity === 'error' ? 7000 : 3500, timer;
    var run = function () { timer = setTimeout(function () { el.remove(); }, left); };
    el.addEventListener('mouseenter', function () { clearTimeout(timer); });
    el.addEventListener('mouseleave', run);
    run();
    return el;
  };

  // ---- DialogHost ----------------------------------------------------------------------
  // Shows a card on the scrim, keeps Tab inside it, resolves with the value of the button
  // pressed (Esc or a click on the scrim: null).
  CDT.dialog = function (o) {
    return new Promise(function (resolve) {
      var host = doc.createElement('div');
      host.className = 'cdt-dialoghost';
      var danger = o.danger;
      host.innerHTML = '<div class="cdt-sheetcard cdt-dialog" role="dialog" aria-modal="true" aria-labelledby="cdt-dlg-t">'
        + '<div class="cdt-card-body"><div class="head"><span class="cdt-ico-tile' + (danger ? ' danger' : '') + '">' + CDT.icon(o.icon || (danger ? 'warning' : 'info'), 'lg') + '</span>'
        + '<div><h2 class="cdt-title" id="cdt-dlg-t">' + esc(o.title) + '</h2>' + (o.message ? '<p class="cdt-muted" style="margin-top:4px">' + esc(o.message) + '</p>' : '') + '</div></div>'
        + (o.field ? '<label class="cdt-field"><span class="cdt-field-label">' + esc(o.field.label) + '</span><input class="cdt-input field" maxlength="' + (o.field.max || 200) + '" value="' + esc(o.field.value || '') + '" placeholder="' + esc(o.field.placeholder || '') + '" data-dlg-field></label>' : '')
        + '</div><div class="cdt-card-foot"><button type="button" class="cdt-btn" data-v="">' + esc(o.cancel || 'Cancel') + '</button>'
        + '<button type="button" class="cdt-btn ' + (danger ? 'danger' : 'primary') + '" data-v="ok">' + esc(o.ok || 'OK') + '</button></div></div>';
      var before = doc.activeElement;
      doc.body.appendChild(host);
      var field = host.querySelector('[data-dlg-field]');
      var okBtn = host.querySelector('[data-v=ok]');
      (field || okBtn).focus();
      if (field) field.select();
      function close(v) {
        host.remove();
        doc.removeEventListener('keydown', key, true);
        if (before && before.focus) before.focus();
        resolve(v);
      }
      function key(ev) {
        if (ev.key === 'Escape') { ev.preventDefault(); close(null); }
        else if (ev.key === 'Enter' && field && doc.activeElement === field) { ev.preventDefault(); if (field.value.trim()) close(field.value.trim()); }
        else if (ev.key === 'Tab') {
          var f = host.querySelectorAll('button, input');
          var first = f[0], last = f[f.length - 1];
          if (ev.shiftKey && doc.activeElement === first) { ev.preventDefault(); last.focus(); }
          else if (!ev.shiftKey && doc.activeElement === last) { ev.preventDefault(); first.focus(); }
        }
      }
      doc.addEventListener('keydown', key, true);
      host.addEventListener('click', function (ev) {
        if (ev.target === host) return close(null);
        var b = ev.target.closest('[data-v]');
        if (!b) return;
        if (b.dataset.v !== 'ok') return close(null);
        if (field) { if (field.value.trim()) close(field.value.trim()); else field.focus(); }
        else close(true);
      });
    });
  };
  CDT.confirm = function (o) { return CDT.dialog(o).then(function (v) { return v === true; }); };
  CDT.prompt = function (o) { return CDT.dialog(o); };

  // A form with data-confirm asks first, in a dialog (data-confirm-ok names the button).
  doc.addEventListener('submit', function (ev) {
    var f = ev.target.closest('form[data-confirm]');
    if (!f || f.dataset.confirmed) return;
    ev.preventDefault();
    CDT.confirm({ title: f.dataset.confirm, message: f.dataset.confirmMessage, ok: f.dataset.confirmOk || 'OK', danger: 'confirmDanger' in f.dataset })
      .then(function (yes) { if (yes) { f.dataset.confirmed = '1'; f.submit(); } });
  }, true);

  // ---- DropDownButton ------------------------------------------------------------------
  function closeMenus(except) {
    doc.querySelectorAll('[data-cdt-dropdown][aria-expanded="true"]').forEach(function (b) {
      if (b === except) return;
      b.setAttribute('aria-expanded', 'false');
      var m = b.nextElementSibling; if (m) m.hidden = true;
    });
  }
  doc.addEventListener('click', function (ev) {
    var b = ev.target.closest('[data-cdt-dropdown]');
    if (b) {
      var m = b.nextElementSibling, open = b.getAttribute('aria-expanded') !== 'true';
      closeMenus(b);
      b.setAttribute('aria-expanded', open ? 'true' : 'false');
      m.hidden = !open;
      if (open) { var first = m.querySelector('button, a'); if (first && ev.detail === 0) first.focus(); }
      return;
    }
    var inMenu = ev.target.closest('.cdt-menu');
    if (!inMenu || ev.target.closest('.cdt-menurow')) closeMenus();
  });
  doc.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape') {
      var open = doc.querySelector('[data-cdt-dropdown][aria-expanded="true"]');
      if (open) { closeMenus(); open.focus(); return; }
      // FocusAssist: Esc takes the focus out of a field.
      var a = doc.activeElement;
      if (a && /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName) && !a.closest('.cdt-dialoghost')) a.blur();
    }
    var row = ev.target.closest && ev.target.closest('.cdt-menu');
    if (row && (ev.key === 'ArrowDown' || ev.key === 'ArrowUp')) {
      ev.preventDefault();
      var items = Array.prototype.slice.call(row.querySelectorAll('.cdt-menurow, .cdt-segment > button'));
      var i = items.indexOf(doc.activeElement);
      items[(i + (ev.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length].focus();
    }
  });

  // ---- SegmentedList: arrow keys move between segments ------------------------------------
  doc.addEventListener('keydown', function (ev) {
    var seg = ev.target.closest && ev.target.closest('.cdt-segment[data-cdt-segment]');
    if (!seg || (ev.key !== 'ArrowRight' && ev.key !== 'ArrowLeft')) return;
    var b = Array.prototype.slice.call(seg.querySelectorAll('button'));
    var i = b.indexOf(doc.activeElement);
    if (i < 0) return;
    ev.preventDefault();
    var n = b[(i + (ev.key === 'ArrowRight' ? 1 : -1) + b.length) % b.length];
    n.focus(); n.click();
  });

  // ---- NumberBox -----------------------------------------------------------------------
  // Reads '.' or ',' as the decimal point and takes arithmetic (6000/4 + 25). Commits on
  // Enter or leaving the field; Esc puts back the last good value.
  function evaluate(text) {
    var s = String(text).trim().replace(/\s+/g, '').replace(/,/g, '.');
    if (s === '') return null;
    if (!/^[-+*/().0-9eE]+$/.test(s)) return NaN;
    var i = 0;
    function num() {
      var m = /^[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|^[0-9]+\.?/.exec(s.slice(i));
      if (!m) throw 0; i += m[0].length; return parseFloat(m[0]);
    }
    function atom() {
      if (s[i] === '(') { i++; var v = sum(); if (s[i] !== ')') throw 0; i++; return v; }
      if (s[i] === '-') { i++; return -atom(); }
      if (s[i] === '+') { i++; return atom(); }
      return num();
    }
    function prod() { var v = atom(); while (s[i] === '*' || s[i] === '/') { var o = s[i++], r = atom(); v = o === '*' ? v * r : v / r; } return v; }
    function sum() { var v = prod(); while (s[i] === '+' || s[i] === '-') { var o = s[i++], r = prod(); v = o === '+' ? v + r : v - r; } return v; }
    try { var v = sum(); return i === s.length && isFinite(v) ? v : NaN; } catch (e) { return NaN; }
  }
  CDT.evaluate = evaluate;
  CDT.numberBox = function (input) {
    if (input.dataset.cdtReady) return;
    input.dataset.cdtReady = '1';
    var box = doc.createElement('span');
    box.className = 'cdt-numberbox';
    input.parentNode.insertBefore(box, input);
    box.appendChild(input);
    input.classList.remove('cdt-input');
    input.setAttribute('autocomplete', 'off');
    var unit = input.dataset.unit;
    if (unit) { var u = doc.createElement('span'); u.className = 'unit'; u.textContent = unit; box.appendChild(u); }
    var spin = doc.createElement('span');
    spin.className = 'spin';
    spin.innerHTML = '<button type="button" tabindex="-1" aria-label="Up">▲</button><button type="button" tabindex="-1" aria-label="Down">▼</button>';
    box.appendChild(spin);
    var step = parseFloat(input.dataset.step || '1'), min = parseFloat(input.dataset.min), max = parseFloat(input.dataset.max);
    var decimals = input.dataset.decimals != null ? parseInt(input.dataset.decimals, 10) : null;
    var good = input.value;
    var err = doc.createElement('span'); err.className = 'cdt-field-error'; err.hidden = true;
    box.parentNode.insertBefore(err, box.nextSibling);
    function fmt(v) { if (decimals == null) return String(+v.toPrecision(12)); return v.toFixed(decimals).replace(/\.?0+$/, '') || '0'; }
    function bad(why) { box.classList.add('invalid'); input.setAttribute('aria-invalid', 'true'); err.textContent = why; err.hidden = false; }
    function ok() { box.classList.remove('invalid'); input.removeAttribute('aria-invalid'); err.hidden = true; }
    function commit(fire) {
      var raw = input.value;
      if (raw.trim() === '') { if (input.required) { bad('A number is needed here.'); return false; } ok(); good = ''; return true; }
      var v = evaluate(raw);
      if (isNaN(v)) { bad('Not a number: ' + raw); return false; }
      if (!isNaN(min) && v < min) { bad('At least ' + min + (unit ? ' ' + unit : '')); return false; }
      if (!isNaN(max) && v > max) { bad('At most ' + max + (unit ? ' ' + unit : '')); return false; }
      ok();
      var t = fmt(v);
      if (t !== raw) input.value = t;
      if (fire && t !== good) input.dispatchEvent(new Event('change', { bubbles: true }));
      good = t;
      return true;
    }
    function bump(dir, times) {
      var v = evaluate(input.value); if (v == null || isNaN(v)) v = 0;
      v = v + dir * step * (times || 1);
      if (!isNaN(min)) v = Math.max(min, v);
      if (!isNaN(max)) v = Math.min(max, v);
      input.value = fmt(v); commit(true);
    }
    input.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter') { if (!commit(true)) ev.preventDefault(); }
      else if (ev.key === 'Escape') { input.value = good; ok(); }
      else if (ev.key === 'ArrowUp') { ev.preventDefault(); bump(1); }
      else if (ev.key === 'ArrowDown') { ev.preventDefault(); bump(-1); }
      else if (ev.key === 'PageUp') { ev.preventDefault(); bump(1, 10); }
      else if (ev.key === 'PageDown') { ev.preventDefault(); bump(-1, 10); }
    }, true);
    input.addEventListener('blur', function () { commit(true); });
    input.addEventListener('wheel', function (ev) { if (doc.activeElement !== input) return; ev.preventDefault(); bump(ev.deltaY < 0 ? 1 : -1); }, { passive: false });
    spin.addEventListener('click', function (ev) { var b = ev.target.closest('button'); if (b) bump(b.getAttribute('aria-label') === 'Up' ? 1 : -1); });
    input.form && input.form.addEventListener('submit', function (ev) { if (!commit(false)) { ev.preventDefault(); ev.stopImmediatePropagation(); input.focus(); } }, true);
    // Code that sets .value from outside (the map pin) is the new good value.
    CDT.watchValue(input, function () { good = input.value; ok(); });
  };
  CDT.watchValue = function (input, fn) {
    var d = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
    Object.defineProperty(input, 'value', { configurable: true, get: function () { return d.get.call(this); },
      set: function (v) { d.set.call(this, v); fn(); } });
  };

  // ---- Slider: the filled part of the track ----------------------------------------------
  function fill(r) {
    var lo = +r.min || 0, hi = +r.max || 100, v = +r.value;
    r.style.setProperty('--pct', ((v - lo) / (hi - lo || 1) * 100) + '%');
  }
  CDT.slider = function (r) {
    if (r.dataset.cdtReady) return; r.dataset.cdtReady = '1';
    fill(r);
    r.addEventListener('input', function () { fill(r); });
    CDT.watchValue(r, function () { fill(r); });
  };
  CDT.fill = fill;

  // ---- DataGrid sorting ----------------------------------------------------------------
  CDT.grid = function (table) {
    if (table.dataset.cdtReady) return; table.dataset.cdtReady = '1';
    table.querySelectorAll('thead th[data-sort]').forEach(function (th, _, all) {
      th.tabIndex = 0;
      function sort() {
        var col = Array.prototype.indexOf.call(th.parentNode.children, th);
        var dir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
        all.forEach(function (h) { h.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', dir);
        var body = table.tBodies[0], rows = Array.prototype.slice.call(body.rows);
        var numeric = th.dataset.sort === 'num';
        rows.sort(function (a, b) {
          var x = a.cells[col].dataset.v || a.cells[col].textContent, y = b.cells[col].dataset.v || b.cells[col].textContent;
          var r = numeric ? parseFloat(x) - parseFloat(y) : x.localeCompare(y, undefined, { numeric: true });
          return dir === 'ascending' ? r : -r;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      }
      th.addEventListener('click', sort);
      th.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); sort(); } });
    });
  };

  // FocusAssist: a click on nothing takes the focus out of a field.
  doc.addEventListener('mousedown', function (ev) {
    var a = doc.activeElement;
    if (!a || !/^(INPUT|TEXTAREA)$/.test(a.tagName)) return;
    if (ev.target === doc.body || ev.target.classList.contains('cdt-page')) a.blur();
  });

  CDT.init = function (scope) {
    scope = scope || doc;
    scope.querySelectorAll('[data-cdt-number]').forEach(CDT.numberBox);
    scope.querySelectorAll('input.cdt-slider').forEach(CDT.slider);
    scope.querySelectorAll('table.cdt-grid').forEach(CDT.grid);
  };
  root.setAttribute('data-brand', CDT.brand());
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', function () { CDT.init(); sync(); });
  else { CDT.init(); sync(); }
})();
