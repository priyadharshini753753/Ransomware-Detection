/* Shared helpers */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const pct = (v, d = 2) => (v * 100).toFixed(d) + '%';
const num = v => Number(v).toLocaleString('en-US', { maximumFractionDigits: 2 });

async function api(path, { method = 'GET', body, form } = {}) {
  const headers = { 'X-CSRF-Token': $('meta[name=csrf-token]').content };
  let payload;
  if (form) payload = form;
  else if (body) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body); }
  const r = await fetch(path, { method, headers, body: payload, credentials: 'same-origin' });
  let data = {};
  try { data = await r.json(); } catch (e) { /* empty body */ }
  if (r.status === 401 && !location.pathname.startsWith('/login')) { location.href = '/login'; return new Promise(() => {}); }
  if (!r.ok) throw new Error(data.error || 'Something went wrong. Try again.');
  return data;
}

function toast(message, kind = '') {
  const host = $('#toasts');
  if (!host) return;
  const t = document.createElement('div');
  t.className = 'toast ' + kind;
  t.textContent = message;
  host.appendChild(t);
  setTimeout(() => t.remove(), 4500);
}

function labelsFor(target) {
  return String(target || '').toLowerCase() === 'benign'
    ? { 0: 'Malicious', 1: 'Benign' } : { 0: 'Class 0', 1: 'Class 1' };
}

function initTabs(onChange) {
  const tabs = $$('.tabs [role=tab]');
  tabs.forEach(b => b.addEventListener('click', () => {
    tabs.forEach(x => x.setAttribute('aria-selected', String(x === b)));
    tabs.forEach(x => { const p = $('#tab-' + x.dataset.tab); if (p) p.hidden = x !== b; });
    if (onChange) onChange(b.dataset.tab);
  }));
}

function table(headers, rows, numericCols = []) {
  const th = headers.map((h, i) => `<th class="${numericCols.includes(i) ? 'num' : ''}">${esc(h)}</th>`).join('');
  const tr = rows.map(r => '<tr>' + r.map((c, i) =>
    `<td class="${numericCols.includes(i) ? 'num' : ''}">${c}</td>`).join('') + '</tr>').join('');
  return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

function dropzone(zone, input, onFile) {
  ['dragenter', 'dragover'].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', ev => { if (ev.dataTransfer.files[0]) onFile(ev.dataTransfer.files[0]); });
  input.addEventListener('change', () => { if (input.files[0]) onFile(input.files[0]); });
}

const logoutBtn = $('#logout');
if (logoutBtn) logoutBtn.addEventListener('click', async () => {
  try { await api('/api/logout', { method: 'POST' }); } finally { location.href = '/login'; }
});
