let schema = null;
let lastBatch = null;
initTabs();

const fieldsEl = $('#fields');

function buildForm() {
  fieldsEl.innerHTML = schema.fields.map(f => {
    const id = 'f-' + f.name;
    if (f.kind === 'text') {
      return `<label class="field wide" for="${esc(id)}">${esc(f.name)}
        <input type="text" id="${esc(id)}" name="${esc(f.name)}" placeholder="for example setup.exe" autocomplete="off">
        <span class="hint">Words in the file name are one of the signals the model uses.</span></label>`;
    }
    return `<label class="field" for="${esc(id)}">${esc(f.name)}
      <input type="number" step="any" id="${esc(id)}" name="${esc(f.name)}" value="${esc(f.median)}" required>
      <span class="hint">Typical: ${num(f.min)} to ${num(f.max)}</span></label>`;
  }).join('');
  const numeric = schema.fields.filter(f => f.kind === 'number').map(f => f.name);
  $('#batch-cols').innerHTML = 'The CSV needs these columns: ' + numeric.map(c => `<code>${esc(c)}</code>`).join(', ') +
    (schema.fields.some(f => f.kind === 'text') ? '. A file name column is optional but improves accuracy.' : '.');
}

function readValues() {
  const v = {};
  $$('input', fieldsEl).forEach(i => v[i.name] = i.value);
  return v;
}

function verdictHTML(r) {
  const malP = 1 - r.p_class1, benP = r.p_class1;
  const modelName = r.model === 'dnn' ? 'The neural network' : 'AdaBoost';
  const conf = Math.max(malP, benP);
  const cls = r.label === 1 ? 'benign' : 'malicious';
  const rows = schema.fields.filter(f => f.kind === 'number').map(f => {
    const v = Number($('#f-' + CSS.escape(f.name)).value);
    return [esc(f.name), num(v), num(f.median_class1), num(f.median_class0)];
  });
  return `<div class="verdict ${cls}">
    <div class="muted small" style="font-weight:600">${esc(modelName)} says this file is</div>
    <div class="word">${esc(r.verdict)}</div>
    <div>${pct(conf, 1)} confident</div>
    <div class="meter" role="img" aria-label="${pct(malP, 1)} malicious, ${pct(benP, 1)} benign">
      <i class="m" style="width:${malP * 100}%"></i><i class="b" style="width:${benP * 100}%"></i></div>
    <div class="meter-key"><span class="m">${esc(r.labels[0])} ${pct(malP, 1)}</span><span class="b">${esc(r.labels[1])} ${pct(benP, 1)}</span></div>
    <div class="compare">${table(['Field', 'This file', 'Typical ' + r.labels[1].toLowerCase(), 'Typical ' + r.labels[0].toLowerCase()], rows, [1, 2, 3])}</div>
    <p class="small muted" style="margin:12px 0 0">Typical values are medians from the training data. This is a statistical estimate, not a guarantee. Treat suspicious files with care.</p>
  </div>`;
}

$('#predict-form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = e.target.querySelector('button[type=submit]');
  btn.disabled = true;
  try {
    const r = await api('/api/predict', { method: 'POST', body: { values: readValues(), model: e.target.model.value } });
    $('#verdict-host').innerHTML = verdictHTML(r);
  } catch (err) { toast(err.message, 'error'); }
  btn.disabled = false;
});

async function loadExample(label) {
  try {
    const row = await api('/api/sample?label=' + label);
    $$('input', fieldsEl).forEach(i => { if (i.name in row) i.value = row[i.name]; });
    toast('Example loaded. Choose Check this file.');
  } catch (e) { toast(e.message, 'error'); }
}
$('#ex-benign').addEventListener('click', () => loadExample(1));
$('#ex-malicious').addEventListener('click', () => loadExample(0));

/* ---- batch ---- */
let batchFile = null;
function setBatchFile(f) { batchFile = f; $('#drop-label').textContent = f.name; }
dropzone($('#drop'), $('#batch-file'), setBatchFile);

$('#batch-go').addEventListener('click', async () => {
  if (!batchFile) { toast('Choose a CSV file first.', 'error'); return; }
  const fd = new FormData();
  fd.append('file', batchFile);
  fd.append('model', $('input[name=bmodel]:checked').value);
  const btn = $('#batch-go'); btn.disabled = true;
  try {
    const r = await api('/api/predict/batch', { method: 'POST', form: fd });
    lastBatch = r;
    const shown = r.rows.slice(0, 200);
    const out = $('#batch-out');
    out.hidden = false;
    out.innerHTML = `<h2>Results</h2>
      <p class="lead">${num(r.rows.length)} files checked: ` +
      Object.entries(r.counts).map(([k, v]) => `<b>${num(v)}</b> ${esc(k.toLowerCase())}`).join(', ') + '.</p>' +
      `<div class="btn-row" style="margin-bottom:14px"><button class="btn ghost small" id="dl">Download results as CSV</button></div>` +
      table(['Row', 'File name', 'Verdict', 'Chance of ' + r.labels[1].toLowerCase()],
        shown.map(x => [x.row, esc(x.name), `<span class="pill ${x.verdict === r.labels[1] ? 'ok' : 'bad'}">${esc(x.verdict)}</span>`, pct(x.p_class1, 1)]), [0, 3]) +
      (r.rows.length > shown.length ? `<p class="small muted" style="margin-top:10px">Showing the first ${shown.length} rows. The download has all of them.</p>` : '');
    $('#dl').addEventListener('click', downloadBatch);
  } catch (e) { toast(e.message, 'error'); }
  btn.disabled = false;
});

function downloadBatch() {
  const q = v => '"' + String(v).replace(/"/g, '""').replace(/^([=+\-@])/, "'$1") + '"';
  const lines = ['row,file_name,verdict,probability'].concat(
    lastBatch.rows.map(x => [x.row, q(x.name), q(x.verdict), x.p_class1].join(',')));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
  a.download = 'predictions.csv'; a.click();
  URL.revokeObjectURL(a.href);
}

(async function init() {
  try {
    schema = await api('/api/schema');
  } catch (e) {
    $('#no-model').hidden = false; return;
  }
  $('#predict-ui').hidden = false;
  buildForm();
})();
