initTabs();
let summary;

async function load() {
  summary = await api('/api/dataset');
  $('#ds-title').textContent = summary.name;
  $('#ds-sub').textContent = summary.missing_total === 0 ? 'No missing values found.' : `${num(summary.missing_total)} missing values found. They are filled with the column median during training.`;
  $('#ds-facts').innerHTML =
    `<div class="fact"><b>${num(summary.rows)}</b><span>rows</span></div>` +
    `<div class="fact"><b>${summary.cols}</b><span>columns</span></div>` +
    `<div class="fact"><b>${num(summary.missing_total)}</b><span>missing values</span></div>`;

  const h = summary.head;
  $('#tab-preview').innerHTML = `<p class="muted small">First ${h.data.length} rows</p>` +
    table(h.columns, h.data.map(r => r.map(c => esc(c))));

  $('#tab-columns').innerHTML = table(['Column', 'Type', 'Distinct values', 'Missing'],
    summary.columns.map(c => [esc(c.name), esc(c.dtype), num(c.unique), c.missing ? `<b>${num(c.missing)}</b>` : '0']), [2, 3]);

  $('#tab-stats').innerHTML = summary.stats.length
    ? table(['Column', 'Mean', 'Std. deviation', 'Min', 'Max'], summary.stats.map(r => [esc(r.name), num(r.mean), num(r.std), num(r.min), num(r.max)]), [1, 2, 3, 4])
    : '<p class="muted">This dataset has no numeric columns.</p>';

  if (summary.class_counts) {
    const names = labelsFor(summary.default_target);
    const items = Object.entries(summary.class_counts).map(([k, v]) => ({ label: `${names[k] ?? k} (${summary.default_target} = ${k})`, value: v }));
    $('#tab-balance').innerHTML = '<div class="chart" id="bal"></div>' +
      `<p class="small muted">Counts of each value in the "${esc(summary.default_target)}" column. A big gap between the two would bias the models.</p>`;
    hbars($('#bal'), items, { fmt: v => num(v) });
  } else {
    $('#tab-balance').innerHTML = '<p class="muted">No two-class column found to chart.</p>';
  }
}

async function upload(file) {
  const fd = new FormData(); fd.append('file', file);
  try {
    await api('/api/dataset/upload', { method: 'POST', form: fd });
    toast('Dataset loaded. Train the models again to use it.');
    await load();
  } catch (e) { toast(e.message, 'error'); }
}
dropzone($('#drop'), $('#file'), upload);
$('#reset').addEventListener('click', async () => {
  try { await api('/api/dataset/reset', { method: 'POST' }); toast('Bundled dataset restored.'); await load(); }
  catch (e) { toast(e.message, 'error'); }
});
load().catch(e => toast(e.message, 'error'));
