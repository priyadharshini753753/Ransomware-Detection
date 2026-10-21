(async function () {
  const s = await api('/api/status');
  const m = s.metrics;
  if (!m) { $('#no-model').hidden = false; return; }
  $('#results-ui').hidden = false;
  if (s.stale) $('#stale').innerHTML = '<div class="notice">These results come from a different dataset than the one loaded now. <a href="/train">Train again</a> to update them.</div>';

  const names = labelsFor(m.params.target);
  const measures = [['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1 score'], ['fpr', 'False positive rate']];
  const better = (k, a, b) => (k === 'fpr' ? a < b : a > b);
  const rows = measures.map(([k, label]) => {
    const a = m.adaboost[k], d = m.dnn[k];
    const aBest = better(k, a, d), dBest = better(k, d, a);
    return [label, `<span class="${aBest ? 'best' : ''}" style="${aBest ? 'font-weight:700;color:var(--benign)' : ''}">${pct(a)}</span>`,
      `<span style="${dBest ? 'font-weight:700;color:var(--benign)' : ''}">${pct(d)}</span>`];
  });
  $('#cmp-table').innerHTML = table(['Measure', 'AdaBoost', 'Neural network'], rows, [1, 2]);
  $('#cmp-lead').textContent = `Measured on ${num(m.dataset.test_rows)} test files the models never trained on. The better score in each row is in green.`;

  groupedBars($('#cmp-chart'), measures.map(x => x[1]), [
    { name: 'AdaBoost', color: '#2a4bd7', values: measures.map(([k]) => m.adaboost[k]) },
    { name: 'Neural network', color: '#e08a00', values: measures.map(([k]) => m.dnn[k]) }]);

  confusionMatrix($('#cm-ada'), m.adaboost.confusion, [names[0], names[1]]);
  confusionMatrix($('#cm-dnn'), m.dnn.confusion, [names[0], names[1]]);
  const wrong = c => c.fp + c.fn;
  $('#cm-ada-lead').textContent = `${num(wrong(m.adaboost.confusion))} wrong out of ${num(m.dataset.test_rows)} test files.`;
  $('#cm-dnn-lead').textContent = `${num(wrong(m.dnn.confusion))} wrong out of ${num(m.dataset.test_rows)} test files.`;

  hbars($('#importance'), m.importances.filter(i => i.value > 0).slice(0, 14).map(i => ({ label: i.feature, value: i.value })), { fmt: v => (v * 100).toFixed(1) + '%' });

  const engine = m.dnn_engine === 'keras' ? 'TensorFlow / Keras' : 'scikit-learn MLPClassifier';
  const info = [
    ['Dataset', m.dataset.name], ['Files', `${num(m.dataset.rows)} (${num(m.dataset.train_rows)} train, ${num(m.dataset.test_rows)} test)`],
    ['Label column', m.params.target], ['Features kept', m.params.k],
    ['Neural network engine', engine], ['Standardised input', m.dnn_scaled ? 'Yes' : 'No'],
    ['Training time', m.seconds + ' seconds'], ['Trained at', new Date(m.trained_at).toLocaleString()],
  ];
  $('#run-info').innerHTML = info.map(([k, v]) =>
    `<div style="display:flex;justify-content:space-between;gap:16px;padding:9px 0;border-bottom:1px solid var(--line)"><dt class="muted">${esc(k)}</dt><dd style="margin:0;text-align:right;font-weight:600">${esc(v)}</dd></div>`).join('');
})().catch(e => toast(e.message, 'error'));
