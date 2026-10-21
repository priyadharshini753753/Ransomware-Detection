(async function () {
  const s = await api('/api/status');
  const m = s.metrics;
  const trained = !!m;

  if (s.job.running) $('#notice').innerHTML = '<div class="notice info">Training is running. Follow it on the <a href="/train">Train models</a> page.</div>';
  else if (s.stale) $('#notice').innerHTML = '<div class="notice">The dataset changed after the models were trained. <a href="/train">Train again</a> to use the new data.</div>';

  const steps = [
    { t: 'Dataset', d: `${num(s.dataset.rows)} files loaded`, done: true, href: '/dataset', link: 'Look at the data' },
    { t: 'Train models', d: trained ? 'AdaBoost and neural network trained' : 'Not trained yet', done: trained, href: '/train', link: trained ? 'Train again' : 'Start training' },
    { t: 'Compare results', d: trained ? 'Metrics are ready' : 'Available after training', done: trained, href: '/results', link: 'Open results' },
    { t: 'Check a file', d: trained ? 'Ready to classify files' : 'Available after training', done: false, href: '/predict', link: 'Check a file' },
  ];
  const firstOpen = steps.findIndex(x => !x.done);
  $('#pipeline').innerHTML = steps.map((x, i) =>
    `<li class="${x.done ? 'done' : (i === (trained ? 3 : 1) ? 'now' : '')}"><h3>${esc(x.t)}</h3><p>${esc(x.d)}</p>` +
    ((x.done || i === (trained ? 3 : 1)) ? `<a href="${x.href}">${esc(x.link)}</a>` : '') + '</li>').join('');

  $('#ds-name').textContent = s.dataset.name;
  $('#ds-facts').innerHTML =
    `<div class="fact"><b>${num(s.dataset.rows)}</b><span>rows</span></div>` +
    `<div class="fact"><b>${s.dataset.cols}</b><span>columns</span></div>`;

  if (!trained) {
    $('#res-lead').textContent = 'No models have been trained yet.';
    $('#res-body').innerHTML = '<a class="btn" href="/train">Train models</a>';
    return;
  }
  const best = m.adaboost.accuracy >= m.dnn.accuracy ? 'AdaBoost' : 'Neural network';
  $('#res-lead').textContent = `${best} scored highest on accuracy.`;
  $('#res-body').innerHTML = table(['Measure', 'AdaBoost', 'Neural network'], [
    ['Accuracy', pct(m.adaboost.accuracy), pct(m.dnn.accuracy)],
    ['F1 score', pct(m.adaboost.f1), pct(m.dnn.f1)],
    ['False positive rate', pct(m.adaboost.fpr), pct(m.dnn.fpr)],
  ], [1, 2]) + `<p class="small muted" style="margin-top:12px">Trained on ${num(m.dataset.train_rows)} files, tested on ${num(m.dataset.test_rows)}. <a href="/results">Full results</a></p>`;
})().catch(e => toast(e.message, 'error'));
