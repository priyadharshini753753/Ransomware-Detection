const stepsEl = $('#steps');
let timer = null;

function renderSteps(job) {
  const names = job.steps && job.steps.length ? job.steps : ['Prepare data', 'Select features', 'Split train / test', 'Train AdaBoost', 'Train neural network', 'Save models'];
  stepsEl.innerHTML = names.map((n, i) => {
    const cls = job.done || i < job.step ? 'done' : (job.running && i === job.step ? 'now' : '');
    return `<li class="${cls}"><span class="dot">${cls === 'done' ? '✓' : ''}</span>${esc(n)}</li>`;
  }).join('');
}

async function poll() {
  const job = await api('/api/train/status');
  renderSteps(job);
  $('#go').disabled = job.running;
  if (job.running) {
    $('#prog-lead').textContent = 'Training is running. The neural network step takes the longest.';
    $('#train-done').hidden = true; $('#train-error').innerHTML = '';
    if (!timer) timer = setInterval(() => poll().catch(() => {}), 1500);
  } else {
    if (timer) { clearInterval(timer); timer = null; }
    if (job.error) {
      $('#prog-lead').textContent = 'Training stopped.';
      $('#train-error').innerHTML = `<div class="notice error">${esc(job.error)}</div>`;
    } else if (job.done) {
      $('#prog-lead').textContent = 'Done.';
      $('#train-done').hidden = false;
    }
  }
}

(async function init() {
  const [ds, st] = await Promise.all([api('/api/dataset'), api('/api/status')]);
  const sel = $('#target');
  sel.innerHTML = ds.columns.map(c => `<option ${c.name === ds.default_target ? 'selected' : ''}>${esc(c.name)}</option>`).join('');
  $('#k').max = Math.max(5, ds.cols * 4);
  $('#engine-note').textContent = st.keras
    ? 'The neural network will be trained with TensorFlow / Keras.'
    : 'TensorFlow is not installed, so the neural network will use scikit-learn (same layers, similar results). Install tensorflow to use Keras.';
  renderSteps(st.job);
  if (st.job.running || st.job.done || st.job.error) await poll();
})().catch(e => toast(e.message, 'error'));

$('#k').addEventListener('input', e => $('#k-out').textContent = e.target.value);
$('#test_size').addEventListener('input', e => $('#t-out').textContent = e.target.value + '%');

$('#train-form').addEventListener('submit', async e => {
  e.preventDefault();
  $('#train-error').innerHTML = ''; $('#train-done').hidden = true;
  try {
    await api('/api/train', { method: 'POST', body: {
      target: $('#target').value, k: +$('#k').value, test_size: +$('#test_size').value / 100,
      epochs: +$('#epochs').value, scale_dnn: $('#scale').checked } });
    $('#go').disabled = true;
    await poll();
  } catch (err) { toast(err.message, 'error'); }
});
