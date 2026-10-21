const form = $('#auth-form');
const errBox = $('#form-error');
if (new URLSearchParams(location.search).get('registered')) {
  toast('Account created. Sign in to continue.');
}
form.addEventListener('submit', async e => {
  e.preventDefault();
  errBox.textContent = '';
  const f = Object.fromEntries(new FormData(form));
  if (!f.username || !f.password) { errBox.textContent = 'Enter a username and a password.'; return; }
  if ('confirm' in f && f.confirm !== f.password) { errBox.textContent = 'The two passwords do not match.'; return; }
  const btn = form.querySelector('button[type=submit]');
  btn.disabled = true;
  try {
    await api(form.dataset.endpoint, { method: 'POST', body: { username: f.username, password: f.password } });
    location.href = form.dataset.next;
  } catch (err) {
    errBox.textContent = err.message;
    btn.disabled = false;
  }
});
