/* Small dependency-free chart helpers (SVG / HTML strings) */
const NS_STYLE = 'font-family:var(--font);fill:#0f1b2d';

function groupedBars(el, labels, series) {
  const W = 720, H = 320, L = 46, R = 12, T = 22, B = 40;
  const iw = W - L - R, ih = H - T - B;
  const gw = iw / labels.length, bw = Math.min(52, (gw - 28) / series.length);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Grouped bar chart comparing the two models on each measure">`;
  [0, .25, .5, .75, 1].forEach(t => {
    const y = T + ih - t * ih;
    s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#d9dfe7"/>`;
    s += `<text x="${L - 8}" y="${y + 4}" text-anchor="end" font-size="12" style="${NS_STYLE};fill:#5a6577">${t * 100}%</text>`;
  });
  labels.forEach((lab, i) => {
    const gx = L + i * gw + (gw - bw * series.length - 6 * (series.length - 1)) / 2;
    series.forEach((se, j) => {
      const v = se.values[i], h = Math.max(v * ih, 2), x = gx + j * (bw + 6), y = T + ih - h;
      s += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="3" fill="${se.color}"/>`;
      s += `<text x="${x + bw / 2}" y="${y - 6}" text-anchor="middle" font-size="12" font-weight="600" style="${NS_STYLE}">${(v * 100).toFixed(v < .01 || v > .99 ? 2 : 1)}%</text>`;
    });
    s += `<text x="${L + i * gw + gw / 2}" y="${H - 14}" text-anchor="middle" font-size="13" style="${NS_STYLE}">${esc(lab)}</text>`;
  });
  el.innerHTML = s + '</svg>';
}

function hbars(el, items, { fmt = v => v.toFixed(3), color = '#2a4bd7', max } = {}) {
  const rowH = 36, L = 200, R = 64, W = 560, H = items.length * rowH + 6;
  const m = max || Math.max(...items.map(i => i.value), 1e-9);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Horizontal bar chart">`;
  items.forEach((it, i) => {
    const y = i * rowH + 4, w = Math.max((W - L - R) * it.value / m, 2);
    const label = it.label.length > 24 ? it.label.slice(0, 23) + '…' : it.label;
    s += `<text x="${L - 10}" y="${y + 19}" text-anchor="end" font-size="15" style="${NS_STYLE}"><title>${esc(it.label)}</title>${esc(label)}</text>`;
    s += `<rect x="${L}" y="${y + 5}" width="${w}" height="20" rx="3" fill="${color}"/>`;
    s += `<text x="${L + w + 8}" y="${y + 19}" font-size="14.5" font-weight="600" style="${NS_STYLE}">${esc(fmt(it.value))}</text>`;
  });
  el.innerHTML = s + '</svg>';
}

/* c = {tn, fp, fn, tp}; class 1 is "positive". Rows = actual, columns = predicted */
function confusionMatrix(el, c, names) {
  el.innerHTML = `
    <div class="cm">
      <div></div><div class="h">Predicted<br>${esc(names[0])}</div><div class="h">Predicted<br>${esc(names[1])}</div>
      <div class="h">Actually<br>${esc(names[0])}</div>
      <div class="c good"><b>${num(c.tn)}</b>correct</div><div class="c bad"><b>${num(c.fp)}</b>wrong</div>
      <div class="h">Actually<br>${esc(names[1])}</div>
      <div class="c bad"><b>${num(c.fn)}</b>wrong</div><div class="c good"><b>${num(c.tp)}</b>correct</div>
    </div>`;
}
