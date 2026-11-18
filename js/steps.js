/**
 * steps.js - guided, one-module-at-a-time walkthrough with REAL numbers.
 * Each step reveals its tensors in order (input -> operation -> result) like a short
 * video, auto-advances, and can be paused, stepped back and forth, or jumped to.
 * All values come from model.js (a tiny Transformer: d_model 4, one head, d_ff 8).
 */
(() => {
  'use strict';
  const TT = window.TinyTransformer;
  const SRC = TT.SRC, VOCAB = TT.VOCAB, T = TT.T, r2 = TT.r2;
  const C4 = ['d1', 'd2', 'd3', 'd4'], C8 = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8'], DL = ['<s>', 'Le'];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f = (v) => (typeof v === 'string' ? v : v === -Infinity ? '−∞' : (Number.isInteger(v) ? String(v) : v.toFixed(2)).replace('-', '−'));
  const p = (v) => (v < 0 ? `(${f(v)})` : f(v));
  const M = (label, m, rows, cols, kind) => ({ label, m, rows, cols, kind: kind || 'val' });
  const col = (A, j) => A.map((r) => r[j]);
  const dot = (name, a, b, res) => `${name} = ${a.map((x, i) => `${p(x)}×${p(b[i])}`).join(' + ')} = ${f(res)}`;
  const sum = (a) => a.reduce((s, v) => s + v, 0);

  function defs(m) {
    const S = [];
    const mu = sum(m.R1[0]) / 4, sd = Math.sqrt(sum(m.R1[0].map((v) => (v - mu) ** 2)) / 4 + 1e-5);
    const zeros = m.Hpre.flat().filter((v) => v <= 0).length;
    const ci = VOCAB.indexOf('chat');
    const exps = m.logits[0].map(Math.exp), P = (m.probs[0][ci] * 100).toFixed(0);

    S.push({ title: 'Words become token ids', mod: 'Input', side: 'enc', blocks: [],
      plain: 'A computer cannot read letters. Each word is cut into a token and replaced by its number in the vocabulary (the paper uses about 37,000 tokens). These ids are made up for the example.',
      rows: [[M('token ids', [[12, 87, 304]], ['id'], SRC, 'plain')]],
      ex: 'Nothing is learned yet. This is just a lookup in a dictionary.' });

    S.push({ title: 'Embedding: id → vector', mod: 'Input Embedding', side: 'enc', blocks: ['enc_emb'],
      plain: 'Each id picks one row from a learned table. That row is the word as a list of numbers (d_model of them: 4 here, 512 in the paper). The paper then multiplies by √d_model so the word signal is not drowned by the position signal added next.',
      formula: 'x₀ = Embedding[id] × √d_model',
      rows: [[M('Embedding table rows', m.Emb, SRC, C4, 'w'), '× 2 =', M('x₀', m.x0, SRC, C4)]],
      ex: `x₀[The, d1] = ${f(m.Emb[0][0])} × 2 = ${f(m.x0[0][0])}` });

    S.push({ title: 'Positional encoding: add word order', mod: 'Positional Encoding', side: 'enc', blocks: ['enc_pos'],
      plain: 'Attention looks at all words at once, so it has no idea which word came first. A fixed fingerprint of sine and cosine waves for each position is added to the embedding. Same word at a different position now has a different vector.',
      formula: 'PE(pos, 2i) = sin(pos / 10000^(2i/d_model)),  PE(pos, 2i+1) = cos(…)',
      rows: [[M('x₀', m.x0, SRC, C4), '+', M('PE (by position)', m.pe, SRC, C4, 'w'), '=', M('x₁ (layer input)', m.x1, SRC, C4)]],
      ex: `PE(pos 1, d1) = sin(1) = ${f(m.pe[1][0])}  ·  x₁[cat, d1] = ${f(m.x0[1][0])} + ${f(m.pe[1][0])} = ${f(m.x1[1][0])}` });

    S.push({ title: 'Self-attention: make Q, K and V', mod: 'Multi-Head Attention', side: 'enc', blocks: ['enc_mha'],
      plain: 'Every word makes three vectors by multiplying with three learned weight matrices: a Query (what am I looking for?), a Key (what do I offer?) and a Value (what information do I carry?). The paper splits this into 8 heads; here one head keeps the numbers readable.',
      formula: 'Q = x₁ W_Q,   K = x₁ W_K,   V = x₁ W_V',
      rows: [[M('x₁', m.x1, SRC, C4), '×', M('W_Q', m.Wq, C4, C4, 'w'), '=', M('Q', m.Q, SRC, C4)],
             [M('x₁', m.x1, SRC, C4), '×', M('W_K', m.Wk, C4, C4, 'w'), '=', M('K', m.K, SRC, C4)],
             [M('x₁', m.x1, SRC, C4), '×', M('W_V', m.Wv, C4, C4, 'w'), '=', M('V', m.V, SRC, C4)]],
      ex: dot('Q[The, d1]', m.x1[0], col(m.Wq, 0), m.Q[0][0]) });

    S.push({ title: 'Attention scores → weights', mod: 'Multi-Head Attention', side: 'enc', blocks: ['enc_mha'],
      plain: 'Each word compares its Query with every Key (a dot product). Big score = good match. We divide by √d_k (=2 here) so scores stay small, then softmax turns each row into percentages that add up to 1. These are the attention weights.',
      formula: 'A = softmax( Q Kᵀ / √d_k )',
      rows: [[M('Q', m.Q, SRC, C4), '×', M('Kᵀ', T(m.K), C4, SRC), '=', M('raw scores', m.S, SRC, SRC)],
             [M('raw scores', m.S, SRC, SRC), '× 0.5 =', M('scaled', m.Ss, SRC, SRC)],
             [M('scaled', m.Ss, SRC, SRC), 'softmax per row =', M('A: attention weights', m.A, SRC, SRC, 'prob')]],
      ex: dot('score[sleeps → cat]', m.Q[2], m.K[1], m.S[2][1]) + `.  Row sums: ${m.A.map((r) => r2(sum(r)).toFixed(2)).join(', ')}` });

    S.push({ title: 'Blend the Values', mod: 'Multi-Head Attention', side: 'enc', blocks: ['enc_mha'],
      plain: `Each word's new vector is a weighted mix of all Value vectors, using its attention weights. "sleeps" gives ${(m.A[2][1] * 100).toFixed(0)}% of its attention to "cat", so it now carries information about its subject. This is how words get context.`,
      formula: 'Z = A V',
      rows: [[M('A', m.A, SRC, SRC, 'prob'), '×', M('V', m.V, SRC, C4), '=', M('Z (context-aware)', m.Z, SRC, C4)]],
      ex: dot('Z[sleeps, d1]', m.A[2], col(m.V, 0), m.Z[2][0]) });

    S.push({ title: 'Add & Norm (residual + LayerNorm)', mod: 'Add & Norm', side: 'enc', blocks: ['enc_an1'],
      plain: 'The layer input is added back to its output (a residual connection, so nothing is lost), then each word vector is rescaled to mean 0 and spread 1 (LayerNorm). This keeps training stable.',
      formula: 'LayerNorm(x + Sublayer(x))',
      rows: [[M('x₁', m.x1, SRC, C4), '+', M('Z', m.Z, SRC, C4), '=', M('x₁ + Z', m.R1, SRC, C4)],
             [M('x₁ + Z', m.R1, SRC, C4), 'LayerNorm per row =', M('N₁', m.N1, SRC, C4)]],
      ex: `Row "The": mean = ${f(r2(mu))}, spread = ${f(r2(sd))}.  N₁[The, d1] = (${f(m.R1[0][0])} − ${f(r2(mu))}) ÷ ${f(r2(sd))} = ${f(m.N1[0][0])}` });

    S.push({ title: 'Feed-forward network', mod: 'Feed Forward', side: 'enc', blocks: ['enc_ffn'],
      plain: `Now each word is processed on its own by the same small network: expand to more numbers (8 here, 2048 in the paper), switch off negatives with ReLU, then shrink back. ${zeros} of the 24 hidden numbers were negative and became 0. W₁ is 4×8 and W₂ is 8×4 (learned).`,
      formula: 'FFN(x) = max(0, x W₁ + b₁) W₂ + b₂',
      rows: [[M('N₁', m.N1, SRC, C4), '× W₁ + b₁ =', M('expanded', m.Hpre, SRC, C8)],
             [M('expanded', m.Hpre, SRC, C8), 'ReLU: negatives → 0 =', M('H', m.H, SRC, C8)],
             [M('H', m.H, SRC, C8), '× W₂ =', M('F', m.F, SRC, C4)]],
      ex: `ReLU keeps ${f(m.H[0][4])} and zeroes every negative value.` });

    S.push({ title: 'Add & Norm → encoder output', mod: 'Add & Norm', side: 'enc', blocks: ['enc_an2'],
      plain: 'Another residual add and LayerNorm. The result is one context-rich vector per word: the encoder output, called the memory. The paper repeats steps 4 to 9 six times with different weights; here one layer is shown. The decoder will read this memory.',
      formula: 'memory = LayerNorm(N₁ + F)',
      rows: [[M('N₁', m.N1, SRC, C4), '+', M('F', m.F, SRC, C4), '=', M('N₁ + F', m.R2, SRC, C4)],
             [M('N₁ + F', m.R2, SRC, C4), 'LayerNorm per row =', M('Encoder output (memory)', m.mem, SRC, C4)]],
      ex: 'The encoder is done. Its output is computed once and reused for every word the decoder writes.' });

    S.push({ title: 'Decoder input: what we have written', mod: 'Output Embedding + Position', side: 'dec', blocks: ['dec_emb', 'dec_pos'],
      plain: 'The decoder writes one word at a time. To predict the 2nd word, its input is what exists so far: a start token <s> and "Le" (the paper calls this the outputs shifted right). They get embedded and positioned exactly like the source words.',
      formula: 'y₁ = Embedding × √d_model + PE',
      rows: [[M('Embedding rows', [m.E[0], m.E[1]], DL, C4, 'w'), '× 2 =', M('y₀', m.y0, DL, C4), '+', M('PE', m.ype, DL, C4, 'w'), '=', M('y₁', m.y1, DL, C4)]],
      ex: `y₁[Le, d1] = ${f(m.y0[1][0])} + ${f(m.ype[1][0])} = ${f(m.y1[1][0])}` });

    S.push({ title: 'Masked self-attention: no peeking ahead', mod: 'Masked Multi-Head Attention', side: 'dec', blocks: ['dec_mmha'],
      plain: 'Same recipe as before (Q, K, V from y₁), with one rule: a word may only look at itself and earlier words. Future spots are set to −∞, so softmax gives them exactly 0. Here <s> sees only itself, while "Le" sees <s> and itself.',
      formula: 'softmax( Q Kᵀ / √d_k + mask ) V',
      rows: [[M('scaled scores', m.dSs, DL, DL), 'mask future →', M('masked', m.dMasked, DL, DL)],
             [M('masked', m.dMasked, DL, DL), 'softmax per row =', M('weights', m.dA, DL, DL, 'prob')],
             [M('weights', m.dA, DL, DL, 'prob'), '× V =', M('Z', m.dZ, DL, C4)],
             [M('y₁ + Z', add2(m.y1, m.dZ), DL, C4), 'LayerNorm =', M('output', m.dN, DL, C4)]],
      ex: `e^(−∞) = 0, so the "<s>" row is [1.00, 0]. "Le" splits ${(m.dA[1][0] * 100).toFixed(0)}% on <s> and ${(m.dA[1][1] * 100).toFixed(0)}% on itself.` });

    S.push({ title: 'Cross-attention: read the encoder', mod: 'Cross-Attention', side: 'dec', blocks: ['dec_cross'],
      plain: `This is where the decoder looks at the source sentence. Queries come from the decoder; Keys and Values come from the encoder memory. "Le" puts ${(m.cA[1][0] * 100).toFixed(0)}% of its attention on "The", the word it translates.`,
      formula: 'softmax( Q_dec · K_memᵀ / √d_k ) · V_mem',
      rows: [[M('decoder out', m.dN, DL, C4), '× W_Q =', M('Q', m.cQ, DL, C4)],
             [M('memory', m.mem, SRC, C4), '× W_K =', M('K', m.cK, SRC, C4)],
             [M('memory', m.mem, SRC, C4), '× W_V =', M('V', m.cV, SRC, C4)],
             [M('Q Kᵀ × 0.5', m.cSs, DL, SRC), 'softmax =', M('weights', m.cA, DL, SRC, 'prob')],
             [M('weights', m.cA, DL, SRC, 'prob'), '× V =', M('Z', m.cZ, DL, C4)],
             [M('decoder out + Z', add2(m.dN, m.cZ), DL, C4), 'LayerNorm =', M('output', m.cN, DL, C4)]],
      ex: dot('score[Le → The]', m.cQ[1], m.cK[0], r2(sum(m.cQ[1].map((v, i) => v * m.cK[0][i]))) ) });

    S.push({ title: 'Feed-forward + Add & Norm', mod: 'Feed Forward', side: 'dec', blocks: ['dec_ffn', 'dec_an3'],
      plain: 'The same expand, ReLU, shrink network runs on each position, followed by a residual add and LayerNorm. The decoder stack repeats the last three modules 6 times in the paper. The last row, for "Le", now holds everything needed to guess the next word.',
      formula: 'out = LayerNorm(x + FFN(x))',
      rows: [[M('cross output', m.cN, DL, C4), '→ FFN =', M('F', m.fF, DL, C4)],
             [M('cross output + F', add2(m.cN, m.fF), DL, C4), 'LayerNorm =', M('decoder output', m.out, DL, C4)]],
      ex: `We keep only the last row, h = [${m.h.map(f).join(', ')}].` });

    S.push({ title: 'Linear: score every word', mod: 'Linear', side: 'dec', blocks: ['dec_lin'],
      plain: 'The vector h is compared with a row for every word in the vocabulary (6 here, ~37,000 in the paper). The paper reuses the embedding table for this (weight tying). These toy rows were set so "chat" fits h best; a real model learns this. The scores are called logits.',
      formula: 'logits = h Eᵀ',
      rows: [[M('h (for "Le")', [m.h], ['h'], C4), '·', M('Eᵀ: one row per word', m.E, VOCAB, C4, 'w'), '=', M('logits', m.logits, ['score'], VOCAB)]],
      ex: dot('logit[chat]', m.h, m.E[ci], m.logits[0][ci]) });

    S.push({ title: 'Softmax: pick the next word', mod: 'Softmax', side: 'dec', blocks: ['dec_soft'],
      plain: `Softmax turns the scores into probabilities that add up to 1. "chat" wins with ${P}%. (In the paper's translation task this is the model's guess for the next French word.)`,
      formula: 'P(word) = e^score ÷ Σ e^score',
      rows: [[M('logits', m.logits, ['score'], VOCAB), 'softmax =', M('probabilities', m.probs, ['P'], VOCAB, 'prob')]],
      ex: `P(chat) = e^${f(m.logits[0][ci])} ÷ (${exps.map((e) => e.toFixed(1)).join(' + ')}) = ${(m.probs[0][ci]).toFixed(2)}` });

    S.push({ title: 'Feed the word back, repeat', mod: 'Auto-regression', side: 'full', blocks: [],
      plain: '"chat" is written to the output and appended to the decoder input. Steps 10 to 15 run again with three words, and so on until the model picks <eos>. Encoding happened once; decoding repeats once per output word. Watch it all at once in the full run.',
      rows: [[M('decoder input next round', [['<s>', 'Le', 'chat']], ['tokens'], ['1', '2', '3'], 'plain')]],
      ex: '', full: true });
    return S;
  }
  const add2 = (A, B) => A.map((r, i) => r.map((v, j) => r2(v + B[i][j])));

  // ---------- rendering ----------
  const cellStyle = (v, kind) => {
    if (typeof v !== 'number' || kind === 'plain') return '';
    if (v === -Infinity) return 'background:repeating-linear-gradient(45deg,#e4ded2 0 4px,#f3eee4 4px 8px)';
    if (kind === 'prob') return `background:rgba(74,160,90,${(0.1 + Math.min(1, v) * 0.7).toFixed(2)})`;
    const a = Math.min(0.6, Math.abs(v) * (kind === 'w' ? 0.25 : 0.22) + (v ? 0.04 : 0));
    return v > 0 ? `background:rgba(74,163,210,${a.toFixed(2)})` : v < 0 ? `background:rgba(224,138,151,${a.toFixed(2)})` : '';
  };
  function table(it) {
    const head = it.cols ? `<tr><th></th>${it.cols.map((c) => `<th scope="col">${esc(c)}</th>`).join('')}</tr>` : '';
    const body = it.m.map((r, i) => `<tr><th scope="row">${esc(it.rows[i] ?? '')}</th>${r.map((v) => `<td style="${cellStyle(v, it.kind)}">${esc(f(v))}</td>`).join('')}</tr>`).join('');
    return `<table class="mx">${head}${body}</table>`;
  }

  let model = null, steps = null, cur = 0, playing = true, speedF = 1, timer = 0, tStart = 0, remain = 0, dom = null;
  const VB = { enc: '50 305 300 500', dec: '505 165 300 640', full: '0 0 860 800' };

  function show(i) {
    cur = Math.max(0, Math.min(steps.length - 1, i));
    const s = steps[cur];
    let idx = 0;
    dom.count.textContent = `Step ${cur + 1} of ${steps.length}`;
    dom.title.textContent = s.title;
    dom.mod.textContent = s.mod;
    dom.plain.textContent = s.plain;
    dom.formula.textContent = s.formula || '';
    dom.formula.hidden = !s.formula;
    dom.parts.innerHTML = s.rows.map((row) => `<div class="st-row">${row.map((it) => {
      const i2 = idx++;
      return typeof it === 'string' ? `<span class="st-op" style="--i:${i2}">${esc(it)}</span>` : `<figure class="st-item" style="--i:${i2}"><figcaption>${esc(it.label)}</figcaption>${table(it)}</figure>`;
    }).join('')}</div>`).join('');
    dom.ex.textContent = s.ex || '';
    dom.ex.hidden = !s.ex;
    dom.ex.style.setProperty('--i', idx);
    dom.full.hidden = !s.full;
    dom.svg.setAttribute('viewBox', VB[s.side]);
    dom.svg.querySelectorAll('.fb').forEach((g) => g.classList.toggle('on', s.blocks.includes(g.dataset.b)));
    dom.prev.disabled = cur === 0; dom.next.disabled = cur === steps.length - 1;
    dom.dots.forEach((d, k) => { d.classList.toggle('on', k === cur); d.classList.toggle('done', k < cur); d.setAttribute('aria-current', k === cur ? 'step' : 'false'); });
    dom.body.scrollTop = 0;
    const wrap = dom.body.closest('#fl_wrap'); if (wrap) wrap.scrollTop = 0;
    schedule((idx * 0.35 + 7) * speedF * 1000);
  }

  function schedule(ms) {
    clearTimeout(timer);
    remain = ms; tStart = performance.now();
    dom.bar.style.animation = 'none'; void dom.bar.offsetWidth;
    dom.bar.style.animation = `stfill ${ms}ms linear forwards`;
    dom.bar.style.animationPlayState = playing ? 'running' : 'paused';
    if (playing && cur < steps.length - 1) timer = setTimeout(() => show(cur + 1), ms);
    else if (cur >= steps.length - 1) { playing = false; sync(); }
  }
  function sync() { dom.play.textContent = playing ? '❚❚ Pause' : '▶ Play'; dom.play.setAttribute('aria-pressed', String(!playing)); dom.bar.style.animationPlayState = playing ? 'running' : 'paused'; }
  function toggle() {
    if (playing) { playing = false; clearTimeout(timer); remain -= performance.now() - tStart; }
    else { if (cur >= steps.length - 1) { show(0); playing = true; sync(); return; } playing = true; tStart = performance.now(); timer = setTimeout(() => show(cur + 1), Math.max(300, remain)); }
    sync();
  }

  function mount(root) {
    stop();
    if (!model) model = TT.build(21, 25);
    if (!steps) steps = defs(model);
    root.innerHTML = `
      <div class="steps">
        <div class="st-bar">
          <button type="button" class="fl-btn" id="st_prev">← Back</button>
          <button type="button" class="fl-btn" id="st_play">❚❚ Pause</button>
          <button type="button" class="fl-btn" id="st_next">Next →</button>
          <label class="fl-speed">Pace <select id="st_speed" aria-label="Auto-play pace"><option value="1.7">Slow</option><option value="1" selected>Normal</option><option value="0.6">Fast</option></select></label>
          <span class="st-count" id="st_count"></span>
        </div>
        <div class="st-progress" aria-hidden="true"><i id="st_bar"></i></div>
        <nav class="st-dots" id="st_dots" aria-label="Steps">${steps.map((s, k) => `<button type="button" class="st-dot" title="${esc(`${k + 1}. ${s.title}`)}" aria-label="Go to step ${k + 1}: ${esc(s.title)}">${k + 1}</button>`).join('')}</nav>
        <div class="st-grid">
          <div class="st-diagram"><svg class="flow-svg st-svg" viewBox="0 0 860 800" role="img" aria-label="Where this step happens in the Transformer">${window.Flow.archSvg()}</svg></div>
          <div class="st-body" id="st_body">
            <span class="st-module" id="st_mod"></span>
            <h3 id="st_title"></h3>
            <p class="st-plain" id="st_plain"></p>
            <code class="st-formula" id="st_formula"></code>
            <div class="st-parts" id="st_parts"></div>
            <p class="st-ex" id="st_ex"></p>
            <button type="button" class="fl-btn" id="st_full" hidden>Watch the full run →</button>
            <p class="st-note">Toy model: d_model = 4, one head, d_ff = 8 (paper: 512, 8 heads, 2048). Weights are small hand-friendly numbers, not trained. Every value is rounded to 2 decimals so you can redo any multiplication.</p>
          </div>
        </div>
      </div>`;
    const q = (id) => root.querySelector('#' + id);
    dom = { prev: q('st_prev'), next: q('st_next'), play: q('st_play'), count: q('st_count'), bar: q('st_bar'), title: q('st_title'), mod: q('st_mod'), plain: q('st_plain'), formula: q('st_formula'), parts: q('st_parts'), ex: q('st_ex'), full: q('st_full'), body: q('st_body'), svg: root.querySelector('.st-svg'), dots: [...root.querySelectorAll('.st-dot')] };
    if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) playing = false;
    dom.prev.addEventListener('click', () => show(cur - 1));
    dom.next.addEventListener('click', () => show(cur + 1));
    dom.play.addEventListener('click', toggle);
    dom.full.addEventListener('click', () => window.Flow.setMode('full'));
    q('st_speed').addEventListener('change', (e) => { speedF = parseFloat(e.target.value) || 1; show(cur); });
    dom.dots.forEach((d, k) => d.addEventListener('click', () => show(k)));
    sync(); show(cur);
  }
  function stop() { clearTimeout(timer); timer = 0; }

  window.Steps = { mount, stop, step: (d) => dom && show(cur + d), goto: (i) => dom && show(i), count: () => (steps ? steps.length : 0), pause: () => { if (playing && dom) toggle(); } };
})();
