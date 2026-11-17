/**
 * flow.js - Animated Transformer data flow.
 * Words enter the encoder, travel up through every block, the encoder output
 * feeds the decoder's cross-attention, and the decoder writes the translation
 * one word at a time. Everything is derived from one clock `t`, so play,
 * pause, replay, speed and scrubbing all stay in sync.
 *
 * NOTE: attention weights and word probabilities are hand-written for teaching.
 * They are not taken from a trained model.
 */
(() => {
  'use strict';

  const ENC = 215, DEC = 648;
  const SRC = ['The', 'cat', 'sleeps'];
  const OUT = ['Le', 'chat', 'dort', '<eos>'];
  const DEC_IN = ['<s>', 'Le', 'chat', 'dort'];
  const ENC_MOVE = 5.4, ENC_DUR = 6, STEP = 6, TOTAL = ENC_DUR + STEP * OUT.length;

  const C = { embed: '#f7ccd2', attn: '#ffddae', norm: '#fdfdb3', ffn: '#bfe3f5', lin: '#dcdcf1', soft: '#cbeacb', ink: '#1b1b1f', mem: '#ffd7a0' };

  // Teaching weights (illustrative only)
  const W_ENC = [[.6, .3, .1], [.3, .5, .2], [.1, .4, .5]];
  const W_CAUSAL = [[1], [.3, .7], [.2, .3, .5], [.1, .2, .3, .4]];
  const W_CROSS = [[.8, .15, .05], [.2, .7, .1], [.05, .25, .7], [.2, .3, .5]];
  const CAND = [
    [['Le', .72], ['Un', .14], ['La', .09]],
    [['chat', .81], ['félin', .10], ['chien', .05]],
    [['dort', .77], ['dormait', .12], ['repose', .07]],
    [['<eos>', .93], ['.', .05], ['!', .01]]
  ];

  // [id, label, cx, cy, w, h, colour, inspector key]
  const BLOCKS = [
    ['enc_emb', 'Input Embedding', ENC, 690, 200, 34, C.embed, 'input_embed'],
    ['enc_mha', 'Multi-Head Attention', ENC, 548, 200, 36, C.attn, 'enc_self_attn'],
    ['enc_an1', 'Add & Norm', ENC, 506, 200, 24, C.norm, null],
    ['enc_ffn', 'Feed Forward', ENC, 452, 200, 34, C.ffn, 'enc_ffn'],
    ['enc_an2', 'Add & Norm', ENC, 410, 200, 24, C.norm, null],
    ['dec_emb', 'Output Embedding', DEC, 690, 214, 34, C.embed, 'input_embed'],
    ['dec_mmha', 'Masked Multi-Head Attention', DEC, 548, 214, 36, C.attn, 'dec_masked_self_attn'],
    ['dec_an1', 'Add & Norm', DEC, 506, 214, 24, C.norm, null],
    ['dec_cross', 'Cross-Attention', DEC, 452, 214, 34, C.attn, 'dec_cross_attn'],
    ['dec_an2', 'Add & Norm', DEC, 410, 214, 24, C.norm, null],
    ['dec_ffn', 'Feed Forward', DEC, 356, 214, 34, C.ffn, 'dec_ffn'],
    ['dec_an3', 'Add & Norm', DEC, 314, 214, 24, C.norm, null],
    ['dec_lin', 'Linear', DEC, 240, 214, 28, C.lin, 'linear_softmax'],
    ['dec_soft', 'Softmax', DEC, 190, 214, 28, C.soft, 'linear_softmax']
  ];
  const POS = [['enc_pos', ENC, 636], ['dec_pos', DEC, 636]];

  const EW = [[748], [690, 'enc_emb'], [636, 'enc_pos'], [548, 'enc_mha'], [506, 'enc_an1'], [452, 'enc_ffn'], [410, 'enc_an2'], [345, 'mem']].map(([y, b]) => ({ y, b }));
  const DW = [[748], [690, 'dec_emb'], [636, 'dec_pos'], [548, 'dec_mmha'], [506, 'dec_an1'], [452, 'dec_cross'], [410, 'dec_an2'], [356, 'dec_ffn'], [314, 'dec_an3'], [240, 'dec_lin'], [190, 'dec_soft']].map(([y, b]) => ({ y, b }));

  const BD = {
    enc_emb: 'Embedding: each word becomes a vector of 512 numbers (scaled by √d_model).',
    enc_pos: 'Positional encoding: sine and cosine waves are added so the model knows word order.',
    enc_mha: 'Self-attention: every word looks at every other word at the same time. Thicker arcs mean more attention.',
    enc_an1: 'Add & Norm: add the sub-layer input back (residual), then LayerNorm.',
    enc_ffn: 'Feed-forward: the same small network (512 → 2048 → 512) runs on each word.',
    enc_an2: 'Add & Norm again. This layer repeats N = 6 times in the paper.',
    mem: 'Encoder output is ready: one context-rich vector per word. The decoder reads these as keys and values.',
    dec_emb: 'The words written so far (starting with <s>) are embedded.',
    dec_pos: 'Positional encoding is added here too.',
    dec_mmha: 'Masked self-attention: a word may only look at earlier words, never at the future.',
    dec_an1: 'Add & Norm.',
    dec_cross: 'Cross-attention: decoder queries look at the encoder output. Orange lines show which source word matters.',
    dec_an2: 'Add & Norm.',
    dec_ffn: 'Feed-forward network on each position.',
    dec_an3: 'Add & Norm. The decoder block also repeats N = 6 times.',
    dec_lin: 'Linear: project to a score for every word in the vocabulary.',
    dec_soft: 'Softmax: scores become probabilities. The most likely word is chosen.'
  };

  const cl = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, k) => a + (b - a) * k;
  const sm = (k) => k * k * (3 - 2 * k);
  const tx = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const yAt = (wp, f) => {
    f = cl(f, 0, wp.length - 1);
    const i = Math.min(Math.floor(f), wp.length - 2);
    return lerp(wp[i].y, wp[i + 1].y, sm(f - i));
  };

  function chip(x, y, text, fill, op = 1, sc = 1) {
    return `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) scale(${sc})" opacity="${op.toFixed(2)}"><rect x="-26" y="-10" width="52" height="20" rx="10" fill="${fill}" stroke="${C.ink}" stroke-width="1.5"/><text y="4" text-anchor="middle" font-size="10.5" font-weight="700" fill="${C.ink}">${tx(text)}</text></g>`;
  }
  function arc(x1, y1, x2, y2, w, col) {
    const mx = (x1 + x2) / 2, lift = 20 + Math.abs(x2 - x1) * 0.22;
    return `<path d="M${x1.toFixed(1)} ${(y1 - 11).toFixed(1)} Q${mx.toFixed(1)} ${(Math.min(y1, y2) - 11 - lift).toFixed(1)} ${x2.toFixed(1)} ${(y2 - 11).toFixed(1)}" fill="none" stroke="${col}" stroke-width="${(1 + w * 7).toFixed(1)}" stroke-linecap="round" opacity="${(0.25 + w * 0.75).toFixed(2)}"/>`;
  }

  function staticSvg() {
    let s = `<defs><marker id="fl_ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="${C.ink}"/></marker></defs>`;
    s += `<rect x="${ENC - 112}" y="388" width="224" height="206" rx="16" fill="#f1f1f1" stroke="${C.ink}" stroke-width="1.5"/>`;
    s += `<rect x="${DEC - 118}" y="290" width="236" height="304" rx="16" fill="#f1f1f1" stroke="${C.ink}" stroke-width="1.5"/>`;
    s += `<text x="${ENC - 122}" y="494" text-anchor="end" font-size="14" font-weight="700">N×</text><text x="${DEC + 128}" y="446" font-size="14" font-weight="700">N×</text>`;
    s += `<path d="M${ENC} 730V356" stroke="${C.ink}" stroke-width="1.5" marker-end="url(#fl_ar)"/><path d="M${DEC} 730V206" stroke="${C.ink}" stroke-width="1.5" marker-end="url(#fl_ar)"/>`;
    s += `<path d="M${ENC + 100} 345H478V452H${DEC - 110}" fill="none" stroke="${C.ink}" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#fl_ar)"/><text x="486" y="404" font-size="11" fill="#4d4d57">K, V</text>`;
    s += `<path d="M${DEC + 112} 62H806V748H${DEC + 114}" fill="none" stroke="#6e6e79" stroke-width="1.5" stroke-dasharray="4 4" marker-end="url(#fl_ar)"/><text transform="translate(822 400) rotate(90)" text-anchor="middle" font-size="11" fill="#4d4d57">fed back (shifted right)</text>`;
    s += `<text x="${ENC}" y="322" text-anchor="middle" font-size="12" font-weight="700">Encoder output (memory)</text>`;
    s += `<text x="${ENC}" y="788" text-anchor="middle" font-size="12" font-weight="700">Inputs (English)</text><text x="${DEC}" y="788" text-anchor="middle" font-size="12" font-weight="700">Outputs so far (French)</text>`;
    s += `<text x="${DEC}" y="26" text-anchor="middle" font-size="12" font-weight="700">Output: the translation</text>`;
    BLOCKS.forEach(([id, label, cx, cy, w, h, fill, key]) => {
      s += `<g class="fb" data-b="${id}"${key ? ` data-key="${key}" tabindex="0" role="button" aria-label="Inspect ${tx(label)}"` : ''}><rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="8" fill="${fill}"/><text x="${cx}" y="${cy + 4}" text-anchor="middle" font-size="${label.length > 26 ? 11 : 12}" font-weight="600">${tx(label)}</text></g>`;
    });
    POS.forEach(([id, cx, cy]) => {
      const left = id === 'enc_pos';
      s += `<g class="fb" data-b="${id}" data-key="input_pe" tabindex="0" role="button" aria-label="Inspect Positional Encoding"><circle cx="${cx}" cy="${cy}" r="15" fill="#fff"/><text x="${cx}" y="${cy + 6}" text-anchor="middle" font-size="18" font-weight="700">⊕</text><text x="${cx + (left ? -24 : 24)}" y="${cy + 4}" text-anchor="${left ? 'end' : 'start'}" font-size="11" fill="#4d4d57">Positional Encoding</text></g>`;
    });
    return s;
  }

  let dom = null, raf = 0, t = 0, last = 0, playing = true, speed = 1, lastActive = null, lastCap = '';

  function draw() {
    let o = '', active = null, phase = '', text = '';
    const inEnc = t < ENC_DUR;

    // ---- encoder chips (always drawn; they end as the "memory") ----
    const fe = cl(t / ENC_MOVE) * (EW.length - 1 + 0.2);
    const ex = (i) => ENC + (i - (SRC.length - 1) / 2) * 58;
    const ey = (i) => yAt(EW, fe - i * 0.1);
    if (inEnc) {
      const fm = fe - 0.1, r = Math.round(fm);
      if (EW[r] && Math.abs(fm - r) < 0.32 && t > 0.1) active = EW[r].b;
    }
    if (active === 'enc_mha') {
      for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) o += arc(ex(i), ey(i), ex(j), ey(j), (W_ENC[i][j] + W_ENC[j][i]) / 2, '#e9a24a');
    }
    SRC.forEach((w, i) => { o += chip(ex(i), ey(i) , w, ey(i) < 380 ? C.mem : C.embed); });

    // ---- decoder ----
    let k = 0, tau = 0;
    if (t >= TOTAL) { k = OUT.length; }
    else if (!inEnc) { k = Math.min(OUT.length - 1, Math.floor((t - ENC_DUR) / STEP)); tau = t - ENC_DUR - k * STEP; }

    if (inEnc) {
      o += chip(DEC, 748, DEC_IN[0], C.lin, 0.35);
      phase = 'Encoder · all words move up together';
      text = active ? BD[active] : 'Words enter the model as tokens and start climbing the encoder stack.';
    } else if (k >= OUT.length) {
      phase = 'Done';
      text = 'Translation complete: “Le chat dort”. Press Replay to watch it again.';
    } else {
      const n = k + 1;
      const dx = (i) => DEC + (i - (n - 1) / 2) * 54;
      const fd = cl(tau / 4.3) * (DW.length - 1 + 0.1 * n);
      const dy = (i) => yAt(DW, fd - i * 0.1);
      const fm = fd - 0.1 * (n - 1) / 2, r = Math.round(fm);
      if (tau < 4.3 && DW[r] && Math.abs(fm - r) < 0.32 && tau > 0.15) active = DW[r].b;
      else if (tau >= 4.3 && tau < 5.3) active = 'dec_soft';

      if (active === 'dec_mmha') {
        for (let i = 1; i < n; i++) for (let j = 0; j < i; j++) o += arc(dx(j), dy(j), dx(i), dy(i), W_CAUSAL[i][j], '#e9a24a');
      }
      if (active === 'dec_cross') {
        for (let i = 0; i < n; i++) for (let j = 0; j < SRC.length; j++) {
          const w = W_CROSS[i][j], mx = ex(j);
          o += `<path d="M${mx.toFixed(1)} 345C${(mx + 170).toFixed(1)} 345 ${(dx(i) - 170).toFixed(1)} ${dy(i).toFixed(1)} ${(dx(i) - 24).toFixed(1)} ${dy(i).toFixed(1)}" fill="none" stroke="#e9a24a" stroke-width="${(0.8 + w * 5).toFixed(1)}" stroke-dasharray="7 5" stroke-dashoffset="${(-t * 45).toFixed(0)}" opacity="${(0.15 + w * 0.85).toFixed(2)}"/>`;
        }
      }
      const fadeIn = k > 0 ? cl(tau / 0.35) : 1, fadeOut = 1 - cl((tau - 4.4) / 0.3);
      if (fadeOut > 0) for (let i = 0; i < n; i++) o += chip(dx(i), dy(i), DEC_IN[i], dy(i) < 250 ? C.soft : C.lin, fadeIn * fadeOut);

      // probability bars
      if (tau >= 4.3 && tau < 5.7) {
        const bp = cl((tau - 4.3) / 0.5), bo = 1 - cl((tau - 5.4) / 0.3);
        o += `<g opacity="${bo.toFixed(2)}">`;
        CAND[k].forEach(([w, p], i) => {
          const y = 104 + i * 22, win = i === 0;
          o += `<text x="${DEC - 100}" y="${y + 11}" font-size="11" font-weight="${win ? 700 : 400}">${tx(w)}</text><rect x="${DEC - 30}" y="${y}" width="${(p * 130 * bp).toFixed(1)}" height="14" rx="4" fill="${win ? C.soft : '#ece6d9'}" stroke="${C.ink}" stroke-width="${win ? 1.5 : 1}"/><text x="${DEC - 30 + p * 130 * bp + 6}" y="${y + 11}" font-size="10" fill="#4d4d57">${Math.round(p * 100 * bp)}%</text>`;
        });
        o += '</g>';
      }

      if (k < 3 && tau >= 5.4) {            // feedback ghost chip: top -> right lane -> bottom slot
        const p = cl((tau - 5.4) / 0.6), x0 = DEC + (k - 1.5) * 54, pts = [[x0, 62], [806, 62], [806, 748], [DEC + ((k + 1) / 2) * 54, 748]];
        const lens = [pts[1][0] - pts[0][0], 686, Math.abs(pts[3][0] - 806)], tot = lens[0] + lens[1] + lens[2];
        let d = p * tot, x, y;
        if (d <= lens[0]) { x = pts[0][0] + d; y = 62; }
        else if (d <= lens[0] + lens[1]) { x = 806; y = 62 + (d - lens[0]); }
        else { x = 806 - (d - lens[0] - lens[1]); y = 748; }
        o += chip(x, y, OUT[k], C.soft, 1);
      }

      phase = `Decoder · step ${k + 1} of ${OUT.length} · predicting “${OUT[k]}”`;
      if (tau >= 5.1) text = k < 3 ? `“${OUT[k]}” is written to the output, then fed back in as the next decoder input.` : 'End-of-sentence chosen. The translation is finished.';
      else if (tau >= 4.3) text = BD.dec_soft;
      else text = active ? BD[active] : 'The words written so far enter the decoder.';
    }

    // ---- output row (top) ----
    const done = k >= OUT.length ? OUT.length : k;
    for (let j = 0; j < done; j++) o += chip(DEC + (j - 1.5) * 54, 62, OUT[j], C.soft);
    if (k < OUT.length && tau >= 5.1) {
      const p = sm(cl((tau - 5.1) / 0.4));
      o += chip(lerp(DEC, DEC + (k - 1.5) * 54, p), lerp(190, 62, p), OUT[k], C.soft);
    }

    dom.dyn.innerHTML = o;
    if (active !== lastActive) {
      dom.blocks.forEach((g) => g.classList.toggle('on', g.dataset.b === active));
      lastActive = active;
    }
    const cap = phase + '|' + text;
    if (cap !== lastCap) { dom.phase.textContent = phase; dom.text.textContent = text; lastCap = cap; }
    if (!dom.seeking) dom.seek.value = String(Math.round(t * 100));
    dom.time.textContent = `${Math.min(t, TOTAL).toFixed(1)}s / ${TOTAL}s`;
  }

  function tick(ts) {
    const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0;
    last = ts;
    if (playing) { t += dt * speed; if (t > TOTAL + 2.5) t = 0; }
    draw();
    raf = requestAnimationFrame(tick);
  }

  function setPlaying(p) {
    playing = p;
    dom.play.textContent = p ? '❚❚ Pause' : '▶ Play';
    dom.play.setAttribute('aria-pressed', String(!p));
  }

  function mountFull(root) {
    stop();
    root.innerHTML = `
      <div class="flow">
        <div class="flow-bar">
          <button type="button" class="fl-btn" id="fl_play">❚❚ Pause</button>
          <button type="button" class="fl-btn" id="fl_replay">↺ Replay</button>
          <label class="fl-speed">Speed
            <select id="fl_speed" aria-label="Animation speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
          <input id="fl_seek" type="range" min="0" max="${TOTAL * 100}" value="0" aria-label="Animation timeline">
          <span class="fl-time" id="fl_time">0.0s</span>
        </div>
        <div class="flow-cap"><b id="fl_phase"></b><span id="fl_text"></span></div>
        <div class="flow-scroll">
          <svg class="flow-svg" viewBox="0 0 860 800" role="img" aria-label="Animated Transformer: the sentence The cat sleeps travels through the encoder, then the decoder writes Le chat dort one word at a time.">${staticSvg()}<g id="fl_dyn"></g></svg>
        </div>
        <p class="flow-legend"><span class="sw" style="background:${C.embed}"></span>Embedding <span class="sw" style="background:${C.attn}"></span>Attention <span class="sw" style="background:${C.norm}"></span>Add &amp; Norm <span class="sw" style="background:${C.ffn}"></span>Feed-forward <span class="sw" style="background:${C.lin}"></span>Linear <span class="sw" style="background:${C.soft}"></span>Softmax / output.
        Click any block to inspect it. Attention strengths and word probabilities are hand-made for teaching, not from a trained model.</p>
      </div>`;
    dom = {
      dyn: root.querySelector('#fl_dyn'), blocks: [...root.querySelectorAll('.fb')],
      play: root.querySelector('#fl_play'), seek: root.querySelector('#fl_seek'), time: root.querySelector('#fl_time'),
      phase: root.querySelector('#fl_phase'), text: root.querySelector('#fl_text'), seeking: false
    };
    lastActive = null; lastCap = ''; last = 0;
    if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) { t = ENC_DUR + STEP * 1.2; playing = false; dom.play.textContent = '▶ Play'; }
    root.querySelector('#fl_replay').addEventListener('click', () => { t = 0; setPlaying(true); });
    dom.play.addEventListener('click', () => setPlaying(!playing));
    root.querySelector('#fl_speed').addEventListener('change', (e) => { speed = parseFloat(e.target.value) || 1; });
    dom.seek.addEventListener('input', () => { dom.seeking = true; t = dom.seek.value / 100; draw(); });
    dom.seek.addEventListener('change', () => { dom.seeking = false; });
    root.querySelectorAll('.fb[data-key]').forEach((g) => {
      const open = () => { if (typeof openInspector === 'function') openInspector(g.dataset.key); };
      g.addEventListener('click', open);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    });
    raf = requestAnimationFrame(tick);
  }

  function stop() { if (raf) cancelAnimationFrame(raf); raf = 0; if (window.Steps) window.Steps.stop(); }

  // ---- mode switch: guided steps (default) or the full continuous run ----
  let mode = 'steps';
  function render() {
    stop();
    const body = document.getElementById('fl_body');
    if (!body) return;
    document.querySelectorAll('.mode-btn').forEach((b) => { const on = b.dataset.m === mode; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
    if (mode === 'steps' && window.Steps) window.Steps.mount(body); else mountFull(body);
  }
  function setMode(m) { mode = m; render(); }
  function mount() {
    const root = document.getElementById('stage_content');
    root.innerHTML = `<div class="mode-bar" role="group" aria-label="Visualisation mode">
        <button type="button" class="mode-btn" data-m="steps">① Step by step <small>real numbers, one module at a time</small></button>
        <button type="button" class="mode-btn" data-m="full">② Full run <small>the whole translation</small></button>
      </div><div id="fl_body"></div>`;
    root.querySelectorAll('.mode-btn').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.m)));
    render();
  }

  window.Flow = { mount, stop, setMode, archSvg: staticSvg, TOTAL, seek: (v) => { t = v; if (dom) draw(); }, pause: () => { if (dom) setPlaying(false); } };
})();
