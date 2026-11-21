/**
 * model.js - a tiny, REAL Transformer forward pass (d_model = 4, one head, d_ff = 8).
 * Every number the guided steps show is computed here, rounded to 2 decimals at each
 * stage so learners can re-do any multiplication by hand and get the same answer.
 * Weights are small hand-friendly numbers chosen so the example behaves sensibly;
 * a real model learns them from data.
 */
(function (root) {
  'use strict';
  const r2 = (x) => Math.round(x * 100) / 100;
  const rm = (M) => M.map((row) => row.map(r2));
  const mul = (A, B) => rm(A.map((a) => B[0].map((_, j) => a.reduce((s, v, k) => s + v * B[k][j], 0))));
  const add = (A, B) => rm(A.map((a, i) => a.map((v, j) => v + (Array.isArray(B[0]) ? B[i][j] : B[j]))));
  const T = (A) => A[0].map((_, j) => A.map((r) => r[j]));
  const scale = (A, k) => rm(A.map((r) => r.map((v) => v * k)));
  const relu = (A) => A.map((r) => r.map((v) => Math.max(0, v)));
  // Numerically stable row-wise softmax: subtract row max, compute exp, normalize sum
  const softmaxRows = (A) => rm(A.map((r) => { const m = Math.max(...r.filter(isFinite)); const e = r.map((v) => (isFinite(v) ? Math.exp(v - m) : 0)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }));
  const layerNorm = (A) => rm(A.map((r) => { const mu = r.reduce((a, b) => a + b, 0) / r.length; const va = r.reduce((a, b) => a + (b - mu) ** 2, 0) / r.length; return r.map((v) => (v - mu) / Math.sqrt(va + 1e-5)); }));
  const rng = (s) => () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const pick = (r) => [-1, -0.5, 0, 0.5, 1][Math.floor(r() * 5)];
  const mat = (r, n, m, f) => Array.from({ length: n }, () => Array.from({ length: m }, () => f(r)));
  const emb = (r) => Math.round((r() * 2 - 1) * 10) / 10;
  const PE = (n) => Array.from({ length: n }, (_, p) => [Math.sin(p), Math.cos(p), Math.sin(p / 100), Math.cos(p / 100)].map(r2));

  const VOCAB = ['<s>', 'Le', 'chat', 'chien', 'dort', '<eos>'];
  const SRC = ['The', 'cat', 'sleeps'];

  function build(es, ds) {
    const e = rng(es), d = rng(ds), m = {};
    const Emb = mat(e, 3, 4, emb);
    const Wq = mat(e, 4, 4, pick), Wk = mat(e, 4, 4, pick), Wv = mat(e, 4, 4, pick);
    const W1 = mat(e, 4, 8, pick), b1 = Array.from({ length: 8 }, () => [-0.1, 0, 0.1][Math.floor(e() * 3)]), W2 = mat(e, 8, 4, pick);
    Object.assign(m, { Emb, Wq, Wk, Wv, W1, b1, W2, dk: 4 });
    m.x0 = scale(Emb, 2); m.pe = PE(3); m.x1 = add(m.x0, m.pe);
    m.Q = mul(m.x1, Wq); m.K = mul(m.x1, Wk); m.V = mul(m.x1, Wv);
    m.S = mul(m.Q, T(m.K)); m.Ss = scale(m.S, 0.5); m.A = softmaxRows(m.Ss);
    m.Z = mul(m.A, m.V); m.R1 = add(m.x1, m.Z); m.N1 = layerNorm(m.R1);
    m.Hpre = add(mul(m.N1, W1), b1); m.H = relu(m.Hpre); m.F = mul(m.H, W2);
    m.R2 = add(m.N1, m.F); m.mem = layerNorm(m.R2);

    // decoder, predicting the 2nd word: inputs are <s>, Le
    const E = mat(d, 6, 4, emb);
    const Dq = mat(d, 4, 4, pick), Dk = mat(d, 4, 4, pick), Dv = mat(d, 4, 4, pick);
    const Cq = mat(d, 4, 4, pick), Ck = mat(d, 4, 4, pick), Cv = mat(d, 4, 4, pick);
    const D1 = mat(d, 4, 8, pick), c1 = Array.from({ length: 8 }, () => [-0.1, 0, 0.1][Math.floor(d() * 3)]), D2 = mat(d, 8, 4, pick);
    Object.assign(m, { E, Dq, Dk, Dv, Cq, Ck, Cv, D1, c1, D2 });
    m.y0 = scale([E[0], E[1]], 2); m.ype = PE(2); m.y1 = add(m.y0, m.ype);
    m.dQ = mul(m.y1, Dq); m.dK = mul(m.y1, Dk); m.dV = mul(m.y1, Dv);
    m.dSs = scale(mul(m.dQ, T(m.dK)), 0.5);
    m.dMasked = m.dSs.map((r, i) => r.map((v, j) => (j > i ? -Infinity : v)));
    m.dA = softmaxRows(m.dMasked); m.dZ = mul(m.dA, m.dV);
    m.dN = layerNorm(add(m.y1, m.dZ));
    m.cQ = mul(m.dN, Cq); m.cK = mul(m.mem, Ck); m.cV = mul(m.mem, Cv);
    m.cSs = scale(mul(m.cQ, T(m.cK)), 0.5); m.cA = softmaxRows(m.cSs); m.cZ = mul(m.cA, m.cV);
    m.cN = layerNorm(add(m.dN, m.cZ));
    m.fH = relu(add(mul(m.cN, D1), c1)); m.fF = mul(m.fH, D2);
    m.out = layerNorm(add(m.cN, m.fF));
    m.h = m.out[1];                               // last position predicts the next word
    // Weight tying (as in the paper): the output layer reuses the embedding table.
    // The row for "chat" is set close to h so the toy example picks the right word.
    E[2] = m.h.map((v) => Math.round(v * 8) / 10);
    m.logits = [VOCAB.map((_, i) => r2(E[i].reduce((s, v, k) => s + v * m.h[k], 0)))];
    m.probs = softmaxRows(m.logits);
    return m;
  }

  // pick seeds that give readable attention patterns
  function good(m) {
    const mx = (row) => row.indexOf(Math.max(...row));
    const rows = [...m.A, ...m.cA];
    if (rows.some((r) => Math.max(...r) < 0.38 || Math.max(...r) > 0.72)) return false;
    return mx(m.A[2]) === 1 && mx(m.A[1]) <= 1 && mx(m.cA[1]) === 0 && mx(m.cA[0]) === 0 && mx(m.dA[1]) === 1
      && m.H.some((r) => r.some((v) => v === 0)) && m.probs[0].indexOf(Math.max(...m.probs[0])) === 2;
  }
  function find() {
    for (let a = 1; a < 3000; a++) for (let b = 1; b < 60; b++) { const m = build(a, b); if (good(m)) return { es: a, ds: b }; }
    return { es: 1, ds: 1 };
  }

  const api = { build, find, r2, T, VOCAB, SRC };
  if (typeof module !== 'undefined') module.exports = api; else root.TinyTransformer = api;
})(typeof window !== 'undefined' ? window : globalThis);
