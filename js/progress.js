/**
 * progress.js - remembers where the learner is, tracks what is finished,
 * draws the green ticks (as SVG, no image files) and shows the completion greeting.
 *
 * Storage: window.localStorage, which belongs to ONE browser profile on ONE site address.
 * - A different Chrome profile (or another browser) starts fresh.
 * - Tabs of the same profile stay in sync through the "storage" event.
 * - If storage is blocked (some private modes), everything still works for the session.
 * Only plain numbers and flags are stored. Stored data is validated when read.
 */
(function () {
  'use strict';

  const KEY = 'transformer-explained:progress:v1';
  const VIEWS = ['blueprint', 'attention', 'dimensions', 'benchmark'];
  const MODES = ['steps', 'full'];
  const REPO = 'https://github.com/Anjaliavv51/visualizing-transformer';
  const COLORS = ['#f7ccd2', '#ffddae', '#fdfdb3', '#bfe3f5', '#dcdcf1', '#cbeacb', '#2e8b3d', '#e9a24a'];

  let cfg = { chapters: [], groups: [], steps: 16 };
  let data = blank();
  const changeL = [], restartL = [];

  function blank() {
    return { v: 1, last: { chapter: 1, view: 'blueprint', mode: 'steps', step: 0 }, done: {}, seen: [], celebrated: false, updated: 0 };
  }
  const int = (x, lo, hi, d) => { x = Number(x); return Number.isInteger(x) && x >= lo && x <= hi ? x : d; };

  // Never trust stored data: keep only the expected shapes and ranges.
  function sanitize(o) {
    const d = blank();
    if (!o || typeof o !== 'object') return d;
    const l = o.last && typeof o.last === 'object' ? o.last : {};
    d.last.chapter = int(l.chapter, 1, 99, 1);
    d.last.view = VIEWS.includes(l.view) ? l.view : 'blueprint';
    d.last.mode = MODES.includes(l.mode) ? l.mode : 'steps';
    d.last.step = int(l.step, 0, 99, 0);
    if (o.done && typeof o.done === 'object') {
      Object.keys(o.done).forEach((k) => {
        if (/^\d{1,2}$/.test(k) && Array.isArray(o.done[k])) d.done[k] = [...new Set(o.done[k].filter((n) => Number.isInteger(n) && n >= 0 && n < 50))];
      });
    }
    if (Array.isArray(o.seen)) d.seen = [...new Set(o.seen.filter((n) => Number.isInteger(n) && n >= 0 && n < 99))];
    d.celebrated = o.celebrated === true;
    d.updated = Number(o.updated) || 0;
    return d;
  }
  function read() {
    try { const raw = window.localStorage.getItem(KEY); return raw ? sanitize(JSON.parse(raw)) : blank(); } catch (e) { return data; }
  }
  function write() {
    data.updated = Date.now();
    try { window.localStorage.setItem(KEY, JSON.stringify(data)); } catch (e) { /* storage blocked: keep in memory */ }
  }
  const fire = () => changeL.slice().forEach((fn) => { try { fn(); } catch (e) { /* keep going */ } });
  data = read();
  window.addEventListener('storage', (e) => { if (e.key === KEY || e.key === null) { data = read(); fire(); } });

  // ---------- reading ----------
  const total = (id) => { const c = cfg.chapters.find((x) => x.id === id); return c ? Math.max(1, c.sections) : 0; };
  const doneList = (id) => (data.done[id] || []).filter((i) => i < total(id));
  const api = {
    configure(c) { cfg = Object.assign(cfg, c); },
    last() { return Object.assign({}, data.last); },
    sectionDone: (id, i) => (data.done[id] || []).includes(i),
    chapterProgress: (id) => ({ done: doneList(id).length, total: total(id) }),
    chapterDone: (id) => total(id) > 0 && doneList(id).length >= total(id),
    groupDone: (ids) => ids.length > 0 && ids.every((id) => api.chapterDone(id)),
    courseDone: () => cfg.chapters.length > 0 && cfg.chapters.every((c) => api.chapterDone(c.id)),
    chaptersDone: () => cfg.chapters.filter((c) => api.chapterDone(c.id)).length,
    groupsDone: () => cfg.groups.filter((g) => api.groupDone(g.ids)).length,
    sectionsDone: () => cfg.chapters.reduce((n, c) => n + doneList(c.id).length, 0),
    stepSeen: (i) => data.seen.includes(i),
    stepsSeen: () => data.seen.length,
    celebrated: () => data.celebrated,

    // ---------- writing ----------
    setLast(p) {
      const l = data.last;
      if (p.chapter != null) l.chapter = int(p.chapter, 1, 99, l.chapter);
      if (p.view != null && VIEWS.includes(p.view)) l.view = p.view;
      if (p.mode != null && MODES.includes(p.mode)) l.mode = p.mode;
      if (p.step != null) l.step = int(p.step, 0, 99, l.step);
      write();
    },
    setSection(id, i, val) {
      const arr = new Set(data.done[id] || []);
      if (val) arr.add(i); else arr.delete(i);
      data.done[id] = [...arr];
      write(); fire();
    },
    markStep(i) { if (!data.seen.includes(i)) { data.seen.push(i); write(); } },
    setCelebrated(v) { data.celebrated = !!v; write(); },
    reset() { data = blank(); write(); fire(); },
    onChange(fn) { changeL.push(fn); },
    onRestart(fn) { restartL.push(fn); }
  };

  // ---------- the green tick (drawn with SVG, animates when newly earned) ----------
  api.tick = function (animate) {
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('class', 'tick' + (animate ? '' : ' static'));
    svg.setAttribute('aria-hidden', 'true');
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', '12'); c.setAttribute('cy', '12'); c.setAttribute('r', '10'); c.setAttribute('class', 'tick-c');
    const p = document.createElementNS(NS, 'path');
    p.setAttribute('d', 'M7.2 12.6l3.1 3.1 6.4-6.9'); p.setAttribute('class', 'tick-p');
    svg.append(c, p);
    return svg;
  };

  // ---------- small toast ----------
  let toastT = 0;
  api.toast = function (msg, ms) {
    let t = document.getElementById('toast');
    if (!t) { t = document.createElement('div'); t.id = 'toast'; t.className = 'toast'; t.setAttribute('role', 'status'); document.body.appendChild(t); }
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), ms || 4200);
  };

  // ---------- completion greeting ----------
  let confettiRaf = 0, lastFocus = null;
  function confetti(canvas) {
    const ctx = canvas.getContext('2d');
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = (canvas.width = innerWidth * dpr), H = (canvas.height = innerHeight * dpr);
    const parts = Array.from({ length: 150 }, () => ({
      x: Math.random() * W, y: -Math.random() * H * 0.8, w: (6 + Math.random() * 7) * dpr, h: (9 + Math.random() * 9) * dpr,
      vx: (Math.random() - 0.5) * 3 * dpr, vy: (2 + Math.random() * 4) * dpr, r: Math.random() * 6.28, vr: (Math.random() - 0.5) * 0.3,
      c: COLORS[Math.floor(Math.random() * COLORS.length)]
    }));
    const t0 = performance.now();
    (function frame(now) {
      ctx.clearRect(0, 0, W, H);
      let alive = 0;
      parts.forEach((p) => {
        p.x += p.vx; p.y += p.vy; p.vy += 0.03 * dpr; p.r += p.vr;
        if (p.y < H + 30) alive++;
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r); ctx.fillStyle = p.c; ctx.strokeStyle = '#1b1b1f'; ctx.lineWidth = dpr;
        ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h); ctx.strokeRect(-p.w / 2, -p.h / 2, p.w, p.h); ctx.restore();
      });
      if (alive && now - t0 < 9000) confettiRaf = requestAnimationFrame(frame); else ctx.clearRect(0, 0, W, H);
    })(t0);
  }
  function closeCelebration() {
    const m = document.getElementById('cele');
    if (m) m.hidden = true;
    cancelAnimationFrame(confettiRaf);
    document.body.classList.remove('cele-open');
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  api.celebrate = function () {
    let m = document.getElementById('cele');
    if (!m) {
      m = document.createElement('div');
      m.id = 'cele'; m.className = 'cele'; m.setAttribute('role', 'dialog'); m.setAttribute('aria-modal', 'true'); m.setAttribute('aria-labelledby', 'cele_t');
      m.innerHTML = '<canvas class="cele-cv" aria-hidden="true"></canvas><div class="cele-card">' +
        '<div class="cele-tick"></div><h2 id="cele_t">Course complete!</h2><p class="cele-sub"></p><ul class="cele-stats"></ul>' +
        '<p class="cele-msg">You can now follow a sentence through every stage of a Transformer: embeddings, attention, the encoder, the decoder and the next-word prediction.</p>' +
        '<div class="cele-actions"><button type="button" class="fl-btn" id="cele_close">Keep exploring</button>' +
        '<a class="fl-btn alt" href="' + REPO + '" target="_blank" rel="noopener noreferrer">⭐ Star on GitHub</a>' +
        '<button type="button" class="fl-btn alt" id="cele_again">Start over</button></div>' +
        '<p class="cele-by">Designed by Pavananjali</p></div>';
      document.body.appendChild(m);
      m.querySelector('#cele_close').addEventListener('click', closeCelebration);
      m.querySelector('#cele_again').addEventListener('click', () => { closeCelebration(); restartL.slice().forEach((fn) => fn()); });
      m.addEventListener('click', (e) => { if (e.target === m) closeCelebration(); });
      m.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') { e.stopPropagation(); closeCelebration(); return; }
        if (e.key === 'Tab') {
          const f = [...m.querySelectorAll('button, a[href]')]; if (!f.length) return;
          const first = f[0], last = f[f.length - 1];
          if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
          else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
      });
    }
    const nCh = cfg.chapters.length;
    m.querySelector('.cele-sub').textContent = `You completed all ${nCh} chapters of Transformer Explained.`;
    const stats = [`${api.chaptersDone()} / ${nCh} chapters`, `${api.groupsDone()} / ${cfg.groups.length} modules`, `${api.sectionsDone()} sections read`, `${api.stepsSeen()} / ${cfg.steps} guided steps seen`];
    m.querySelector('.cele-stats').replaceChildren(...stats.map((s) => { const li = document.createElement('li'); li.textContent = s; return li; }));
    m.querySelector('.cele-tick').replaceChildren(api.tick(true));
    lastFocus = document.activeElement;
    m.hidden = false;
    document.body.classList.add('cele-open');
    m.querySelector('#cele_close').focus();
    const cv = m.querySelector('.cele-cv');
    const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    cancelAnimationFrame(confettiRaf);
    if (reduce) cv.getContext('2d').clearRect(0, 0, cv.width, cv.height); else confetti(cv);
  };

  window.Progress = api;
})();
