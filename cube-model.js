// Pocket-cube model shared by playground.html and dashboard.html.
// Mirrors rubik.py: sticker order, the nine actions, check(), the perfect
// hash encode(), plus helpers for the trained policy and for real cubes.
(function (global) {
  'use strict';
  const F = 0, R = 1, T = 2, B = 3, L = 4, D = 5;
  const LT = 0, RT = 1, RB = 2, LB = 3;
  const FACE_NAMES = ['front', 'right', 'top', 'back', 'left', 'bottom'];
  const ACTIONS = ['t1', 't2', 't3', 'r1', 'r2', 'r3', 'f1', 'f2', 'f3'];
  const NOTATION = ['U', 'U2', "U'", 'R', 'R2', "R'", 'F', 'F2', "F'"];
  const INVERSE = [2, 1, 0, 5, 4, 3, 8, 7, 6];
  const INIT = 'wggbrymrgrwrbmbwybywmygm';
  const SOLVED = 'ggggyyyyrrrrbbbbwwwwmmmm';
  const ENV_LETTERS = 'wgbrym';
  const ENV_COLORS = { w: '#F4F3EE', g: '#27A64A', b: '#2E67D9', r: '#D93A31', y: '#F5C531', m: '#D6449C' };
  // a physical cube: white, yellow, green, blue, red, orange
  const REAL_LETTERS = 'WYGBRO';
  const REAL_COLORS = { W: '#F4F3EE', Y: '#F5C531', G: '#27A64A', B: '#2E67D9', R: '#D93A31', O: '#F07E1A' };
  const REAL_NAMES = { W: '白', Y: '黄', G: '绿', B: '蓝', R: '红', O: '橙' };

  const clone = s => s.map(f => f.slice());
  const key = s => s.map(f => f.join('')).join('');
  const fromKey = k => [0, 1, 2, 3, 4, 5].map(f => k.slice(f * 4, f * 4 + 4).split(''));

  // ---------- moves (verbatim from rubik.py) ----------
  function twistTop(s) {
    const p = clone(s);
    s[F][LT] = p[R][LT]; s[F][RT] = p[R][RT];
    s[L][LT] = p[F][LT]; s[L][RT] = p[F][RT];
    s[B][LT] = p[L][LT]; s[B][RT] = p[L][RT];
    s[R][LT] = p[B][LT]; s[R][RT] = p[B][RT];
    s[T][LT] = p[T][LB]; s[T][RT] = p[T][LT]; s[T][RB] = p[T][RT]; s[T][LB] = p[T][RB];
  }
  function twistRight(s) {
    const p = clone(s);
    s[F][RT] = p[D][RT]; s[F][RB] = p[D][RB];
    s[T][RT] = p[F][RT]; s[T][RB] = p[F][RB];
    s[B][LB] = p[T][RT]; s[B][LT] = p[T][RB];
    s[D][RT] = p[B][LB]; s[D][RB] = p[B][LT];
    s[R][LT] = p[R][LB]; s[R][RT] = p[R][LT]; s[R][RB] = p[R][RT]; s[R][LB] = p[R][RB];
  }
  function twistFront(s) {
    const p = clone(s);
    s[R][LT] = p[T][LB]; s[R][LB] = p[T][RB];
    s[T][LB] = p[L][RB]; s[T][RB] = p[L][RT];
    s[L][RT] = p[D][LT]; s[L][RB] = p[D][RT];
    s[D][LT] = p[R][LB]; s[D][RT] = p[R][LT];
    s[F][LT] = p[F][LB]; s[F][RT] = p[F][LT]; s[F][RB] = p[F][RT]; s[F][LB] = p[F][RB];
  }
  const TWIST = [twistTop, twistRight, twistFront];
  function applyAction(s, a) { const k = a % 3 + 1; for (let i = 0; i < k; i++) TWIST[Math.floor(a / 3)](s); }
  // PERM[a][i] = index of the sticker that moves into slot i
  const PERM = ACTIONS.map((_, a) => { const idx = fromKey('abcdefghijklmnopqrstuvwx'); applyAction(idx, a); return key(idx).split('').map(c => c.charCodeAt(0) - 97); });
  const applyStr = (str, a) => PERM[a].map(i => str[i]).join('');
  function check(s) { if (typeof s === 'string') s = fromKey(s); let n = 0; for (let f = 0; f < 6; f++) if (new Set(s[f]).size === 1) n++; return n; }

  // ---------- geometry of the 24 sticker slots (from rubik.py's old renderer) ----------
  // centre and outward normal per slot index (face*4+pos); x -> right, y -> back, z -> top
  const SLOT_POS = [], SLOT_NORMAL = [];
  const G = {};
  G[F] = { [LT]: [-0.5, -1, 0.5], [LB]: [-0.5, -1, -0.5], [RT]: [0.5, -1, 0.5], [RB]: [0.5, -1, -0.5] };
  G[B] = { [RT]: [-0.5, 1, 0.5], [RB]: [-0.5, 1, -0.5], [LT]: [0.5, 1, 0.5], [LB]: [0.5, 1, -0.5] };
  G[R] = { [LT]: [1, -0.5, 0.5], [RT]: [1, 0.5, 0.5], [LB]: [1, -0.5, -0.5], [RB]: [1, 0.5, -0.5] };
  G[L] = { [RT]: [-1, -0.5, 0.5], [LT]: [-1, 0.5, 0.5], [RB]: [-1, -0.5, -0.5], [LB]: [-1, 0.5, -0.5] };
  G[T] = { [LT]: [-0.5, 0.5, 1], [RT]: [0.5, 0.5, 1], [RB]: [0.5, -0.5, 1], [LB]: [-0.5, -0.5, 1] };
  G[D] = { [LB]: [-0.5, 0.5, -1], [RB]: [0.5, 0.5, -1], [RT]: [0.5, -0.5, -1], [LT]: [-0.5, -0.5, -1] };
  const NORMALS = { [F]: [0, -1, 0], [B]: [0, 1, 0], [R]: [1, 0, 0], [L]: [-1, 0, 0], [T]: [0, 0, 1], [D]: [0, 0, -1] };
  for (let f = 0; f < 6; f++) for (let p = 0; p < 4; p++) { SLOT_POS[f * 4 + p] = G[f][p]; SLOT_NORMAL[f * 4 + p] = NORMALS[f]; }

  // ---------- corners ----------
  // slots per corner position ordered [top/bottom face, front/back face, left/right face];
  // the last corner (back-left-bottom) never moves.
  const CORNER_SLOTS = [[10, 1, 4], [11, 0, 17], [21, 2, 7], [20, 3, 18], [9, 12, 5], [8, 13, 16], [22, 15, 6], [23, 14, 19]];
  const ORI_SIGN = [-1, 1, 1, -1, 1, -1, -1], ORI_CONST = 0;
  const UD = new Set(['r', 'm']);
  const CUBIE_ID = {};
  for (let i = 0; i < 8; i++) CUBIE_ID[CORNER_SLOTS[i].map(j => SOLVED[j]).sort().join('')] = i;
  const FACT = [1, 1, 2, 6, 24, 120, 720];

  function cornerColours(str) { return CORNER_SLOTS.map(sl => sl.map(j => str[j])); }

  /** Is this 24-letter env state a reachable cube state?  Returns {ok, reason}. */
  function legality(str) {
    if (typeof str !== 'string' || str.length !== 24 || /[^wgbrym]/.test(str)) return { ok: false, reason: '需要 24 个字母（w g b r y m）' };
    const cols = cornerColours(str);
    const ids = [], oris = [];
    for (let i = 0; i < 8; i++) {
      const id = CUBIE_ID[cols[i].slice().sort().join('')];
      if (id === undefined) return { ok: false, reason: `${['前右上', '前左上', '前右下', '前左下', '后右上', '后左上', '后右下', '后左下'][i]}角块的颜色组合 ${cols[i].join('')} 不存在于这个魔方` };
      ids.push(id); oris.push(cols[i].findIndex(c => UD.has(c)));
    }
    if (new Set(ids).size !== 8) return { ok: false, reason: '有角块重复出现' };
    if (ids[7] !== 7 || oris[7] !== 0) return { ok: false, reason: '固定角块（后左下）必须是 back=b、left=w、bottom=m' };
    let sum = 0; for (let i = 0; i < 7; i++) sum += ORI_SIGN[i] * oris[i];
    if (((sum % 3) + 3) % 3 !== ORI_CONST) return { ok: false, reason: '角块方向奇偶性不对：有一个角块被单独拧转过，这个状态无法复原' };
    return { ok: true, ids, oris };
  }

  /** Perfect hash of a legal env state, identical to rubik.encode. */
  function encode(str) {
    const lg = legality(str);
    if (!lg.ok) throw new Error(lg.reason);
    const perm = lg.ids.slice(0, 7);
    let rank = 0;
    for (let i = 0; i < 7; i++) { let c = 0; for (let j = i + 1; j < 7; j++) if (perm[j] < perm[i]) c++; rank += c * FACT[6 - i]; }
    let ocode = 0; for (let i = 0; i < 6; i++) ocode = ocode * 3 + lg.oris[i];
    return rank * 729 + ocode;
  }
  const N_STATES = 3674160;

  // ---------- policy (two 4-bit actions per byte, from play_rubik.py) ----------
  function policyAction(packed, index) { const b = packed[index >> 1]; return index & 1 ? b >> 4 : b & 15; }
  function policySolve(str, packed, maxSteps) {
    const moves = []; let s = str;
    for (let i = 0; i < (maxSteps || 30); i++) {
      if (s === SOLVED) return { moves, solved: true, final: s };
      const a = policyAction(packed, encode(s));
      moves.push(a); s = applyStr(s, a);
    }
    return { moves, solved: s === SOLVED, final: s };
  }

  // ---------- optimal solver (bidirectional BFS, 6 + 6 >= God's number 11) ----------
  function solvedTarget(str) {
    const cols = cornerColours(str);
    const adj = {};
    for (const c of cols) {
      if (new Set(c).size !== 3) return null;
      for (const a of c) { adj[a] = adj[a] || new Set(); for (const b of c) if (a !== b) adj[a].add(b); }
    }
    const all = Object.keys(adj);
    if (all.length !== 6) return null;
    const opp = {};
    for (const a of all) { const rest = all.filter(b => b !== a && !adj[a].has(b)); if (rest.length !== 1) return null; opp[a] = rest[0]; }
    const back = str[14], left = str[19], bottom = str[23];
    const t = []; t[F] = opp[back]; t[R] = opp[left]; t[T] = opp[bottom]; t[B] = back; t[L] = left; t[D] = bottom;
    return t.map(c => c.repeat(4)).join('');
  }
  function bfsLayer(start, depth) {
    const seen = new Map([[start, [null, -1]]]);
    let frontier = [start];
    for (let d = 0; d < depth; d++) {
      const next = [];
      for (const s of frontier) for (let a = 0; a < 9; a++) { const n = applyStr(s, a); if (!seen.has(n)) { seen.set(n, [s, a]); next.push(n); } }
      frontier = next;
    }
    return seen;
  }
  function pathTo(seen, k) { const out = []; while (seen.get(k)[0] !== null) { const [prev, a] = seen.get(k); out.push(a); k = prev; } return out.reverse(); }
  function optimalSolve(str) {
    const target = solvedTarget(str);
    if (!target) return { error: 'invalid' };
    if (str === target) return { moves: [] };
    const fwd = bfsLayer(str, 6), bwd = bfsLayer(target, 6);
    let best = null;
    for (const k of fwd.keys()) {
      if (!bwd.has(k)) continue;
      const p = pathTo(fwd, k).concat(pathTo(bwd, k).reverse().map(a => INVERSE[a]));
      if (!best || p.length < best.length) best = p;
    }
    return best ? { moves: best } : { error: 'unreachable' };
  }

  // ---------- real cubes ----------
  /**
   * Turn a physical cube's stickers (24 letters out of W Y G B R O, in env
   * slot order) into an env state.  Colours are relabelled so that the fixed
   * back-left-bottom cubie becomes b/w/m and each colour's opposite maps to
   * the env's opposite; anchoring on that corner also fixes the handedness,
   * so the relabelled state is legal whenever the input describes a real
   * cube.  Moves then apply unchanged to the physical cube.
   * Returns {env, map, unmap} or {error}.
   */
  function fromReal(realStr) {
    realStr = (realStr || '').toUpperCase();
    if (realStr.length !== 24 || /[^WYGBRO]/.test(realStr)) return { error: '需要 24 个字母，只能用 W Y G B R O' };
    for (const c of REAL_LETTERS) if ((realStr.match(new RegExp(c, 'g')) || []).length !== 4) return { error: `${REAL_NAMES[c]}色应该恰好有 4 张贴纸，现在有 ${(realStr.match(new RegExp(c, 'g')) || []).length} 张` };
    const cols = cornerColours(realStr);
    const adj = {};
    for (const c of cols) { if (new Set(c).size !== 3) return { error: `有一个角块出现了两张同色贴纸（${c.map(x => REAL_NAMES[x]).join('')}）` }; for (const a of c) { adj[a] = adj[a] || new Set(); for (const b of c) if (a !== b) adj[a].add(b); } }
    const opp = {};
    for (const a of REAL_LETTERS) { const rest = [...REAL_LETTERS].filter(b => b !== a && !adj[a].has(b)); if (rest.length !== 1) return { error: `无法确定 ${REAL_NAMES[a]}色的对面颜色，请检查输入` }; opp[a] = rest[0]; }
    const back = realStr[14], left = realStr[19], bottom = realStr[23];
    const map = {}; map[back] = 'b'; map[left] = 'w'; map[bottom] = 'm'; map[opp[back]] = 'g'; map[opp[left]] = 'y'; map[opp[bottom]] = 'r';
    const env = [...realStr].map(c => map[c]).join('');
    const lg = legality(env);
    if (!lg.ok) return { error: lg.reason };
    const unmap = {}; for (const k in map) unmap[map[k]] = k;
    return { env, map, unmap };
  }

  global.CubeModel = {
    F, R, T, B, L, D, LT, RT, RB, LB, FACE_NAMES, ACTIONS, NOTATION, INVERSE, INIT, SOLVED,
    ENV_LETTERS, ENV_COLORS, REAL_LETTERS, REAL_COLORS, REAL_NAMES,
    clone, key, fromKey, applyAction, applyStr, PERM, check,
    SLOT_POS, SLOT_NORMAL, CORNER_SLOTS, legality, encode, N_STATES,
    policyAction, policySolve, solvedTarget, optimalSolve,
    fromReal,
  };
})(typeof window !== 'undefined' ? window : globalThis);
