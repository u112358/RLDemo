"""Tabular Q-learning that solves the 2x2x2 cube from *any* scramble.

What changed compared with the original demo and why:

* The Q-table is a dense ``(3,674,160 x 9)`` float32 array indexed by the
  perfect hash from ``rubik.encode`` instead of a pandas frame keyed by the
  state string.  Every legal state has a row, so nothing is "unseen".
* Episodes do not start from one fixed scramble.  Each environment starts
  from the solved cube scrambled by ``k`` random moves with ``k`` drawn from
  ``1..K``; ``K`` grows from 1 to 11 as soon as the greedy policy solves
  distance-``K`` states reliably (reverse-scramble curriculum).  That keeps
  every training state within reach of states whose values are already
  right, so the reward always propagates.
* Reward is -1 per move and the episode ends at the fully solved cube, so
  with gamma = 1 the optimal Q-value is exactly minus the distance to solved
  and can be checked against a BFS.  Transitions are deterministic, so a
  learning rate of 1 is a plain Bellman backup.  The table starts at -12,
  below every true value, so untried actions never look attractive.
* The behaviour policy tries an action whose value is still unknown before
  exploiting a learned one (plus epsilon-greedy noise), so every state's
  nine actions get tried within a few visits and the correct one is found
  without waiting for random exploration.
* Thousands of environments step in lock-step as numpy arrays (the update
  rule is the ordinary one-step Q-learning rule, just batched), which is what
  makes filling 33 million table entries take a minute rather than a day.

Usage::

    python play_rubik.py train [--dashboard 8000]     # train the Q-table, optional live page
    python play_rubik.py train --agent net [...]       # same learner with a neural network (see qnet.py)
    python play_rubik.py train --agent net --backend torch   # the network on a GPU (Apple MPS / CUDA), see qnet_torch.py
    python play_rubik.py serve [--port 8000]           # pages + trained table, no training
    python play_rubik.py eval                          # success / length per depth
    python play_rubik.py solve wggbrymrgrwrbmbwybywmygm
"""
import argparse
import http.server
import json
import os
import shutil
import sys
import threading
import time
import unicodedata

import numpy as np

import rubik as rb

CACHE = 'cache'
Q_PATH = os.path.join(CACHE, 'q_table.npy')
POLICY_PATH = os.path.join(CACHE, 'policy.npy')
METRICS_PATH = os.path.join(CACHE, 'metrics.json')
NET_PATH = os.path.join(CACHE, 'net.npz')
NET_POLICY_PATH = os.path.join(CACHE, 'net_policy.npy')
NET_METRICS_PATH = os.path.join(CACHE, 'net_metrics.json')
NET_SEEN_PATH = os.path.join(CACHE, 'net_seen.npy')          # packed bitmap of states sampled during training
MAX_DEPTH = 11


def set_tag(tag):
    """Keep one experiment's files apart from another's: cache/<tag>/... (the
    transition table stays shared in cache/)."""
    global Q_PATH, POLICY_PATH, METRICS_PATH, NET_PATH, NET_POLICY_PATH, NET_METRICS_PATH, NET_SEEN_PATH
    if not tag:
        return
    d = os.path.join(CACHE, tag)
    os.makedirs(d, exist_ok=True)
    Q_PATH, POLICY_PATH, METRICS_PATH = os.path.join(d, 'q_table.npy'), os.path.join(d, 'policy.npy'), os.path.join(d, 'metrics.json')
    NET_PATH, NET_POLICY_PATH = os.path.join(d, 'net.npz'), os.path.join(d, 'net_policy.npy')
    NET_METRICS_PATH, NET_SEEN_PATH = os.path.join(d, 'net_metrics.json'), os.path.join(d, 'net_seen.npy')


def paths(agent):
    """Cache files of an agent: 'table' (Q-table) or 'net' (neural network)."""
    if agent == 'net':
        return {'model': NET_PATH, 'policy': NET_POLICY_PATH, 'metrics': NET_METRICS_PATH}
    return {'model': Q_PATH, 'policy': POLICY_PATH, 'metrics': METRICS_PATH}


def scramble(T, rng, depths):
    """Vectorised: solved cube scrambled by ``depths[i]`` random moves."""
    n = depths.shape[0]
    s = np.full(n, rb.SOLVED_INDEX, dtype=np.int64)
    for j in range(int(depths.max())):
        a = rng.integers(9, size=n)
        m = depths > j
        s[m] = T[s[m], a[m]]
    return s


def greedy_rollout(Q, T, s, max_steps=30):
    """Follow argmax Q from states ``s``; returns solve length per state (-1 = failed).
    ``Q`` is the (N, 9) table or a callable idx -> (n, 9) values."""
    qf = Q if callable(Q) else (lambda idx: Q[idx])
    s = s.copy()
    length = np.full(s.shape[0], -1, dtype=np.int64)
    for t in range(max_steps):
        active = length < 0
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        a = np.argmax(qf(s[idx]), axis=1)
        s[idx] = T[s[idx], a]
        just = idx[s[idx] == rb.SOLVED_INDEX]
        length[just] = t + 1
    return length


def beam_solve(qf, T, start, width=32, max_depth=30):
    """Beam search on the value function: keep the ``width`` states with the
    highest max_a Q(s, a) at every depth, expand all nine children, stop at the
    solved state.  Returns the action list, or None.  Turns an approximate
    value function into a solver (DeepCube's insight, in its simplest form)."""
    if start == rb.SOLVED_INDEX:
        return []
    T = np.asarray(T)
    frontier = np.array([start], dtype=np.int64)
    layers = []                                     # per depth: (states, parent index, action)
    seen = {int(start)}
    for depth in range(max_depth):
        children = T[frontier]                      # (n, 9)
        cand = children.reshape(-1)
        parent = np.repeat(np.arange(frontier.shape[0]), 9)
        action = np.tile(np.arange(9), frontier.shape[0])
        keep = np.array([c not in seen for c in cand.tolist()])
        cand, parent, action = cand[keep], parent[keep], action[keep]
        if cand.size == 0:
            return None
        hit = np.nonzero(cand == rb.SOLVED_INDEX)[0]
        if hit.size:
            layers.append((cand, parent, action))
            i = int(hit[0]); moves = []
            for states, par, act in reversed(layers):
                moves.append(int(act[i])); i = int(par[i])
            return moves[::-1]
        # dedupe within the layer, score by value, keep the best `width`
        cand, first = np.unique(cand, return_index=True)
        parent, action = parent[first], action[first]
        v = qf(cand).max(axis=1)
        top = np.argsort(-v)[:width]
        cand, parent, action = cand[top], parent[top], action[top]
        seen.update(cand.tolist())
        layers.append((cand, parent, action))
        frontier = cand
    return None


class Trainer:
    def __init__(self, n_envs=8192, alpha=1.0, gamma=1.0, eps=0.1, promote=0.97,
                 episode_cap=25, eval_every=50, seed=0):
        self.rng = np.random.default_rng(seed)
        self.T = np.ascontiguousarray(rb.transitions())
        self.dist = rb.bfs_distances(self.T)                      # ground truth, for metrics only
        self.by_depth = [np.nonzero(self.dist == d)[0] for d in range(MAX_DEPTH + 1)]
        self.depth_weight = np.bincount(self.dist, minlength=MAX_DEPTH + 1) / rb.N_STATES
        # Pessimistic initialisation: every true Q-value lies in [-12, -1], so
        # an untried action always looks worse than a learned one and the greedy
        # policy locks onto the solution path as soon as it is found.  (Optimistic
        # zeros would instead drive exploration over the whole state space and
        # defeat the curriculum.)
        self.Q0 = -float(MAX_DEPTH + 1)
        self.Q = np.full((rb.N_STATES, 9), self.Q0, dtype=np.float32)
        self.n, self.alpha, self.gamma, self.eps = n_envs, alpha, gamma, eps
        self.promote, self.cap, self.eval_every = promote, episode_cap, eval_every
        self.K = 1
        self.steps = 0
        self.episodes = 0
        self.solved = 0
        self.t0 = time.time()
        self.metrics = []
        self.k = np.zeros(n_envs, dtype=np.int64)        # scramble depth of each env's episode
        self.s = self._new_starts(np.arange(n_envs))
        self.age = np.zeros(n_envs, dtype=np.int64)
        self.recorder = rb.EpisodeRecorder()            # env 0's episodes, for the dashboard replay
        self.policy_cache = None

    @property
    def trace(self):
        return self.recorder.last

    def qvalues(self, idx):
        return self.Q[idx]

    def full_policy(self):
        return np.argmax(self.Q, axis=1).astype(np.uint8)

    def _new_starts(self, which):
        # k random moves from solved, k in 1..K.  Once the curriculum is
        # complete, longer walks (up to 3x the diameter) approximate uniformly
        # random states, which are mostly 8-10 moves deep.
        hi = self.K + 1 if self.K < MAX_DEPTH else 3 * MAX_DEPTH
        depths = self.rng.integers(1, hi, size=which.shape[0])
        self.k[which] = depths
        return scramble(self.T, self.rng, depths)

    def step(self):
        Q, T, s, rng = self.Q, self.T, self.s, self.rng
        q = Q[s]
        # Behaviour policy: try actions whose value is still unknown (== Q0)
        # before exploiting a learned one, random tie-break among equals.  The
        # bonus only steers exploration; the values themselves stay pessimistic.
        score = q + (q == self.Q0) * 100.0 + rng.random(q.shape, dtype=np.float32) * 1e-3
        greedy = np.argmax(score, axis=1)
        explore = rng.random(self.n) < self.eps
        a = np.where(explore, rng.integers(9, size=self.n), greedy)
        s2 = T[s, a]
        done = s2 == rb.SOLVED_INDEX
        target = -1.0 + self.gamma * np.where(done, 0.0, Q[s2].max(axis=1))
        # Every true Q-value is >= -12 (11 moves suffice from anywhere), so the
        # estimate is clamped there: an action tried before its successor was
        # learned then counts as "unknown" again instead of sinking for good.
        Q[s, a] = np.maximum(Q[s, a] + self.alpha * (target - Q[s, a]), self.Q0)
        self.age += 1
        reset = done | (self.age >= self.cap)
        self.recorder.push(self.steps, s[0], a[0], s2[0], done[0], reset[0], self.k[0], self.K,
                           explore[0], q[0, a[0]] == self.Q0)
        n_reset = int(reset.sum())
        if n_reset:
            s2 = s2.copy()
            s2[reset] = self._new_starts(np.nonzero(reset)[0])
            self.age[reset] = 0
            self.episodes += n_reset
            self.solved += int(done.sum())
        self.s = s2
        self.steps += self.n

    def evaluate(self, n_per_depth=400, sample_states=200000):
        succ, mean_len = [], []
        for d in range(1, MAX_DEPTH + 1):
            pool = self.by_depth[d]
            pick = pool[self.rng.integers(pool.shape[0], size=min(n_per_depth, pool.shape[0]))]
            length = greedy_rollout(self.Q, self.T, pick)
            ok = length > 0
            succ.append(float(ok.mean()))
            mean_len.append(float(length[ok].mean()) if ok.any() else None)
        succ_random = float(sum(self.depth_weight[d] * succ[d - 1] for d in range(1, MAX_DEPTH + 1)) + self.depth_weight[0])
        # coverage = states whose best action has a learned (> Q0) value;
        # q_error = mean |max_a Q(s,a) + distance(s)| over those states
        sample = self.rng.integers(rb.N_STATES, size=sample_states)
        v = self.Q[sample].max(axis=1)
        known = v > self.Q0
        coverage = float(known.mean())
        q_err = float(np.abs(v[known] + self.dist[sample][known]).mean()) if known.any() else None
        seen_by_depth = []                    # share of the states at each distance with a learned value (20k sampled per depth)
        for d in range(1, MAX_DEPTH + 1):
            pool = self.by_depth[d]
            pick = pool if pool.size <= 20000 else pool[self.rng.integers(pool.size, size=20000)]
            seen_by_depth.append(float((self.Q[pick].max(axis=1) > self.Q0).mean()))
        rec = {
            'step': self.steps, 'time': round(time.time() - self.t0, 1), 'K': self.K,
            'episodes': self.episodes, 'solved': self.solved,
            'coverage': coverage, 'q_error': q_err, 'success_random': succ_random, 'seen_by_depth': seen_by_depth,
            'success_by_depth': succ, 'mean_len_by_depth': mean_len,
            'optimal_len_by_depth': list(range(1, MAX_DEPTH + 1)),
            'sps': round(self.steps / max(time.time() - self.t0, 1e-9)),
        }
        self.metrics.append(rec)
        # curriculum: promote when the greedy policy is reliable at the current depth
        if self.K < MAX_DEPTH and succ[self.K - 1] >= self.promote:
            self.K += 1
        return rec

    def save(self):
        os.makedirs(CACHE, exist_ok=True)
        np.save(Q_PATH, self.Q)
        self.policy_cache = self.full_policy()
        np.save(POLICY_PATH, self.policy_cache)
        write_metrics(self.metrics, METRICS_PATH)


def write_metrics(metrics, path):
    os.makedirs(CACHE, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'config': CONFIG, 'records': metrics}, f)
    os.replace(tmp, path)


CONFIG = {}


def pack_policy(policy):
    """uint8 actions (0..8) -> two per byte, low nibble first."""
    p = np.asarray(policy, dtype=np.uint8)
    if p.shape[0] % 2:
        p = np.concatenate([p, np.zeros(1, np.uint8)])
    return (p[0::2] | (p[1::2] << 4)).tobytes()


class Serving:
    """Data behind the HTTP endpoints: a live trainer, or the saved cache of one agent."""

    def __init__(self, trainer=None, agent='table'):
        self.trainer = trainer
        self.agent = agent
        self.Q = self.policy = self.net = None
        self.records, self.config = [], {}
        self._packed, self._packed_at = None, 0.0
        if trainer is None:
            pt = paths(agent)
            if os.path.exists(pt['metrics']):
                with open(pt['metrics']) as f:
                    m = json.load(f)
                self.records, self.config = m.get('records', []), m.get('config', {})
            if agent == 'net':
                if os.path.exists(NET_PATH):
                    from qnet import MLP, onehot
                    net = MLP.from_file(NET_PATH)
                    self.net = lambda idx: net.forward(onehot(idx))
                if os.path.exists(NET_POLICY_PATH):
                    self.policy = np.load(NET_POLICY_PATH, mmap_mode='r')
            else:
                if os.path.exists(Q_PATH):
                    self.Q = np.load(Q_PATH, mmap_mode='r')
                elif os.path.exists(POLICY_PATH):
                    self.policy = np.load(POLICY_PATH, mmap_mode='r')

    def has_table(self):
        return self.trainer is not None or self.Q is not None or self.policy is not None or self.net is not None

    def metrics(self):
        if self.trainer is not None:
            return {'config': CONFIG, 'records': self.trainer.metrics}
        return {'config': self.config, 'records': self.records}

    def _qrow(self, idx):
        if self.trainer is not None:
            return np.asarray(self.trainer.qvalues(np.array([idx]))[0], dtype=np.float64)
        if self.net is not None:
            return np.asarray(self.net(np.array([idx]))[0], dtype=np.float64)
        if self.Q is not None:
            return np.asarray(self.Q[idx], dtype=np.float64)
        return None

    def act(self, state):
        """Greedy action (and the Q row when available) for a 24-letter state."""
        idx = rb.encode(state)
        row = self._qrow(idx)
        if row is not None:
            best = np.nonzero(row == row.max())[0]
            a = int(np.random.choice(best))
            return {'index': int(idx), 'action': a, 'q': [round(float(v), 3) for v in row],
                    'unknown': bool(row.max() <= -(MAX_DEPTH + 1))}
        if self.policy is not None:
            return {'index': int(idx), 'action': int(self.policy[idx]), 'q': None, 'unknown': False}
        raise RuntimeError('no trained agent')

    def policy_bytes(self):
        if self.trainer is not None:
            # a full policy of the network costs ~30 s; refresh at most once a minute
            if self._packed is None or (self.trainer.policy_cache is None and time.time() - self._packed_at > 60):
                pol = self.trainer.policy_cache if self.trainer.policy_cache is not None else self.trainer.full_policy()
                self._packed, self._packed_at = pack_policy(pol), time.time()
            return self._packed
        if self._packed is None:
            if self.policy is not None:
                src = self.policy
            elif self.Q is not None:
                src = np.argmax(self.Q, axis=1)
            else:
                raise RuntimeError('no policy')
            self._packed = pack_policy(src)
        return self._packed

    def trace(self):
        """The last complete episode of environment #0, states as 24-letter strings."""
        t = self.trainer.trace if self.trainer is not None else None
        if not t:
            return {}
        out = dict(t)
        out['states'] = [rb.to_str(x) for x in rb.decode(np.array(t['states']))]
        return out


def serve(port, serving):
    """``/`` is dashboard.html, other files are served from the repo directory;
    ``/metrics``, ``/trace``, ``/spectator?state=...`` and ``/policy.bin`` are
    the data endpoints used by dashboard.html and playground.html."""
    from urllib.parse import urlparse, parse_qs
    here = os.path.dirname(os.path.abspath(__file__))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == '/metrics':
                    return self._send(json.dumps(serving.metrics()).encode(), 'application/json')
                if u.path == '/trace':
                    return self._send(json.dumps(serving.trace()).encode(), 'application/json')
                if u.path == '/spectator':
                    state = parse_qs(u.query).get('state', [''])[0]
                    return self._send(json.dumps(serving.act(state)).encode(), 'application/json')
                if u.path == '/policy.bin':
                    return self._send(serving.policy_bytes(), 'application/octet-stream')
            except Exception as e:  # bad state string, no table, ...
                return self._send(json.dumps({'error': str(e)}).encode(), 'application/json')
            if u.path in ('/', '/index.html'):
                self.path = '/dashboard.html'
            return super().do_GET()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('0.0.0.0', port), lambda *a, **k: Handler(*a, directory=here, **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print('dashboard:   http://localhost:%d/' % port)
    print('playground:  http://localhost:%d/playground.html' % port)
    return server


def enable_ansi():
    if os.name == 'nt':
        os.system('')                     # turns on VT escape processing in the Windows console
    return sys.stdout.isatty()


def fmt_time(sec):
    sec = max(0, int(sec))
    return '%d:%02d:%02d' % (sec // 3600, sec % 3600 // 60, sec % 60) if sec >= 3600 else '%d:%02d' % (sec // 60, sec % 60)


def dwidth(text):
    """Display width of a string in terminal cells (CJK characters take two)."""
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 0 if unicodedata.combining(ch) else 1 for ch in text)


SPARK = '▁▂▃▄▅▆▇█'


def pad(text, cells):
    return text + ' ' * max(0, cells - dwidth(text))


def sparkline(values, n=16):
    vals = [v for v in values[-n:] if v is not None]
    if len(vals) < 2:
        return ''
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return SPARK[3] * len(vals)
    return ''.join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


# what every number on the status panel means (the dashboard shows the same definitions)
METRIC_HELP = [
    ('K',     '课程深度：训练起点 = 复原态随机打乱 1…K 步；当前深度的解出率过了门槛就 K+1'),
    ('解出率', '随机打乱的魔方（按全部 3,674,160 个状态的真实分布加权），贪心策略 30 步内复原的比例'),
    ('泛化',  '训练里从未采样过的状态，贪心策略 30 步内复原的比例 —— 只有网络智能体有'),
    ('已采样', '训练里采样过的状态占全部状态的比例；Q 表智能体记为「覆盖」= 至少学过一次的状态比例'),
    ('|Q+d|', '价值误差：|max_a Q(s,a) + 真实 BFS 距离| 的均值，Q 值本应等于「离复原还差几步」的相反数，0 = 完美'),
    ('loss',  '最近 200 次更新的 TD 误差均方 —— 只有网络智能体有'),
    ('吞吐',  'upd/s = 每秒梯度更新次数（网络）；步/s = 每秒环境转动次数'),
]


class Progress:
    """pip-style progress bar plus a small status panel, redrawn in place at every evaluation (TTY only).

    Curriculum promotions and other events scroll past above the panel like pip's "Collecting ..." lines.
    """
    # 256-colour palette shared with the dashboard: solve = cyan, curriculum = amber, sampled = violet,
    # generalisation = green, error = rose; the bar itself is pip's magenta / green when finished
    C = {'dim': '\033[2m', 'bold': '\033[1m', 'off': '\033[0m',
         'bar': '\033[38;5;197m', 'bar_bg': '\033[38;5;238m', 'bar_done': '\033[38;5;78m',
         'cyan': '\033[38;5;51m', 'amber': '\033[38;5;214m', 'violet': '\033[38;5;141m',
         'green': '\033[38;5;78m', 'rose': '\033[38;5;204m', 'grey': '\033[38;5;245m', 'white': '\033[38;5;255m'}
    BAR_W = 40

    def __init__(self, args, tr, plain):
        self.args, self.tr, self.plain, self.t0 = args, tr, plain, time.time()
        self.max_k = tr_max_k(tr)
        self.start_steps = tr.steps
        self.drawn = 0                     # lines of the panel currently on screen
        self.prev_k = tr.K
        self.hist = {'success_random': [], 'unseen_success': [], 'coverage': [], 'q_error': [], 'loss': []}
        self.is_net = args.agent == 'net'

    # ---- styling helpers ----
    def col(self, name, text):
        return text if self.plain else self.C[name] + text + self.C['off']

    def columns(self):
        try:
            return max(60, shutil.get_terminal_size((100, 24)).columns)
        except Exception:
            return 100

    def bar(self, frac):
        frac = min(1.0, max(0.0, frac))
        filled = frac * self.BAR_W
        full = int(filled)
        half = 1 if filled - full >= 0.5 and full < self.BAR_W else 0
        done = '━' * full + ('╸' if half else '')
        rest = '━' * (self.BAR_W - full - half)
        return self.col('bar_done' if frac >= 1 else 'bar', done) + self.col('bar_bg', rest)

    def fit(self, segs):
        """segs: list of (text, style-or-None, droppable).  Fit the line into the terminal width."""
        budget = self.columns() - 1
        keep = list(segs)
        while sum(dwidth(t) for t, _, _ in keep) > budget and any(d for _, _, d in keep):
            for i in range(len(keep) - 1, -1, -1):
                if keep[i][2]:
                    del keep[i]
                    break
        out, used = [], 0
        for text, style, _ in keep:
            w = dwidth(text)
            if used + w > budget:
                cut = ''
                for ch in text:
                    if used + dwidth(cut + ch) > budget:
                        break
                    cut += ch
                text, w = cut, dwidth(cut)
            used += w
            out.append(self.col(style, text) if style else text)
            if used >= budget:
                break
        return ''.join(out)

    # ---- content ----
    def header_segs(self, rec):
        a = self.args
        if self.is_net and a.backend == 'torch':
            who = 'net/torch %s · %s' % (rec.get('device', ''), '-'.join(map(str, [144] + list(self.tr.layers) + [9])))
        elif self.is_net:
            who = 'net/numpy · 144-%d-%d-9' % (a.hidden, a.hidden)
        else:
            who = 'Q 表 · 3,674,160 × 9'
        tag = ' · tag %s' % a.tag if getattr(a, 'tag', None) else ''
        return [(' ▍ ', 'bar', False), ('2×2 cube', 'bold', False), ('  %s%s' % (who, tag), 'grey', False),
                ('   step %s' % format(rec['step'], ','), 'dim', True)]

    def bar_segs(self, rec):
        a = self.args
        elapsed = time.time() - self.t0
        if a.minutes:
            frac, eta = min(1.0, elapsed / (60 * a.minutes)), 60 * a.minutes - elapsed
            budget = '时长预算 %d min' % a.minutes
        else:
            done_steps = rec['step'] - self.start_steps
            frac = min(1.0, done_steps / a.steps)
            eta = (a.steps - done_steps) / max(rec['sps'], 1)
            budget = '步数预算 %s' % format(a.steps, ',')
        goal = min(1.0, rec['success_random'] / a.target) * (rec['K'] / self.max_k)
        if goal > frac:                     # closer to the target than to the budget: show that instead
            frac, eta = goal, eta * (1 - goal) / max(1e-9, 1 - frac) if frac < 1 else 0
        eta_text = '0:00' if frac >= 1 else fmt_time(eta)
        return [(' ', None, False), (self.bar(frac), None, False), (' %3d%%' % round(100 * frac), 'bold', False),
                ('  已用 %s' % fmt_time(elapsed), 'white', False), ('  剩余 %s' % eta_text, 'white', False),
                ('   ' + budget, 'dim', True)]

    def metric_segs(self, label, style, value, spark, desc, extra=''):
        segs = [(' ' + pad(label, 7), 'grey', False), (value, style, False)]
        segs.append(('  ' + spark if spark else '  ' + ' ' * 16, 'dim', True))
        segs.append(('  ' + desc, 'dim', True))
        if extra:
            segs.append(('   ' + extra, 'grey', True))
        return segs

    def lines(self, rec):
        K, succ = rec['K'], rec['success_random']
        for k in self.hist:
            self.hist[k].append(rec.get(k))
        at_k = rec['success_by_depth'][K - 1]
        promote = getattr(self.tr, 'promote', 0.97)
        pips = '▮' * K + '▯' * (self.max_k - K)
        k_state = ('深度 %d 解出 %.1f%% / 升级门槛 %.0f%%' % (K, 100 * at_k, 100 * promote)) if K < self.max_k \
            else '已到最深，深度 %d 解出 %.1f%%' % (K, 100 * at_k)
        out = [self.header_segs(rec), self.bar_segs(rec),
               [(' ' + pad('课程 K', 7), 'grey', False), ('%2d/%d ' % (K, self.max_k), 'amber', False), (pips, 'amber', False),
                ('   起点 = 复原态打乱 1…%d 步' % K, 'dim', True), ('   ' + k_state, 'grey', True)]]
        scol = 'green' if succ >= 0.95 else 'cyan' if succ >= 0.5 else 'rose'
        out.append(self.metric_segs('解出率', scol, '%6.2f%%' % (100 * succ), sparkline(self.hist['success_random']),
                                    '随机打乱的魔方，贪心 30 步内复原的比例（按 367 万状态分布加权）'))
        if self.is_net:
            u = rec.get('unseen_success')
            out.append(self.metric_segs('泛化', 'green', '%6.2f%%' % (100 * u) if u is not None else '   n/a ',
                                        sparkline(self.hist['unseen_success']), '训练里从未采样过的状态，贪心能复原的比例'))
            out.append(self.metric_segs('已采样', 'violet', '%6.2f%%' % (100 * rec['coverage']), sparkline(self.hist['coverage']),
                                        '训练里采样过的状态 / 3,674,160 个状态'))
        else:
            out.append(self.metric_segs('覆盖', 'violet', '%6.2f%%' % (100 * rec['coverage']), sparkline(self.hist['coverage']),
                                        'Q 表里至少学过一次的状态 / 3,674,160 个状态'))
        qe = rec['q_error']
        extra = ('loss %.4f' % rec['loss']) if rec.get('loss') is not None else ''
        out.append(self.metric_segs('|Q+d|', 'rose', '%6.3f ' % qe if qe is not None else '   n/a ', sparkline(self.hist['q_error']),
                                    '价值误差 = |Q 值 + 真实距离| 均值，0 = 完美', extra))
        thr = ('%.1f upd/s · %s 步/s' % (rec['updates'] / max(rec['time'], 1e-9), format(rec['sps'], ','))
               if rec.get('updates') else '%s 步/s' % format(rec['sps'], ','))
        out.append([(' ' + pad('吞吐', 7), 'grey', False), (thr, 'white', False),
                    ('   %s 回合 · 复原 %s · 训练用时 %s' % (format(rec['episodes'], ','), format(rec['solved'], ','), fmt_time(rec['time'])), 'dim', True)])
        return [self.fit(s) for s in out]

    # ---- drawing ----
    def clear(self):
        if self.drawn and not self.plain:
            sys.stdout.write('\033[%dA' % self.drawn + '\r' + '\033[J')
            self.drawn = 0

    def log(self, text, style='grey'):
        """Print an event line above the panel (survives in the scrollback)."""
        stamp = fmt_time(time.time() - self.t0)
        if self.plain:
            print('[%s] %s' % (stamp, text))
            return
        self.clear()
        sys.stdout.write(self.col('dim', ' [%s] ' % stamp) + self.col(style, text) + '\n')
        sys.stdout.flush()

    def show(self, rec, final=False):
        if self.tr.K > self.prev_k:
            d = rec['K']
            self.log('↑ 课程升级  K %d → %d   深度 %d 解出 %.1f%% ≥ 门槛 %.0f%%'
                     % (self.prev_k, self.tr.K, d, 100 * rec['success_by_depth'][d - 1], 100 * getattr(self.tr, 'promote', 0.97)), 'amber')
            self.prev_k = self.tr.K
        if self.plain:
            print(fmt_row(rec))
            return
        lines = self.lines(rec)
        try:
            self.clear()
            sys.stdout.write('\n'.join(lines) + '\n')
            self.drawn = len(lines)
            if final:
                self.drawn = 0
        except UnicodeEncodeError:
            sys.stdout.write(fmt_row(rec) + '\n')
        sys.stdout.flush()

    def legend(self):
        rows = METRIC_HELP if self.is_net else [r for r in METRIC_HELP if r[0] not in ('泛化', 'loss')]
        if self.plain:
            for k, v in rows:
                print('  %-6s %s' % (k, v))
            return
        sys.stdout.write(self.col('bold', ' 指标说明') + '\n')
        for k, v in rows:
            sys.stdout.write('  ' + self.col('white', '%-6s' % k) + ' ' + self.col('dim', v) + '\n')
        sys.stdout.write('\n')
        sys.stdout.flush()


def fmt_row(rec):
    """One plain line per evaluation (log files, --plain, non-TTY)."""
    parts = ['step %11s' % format(rec['step'], ','), 'time %8s' % fmt_time(rec['time']), 'K %2d' % rec['K'],
             'solve %6.2f%%' % (100 * rec['success_random'])]
    if rec.get('unseen_success') is not None:
        parts.append('sampled %5.1f%%' % (100 * rec['coverage']))
        parts.append('unseen-solve %6.2f%%' % (100 * rec['unseen_success']))
    else:
        parts.append('coverage %5.1f%%' % (100 * rec['coverage']))
    parts.append('|Q+d| %s' % ('%.3f' % rec['q_error'] if rec['q_error'] is not None else 'n/a'))
    if rec.get('loss') is not None:
        parts.append('loss %.4f' % rec['loss'])
    if rec.get('updates'):
        parts.append('%.1f upd/s' % (rec['updates'] / max(rec['time'], 1e-9)))
    parts.append('%s steps/s' % format(rec['sps'], ','))
    return '  '.join(parts)


def train(args):
    global CONFIG
    torch_backend = args.agent == 'net' and args.backend == 'torch'
    if args.envs is None:
        args.envs = 64 if torch_backend else 256 if args.agent == 'net' else 8192
    if args.batch is None:
        args.batch = 8192 if torch_backend else 256
    if args.k_margin is None:
        args.k_margin = 2 if torch_backend else 0
    if args.target_every is None:
        args.target_every = 200 if torch_backend else 1000
    if args.updates_per_step is None:
        args.updates_per_step = 1 if torch_backend else 4
    if args.steps is None:
        args.steps = 10 ** 15 if args.minutes else (40_000_000 if args.agent == 'net' else 400_000_000)
    if args.agent == 'net' and args.eval_every == 50:
        args.eval_every = 50 if torch_backend else 200      # an evaluation costs ~0.7 s with the numpy network
    CONFIG = {k: v for k, v in vars(args).items() if k != 'cmd'}
    pt = paths(args.agent)
    if args.agent == 'net' and args.backend == 'torch':
        from qnet_torch import TorchNetTrainer
        tr = TorchNetTrainer(layers=tuple(int(x) for x in args.layers.split(',')), lr=args.lr, batch=args.batch,
                             device=args.device, all_actions=not args.replay, k_margin=args.k_margin,
                             weight_by_depth=args.weight_by_depth, promote=args.promote, max_k=args.max_k,
                             k_start=args.k_start, target_every=args.target_every, n_envs=args.envs, eps=args.eps,
                             episode_cap=args.cap, buffer=args.buffer, updates_per_step=args.updates_per_step, seed=args.seed,
                             amp=args.amp)
        print('torch network agent on %s: %s MLP, batch %d, %s targets, curriculum margin %d'
              % (tr.device, '-'.join(map(str, [24 * 6] + tr.layers + [9])), args.batch,
                 'replay Q-learning' if args.replay else 'all-actions', args.k_margin))
    elif args.agent == 'net':
        from qnet import NetTrainer
        tr = NetTrainer(n_envs=args.envs, gamma=args.gamma, eps=args.eps, promote=args.promote, episode_cap=args.cap,
                        seed=args.seed, hidden=args.hidden, lr=args.lr, batch=args.batch, buffer=args.buffer,
                        target_every=args.target_every, max_k=args.max_k, updates_per_step=args.updates_per_step,
                        k_start=args.k_start, all_actions=args.all_actions, dyn_cap=args.dyn_cap,
                        k_margin=args.k_margin, weight_by_depth=args.weight_by_depth)
        print('network agent: %d-%d-%d-9 MLP, %d parallel environments, curriculum up to K=%d'
              % (24 * 6, args.hidden, args.hidden, args.envs, args.max_k))
    else:
        tr = Trainer(n_envs=args.envs, alpha=args.alpha, gamma=args.gamma, eps=args.eps,
                     promote=args.promote, episode_cap=args.cap, eval_every=args.eval_every, seed=args.seed)
        print('transition table + BFS ready, %d states, %d parallel environments' % (rb.N_STATES, args.envs))
    if args.resume:
        if args.agent != 'net' or not os.path.exists(NET_PATH):
            raise SystemExit('--resume needs a saved network agent (cache/net.npz)')
        tr.load(NET_PATH)
        if os.path.exists(NET_SEEN_PATH):
            tr.seen |= np.unpackbits(np.load(NET_SEEN_PATH))[:rb.N_STATES].astype(bool)
        if os.path.exists(NET_METRICS_PATH):
            with open(NET_METRICS_PATH) as f:
                old = json.load(f).get('records', [])
            if old:
                tr.K = max(tr.K, min(old[-1]['K'], tr_max_k(tr)))
                offset_step, offset_time = old[-1]['step'], old[-1]['time']
                tr.metrics = old
                tr.steps = offset_step
                tr.updates = old[-1].get('updates', 0)
                tr.t0 -= offset_time                 # elapsed time, steps and updates continue from the saved run
        print('resumed from %s: K=%d, %d earlier evaluations' % (NET_PATH, tr.K, len(tr.metrics)))
    if args.dashboard:
        serve(args.dashboard, Serving(tr, args.agent))
    progress = Progress(args, tr, plain=args.plain or not enable_ansi())
    progress.legend()
    run_start, start_steps = time.time(), tr.steps    # --minutes and --steps count this process only, also after --resume
    progress.log('开始训练  ' + ('时长预算 %d 分钟' % args.minutes if args.minutes else '步数预算 %s' % format(args.steps, ','))
                 + '  每 %d 次迭代评估一次' % args.eval_every + ('  · 从 K=%d 续训' % tr.K if args.resume else ''))
    i = 0
    try:
        while tr.steps - start_steps < args.steps and not (args.minutes and time.time() - run_start > 60 * args.minutes):
            tr.step()
            i += 1
            if i % args.eval_every == 0:
                if args.lr_final is not None and args.minutes:
                    frac = min(1.0, (time.time() - run_start) / (60 * args.minutes))
                    tr.set_lr(args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + np.cos(np.pi * frac)))
                rec = tr.evaluate()
                progress.show(rec)
                write_metrics(tr.metrics, pt['metrics'])
                done = rec['success_random'] >= args.target and (args.agent == 'net' or rec['coverage'] >= args.target)
                if rec['K'] == tr_max_k(tr) and done:
                    progress.log('✔ 达到目标：K=%d，解出率 %.2f%% ≥ %.1f%%' % (rec['K'], 100 * rec['success_random'], 100 * args.target), 'green')
                    break
        else:
            progress.log('预算用完，最后评估一次并保存', 'amber')
    except KeyboardInterrupt:
        progress.log('Ctrl-C：最后评估一次并保存', 'amber')
    rec = tr.evaluate()
    progress.show(rec, final=True)
    write_metrics(tr.metrics, pt['metrics'])
    if args.agent == 'net':
        progress.log('导出每个状态的贪心动作（约 30 s）')
        tr.save(NET_PATH, NET_POLICY_PATH)
        np.save(NET_SEEN_PATH, np.packbits(tr.seen))
    else:
        tr.save()
    print_table(rec, progress)
    progress.log('已保存 %s  %s  %s' % (pt['model'], pt['policy'], pt['metrics']), 'green')
    if args.dashboard:
        print('still serving; Ctrl-C to quit')
        wait_forever()


def tr_max_k(tr):
    return getattr(tr, 'max_k', MAX_DEPTH)


def wait_forever():
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def serve_cmd(args):
    sv = Serving(agent=args.agent)
    if not sv.has_table():
        print('note: no trained %s agent in %s/ yet, the pages will use their built-in solvers' % (args.agent, CACHE))
    serve(args.port, sv)
    wait_forever()


DEPTH_COUNT = [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]


def print_table(rec, progress=None):
    """Per-distance breakdown: 400 states at each exact distance d, greedy policy, 30-move limit."""
    col = progress.col if progress else (lambda name, text: text)
    print()
    print(col('bold', ' 按距离分解') + col('dim', '   每个距离 d 抽 400 个离复原正好 d 步的状态，贪心走 30 步'))
    print(col('grey', '  距离   状态数      解出率                        平均步数(最优)'))
    for d in range(1, MAX_DEPTH + 1):
        s = rec['success_by_depth'][d - 1]
        ml = rec['mean_len_by_depth'][d - 1]
        n = int(round(s * 24))
        bar = col('green' if s >= 0.95 else 'cyan' if s >= 0.5 else 'rose', '█' * n) + col('bar_bg', '░' * (24 - n))
        print('  %2d   %9s   %6.1f%%  %s  %s' % (d, format(DEPTH_COUNT[d], ','), 100 * s, bar, '%.2f (%d)' % (ml, d) if ml else '-'))


def load_policy(agent):
    pt = paths(agent)
    if os.path.exists(pt['policy']):
        return np.load(pt['policy'], mmap_mode='r')
    if agent == 'table' and os.path.exists(Q_PATH):
        return np.argmax(np.load(Q_PATH, mmap_mode='r'), axis=1)
    sys.exit('no trained %s agent in %s/ - run: python play_rubik.py train --agent %s' % (agent, CACHE, agent))


def solve(args):
    a = rb.to_array(args.state)
    idx = rb.encode(a)
    moves = []
    if args.beam:
        if args.agent != 'net' or not os.path.exists(NET_PATH):
            sys.exit('--beam needs the network agent (cache/net.npz)')
        from qnet import MLP, onehot
        net = MLP.from_file(NET_PATH)
        moves = beam_solve(lambda i: net.forward(onehot(i)), rb.transitions(), idx, width=args.beam)
        if moves is None:
            print('beam search (width %d) found no solution within 30 moves' % args.beam)
            return
        print('%d moves:' % len(moves), ' '.join(rb.ACTIONS[m] for m in moves), '  (%s)' % ' '.join(rb.NOTATION[m] for m in moves))
        dist = rb.bfs_distances(rb.transitions())
        print('optimal:', int(dist[idx]))
        return
    policy = load_policy(args.agent)
    for _ in range(30):
        if idx == rb.SOLVED_INDEX:
            break
        act = int(policy[idx])
        moves.append(act)
        a = a[rb.PERM[act]]
        idx = rb.encode(a)
    if idx != rb.SOLVED_INDEX:
        print('policy did not solve this state within 30 moves')
        return
    print('%d moves:' % len(moves), ' '.join(rb.ACTIONS[m] for m in moves), '  (%s)' % ' '.join(rb.NOTATION[m] for m in moves))
    if os.path.exists(os.path.join(CACHE, 'transitions.npy')):
        dist = rb.bfs_distances(rb.transitions())
        print('optimal:', int(dist[rb.encode(args.state)]))


def evaluate(args):
    if args.agent == 'net':
        return evaluate_net(args)
    if not os.path.exists(Q_PATH):
        sys.exit('no trained table in %s/ - run: python play_rubik.py train' % CACHE)
    tr = Trainer.__new__(Trainer)
    tr.rng = np.random.default_rng(args.seed)
    tr.T = np.ascontiguousarray(rb.transitions())
    tr.dist = rb.bfs_distances(tr.T)
    tr.by_depth = [np.nonzero(tr.dist == d)[0] for d in range(MAX_DEPTH + 1)]
    tr.depth_weight = np.bincount(tr.dist, minlength=MAX_DEPTH + 1) / rb.N_STATES
    tr.Q = np.load(Q_PATH, mmap_mode='r')
    tr.Q0 = -float(MAX_DEPTH + 1)
    tr.steps, tr.episodes, tr.solved, tr.K, tr.t0, tr.metrics, tr.promote = 0, 0, 0, MAX_DEPTH, time.time(), [], 2
    rec = tr.evaluate(n_per_depth=2000)
    print('coverage %.2f%%   success on a uniformly random state %.3f%%   mean |Q + distance| %.4f'
          % (100 * rec['coverage'], 100 * rec['success_random'], rec['q_error']))
    print_table(rec)


def evaluate_net(args):
    if not os.path.exists(NET_PATH):
        sys.exit('no trained network in %s/ - run: python play_rubik.py train --agent net' % CACHE)
    from qnet import MLP, onehot
    T = np.ascontiguousarray(rb.transitions())
    dist = rb.bfs_distances(T)
    net = MLP.from_file(NET_PATH)
    qf = lambda idx: net.forward(onehot(idx))
    rng = np.random.default_rng(args.seed)
    rec = {'success_by_depth': [], 'mean_len_by_depth': []}
    weight = np.bincount(dist, minlength=MAX_DEPTH + 1) / rb.N_STATES
    total = weight[0]
    seen = None
    if args.seen:
        if not os.path.exists(NET_SEEN_PATH):
            sys.exit('no %s: train (or resume) once with the current code to record which states were sampled' % NET_SEEN_PATH)
        seen = np.unpackbits(np.load(NET_SEEN_PATH))[:rb.N_STATES].astype(bool)
        print('states sampled during training: %.2f%%' % (100 * seen.mean()))
        print('\ndistance   seen: n  solved   |  unseen: n  solved   steps-to-seen (median / >=3 / never)   |Q+d| seen / unseen')
        for d in range(1, MAX_DEPTH + 1):
            pool = np.nonzero(dist == d)[0]
            pick = pool[rng.integers(pool.shape[0], size=min(4000, pool.shape[0]))]
            ps, pu = pick[seen[pick]], pick[~seen[pick]]
            ok_s = greedy_rollout(qf, T, ps) > 0 if ps.size else np.zeros(0, bool)
            ok_u = greedy_rollout(qf, T, pu) > 0 if pu.size else np.zeros(0, bool)
            # for unseen states that get solved: how many greedy moves until the path enters a seen state
            hops = []
            for s0 in pu[ok_u][:500]:
                st, h = int(s0), 0
                while st != rb.SOLVED_INDEX and not seen[st] and h < 30:
                    st = int(T[st, int(np.argmax(qf(np.array([st]))[0]))]); h += 1
                hops.append(h if seen[st] else 99)
            hops = np.array(hops)
            err_s = float(np.abs(qf(ps).max(axis=1) + d).mean()) if ps.size else float('nan')
            err_u = float(np.abs(qf(pu).max(axis=1) + d).mean()) if pu.size else float('nan')
            print('  %2d      %5d  %6.1f%%   |  %5d  %6.1f%%   %s   %.2f / %.2f' % (
                d, ps.size, 100 * ok_s.mean() if ps.size else 0, pu.size, 100 * ok_u.mean() if pu.size else 0,
                ('%3d / %4.0f%% / %4.0f%%' % (np.median(hops[hops < 99]) if (hops < 99).any() else 0,
                                              100 * (hops[hops < 99] >= 3).mean() if (hops < 99).any() else 0,
                                              100 * (hops == 99).mean())) if hops.size else '   -  /    - /    -',
                err_s, err_u))
        print('steps-to-seen: greedy moves from an unseen solved state until the path first enters a sampled state;'
              ' "never" = solved without touching any sampled state')
        return
    n_eval = 2000 if not args.beam else 200
    for d in range(1, MAX_DEPTH + 1):
        pool = np.nonzero(dist == d)[0]
        pick = pool[rng.integers(pool.shape[0], size=min(n_eval, pool.shape[0]))]
        if args.beam:
            sols = [beam_solve(qf, T, int(s), width=args.beam) for s in pick]
            length = np.array([len(m) if m is not None else -1 for m in sols])
        else:
            length = greedy_rollout(qf, T, pick)
        ok = length > 0
        rec['success_by_depth'].append(float(ok.mean()))
        rec['mean_len_by_depth'].append(float(length[ok].mean()) if ok.any() else None)
        total += weight[d] * ok.mean()
    sample = rng.integers(rb.N_STATES, size=20000)
    err = float(np.abs(qf(sample).max(axis=1) + dist[sample]).mean())
    print('%s   success on a uniformly random state %.3f%%   mean |Q + distance| %.4f'
          % ('beam search, width %d' % args.beam if args.beam else 'greedy', 100 * total, err))
    print_table(rec)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--steps', type=int, default=None, help='max environment steps (default: unlimited with --minutes, else 400M table / 40M net)')
    t.add_argument('--lr-final', type=float, default=None, help='net: cosine-decay the learning rate to this value over --minutes')
    t.add_argument('--envs', type=int, default=None, help='parallel environments (default 8192 table / 1024 net)')
    t.add_argument('--alpha', type=float, default=1.0)
    t.add_argument('--gamma', type=float, default=1.0)
    t.add_argument('--eps', type=float, default=0.1)
    t.add_argument('--promote', type=float, default=0.97, help='success rate at depth K that unlocks K+1')
    t.add_argument('--cap', type=int, default=25, help='max moves per episode')
    t.add_argument('--eval-every', type=int, default=50, help='batched steps between evaluations')
    t.add_argument('--target', type=float, default=0.999, help='stop once random-state success and coverage both reach this')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--dashboard', type=int, default=0, metavar='PORT', help='serve dashboard.html + /metrics on this port')
    t.add_argument('--agent', choices=['table', 'net'], default='table', help='Q-table (default) or neural network')
    t.add_argument('--hidden', type=int, default=256, help='net: hidden layer width')
    t.add_argument('--lr', type=float, default=1e-3, help='net: Adam learning rate')
    t.add_argument('--batch', type=int, default=None, help='net: minibatch size (default 256 numpy / 8192 torch)')
    t.add_argument('--updates-per-step', type=int, default=None, help='net: gradient steps per environment step (default 4 numpy / 1 torch)')
    t.add_argument('--minutes', type=float, default=0, help='stop after this many minutes of training (0 = no limit)')
    t.add_argument('--all-actions', action='store_true', help='net: DeepCube-style targets for all 9 actions of sampled states (uses the transition model)')
    t.add_argument('--dyn-cap', action='store_true', help='net: episode cap = K + 3 so episodes do not wander far beyond the curriculum')
    t.add_argument('--k-margin', type=int, default=None, help='net: sample training states up to K + margin moves deep (default 0 numpy / 2 torch)')
    t.add_argument('--backend', choices=['numpy', 'torch'], default='numpy', help='net: numpy (CPU) or torch (Apple MPS / CUDA / CPU)')
    t.add_argument('--device', default='auto', help='torch: auto | mps | cuda | cpu')
    t.add_argument('--layers', default='1024,1024,512', help='torch: hidden layer widths')
    t.add_argument('--replay', action='store_true', help='torch: model-free replay Q-learning instead of all-actions targets')
    t.add_argument('--resume', action='store_true', help='net: continue from cache/net.npz (weights, curriculum depth, metrics history)')
    t.add_argument('--plain', action='store_true', help='plain one-line-per-evaluation log instead of the progress bar')
    t.add_argument('--amp', action='store_true', help='torch: bf16 autocast for the forward passes (CUDA), ~2x faster')
    t.add_argument('--weight-by-depth', action='store_true', help='net: weight the loss by 1/scramble depth (DeepCube)')
    t.add_argument('--k-start', type=int, default=1, help='net: initial curriculum depth (= --max-k: no curriculum, all depths from the start)')
    t.add_argument('--buffer', type=int, default=200000, help='net: replay buffer size')
    t.add_argument('--target-every', type=int, default=None, help='net: target network refresh in updates (default 1000 numpy / 200 torch)')
    t.add_argument('--max-k', type=int, default=MAX_DEPTH, help='net: cap the curriculum depth (train shallow, test deep)')
    e = sub.add_parser('eval')
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--agent', choices=['table', 'net'], default='table')
    e.add_argument('--beam', type=int, default=0, metavar='W', help='net: beam search of width W on the value function instead of greedy')
    e.add_argument('--seen', action='store_true', help='net: split every distance into states sampled during training vs never sampled')
    s = sub.add_parser('solve')
    s.add_argument('state', help='24 letters, e.g. %s' % rb.INIT)
    s.add_argument('--agent', choices=['table', 'net'], default='table')
    s.add_argument('--beam', type=int, default=0, metavar='W', help='net: beam search of width W')
    v = sub.add_parser('serve', help='serve dashboard.html / playground.html plus the trained agent, no training')
    v.add_argument('--port', type=int, default=8000)
    v.add_argument('--agent', choices=['table', 'net'], default='table')
    for sp in (t, e, s, v):
        sp.add_argument('--tag', default='', help='keep this experiment\'s files in cache/<tag>/ (train, eval, solve, serve)')
    args = p.parse_args()
    set_tag(args.tag)
    {'train': train, 'eval': evaluate, 'solve': solve, 'serve': serve_cmd}[args.cmd](args)


if __name__ == '__main__':
    main()
