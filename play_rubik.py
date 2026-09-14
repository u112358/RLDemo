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

    python play_rubik.py train [--dashboard 8000]     # train, optional live page
    python play_rubik.py serve [--port 8000]           # pages + trained table, no training
    python play_rubik.py eval                          # success / length per depth
    python play_rubik.py solve wggbrymrgrwrbmbwybywmygm
"""
import argparse
import http.server
import json
import os
import sys
import threading
import time

import numpy as np

import rubik as rb

CACHE = 'cache'
Q_PATH = os.path.join(CACHE, 'q_table.npy')
POLICY_PATH = os.path.join(CACHE, 'policy.npy')
METRICS_PATH = os.path.join(CACHE, 'metrics.json')
MAX_DEPTH = 11


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
    """Follow argmax Q from states ``s``; returns solve length per state (-1 = failed)."""
    s = s.copy()
    length = np.full(s.shape[0], -1, dtype=np.int64)
    for t in range(max_steps):
        active = length < 0
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        a = np.argmax(Q[s[idx]], axis=1)
        s[idx] = T[s[idx], a]
        just = idx[s[idx] == rb.SOLVED_INDEX]
        length[just] = t + 1
    return length


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
        self.trace = None                                # last transition of env 0, for the dashboard

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
        self.trace = {'step': self.steps, 'state': int(s[0]), 'action': int(a[0]), 'next': int(s2[0]),
                      'explore': bool(explore[0]), 'unknown': bool(q[0, a[0]] == self.Q0),
                      'done': bool(done[0]), 'k': int(self.k[0]), 'age': int(self.age[0]), 'K': self.K}
        reset = done | (self.age >= self.cap)
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
        rec = {
            'step': self.steps, 'time': round(time.time() - self.t0, 1), 'K': self.K,
            'episodes': self.episodes, 'solved': self.solved,
            'coverage': coverage, 'q_error': q_err, 'success_random': succ_random,
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
        np.save(POLICY_PATH, np.argmax(self.Q, axis=1).astype(np.uint8))
        write_metrics(self.metrics)


def write_metrics(metrics):
    os.makedirs(CACHE, exist_ok=True)
    tmp = METRICS_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'config': CONFIG, 'records': metrics}, f)
    os.replace(tmp, METRICS_PATH)


CONFIG = {}


def pack_policy(policy):
    """uint8 actions (0..8) -> two per byte, low nibble first."""
    p = np.asarray(policy, dtype=np.uint8)
    if p.shape[0] % 2:
        p = np.concatenate([p, np.zeros(1, np.uint8)])
    return (p[0::2] | (p[1::2] << 4)).tobytes()


class Serving:
    """Data behind the HTTP endpoints: live trainer, or the saved cache."""

    def __init__(self, trainer=None):
        self.trainer = trainer
        self.Q = self.policy = None
        self.records, self.config = [], {}
        self._packed = None
        if trainer is None:
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH) as f:
                    m = json.load(f)
                self.records, self.config = m.get('records', []), m.get('config', {})
            if os.path.exists(Q_PATH):
                self.Q = np.load(Q_PATH, mmap_mode='r')
            elif os.path.exists(POLICY_PATH):
                self.policy = np.load(POLICY_PATH, mmap_mode='r')

    def metrics(self):
        if self.trainer is not None:
            return {'config': CONFIG, 'records': self.trainer.metrics}
        return {'config': self.config, 'records': self.records}

    def act(self, state):
        """Greedy action of the current table for a 24-letter state."""
        idx = rb.encode(state)
        Q = self.trainer.Q if self.trainer is not None else self.Q
        if Q is not None:
            row = np.asarray(Q[idx], dtype=np.float64)
            best = np.nonzero(row == row.max())[0]
            a = int(np.random.choice(best))
            return {'index': int(idx), 'action': a, 'q': [round(float(v), 3) for v in row],
                    'unknown': bool(row.max() <= -(MAX_DEPTH + 1))}
        if self.policy is not None:
            return {'index': int(idx), 'action': int(self.policy[idx]), 'q': None, 'unknown': False}
        raise RuntimeError('no table')

    def policy_bytes(self):
        if self.trainer is not None:
            return pack_policy(np.argmax(self.trainer.Q, axis=1))
        if self._packed is None:
            src = self.policy if self.policy is not None else np.argmax(self.Q, axis=1)
            self._packed = pack_policy(src)
        return self._packed

    def trace(self):
        t = self.trainer.trace if self.trainer is not None else None
        if not t:
            return {}
        out = dict(t)
        out['state'] = rb.to_str(rb.decode(t['state'])[0])
        out['next'] = rb.to_str(rb.decode(t['next'])[0])
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


def fmt_row(rec):
    return ('step %10d  %6.0fs  K=%2d  coverage %5.1f%%  success(random) %6.2f%%  Q-err %s  %s env-steps/s'
            % (rec['step'], rec['time'], rec['K'], 100 * rec['coverage'], 100 * rec['success_random'],
               '%.3f' % rec['q_error'] if rec['q_error'] is not None else '  n/a',
               format(rec['sps'], ',')))


def train(args):
    global CONFIG
    CONFIG = {k: v for k, v in vars(args).items() if k != 'cmd'}
    tr = Trainer(n_envs=args.envs, alpha=args.alpha, gamma=args.gamma, eps=args.eps,
                 promote=args.promote, episode_cap=args.cap, eval_every=args.eval_every, seed=args.seed)
    print('transition table + BFS ready, %d states, %d parallel environments' % (rb.N_STATES, args.envs))
    if args.dashboard:
        serve(args.dashboard, Serving(tr))
    i = 0
    while tr.steps < args.steps:
        tr.step()
        i += 1
        if i % args.eval_every == 0:
            rec = tr.evaluate()
            print(fmt_row(rec))
            write_metrics(tr.metrics)
            if rec['K'] == MAX_DEPTH and rec['success_random'] >= args.target and rec['coverage'] >= args.target:
                print('target reached')
                break
    rec = tr.evaluate()
    print(fmt_row(rec))
    tr.save()
    print_table(rec)
    print('saved', Q_PATH, POLICY_PATH, METRICS_PATH)
    if args.dashboard:
        print('still serving; Ctrl-C to quit')
        wait_forever()


def wait_forever():
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def serve_cmd(args):
    sv = Serving()
    if sv.Q is None and sv.policy is None:
        print('note: no trained table in %s/ yet, the pages will use their built-in solvers' % CACHE)
    serve(args.port, sv)
    wait_forever()


def print_table(rec):
    print('\ndistance  success  mean length (optimal)')
    for d in range(1, MAX_DEPTH + 1):
        ml = rec['mean_len_by_depth'][d - 1]
        print('  %2d      %6.1f%%   %s' % (d, 100 * rec['success_by_depth'][d - 1], '%.2f (%d)' % (ml, d) if ml else '-'))


def load_policy():
    if os.path.exists(POLICY_PATH):
        return np.load(POLICY_PATH, mmap_mode='r')
    if os.path.exists(Q_PATH):
        return np.argmax(np.load(Q_PATH, mmap_mode='r'), axis=1)
    sys.exit('no trained table in %s/ - run: python play_rubik.py train' % CACHE)


def solve(args):
    policy = load_policy()
    a = rb.to_array(args.state)
    idx = rb.encode(a)
    moves = []
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--steps', type=int, default=400_000_000, help='max environment steps')
    t.add_argument('--envs', type=int, default=8192)
    t.add_argument('--alpha', type=float, default=1.0)
    t.add_argument('--gamma', type=float, default=1.0)
    t.add_argument('--eps', type=float, default=0.1)
    t.add_argument('--promote', type=float, default=0.97, help='success rate at depth K that unlocks K+1')
    t.add_argument('--cap', type=int, default=25, help='max moves per episode')
    t.add_argument('--eval-every', type=int, default=50, help='batched steps between evaluations')
    t.add_argument('--target', type=float, default=0.999, help='stop once random-state success and coverage both reach this')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--dashboard', type=int, default=0, metavar='PORT', help='serve dashboard.html + /metrics on this port')
    e = sub.add_parser('eval')
    e.add_argument('--seed', type=int, default=0)
    s = sub.add_parser('solve')
    s.add_argument('state', help='24 letters, e.g. %s' % rb.INIT)
    v = sub.add_parser('serve', help='serve dashboard.html / playground.html plus the trained table, no training')
    v.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    {'train': train, 'eval': evaluate, 'solve': solve, 'serve': serve_cmd}[args.cmd](args)


if __name__ == '__main__':
    main()
