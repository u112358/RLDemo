"""Q-learning with a neural network instead of a table (a small DQN).

Same environment, same reverse-scramble curriculum and the same one-step
Q-learning target as the tabular learner in ``play_rubik.py``; the only
change is where Q(s, a) lives.  The table has one row per state and can
only ever *cover* states; the network takes the 24 stickers as a 24 x 6
one-hot vector and has to *generalise*: states it never trained on get a
value from the same weights.  The metric ``unseen_success`` (greedy solve
rate on states that were never visited during training) measures exactly
that, and is 0 by construction for a table.

Implemented in numpy (no PyTorch): a 144 -> hidden -> hidden -> 9 MLP with
ReLU, Adam, a target network and a replay buffer.
"""
import time

import numpy as np

import rubik as rb

MAX_DEPTH = 11
N_IN = 24 * 6


N_IN_CUBIE = 7 * (7 + 3)        # per movable slot: which of the 7 cubies sits there (7) and its twist (3)
FEATURE_DIMS = {'sticker': N_IN, 'cubie': N_IN_CUBIE}


def onehot(idx):
    """State indices -> (N, 144) float32 one-hot of the 24 sticker colours."""
    states = rb.decode(np.asarray(idx, dtype=np.int64))
    n = states.shape[0]
    X = np.zeros((n, N_IN), dtype=np.float32)
    X[np.arange(n)[:, None], np.arange(24)[None, :] * 6 + states] = 1.0
    return X


def cubie_onehot(idx):
    """State indices -> (N, 70) float32: for each of the 7 movable slots, one-hot of the cubie in it
    and one-hot of its orientation.  Makes the piece structure explicit instead of raw stickers."""
    states = rb.decode(np.asarray(idx, dtype=np.int64))
    cid, ori = rb.cubies(states)
    n = states.shape[0]
    X = np.zeros((n, N_IN_CUBIE), dtype=np.float32)
    rows = np.arange(n)[:, None]
    X[rows, np.arange(7)[None, :] * 10 + cid[:, :7]] = 1.0
    X[rows, np.arange(7)[None, :] * 10 + 7 + ori[:, :7]] = 1.0
    return X


def features(idx, kind='sticker'):
    return onehot(idx) if kind == 'sticker' else cubie_onehot(idx)


def feature_kind(n_in):
    for k, v in FEATURE_DIMS.items():
        if v == n_in:
            return k
    raise ValueError('unknown feature width %d' % n_in)


def qfunction(net):
    """idx -> (n, 9) Q-values for a saved MLP, picking the feature kind from its input width."""
    kind = feature_kind(net.W[0].shape[0])
    return lambda idx: net.forward(features(idx, kind))


class MLP:
    """Two hidden layers, ReLU, Adam.  Everything float32."""

    def __init__(self, sizes, rng, lr=1e-3):
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32))
            self.b.append(np.zeros(b, dtype=np.float32))
        self.lr, self.beta1, self.beta2, self.eps = lr, 0.9, 0.999, 1e-8
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X, cache=False):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return (h, acts) if cache else h

    def backward(self, acts, dout):
        """Gradients of a loss whose gradient wrt the output is ``dout``."""
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        return gW + gb

    def adam(self, grads):
        self.t += 1
        params = self.W + self.b
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1; m += (1 - self.beta1) * g
            v *= self.beta2; v += (1 - self.beta2) * (g * g)
            p -= lr_t * m / (np.sqrt(v) + self.eps)

    def params(self):
        return [p.copy() for p in self.W + self.b]

    def load(self, params):
        n = len(self.W)
        self.W = [p.copy() for p in params[:n]]
        self.b = [p.copy() for p in params[n:]]

    def save(self, path):
        np.savez(path, *(self.W + self.b), sizes=np.array([self.W[0].shape[0]] + [w.shape[1] for w in self.W]))

    @classmethod
    def from_file(cls, path):
        z = np.load(path)
        sizes = z['sizes'].tolist()
        net = cls(sizes, np.random.default_rng(0))
        n = len(sizes) - 1
        net.W = [z['arr_%d' % i] for i in range(n)]
        net.b = [z['arr_%d' % (n + i)] for i in range(n)]
        return net


class NetTrainer:
    """Same interface as play_rubik.Trainer (step / evaluate / save / qvalues)."""

    def __init__(self, n_envs=256, gamma=1.0, eps=0.1, promote=0.97, episode_cap=25,
                 seed=0, hidden=256, lr=1e-3, batch=256, buffer=200000, target_every=1000, max_k=MAX_DEPTH,
                 updates_per_step=4, k_start=1, all_actions=False, dyn_cap=False, k_margin=0, weight_by_depth=False):
        from play_rubik import scramble  # shared helper; imported lazily (play_rubik imports this module)
        self._scramble = scramble
        self.rng = np.random.default_rng(seed)
        self.T = np.ascontiguousarray(rb.transitions())
        self.dist = rb.bfs_distances(self.T)
        self.by_depth = [np.nonzero(self.dist == d)[0] for d in range(MAX_DEPTH + 1)]
        self.depth_weight = np.bincount(self.dist, minlength=MAX_DEPTH + 1) / rb.N_STATES
        self.net = MLP([N_IN, hidden, hidden, 9], self.rng, lr=lr)
        self.target = MLP([N_IN, hidden, hidden, 9], self.rng, lr=lr)
        self.target.load(self.net.params())
        self.n, self.gamma, self.eps, self.promote, self.cap = n_envs, gamma, eps, promote, episode_cap
        self.batch, self.target_every, self.max_k = batch, target_every, max_k
        self.updates_per_step = updates_per_step
        self.all_actions, self.dyn_cap = all_actions, dyn_cap
        self.k_margin, self.weight_by_depth = k_margin, weight_by_depth
        self.Q0 = -float(MAX_DEPTH + 1)
        # replay buffer (ring)
        self.buf_n = buffer
        self.buf_s = np.zeros(buffer, dtype=np.int64)
        self.buf_a = np.zeros(buffer, dtype=np.int64)
        self.buf_s2 = np.zeros(buffer, dtype=np.int64)
        self.buf_done = np.zeros(buffer, dtype=bool)
        self.buf_pos, self.buf_len = 0, 0
        self.seen = np.zeros(rb.N_STATES, dtype=bool)
        self.K = min(k_start, max_k)      # k_start = max_k disables the curriculum: all depths from the start
        self.steps = self.updates = self.episodes = self.solved = 0
        self.losses = []
        self.t0 = time.time()
        self.metrics = []
        self.k = np.zeros(n_envs, dtype=np.int64)
        self.s = self._new_starts(np.arange(n_envs))
        self.age = np.zeros(n_envs, dtype=np.int64)
        self.recorder = rb.EpisodeRecorder()
        self.policy_cache = None

    @property
    def trace(self):
        return self.recorder.last

    # ---- interface shared with the tabular trainer ----
    def qvalues(self, idx):
        return self.net.forward(onehot(idx))

    def set_lr(self, lr):
        self.net.lr = float(lr)

    def _depth_hi(self, margin=0):
        # exclusive upper bound of the scramble depth; `margin` lets the training
        # distribution run ahead of the curriculum so that the values one or two
        # levels deeper than K are learned before the greedy policy is tested at K
        if self.K < self.max_k:
            return min(self.K + margin, self.max_k) + 1
        return 3 * MAX_DEPTH if self.max_k == MAX_DEPTH else self.max_k + 1

    def _new_starts(self, which):
        depths = self.rng.integers(1, self._depth_hi(), size=which.shape[0])
        self.k[which] = depths
        return self._scramble(self.T, self.rng, depths)

    def step(self):
        T, s, rng = self.T, self.s, self.rng
        q = self.qvalues(s)
        greedy = np.argmax(q + rng.random(q.shape, dtype=np.float32) * 1e-3, axis=1)
        explore = rng.random(self.n) < self.eps
        a = np.where(explore, rng.integers(9, size=self.n), greedy)
        s2 = T[s, a]
        done = s2 == rb.SOLVED_INDEX
        self.seen[s] = True
        # store transitions
        n = self.n
        pos = (self.buf_pos + np.arange(n)) % self.buf_n
        self.buf_s[pos], self.buf_a[pos], self.buf_s2[pos], self.buf_done[pos] = s, a, s2, done
        self.buf_pos = (self.buf_pos + n) % self.buf_n
        self.buf_len = min(self.buf_len + n, self.buf_n)
        # gradient steps on replay minibatches
        if self.buf_len >= self.batch:
            for _ in range(self.updates_per_step):
                self._update()
        self.age += 1
        cap = min(self.cap, self.K + 3) if self.dyn_cap else self.cap
        reset = done | (self.age >= cap)
        self.recorder.push(self.steps, s[0], a[0], s2[0], done[0], reset[0], self.k[0], self.K, explore[0])
        n_reset = int(reset.sum())
        if n_reset:
            s2 = s2.copy()
            s2[reset] = self._new_starts(np.nonzero(reset)[0])
            self.age[reset] = 0
            self.episodes += n_reset
            self.solved += int(done.sum())
        self.s = s2
        self.steps += n

    def _update(self):
        if self.all_actions:
            return self._update_all_actions()
        i = self.rng.integers(self.buf_len, size=self.batch)
        s, a, s2, done = self.buf_s[i], self.buf_a[i], self.buf_s2[i], self.buf_done[i]
        # double DQN: the online net picks the next action, the target net values it
        X2 = onehot(s2)
        a2 = np.argmax(self.net.forward(X2), axis=1)
        q2 = self.target.forward(X2)[np.arange(self.batch), a2]
        target = -1.0 + self.gamma * np.where(done, 0.0, q2)
        target = np.clip(target, self.Q0, 0.0).astype(np.float32)   # every true value lies in [-12, -1]
        q, acts = self.net.forward(onehot(s), cache=True)
        rows = np.arange(self.batch)
        err = q[rows, a] - target
        self.losses.append(float(np.mean(err * err)))
        dout = np.zeros_like(q)
        dout[rows, a] = 2.0 * err / self.batch
        self.net.adam(self.net.backward(acts, dout))
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.load(self.net.params())

    def _update_all_actions(self):
        """DeepCube-style target: states sampled from the curriculum distribution,
        targets for all nine actions from the transition table (uses the model)."""
        B = self.batch
        depths = self.rng.integers(1, self._depth_hi(self.k_margin), size=B)
        s = self._scramble(self.T, self.rng, depths)
        self.seen[s] = True
        children = self.T[s]                                   # (B, 9)
        Xc = onehot(children.ravel())
        a2 = np.argmax(self.net.forward(Xc), axis=1)
        q2 = self.target.forward(Xc)[np.arange(B * 9), a2].reshape(B, 9)
        done = children == rb.SOLVED_INDEX
        target = np.clip(-1.0 + self.gamma * np.where(done, 0.0, q2), self.Q0, 0.0).astype(np.float32)
        q, acts = self.net.forward(onehot(s), cache=True)
        err = q - target
        w = (1.0 / depths).astype(np.float32) if self.weight_by_depth else np.ones(B, dtype=np.float32)
        w = (w / w.mean())[:, None]
        self.losses.append(float(np.mean(w * err * err)))
        self.net.adam(self.net.backward(acts, 2.0 * w * err / (B * 9)))
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.load(self.net.params())

    def evaluate(self, n_per_depth=400, sample_states=20000):
        from play_rubik import greedy_rollout
        succ, mean_len = [], []
        for d in range(1, MAX_DEPTH + 1):
            pool = self.by_depth[d]
            pick = pool[self.rng.integers(pool.shape[0], size=min(n_per_depth, pool.shape[0]))]
            length = greedy_rollout(self.qvalues, self.T, pick)
            ok = length > 0
            succ.append(float(ok.mean()))
            mean_len.append(float(length[ok].mean()) if ok.any() else None)
        succ_random = float(sum(self.depth_weight[d] * succ[d - 1] for d in range(1, MAX_DEPTH + 1)) + self.depth_weight[0])
        sample = self.rng.integers(rb.N_STATES, size=sample_states)
        v = self.qvalues(sample).max(axis=1)
        q_err = float(np.abs(v + self.dist[sample]).mean())
        coverage = float(self.seen.mean())
        unseen = sample[~self.seen[sample]][:2000]
        unseen_success = float((greedy_rollout(self.qvalues, self.T, unseen) > 0).mean()) if unseen.size else None
        seen_pick = sample[self.seen[sample]][:2000]
        seen_success = float((greedy_rollout(self.qvalues, self.T, seen_pick) > 0).mean()) if seen_pick.size else None
        # share of the states at each exact distance that training has sampled (the "memorised" front)
        seen_by_depth = [float(self.seen[self.by_depth[d]].mean()) for d in range(1, MAX_DEPTH + 1)]
        loss = float(np.mean(self.losses[-200:])) if self.losses else None
        rec = {
            'step': self.steps, 'time': round(time.time() - self.t0, 1), 'K': self.K,
            'episodes': self.episodes, 'solved': self.solved, 'updates': self.updates,
            'coverage': coverage, 'q_error': q_err, 'success_random': succ_random,
            'unseen_success': unseen_success, 'seen_success': seen_success, 'seen_by_depth': seen_by_depth, 'loss': loss,
            'success_by_depth': succ, 'mean_len_by_depth': mean_len,
            'optimal_len_by_depth': list(range(1, MAX_DEPTH + 1)),
            'sps': round(self.steps / max(time.time() - self.t0, 1e-9)),
        }
        self.metrics.append(rec)
        if self.K < self.max_k and succ[self.K - 1] >= self.promote:
            self.K += 1
        return rec

    def load(self, net_path):
        saved = MLP.from_file(net_path)
        if [w.shape for w in saved.W] != [w.shape for w in self.net.W]:
            raise SystemExit('saved network has a different size than --hidden')
        self.net.load(saved.W + saved.b)
        self.target.load(self.net.params())

    def full_policy(self, chunk=100000):
        """Greedy action of the network for every state (used for policy.bin)."""
        out = np.empty(rb.N_STATES, dtype=np.uint8)
        for lo in range(0, rb.N_STATES, chunk):
            idx = np.arange(lo, min(lo + chunk, rb.N_STATES))
            out[lo:lo + idx.shape[0]] = np.argmax(self.qvalues(idx), axis=1)
        return out

    def save(self, net_path, policy_path):
        self.net.save(net_path)
        pol = self.full_policy()
        np.save(policy_path, pol)
        self.policy_cache = pol
        return pol
