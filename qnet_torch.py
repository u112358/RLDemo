"""PyTorch backend for the network Q-learner (GPU: Apple MPS, CUDA, or CPU).

Same interface as ``qnet.NetTrainer`` (step / evaluate / save / qvalues /
metrics / trace), same environment, curriculum and targets.  Differences
that make it worth a GPU:

* the one-hot features of *all* 3,674,160 states (529 MB as uint8) and the
  transition table live on the device, so sampling, children lookup and the
  forward passes never touch the CPU;
* a bigger MLP (default 144-1024-1024-512-9) and batches of thousands of
  states, each contributing targets for all nine actions (DeepCube-style,
  ``--all-actions``); ``--replay`` switches to model-free replay Q-learning
  like the numpy version;
* weights are exported in the numpy ``MLP`` format, so ``serve``, ``eval``
  and ``solve`` with ``--agent net`` work on the result without torch.

    python play_rubik.py train --agent net --backend torch --dashboard 8000 --minutes 20
"""
import time

import numpy as np
import torch
import torch.nn as nn

import rubik as rb
from qnet import MAX_DEPTH, N_IN


def pick_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def all_features(device, chunk=200000):
    """(N_STATES, 144) uint8 one-hot of every state, on the device."""
    feats = torch.empty((rb.N_STATES, N_IN), dtype=torch.uint8, device=device)
    for lo in range(0, rb.N_STATES, chunk):
        hi = min(lo + chunk, rb.N_STATES)
        states = rb.decode(np.arange(lo, hi))
        X = np.zeros((hi - lo, N_IN), dtype=np.uint8)
        X[np.arange(hi - lo)[:, None], np.arange(24)[None, :] * 6 + states] = 1
        feats[lo:hi] = torch.from_numpy(X).to(device)
    return feats


def make_mlp(layers):
    sizes = [N_IN] + list(layers) + [9]
    mods = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        mods.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class TorchNetTrainer:
    def __init__(self, layers=(1024, 1024, 512), lr=1e-3, batch=8192, device='auto', all_actions=True,
                 k_margin=2, weight_by_depth=False, promote=0.97, max_k=MAX_DEPTH, k_start=1,
                 target_every=200, n_envs=64, eps=0.1, episode_cap=25, buffer=500000, updates_per_step=1,
                 seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = pick_device(device)
        self.T = np.ascontiguousarray(rb.transitions())
        self.dist = rb.bfs_distances(self.T)
        self.by_depth = [np.nonzero(self.dist == d)[0] for d in range(MAX_DEPTH + 1)]
        self.depth_weight = np.bincount(self.dist, minlength=MAX_DEPTH + 1) / rb.N_STATES
        self.T_t = torch.from_numpy(self.T.astype(np.int64)).to(self.device)
        self.feats = all_features(self.device)
        self.net = make_mlp(layers).to(self.device)
        self.target = make_mlp(layers).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.layers = list(layers)
        self.batch, self.all_actions, self.k_margin, self.weight_by_depth = batch, all_actions, k_margin, weight_by_depth
        self.promote, self.max_k, self.target_every = promote, max_k, target_every
        self.n, self.eps, self.cap, self.updates_per_step = n_envs, eps, episode_cap, updates_per_step
        self.Q0 = -float(MAX_DEPTH + 1)
        self.K = min(k_start, max_k)
        self.seen = np.zeros(rb.N_STATES, dtype=bool)
        self.steps = self.updates = self.episodes = self.solved = 0
        self.losses = []
        self.metrics = []
        self.trace = None
        self.policy_cache = None
        self.t0 = time.time()
        # replay buffer on the device (model-free mode) and a few environments
        # (always: they feed the dashboard's live cube and, in replay mode, the buffer)
        self.buf_n = buffer
        self.buf_s = torch.zeros(buffer, dtype=torch.int64, device=self.device)
        self.buf_a = torch.zeros(buffer, dtype=torch.int64, device=self.device)
        self.buf_s2 = torch.zeros(buffer, dtype=torch.int64, device=self.device)
        self.buf_done = torch.zeros(buffer, dtype=torch.bool, device=self.device)
        self.buf_pos, self.buf_len = 0, 0
        self.k = np.zeros(n_envs, dtype=np.int64)
        self.age = np.zeros(n_envs, dtype=np.int64)
        self.s = self._new_starts(np.arange(n_envs))

    # ---- helpers ----
    def _depth_hi(self, margin=0):
        if self.K < self.max_k:
            return min(self.K + margin, self.max_k) + 1
        return 3 * MAX_DEPTH if self.max_k == MAX_DEPTH else self.max_k + 1

    def _scramble_t(self, depths):
        """Solved cube scrambled by ``depths`` (device tensor) random moves."""
        n = depths.shape[0]
        s = torch.full((n,), rb.SOLVED_INDEX, dtype=torch.int64, device=self.device)
        for j in range(int(depths.max().item())):
            a = torch.randint(9, (n,), device=self.device)
            m = depths > j
            s = torch.where(m, self.T_t[s, a], s)
        return s

    def _new_starts(self, which):
        depths = self.rng.integers(1, self._depth_hi(), size=which.shape[0])
        self.k[which] = depths
        d = torch.from_numpy(depths).to(self.device)
        return self._scramble_t(d).cpu().numpy()

    def _q(self, idx_t, model=None):
        return (model or self.net)(self.feats[idx_t].float())

    def qvalues(self, idx):
        with torch.no_grad():
            out = []
            idx = np.asarray(idx, dtype=np.int64)
            for lo in range(0, idx.shape[0], 65536):
                t = torch.from_numpy(idx[lo:lo + 65536]).to(self.device)
                out.append(self._q(t).cpu().numpy())
        return np.concatenate(out) if len(out) > 1 else out[0]

    # ---- training ----
    def step(self):
        # a few environments: live trace for the dashboard (and data in replay mode)
        s_t = torch.from_numpy(self.s).to(self.device)
        with torch.no_grad():
            q = self._q(s_t)
        greedy = (q + torch.rand_like(q) * 1e-3).argmax(1)
        explore = torch.rand(self.n, device=self.device) < self.eps
        a = torch.where(explore, torch.randint(9, (self.n,), device=self.device), greedy)
        s2 = self.T_t[s_t, a]
        done = s2 == rb.SOLVED_INDEX
        if not self.all_actions:
            pos = (self.buf_pos + torch.arange(self.n, device=self.device)) % self.buf_n
            self.buf_s[pos], self.buf_a[pos], self.buf_s2[pos], self.buf_done[pos] = s_t, a, s2, done
            self.buf_pos = (self.buf_pos + self.n) % self.buf_n
            self.buf_len = min(self.buf_len + self.n, self.buf_n)
        a0, s20, ex0, d0 = int(a[0]), int(s2[0]), bool(explore[0]), bool(done[0])
        self.trace = {'step': self.steps, 'state': int(self.s[0]), 'action': a0, 'next': s20, 'explore': ex0,
                      'unknown': False, 'done': d0, 'k': int(self.k[0]), 'age': int(self.age[0]), 'K': self.K}
        self.seen[self.s] = True
        for _ in range(self.updates_per_step):
            if self.all_actions:
                self._update_all_actions()
            elif self.buf_len >= self.batch:
                self._update_replay()
        done_np, s2_np = done.cpu().numpy(), s2.cpu().numpy()
        self.age += 1
        reset = done_np | (self.age >= self.cap)
        n_reset = int(reset.sum())
        if n_reset:
            s2_np[reset] = self._new_starts(np.nonzero(reset)[0])
            self.age[reset] = 0
            self.episodes += n_reset
            self.solved += int(done_np.sum())
        self.s = s2_np
        self.steps += self.n

    def _targets(self, children, B):
        """Double-DQN targets for (B, k) children indices."""
        flat = children.reshape(-1)
        with torch.no_grad():
            X = self.feats[flat].float()
            a2 = self.net(X).argmax(1)
            q2 = self.target(X).gather(1, a2[:, None]).squeeze(1)
        done = flat == rb.SOLVED_INDEX
        t = torch.where(done, torch.zeros_like(q2), q2) - 1.0
        return t.clamp(self.Q0, 0.0).reshape(children.shape)

    def _finish(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.losses.append(float(loss.item()))
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.load_state_dict(self.net.state_dict())

    def _update_all_actions(self):
        B = self.batch
        depths = torch.randint(1, self._depth_hi(self.k_margin), (B,), device=self.device)
        s = self._scramble_t(depths)
        if self.updates % 20 == 0:
            self.seen[s.cpu().numpy()] = True
        children = self.T_t[s]                              # (B, 9)
        target = self._targets(children, B)
        q = self._q(s)
        w = (1.0 / depths.float()) if self.weight_by_depth else torch.ones(B, device=self.device)
        w = (w / w.mean())[:, None]
        self._finish((w * (q - target) ** 2).mean())

    def _update_replay(self):
        i = torch.randint(self.buf_len, (self.batch,), device=self.device)
        s, a, s2, done = self.buf_s[i], self.buf_a[i], self.buf_s2[i], self.buf_done[i]
        target = self._targets(s2[:, None], self.batch)[:, 0]
        q = self._q(s).gather(1, a[:, None]).squeeze(1)
        self._finish(((q - target) ** 2).mean())

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
        loss = float(np.mean(self.losses[-200:])) if self.losses else None
        rec = {
            'step': self.steps, 'time': round(time.time() - self.t0, 1), 'K': self.K,
            'episodes': self.episodes, 'solved': self.solved, 'updates': self.updates,
            'coverage': coverage, 'q_error': q_err, 'success_random': succ_random,
            'unseen_success': unseen_success, 'loss': loss,
            'success_by_depth': succ, 'mean_len_by_depth': mean_len,
            'optimal_len_by_depth': list(range(1, MAX_DEPTH + 1)),
            'sps': round(self.steps / max(time.time() - self.t0, 1e-9)),
            'device': str(self.device),
        }
        self.metrics.append(rec)
        if self.K < self.max_k and succ[self.K - 1] >= self.promote:
            self.K += 1
        return rec

    def full_policy(self, chunk=262144):
        out = np.empty(rb.N_STATES, dtype=np.uint8)
        with torch.no_grad():
            for lo in range(0, rb.N_STATES, chunk):
                hi = min(lo + chunk, rb.N_STATES)
                idx = torch.arange(lo, hi, device=self.device)
                out[lo:hi] = self._q(idx).argmax(1).cpu().numpy()
        return out

    def save(self, net_path, policy_path):
        # numpy MLP format: W_i (in, out), b_i, sizes  -> serve / eval / solve without torch
        lin = [m for m in self.net if isinstance(m, nn.Linear)]
        W = [m.weight.detach().cpu().numpy().T.astype(np.float32).copy() for m in lin]
        b = [m.bias.detach().cpu().numpy().astype(np.float32).copy() for m in lin]
        np.savez(net_path, *(W + b), sizes=np.array([N_IN] + self.layers + [9]))
        pol = self.full_policy()
        np.save(policy_path, pol)
        self.policy_cache = pol
        return pol
