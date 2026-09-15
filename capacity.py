"""Supervised capacity ceiling: how many states can this architecture *memorise* when handed the answers?

The same MLP the RL agent uses is trained directly on the BFS-optimal actions of n random states
(any optimal action counts as correct).  This removes Q-learning from the picture and measures the
architecture alone:

  fit      accuracy on the n training states       -> the memorisation ceiling n_max(params)
  held-out accuracy on states it never trained on  -> does supervised fitting generalise at all?
  solve    greedy solve rate on random states       -> what that accuracy is worth as a policy

Compare the RL agent's "sampled states solved" count with n_max: if RL solves far fewer states than
the same network can memorise supervised, the RL bottleneck is optimisation, not capacity.

Every n gets the same optimisation budget (--steps Adam steps, cosine learning rate) and stops early once
the fit is essentially perfect, so the table compares architectures at equal compute.

Usage:  python capacity.py --layers 1024,1024,512 --n 100000,300000,1000000,3674160 --steps 6000
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import rubik as rb
from play_rubik import greedy_rollout
from qnet_torch import pick_device, all_features, make_mlp

MAX_DEPTH = 11


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layers', default='1024,1024,512')
    ap.add_argument('--n', default='100000,300000,1000000,3674160', help='training-set sizes (states)')
    ap.add_argument('--steps', type=int, default=6000, help='Adam steps per training-set size')
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = pick_device(args.device)
    layers = tuple(int(x) for x in args.layers.split(','))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    T = np.ascontiguousarray(rb.transitions())
    dist = rb.bfs_distances(T)
    # optimal-action mask: a move is optimal when it brings the state one step closer to solved
    opt = dist[T] == (dist[:, None] - 1)
    opt[rb.SOLVED_INDEX] = True
    opt_t = torch.from_numpy(opt).to(device)
    feats = all_features(device)
    T_t = torch.from_numpy(T.copy()).to(device)
    depth_weight = np.bincount(dist, minlength=MAX_DEPTH + 1) / rb.N_STATES
    by_depth = [np.nonzero(dist == d)[0] for d in range(MAX_DEPTH + 1)]
    chance = float(opt.sum(1).mean() / 9)          # picking a move at random is right this often (several moves can be optimal)
    n_params = sum(p.numel() for p in make_mlp(layers).parameters())
    print('device %s   MLP 144-%s-9   %s parameters   %.1f Mbit at 2 bit/param' % (device, '-'.join(map(str, layers)), format(n_params, ','), 2 * n_params / 1e6))
    print('a full policy needs about %.1f Mbit (3,674,160 states × log2 9 bits)' % (rb.N_STATES * np.log2(9) / 1e6))
    print('chance level for fit / held-out accuracy: %.1f%% (a random move is optimal this often)\n' % (100 * chance))
    print('%9s %7s %8s %9s %9s %10s %8s' % ('n states', 'steps', 'fit', 'held-out', 'solve', 'Mbit fit', 'time'))

    def accuracy(model, idx):
        ok = 0
        with torch.no_grad():
            for lo in range(0, idx.shape[0], 65536):
                b = torch.from_numpy(idx[lo:lo + 65536]).to(device)
                a = model(feats[b].float()).argmax(1)
                ok += int(opt_t[b, a].sum())
        return ok / idx.shape[0]

    def solve_rate(model):
        def qf(idx):
            with torch.no_grad():
                return model(feats[torch.from_numpy(idx).to(device)].float()).cpu().numpy()
        s = 0.0
        for d in range(1, MAX_DEPTH + 1):
            pool = by_depth[d]
            pick = pool[rng.integers(pool.shape[0], size=min(400, pool.shape[0]))]
            s += depth_weight[d] * float((greedy_rollout(qf, T, pick) > 0).mean())
        return s + depth_weight[0]

    for n in [int(x) for x in args.n.split(',')]:
        n = min(n, rb.N_STATES)
        train = rng.choice(rb.N_STATES, size=n, replace=False) if n < rb.N_STATES else np.arange(rb.N_STATES)
        mask = np.zeros(rb.N_STATES, bool); mask[train] = True
        held = np.nonzero(~mask)[0]
        held = held[rng.integers(held.shape[0], size=min(100000, held.shape[0]))] if held.size else held
        model = make_mlp(layers).to(device)
        opt_ = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=args.steps, eta_min=args.lr / 20)
        train_t = torch.from_numpy(train).to(device)
        fit_probe = train if n <= 200000 else train[rng.integers(n, size=200000)]
        t0 = time.time()
        fit, step, batch = 0.0, 0, min(args.batch, n)
        while step < args.steps:
            perm = train_t[torch.randperm(n, device=device)]        # one pass over the training states
            for lo in range(0, n - batch + 1, batch):
                b = perm[lo:lo + batch]
                logits = model(feats[b].float())
                # -log Σ_{a optimal} p(a): any optimal action is a correct answer
                loss = (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~opt_t[b], -1e9), 1)).mean()
                opt_.zero_grad(set_to_none=True)
                loss.backward()
                opt_.step()
                sched.step()
                step += 1
                if step % args.eval_every == 0 or step == args.steps:
                    fit = accuracy(model, fit_probe)
                    print('\r  n=%s  step %d/%d  loss %.4f  fit %.2f%%  %.0fs   ' % (format(n, ','), step, args.steps, loss.item(), 100 * fit, time.time() - t0), end='', flush=True)
                if step >= args.steps or fit >= 0.999:
                    break
            if fit >= 0.999:
                break
        fit = accuracy(model, fit_probe)
        ho = accuracy(model, held) if held.size else float('nan')
        sr = solve_rate(model)
        print('\r%9s %7d %7.2f%% %8.2f%% %8.2f%% %9.2f %7.0fs' % (format(n, ','), step, 100 * fit, 100 * ho, 100 * sr, fit * n * np.log2(9) / 1e6, time.time() - t0))
    print('\nreading: fit near 100%% = the architecture can memorise that many states; the largest such n is its ceiling.'
          '\n         held-out well above the %.0f%% chance level = supervised training generalises even where RL did not.' % (100 * chance))


if __name__ == '__main__':
    main()
