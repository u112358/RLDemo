"""Symmetry test: does the network solve the *symmetric images* of states it trained on?

The pocket cube with the DBL cubie fixed has 6 whole-cube symmetries that keep the move set
{U, R, F}: the 3-fold rotation about the URF-DBL diagonal (U -> R -> F -> U) and a mirror
(swaps R and F, reverses every turn).  A symmetry maps a state to another state at the same
distance from solved, so the image of a sampled state is a new state the network never saw,
but one whose solution is "the same solution, renamed".

  image solved as often as the original          -> the network learned the structure (invariance)
  image solved as rarely as a random unseen state -> it memorised the sampled states

Usage:  python symmetry_test.py --tag main [--n 2000] [--seed 0]
"""
import argparse
import os
import sys

import numpy as np

import rubik as rb
import play_rubik as pr
from qnet import MLP, qfunction

MAX_DEPTH = 11


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='', help='experiment folder cache/<tag>/')
    ap.add_argument('--n', type=int, default=2000, help='sampled states per distance')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.tag:
        pr.set_tag(args.tag)
    if not os.path.exists(pr.NET_PATH) or not os.path.exists(pr.NET_SEEN_PATH):
        sys.exit('need %s and %s (train the network agent with the current code first)' % (pr.NET_PATH, pr.NET_SEEN_PATH))
    rng = np.random.default_rng(args.seed)
    T = np.ascontiguousarray(rb.transitions())
    dist = rb.bfs_distances(T)
    seen = np.unpackbits(np.load(pr.NET_SEEN_PATH))[:rb.N_STATES].astype(bool)
    net = MLP.from_file(pr.NET_PATH)
    qf = qfunction(net)

    sig = rb.symmetries(T)                       # (6, N): identity, ρ, ρ², μ, μρ, μρ²  (cached in cache/symmetries.npy)
    names = ['ρ', 'ρ²', 'μ', 'μρ', 'μρ²']
    sigmas = {}
    for i, name in enumerate(names):
        sg = np.asarray(sig[i + 1], dtype=np.int64)
        assert np.array_equal(dist[sg], dist)
        sigmas[name] = sg
    print('symmetries verified: %s   (each maps every state to one at the same distance)' % ', '.join(sigmas))
    print('states sampled during training: %.2f%%\n' % (100 * seen.mean()))

    print('distance | sampled states      | their symmetric images, unseen | random unseen states   | ratio')
    print('    d    |    n    solved      |    n    solved                 |    n    solved         | image/random')
    tot = {'seen': [0, 0], 'img': [0, 0], 'rnd': [0, 0]}
    for d in range(1, MAX_DEPTH + 1):
        pool = np.nonzero(dist == d)[0]
        s_pool = pool[seen[pool]]
        u_pool = pool[~seen[pool]]
        if s_pool.size == 0 or u_pool.size == 0:
            print('  %2d     | %7d  %s' % (d, s_pool.size, 'no sampled states' if s_pool.size == 0 else 'every state sampled'))
            continue
        s_pick = s_pool[rng.integers(s_pool.size, size=min(args.n, s_pool.size))]
        images = np.unique(np.concatenate([sg[s_pick] for sg in sigmas.values()]))
        images = images[~seen[images]]
        if images.size > args.n:
            images = images[rng.integers(images.size, size=args.n)]
        r_pick = u_pool[rng.integers(u_pool.size, size=min(args.n, u_pool.size))]
        res = {}
        for key, states in (('seen', s_pick), ('img', images), ('rnd', r_pick)):
            ok = (pr.greedy_rollout(qf, T, states) > 0) if states.size else np.zeros(0, bool)
            res[key] = (states.size, float(ok.mean()) if states.size else float('nan'))
            tot[key][0] += states.size
            tot[key][1] += int(ok.sum())
        ratio = res['img'][1] / res['rnd'][1] if res['rnd'][1] > 0 else float('inf')
        print('  %2d     | %7d  %6.1f%%     | %7d  %6.1f%%                | %7d  %6.1f%%        | %s'
              % (d, res['seen'][0], 100 * res['seen'][1], res['img'][0], 100 * res['img'][1], res['rnd'][0], 100 * res['rnd'][1],
                 '%.1f×' % ratio if np.isfinite(ratio) else 'n/a'))
    print()
    for key, label in (('seen', 'sampled states'), ('img', 'symmetric images (unseen)'), ('rnd', 'random unseen states')):
        n, k = tot[key]
        print('  %-27s %7d states, solved %6.2f%%' % (label, n, 100 * k / max(n, 1)))
    print('\nreading (compare rows at the same distance):'
          '\n  images >> random unseen                 the network learned the cube\'s symmetry structure'
          '\n  sampled >> images ≈ random unseen       it memorised individual states; nothing transfers, even to a renamed copy'
          '\n  sampled ≈ images ≈ random unseen        neither: it interpolates in sticker space (similar stickers, similar value)'
          '\n                                          and knows nothing about the symmetry; check beam search for how far that carries')


if __name__ == '__main__':
    main()
