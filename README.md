# RLDemo — Q-learning on a 2×2×2 pocket cube

Tabular Q-learning that learns to solve the pocket cube from **any** scramble,
plus two browser pages: an interactive 3D playground for the environment and a
live training dashboard.

```
pip install numpy
python play_rubik.py train --dashboard 8000   # ~2 min, open http://localhost:8000/
python play_rubik.py eval                     # success rate / solution length per distance
python play_rubik.py solve wggbrymrgrwrbmbwybywmygm
```

## Files

| file | what it is |
|---|---|
| `rubik.py` | The environment: 6 faces × 4 stickers, 9 actions (`t1 t2 t3 r1 r2 r3 f1 f2 f3` = top / right / front layer turned 90° / 180° / 270°). Also a perfect hash of the 3,674,160 legal states (`encode` / `decode`), a cached transition table and a BFS distance table used as ground truth. `python rubik.py` runs a self-check. |
| `play_rubik.py` | The learner (`train`, `eval`, `solve`). Writes `cache/q_table.npy`, `cache/policy.npy`, `cache/metrics.json`. |
| `playground.html` | 3D visualiser of the environment (three.js from cdnjs): same sticker indices, action ids and reward as `rubik.py`; animated moves, sequence playback, scramble, undo and an optimal solver. Open the file in a browser. |
| `dashboard.html` | Training monitor. Served by `train --dashboard PORT`; polls `/metrics` every 2 s. |

## How the learner works

The original demo kept a pandas Q-table keyed by the state string, always
started from one fixed scramble, and called an episode "solved" once two faces
were uniform. It could only ever learn a path from that one start.

The current version:

- **Every state has a row.** A legal 2×2×2 state is a permutation of the 7
  movable corner cubies times the orientation of 6 of them (7! · 3⁶ =
  3,674,160), so the Q-table is a dense `(3674160, 9)` float32 array indexed by
  that hash. Nothing is "unseen" any more.
- **Reverse-scramble curriculum.** Episodes start from the solved cube scrambled
  by `k` random moves, `k ∈ 1..K`. `K` starts at 1 and is raised as soon as the
  greedy policy solves ≥97 % of states at exact distance `K`. Every training
  state is therefore next to states whose values are already right, so the
  reward always propagates.
- **Reward −1 per move, terminal at the solved cube, γ = 1.** The optimal
  Q-value is then exactly minus the distance to solved, which is checked
  against the BFS table during training (the "value error" curve).
- **Pessimistic, clamped table.** Values start at −12, below every true value,
  so an untried action never looks better than a learned one; estimates are
  clamped at −12 so an action tried before its successor was learned is
  treated as unknown again rather than sinking for good. Transitions are
  deterministic, so α = 1 is a plain Bellman backup.
- **Batched.** 8,192 environments step in lock-step as numpy arrays; the update
  is the ordinary one-step Q-learning rule applied to a batch. About 2.5 M
  environment steps per second on one CPU core.

## Results

One run of `python play_rubik.py train` on a single CPU core (seed 0):

| | |
|---|---|
| wall time | 88 s |
| environment steps | 184 M (2.1 M steps/s) |
| states with a learned value | 99.9 % of 3,674,160 |
| greedy policy solves a uniformly random state | 100.0 % (2,000 samples per distance) |
| mean \|Q + distance\| | 0.41 moves |

| distance | states | solved | mean length |
|---:|---:|---:|---:|
| 1 | 9 | 100 % | 1.00 |
| 2 | 54 | 100 % | 2.00 |
| 3 | 321 | 100 % | 3.00 |
| 4 | 1,847 | 100 % | 4.00 |
| 5 | 9,992 | 100 % | 5.00 |
| 6 | 50,136 | 100 % | 6.00 |
| 7 | 227,536 | 100 % | 7.09 |
| 8 | 870,072 | 100 % | 8.27 |
| 9 | 1,887,748 | 100 % | 9.41 |
| 10 | 623,800 | 100 % | 10.22 |
| 11 | 2,644 | 100 % | 11.05 |

Curriculum timeline: K reached 6 after 1.4 s, 8 after 25 s, 11 after 75 s.
Solutions are near-optimal rather than optimal because a value computed while
its successor was still under-estimated is only refreshed when that action is
taken again (ε-greedy); training longer (`--target 1.0 --steps N`) keeps
shrinking the value error.

Two things that did **not** work on the way, kept here because they are the
interesting part of tabular RL on this problem:

- *Optimistic zeros* (the usual default). Every untried action looks best, so
  the greedy policy is pulled towards unexplored states everywhere and the
  curriculum stops meaning anything: after 30 M steps the table covered 6 % of
  the states and the greedy policy solved no distance-4 state at all.
- *Pessimistic values with plain ε-greedy.* Once one action of a state leads to
  any learned neighbour, greedy locks onto it and the remaining (possibly
  better) actions are only retried with probability ε/9 per visit; after
  400 M steps the curriculum was stuck at K = 7-8 and 27 % of random states
  were solved. Trying unknown actions first fixes this without changing the
  values.

The `reset()` scramble of the original demo is a legal state 9 moves from
solved (one optimal solution: `t3 f3 t2 f3 r1 f2 r1 f3 t2`). Reachable from
it are all 3,674,160 states, of which only 634 have two or more uniform faces,
which is why the original reward was so sparse.
