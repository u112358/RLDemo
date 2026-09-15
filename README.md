# RLDemo — Q-learning on a 2×2×2 pocket cube

Tabular Q-learning that learns to solve the pocket cube from **any** scramble,
plus two browser pages: an interactive 3D playground for the environment and a
live training dashboard.

```
pip install numpy
python play_rubik.py train --dashboard 8000   # ~2 min, open http://localhost:8000/ while it runs
python play_rubik.py serve                    # afterwards: dashboard + playground + trained policy, no training
python play_rubik.py eval                     # success rate / solution length per distance
python play_rubik.py solve wggbrymrgrwrbmbwybywmygm
```

The terminal shows one coloured progress line (elapsed / remaining time,
curriculum depth, success, coverage, value error, throughput); pass `--plain`
or redirect to a file for one line per evaluation instead.

With the server running, http://localhost:8000/ is the training monitor and
http://localhost:8000/playground.html the playground with the trained policy
loaded (the "策略解法" button and the real-cube section need it).

## Files

| file | what it is |
|---|---|
| `rubik.py` | The environment: 6 faces × 4 stickers, 9 actions (`t1 t2 t3 r1 r2 r3 f1 f2 f3` = top / right / front layer turned 90° / 180° / 270°). Also a perfect hash of the 3,674,160 legal states (`encode` / `decode`), a cached transition table and a BFS distance table used as ground truth. `python rubik.py` runs a self-check. |
| `play_rubik.py` | The learner (`train`, `eval`, `solve`). Writes `cache/q_table.npy`, `cache/policy.npy`, `cache/metrics.json`. |
| `playground.html` | 3D visualiser of the environment (three.js from cdnjs): same sticker indices, action ids and reward as `rubik.py`; animated moves, sequence playback, scramble, undo, an optimal solver, the trained policy, and a **real-cube mode**: paint the 24 stickers of a physical cube, the page relabels the colours, looks the state up in the policy and walks you through the moves in your cube's colours. |
| `dashboard.html` | Training monitor: curves for success rate, curriculum depth, coverage and value error, a per-distance breakdown, and a 3D cube that either replays environment #0's most recent complete training episode (start scramble, every move, whether it was solved) or, as a "spectator", scrambles a cube and solves it with the current Q-table so you can watch the policy improve; a speed slider goes from 1 move/s to instant. |
| `qnet.py`, `qnet_torch.py` | The network agents: numpy MLP (CPU) and PyTorch (Apple MPS / CUDA), same interface as the table trainer. |
| `cube-model.js`, `cube-3d.js` | Shared browser code: the cube model (moves, `encode`, legality, policy lookup, BFS solver, real-cube relabelling) and the three.js view. |

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

## The same learner with a neural network

```
python play_rubik.py train --agent net --dashboard 8000     # ~10 min on one CPU
python play_rubik.py train --agent net --max-k 7            # train shallow, test deep
python play_rubik.py serve --agent net                      # pages + the network's policy
python play_rubik.py eval --agent net
```

`qnet.py` keeps everything above (environment, curriculum, reward, ε-greedy
batched Q-learning) and only replaces the table: Q(s, ·) is a
144 → 256 → 256 → 9 MLP over the one-hot stickers, trained on a replay buffer
with a target network (a small DQN, written in numpy so nothing beyond numpy
is needed). The point of the comparison:

- The table can only *cover* states: a state it never visited has no
  information. The network has to *generalise*: every state, seen or not,
  gets a value from the same weights.
- So the dashboard gains a metric the table cannot have: **unseen success**,
  the greedy solve rate on states that were never visited during training.
  For the table this is 0 by construction (`--max-k 7` makes the contrast
  stark: the table never sees distance 8-11 at all).
- The price is exactness. The table converges to Q = −distance in the states
  it has covered; the network's values are approximate everywhere, so its
  solutions are longer and it needs many more gradient steps per state-visit
  than the table needs lookups.

### On a GPU (Apple Silicon or CUDA)

```
pip install torch                                              # Apple Silicon: the default wheel has MPS support
python play_rubik.py train --agent net --backend torch --dashboard 8000 --minutes 20
python play_rubik.py serve --agent net                         # afterwards, pages + the network's policy (no torch needed)
python play_rubik.py train --agent net --backend torch --resume --minutes 30   # continue from cache/net.npz
```

`--resume` reloads the saved weights, the curriculum depth and the metrics
history, so a run can be extended without starting over (`--layers` must
match the saved network). On CUDA, TF32 is on by default and `--amp` adds
bf16 autocast for the forward passes. `--lr-final 1e-4` cosine-decays the
learning rate over `--minutes`, which sharpens the values once the curriculum
has stalled.

**Seen vs unseen.** The set of states sampled during training is saved
with the network (`cache/net_seen.npy`). `eval --agent net --seen` splits
every distance into sampled and never-sampled states, reports the greedy
solve rate of each, and for solved never-sampled states counts how many
greedy moves it took to reach a sampled state: mostly 1-2 means memory plus
one step of local generalisation, "never" means the value function
generalised along the whole path.

**Search on top of the value function.** A greedy walk needs every Q-value
to be right; a beam search only needs the solved state to be reachable
through states the network rates highly. `eval --agent net --beam 32` and
`solve --agent net --beam 32 <state>` keep the 32 best states per depth and
expand all of their children (DeepCube's idea in its simplest form). On a
network whose greedy policy solves 30 % of random states this is what turns
it into a solver.

`qnet_torch.py` is the same learner on PyTorch: the one-hot features of all
3,674,160 states (529 MB) and the transition table live on the device, the
default network is 144-1024-1024-512-9, and each batch of 8,192 sampled
states contributes targets for all nine actions (`--replay` switches to
model-free replay Q-learning). `--device auto` picks MPS on a Mac, CUDA if
present, otherwise CPU; `--layers 2048,2048,1024` widens the net. The
weights are exported in the numpy `MLP` format, so `serve`, `eval` and
`solve` work on the result without torch. Metrics and the dashboard are
identical to the numpy version, plus a `device` field.

### What happened

Runs on one CPU core (numpy, 4 threads unless noted). "d≤4 / d=5 / d=6" is the
greedy solve rate on states at exactly that distance; "random" is the solve
rate on a uniformly random state; "unseen" the solve rate on states never
visited during training.

| variant | minutes | d≤4 | d=5 | d=6 | random | unseen |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning, replay, curriculum (default) | 9 | 100 % | 84 % | 32 % | 3.6 % | 3.5 % |
| + episode cap K+3 (`--dyn-cap`) | 6 | 100 % | 82 % | 38 % | 3.5 % | 2.5 % |
| all-actions targets, curriculum (`--all-actions`) | 6 | 100 % | 81 % | 28 % | 1.2 % | 1.1 % |
| all-actions, sample up to K+2 (`--k-margin 2`) | 6 | 100 % | 91 % | 38 % | 3.6 % | 2.0 % |
| DeepCube-style: all depths, 1/k loss (`--k-start 11 --weight-by-depth`) | 6 | 100 % | 88 % | 42 % | 2.4 % | 1.5 % |
| no curriculum, plain Q-learning (`--k-start 11`) | 6 | 75-100 % | 31 % | 7 % | 0.3 % | 0.3 % |
| default, `--hidden 512`, 15 minutes | 15 | 100 % | 93 % | 44 % | 4.2 % | 3.8 % |

Every variant reaches the same plateau: perfect up to four moves, then a
fast fall-off. Two things were ruled out along the way (the code keeps the
switches so you can reproduce them):

- **Not the number of gradient steps.** Raising updates per environment
  step 8× changed nothing.
- **Not optimistic extrapolation.** The diagnostic in the session showed the
  opposite: values of states just beyond the curriculum are *under*-estimated
  (depth-5 states got about −8 for their correct action, true −5), because
  the replay buffer is dominated by wandering transitions into deep states
  whose targets sit at the −12 clamp, and the network smooths that mass over
  the frontier.

What remains is precision. For the greedy policy to be right, the network
must separate Q-values that differ by exactly 1 across millions of states,
i.e. keep its error below ±0.5 nearly everywhere. A 144-256-256-9 MLP trained
for minutes on a CPU gets the *average* error down to about 1 move (the
"value error" curve) but not the per-state error, and the curriculum cannot
advance past K = 5 because the 97 % gate at depth 5 is never met. The table
gets ±0 by construction. The published DeepCube results use networks two
orders of magnitude larger, GPU-hours, and a search on top of the value
function rather than a greedy walk; that is the actual cost of trading
coverage for generalisation.

The generalisation itself is real but small: 2-4 % of never-visited states
are solved greedily, against 0 % for any table.

## Applying the policy to a real cube

The policy only ever turns the top, right and front layers, so the
back-left-bottom cubie of the physical cube never moves. Hold the cube in any
orientation, enter its 24 sticker colours in the playground, and the page

1. relabels colours so that the fixed cubie's back/left/bottom stickers become
   the env's `b`/`w`/`m` and each colour's opposite maps to the env's opposite
   (this also fixes the handedness, so the result is a legal env state whenever
   the input is a real cube),
2. checks the state is solvable (a corner twisted by hand fails the orientation
   parity test),
3. looks the state up in `policy.bin` (two 4-bit actions per byte, one per
   state, 1.8 MB) and follows the greedy action until solved, and
4. animates each move in the cube's own colours; U / R / F are the standard
   clockwise turns, `'` is anticlockwise, `2` is a half turn.

`python play_rubik.py solve <24 letters>` does the same lookup on the command
line for env-coloured states.

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
