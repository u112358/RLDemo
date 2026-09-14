# RLDemo — Q-learning on a 2×2×2 pocket cube

- `rubik.py` — the environment. 6 faces × 4 stickers, 9 actions (`t1 t2 t3 r1 r2 r3 f1 f2 f3` = top / right / front layer turned 90° / 180° / 270°), reward = number of single-colour faces (`check`). An episode is `done` when the reward is greater than 1.
- `play_rubik.py` — tabular Q-learning agent (pandas Q-table), trains for 100k episodes from the fixed `reset()` scramble and writes the Q-table to CSV.
- `playground.html` — interactive 3D visualiser of the same environment. Open it in a browser (it loads three.js from cdnjs). Same sticker indices, same action ids, same reward; you can click actions, type a sequence such as `t1 r3 f2` or `0 5 7`, scramble, undo, and compute the optimal solution with a bidirectional BFS.

The `reset()` scramble is a legal cube state 9 moves from solved (optimal: `t3 f3 t2 f3 r1 f2 r1 f3 t2`).
