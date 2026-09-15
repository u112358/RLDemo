# MIT License
#
# Copyright (c) 2017 BingZhang Hu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""2x2x2 pocket cube environment.

The cube is 6 faces x 4 stickers.  A state is a length-24 array of colour
ids (see ``COLORS``) in the order front, right, top, back, left, bottom, and
within a face left-top, right-top, right-bottom, left-bottom.  The nine
actions turn the top, right or front layer by 90/180/270 degrees.  Because
the back-left-bottom cubie never moves, those three layers reach every one of
the 3,674,160 legal states.

Besides the ``Rubik`` gym-like class this module provides a perfect hash of
the legal states (``encode`` / ``decode``) and a cached transition table
(``transitions``), which is what makes a full-size tabular Q-learner
practical (see ``play_rubik.py``).
"""
import os
import numpy as np

FRONT, RIGHT, TOP, BACK, LEFT, BOTTOM = range(6)
LEFT_TOP, RIGHT_TOP, RIGHT_BOTTOM, LEFT_BOTTOM = range(4)
FACE_NAMES = ['front', 'right', 'top', 'back', 'left', 'bottom']

COLORS = 'wgbrym'            # colour id -> letter (white green blue red yellow magenta)
ACTIONS = ['t1', 't2', 't3', 'r1', 'r2', 'r3', 'f1', 'f2', 'f3']
NOTATION = ['U', 'U2', "U'", 'R', 'R2', "R'", 'F', 'F2', "F'"]
INVERSE = [2, 1, 0, 5, 4, 3, 8, 7, 6]

# The scramble the original demo always started from.
INIT = 'wggbrymrgrwrbmbwybywmygm'
# The solved cube that scramble belongs to (fixed by the immovable
# back-left-bottom cubie): front g, right y, top r, back b, left w, bottom m.
SOLVED = 'ggggyyyyrrrrbbbbwwwwmmmm'

N_STATES = 3674160           # 7! * 3^6


# --------------------------------------------------------------------------
# Moves, written as sticker cycles (the original implementation) and then
# compiled once into permutation vectors.
# --------------------------------------------------------------------------
def _twist_top(s):
    p = [f[:] for f in s]
    s[FRONT][LEFT_TOP] = p[RIGHT][LEFT_TOP]
    s[FRONT][RIGHT_TOP] = p[RIGHT][RIGHT_TOP]
    s[LEFT][LEFT_TOP] = p[FRONT][LEFT_TOP]
    s[LEFT][RIGHT_TOP] = p[FRONT][RIGHT_TOP]
    s[BACK][LEFT_TOP] = p[LEFT][LEFT_TOP]
    s[BACK][RIGHT_TOP] = p[LEFT][RIGHT_TOP]
    s[RIGHT][LEFT_TOP] = p[BACK][LEFT_TOP]
    s[RIGHT][RIGHT_TOP] = p[BACK][RIGHT_TOP]
    s[TOP][LEFT_TOP] = p[TOP][LEFT_BOTTOM]
    s[TOP][RIGHT_TOP] = p[TOP][LEFT_TOP]
    s[TOP][RIGHT_BOTTOM] = p[TOP][RIGHT_TOP]
    s[TOP][LEFT_BOTTOM] = p[TOP][RIGHT_BOTTOM]


def _twist_right(s):
    p = [f[:] for f in s]
    s[FRONT][RIGHT_TOP] = p[BOTTOM][RIGHT_TOP]
    s[FRONT][RIGHT_BOTTOM] = p[BOTTOM][RIGHT_BOTTOM]
    s[TOP][RIGHT_TOP] = p[FRONT][RIGHT_TOP]
    s[TOP][RIGHT_BOTTOM] = p[FRONT][RIGHT_BOTTOM]
    s[BACK][LEFT_BOTTOM] = p[TOP][RIGHT_TOP]
    s[BACK][LEFT_TOP] = p[TOP][RIGHT_BOTTOM]
    s[BOTTOM][RIGHT_TOP] = p[BACK][LEFT_BOTTOM]
    s[BOTTOM][RIGHT_BOTTOM] = p[BACK][LEFT_TOP]
    s[RIGHT][LEFT_TOP] = p[RIGHT][LEFT_BOTTOM]
    s[RIGHT][RIGHT_TOP] = p[RIGHT][LEFT_TOP]
    s[RIGHT][RIGHT_BOTTOM] = p[RIGHT][RIGHT_TOP]
    s[RIGHT][LEFT_BOTTOM] = p[RIGHT][RIGHT_BOTTOM]


def _twist_front(s):
    p = [f[:] for f in s]
    s[RIGHT][LEFT_TOP] = p[TOP][LEFT_BOTTOM]
    s[RIGHT][LEFT_BOTTOM] = p[TOP][RIGHT_BOTTOM]
    s[TOP][LEFT_BOTTOM] = p[LEFT][RIGHT_BOTTOM]
    s[TOP][RIGHT_BOTTOM] = p[LEFT][RIGHT_TOP]
    s[LEFT][RIGHT_TOP] = p[BOTTOM][LEFT_TOP]
    s[LEFT][RIGHT_BOTTOM] = p[BOTTOM][RIGHT_TOP]
    s[BOTTOM][LEFT_TOP] = p[RIGHT][LEFT_BOTTOM]
    s[BOTTOM][RIGHT_TOP] = p[RIGHT][LEFT_TOP]
    s[FRONT][LEFT_TOP] = p[FRONT][LEFT_BOTTOM]
    s[FRONT][RIGHT_TOP] = p[FRONT][LEFT_TOP]
    s[FRONT][RIGHT_BOTTOM] = p[FRONT][RIGHT_TOP]
    s[FRONT][LEFT_BOTTOM] = p[FRONT][RIGHT_BOTTOM]


def _build_perms():
    perms = []
    for twist in (_twist_top, _twist_right, _twist_front):
        idx = [[f * 4 + p for p in range(4)] for f in range(6)]
        for _ in range(3):
            twist(idx)
            perms.append([v for face in idx for v in face])
    return np.array(perms, dtype=np.int64)


PERM = _build_perms()        # (9, 24): next_state = state[PERM[action]]


def to_array(s):
    """'wggb...' (24 letters) or a list of 6 lists -> uint8 array of colour ids."""
    if isinstance(s, np.ndarray):
        return s.astype(np.uint8)
    if not isinstance(s, str):
        s = ''.join(''.join(f) for f in s)
    s = s.strip()
    if len(s) != 24 or any(c not in COLORS for c in s):
        raise ValueError('a state is 24 letters out of %r' % COLORS)
    return np.array([COLORS.index(c) for c in s], dtype=np.uint8)


def to_str(a):
    return ''.join(COLORS[int(c)] for c in np.asarray(a).reshape(24))


def check(state):
    """Number of faces that are a single colour (0..6), the original reward."""
    a = to_array(state).reshape(6, 4)
    return int(np.sum(np.all(a == a[:, :1], axis=1)))


class Rubik:
    """Gym-like wrapper.  ``state`` is the 24-letter string, ``take_action``
    returns ``(state, reward, done)`` with reward = ``check(state)`` and done
    once ``done_faces`` faces are uniform (6 = fully solved; the original demo
    used 2)."""

    def __init__(self, done_faces=6, seed=None):
        self.action_space = list(ACTIONS)
        self.n_actions = len(ACTIONS)
        self.done_faces = done_faces
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self._a = to_array(INIT)
        self.reset()

    @property
    def state(self):
        return to_str(self._a)

    def reset(self, scramble=None):
        """``scramble=None`` -> the fixed INIT scramble, ``int k`` -> k random
        moves from solved, ``str`` -> that exact state."""
        if scramble is None:
            self._a = to_array(INIT)
        elif isinstance(scramble, (int, np.integer)):
            self._a = to_array(SOLVED)
            for _ in range(int(scramble)):
                self._a = self._a[PERM[self.rng.integers(9)]]
        else:
            self._a = to_array(scramble)
        self.count = 0
        return self.state

    def take_action(self, action):
        self._a = self._a[PERM[int(action)]]
        self.count += 1
        reward = check(self._a)
        return self.state, reward, reward >= self.done_faces


# --------------------------------------------------------------------------
# Perfect hashing of legal states.
#
# A legal state is a permutation of the 7 movable corner cubies (7! = 5040)
# times the orientation of 6 of them (3^6 = 729, the seventh is forced).
# Each corner position lists its three sticker slots as [UD face, FB face,
# RL face]; orientation = which of the three holds the top/bottom colour.
# --------------------------------------------------------------------------
def _slot(face, pos):
    return face * 4 + pos


CORNER_SLOTS = np.array([
    [_slot(TOP, RIGHT_BOTTOM),    _slot(FRONT, RIGHT_TOP),    _slot(RIGHT, LEFT_TOP)],     # front-right-top
    [_slot(TOP, LEFT_BOTTOM),     _slot(FRONT, LEFT_TOP),     _slot(LEFT, RIGHT_TOP)],     # front-left-top
    [_slot(BOTTOM, RIGHT_TOP),    _slot(FRONT, RIGHT_BOTTOM), _slot(RIGHT, LEFT_BOTTOM)],  # front-right-bottom
    [_slot(BOTTOM, LEFT_TOP),     _slot(FRONT, LEFT_BOTTOM),  _slot(LEFT, RIGHT_BOTTOM)],  # front-left-bottom
    [_slot(TOP, RIGHT_TOP),       _slot(BACK, LEFT_TOP),      _slot(RIGHT, RIGHT_TOP)],    # back-right-top
    [_slot(TOP, LEFT_TOP),        _slot(BACK, RIGHT_TOP),     _slot(LEFT, LEFT_TOP)],      # back-left-top
    [_slot(BOTTOM, RIGHT_BOTTOM), _slot(BACK, LEFT_BOTTOM),   _slot(RIGHT, RIGHT_BOTTOM)], # back-right-bottom
    [_slot(BOTTOM, LEFT_BOTTOM),  _slot(BACK, RIGHT_BOTTOM),  _slot(LEFT, LEFT_BOTTOM)],   # back-left-bottom (fixed)
], dtype=np.int64)

_SOLVED_A = to_array(SOLVED)
_UD = np.array([COLORS.index('r'), COLORS.index('m')])           # top / bottom colours
_FACT = np.array([1, 1, 2, 6, 24, 120, 720], dtype=np.int64)
_POW3 = 3 ** np.arange(5, -1, -1)

# cubie id = its home position in the solved cube; lookup by colour triple
_CUBIE_ID = np.full((6, 6, 6), -1, dtype=np.int64)
for _i in range(8):
    _c = _SOLVED_A[CORNER_SLOTS[_i]]
    for _p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        _CUBIE_ID[_c[_p[0]], _c[_p[1]], _c[_p[2]]] = _i


def _cubies(states):
    """(N,24) -> cubie ids (N,8), orientations (N,8)."""
    cols = states[:, CORNER_SLOTS]                                   # (N,8,3)
    cid = _CUBIE_ID[cols[..., 0], cols[..., 1], cols[..., 2]]
    ori = np.argmax(np.isin(cols, _UD), axis=2)
    return cid, ori


def _fill_sticker_table():
    """STICKER[position, cubie, orientation] -> the 3 colours in slot order,
    observed on a random walk (every combination is hit within a few
    thousand moves).  Also finds the orientation-sum invariant."""
    rng = np.random.default_rng(0)
    table = np.full((8, 8, 3, 3), -1, dtype=np.int64)
    seen = np.zeros((8, 8, 3), dtype=bool)
    ori_rows = []
    a = _SOLVED_A.copy()
    for _ in range(6000):
        a = a[PERM[rng.integers(9)]]
        cid, ori = _cubies(a[None])
        cols = a[CORNER_SLOTS]
        for p in range(8):
            table[p, cid[0, p], ori[0, p]] = cols[p]
            seen[p, cid[0, p], ori[0, p]] = True
        ori_rows.append(ori[0])
    assert seen[:7, :7, :].all() and seen[7, 7, 0]
    ori_rows = np.array(ori_rows)
    # orientation invariant: sum_i sign_i * ori_i = const (mod 3)
    for bits in range(1 << 7):
        sign = np.array([1 if bits >> i & 1 else -1 for i in range(7)])
        v = (ori_rows[:, :7] * sign).sum(axis=1) % 3
        if (v == v[0]).all():
            return table, sign, int(v[0])
    raise RuntimeError('no orientation invariant found')


_STICKER, _ORI_SIGN, _ORI_CONST = _fill_sticker_table()


def encode(states):
    """Legal state(s) -> integer index in [0, N_STATES).  Accepts a 24-letter
    string, a (24,) array or an (N,24) array."""
    single = isinstance(states, str) or np.ndim(states) == 1
    a = np.atleast_2d(to_array(states) if isinstance(states, str) else np.asarray(states, dtype=np.uint8))
    cid, ori = _cubies(a)
    perm = cid[:, :7]
    if (perm < 0).any() or (cid[:, 7] != 7).any():
        raise ValueError('not a legal cube state')
    smaller = (perm[:, :, None] > perm[:, None, :]) & (np.arange(7)[None, None, :] > np.arange(7)[None, :, None])
    rank = (smaller.sum(axis=2) * _FACT[6::-1][None, :]).sum(axis=1)
    ocode = (ori[:, :6] * _POW3[None, :]).sum(axis=1)
    idx = rank * 729 + ocode
    return int(idx[0]) if single else idx


def decode(idx):
    """Inverse of ``encode``: index (or array of indices) -> (N,24) uint8 states."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    n = idx.shape[0]
    rank, ocode = idx // 729, idx % 729
    perm = np.zeros((n, 7), dtype=np.int64)
    avail = np.ones((n, 7), dtype=bool)
    rows = np.arange(n)
    for i in range(7):
        f = _FACT[6 - i]
        d, rank = rank // f, rank % f
        cum = np.cumsum(avail, axis=1)
        perm[:, i] = np.argmax((cum == (d + 1)[:, None]) & avail, axis=1)
        avail[rows, perm[:, i]] = False
    ori = np.zeros((n, 8), dtype=np.int64)
    ori[:, :6] = (ocode[:, None] // _POW3[None, :]) % 3
    partial = (ori[:, :6] * _ORI_SIGN[:6]).sum(axis=1)
    ori[:, 6] = ((_ORI_CONST - partial) * _ORI_SIGN[6]) % 3     # sign is +-1 so it is its own inverse mod 3
    states = np.empty((n, 24), dtype=np.uint8)
    for p in range(7):
        states[:, CORNER_SLOTS[p]] = _STICKER[p, perm[:, p], ori[:, p]]
    states[:, CORNER_SLOTS[7]] = _SOLVED_A[CORNER_SLOTS[7]]
    return states


SOLVED_INDEX = encode(SOLVED)


class EpisodeRecorder:
    """Keeps the last complete episode of one environment (for the dashboard's
    replay): start state, every action, whether it was solved."""

    def __init__(self):
        self.states, self.actions, self.tags = [], [], []
        self.last, self.count = None, 0

    def push(self, step, s, a, s2, done, reset, k, K, explore=False, unknown=False):
        if not self.states:
            self.states = [int(s)]
        self.states.append(int(s2))
        self.actions.append(int(a))
        self.tags.append('explore' if explore else 'unknown' if unknown else 'greedy')
        if reset:
            self.count += 1
            self.last = {'id': self.count, 'step': int(step), 'k': int(k), 'K': int(K), 'solved': bool(done),
                         'states': self.states, 'actions': self.actions, 'tags': self.tags}
            self.states, self.actions, self.tags = [], [], []


def transitions(cache_dir='cache'):
    """(N_STATES, 9) int32 table: next index for every state and action.
    Built once (about 20 s) and cached on disk."""
    path = os.path.join(cache_dir, 'transitions.npy')
    if os.path.exists(path):
        return np.load(path, mmap_mode='r')
    T = np.empty((N_STATES, 9), dtype=np.int32)
    chunk = 250000
    for lo in range(0, N_STATES, chunk):
        hi = min(lo + chunk, N_STATES)
        s = decode(np.arange(lo, hi))
        for a in range(9):
            T[lo:hi, a] = encode(s[:, PERM[a]])
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, T)
    return T


def bfs_distances(T):
    """Exact distance-to-solved of every state (0..11), by BFS over ``T``."""
    dist = np.full(N_STATES, -1, dtype=np.int8)
    dist[SOLVED_INDEX] = 0
    frontier = np.array([SOLVED_INDEX])
    d = 0
    while frontier.size:
        d += 1
        nxt = np.unique(np.asarray(T)[frontier].ravel())
        nxt = nxt[dist[nxt] < 0]
        dist[nxt] = d
        frontier = nxt
    return dist


if __name__ == '__main__':
    # self-check: the index is a bijection onto the 3,674,160 legal states and
    # the BFS depth profile is the known one for the pocket cube.
    T = transitions()
    dist = bfs_distances(T)
    hist = np.bincount(dist[dist >= 0])
    print('states reachable from solved:', int((dist >= 0).sum()), 'of', N_STATES)
    print('depth histogram:', hist.tolist())
    assert (dist >= 0).all() and hist.tolist() == [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]
    print('INIT is', int(dist[encode(INIT)]), 'moves from solved')
    print('ok')


def cubies(states):
    """(N,24) sticker states -> (cubie id per slot (N,8), orientation per slot (N,8))."""
    return _cubies(np.atleast_2d(np.asarray(states, dtype=np.uint8)))


# ---- whole-cube symmetries that keep the fixed cubie and the move set {U, R, F} ----
def symmetry_move_perms():
    """The six move permutations: identity, rho (U->R->F->U), rho^2, mu (mirror: R<->F, every turn
    reversed), mu*rho, mu*rho^2.  Returns (6, 9) int64, index 0 = identity."""
    layer, turn = np.arange(9) // 3, np.arange(9) % 3
    rho = (layer + 1) % 3 * 3 + turn
    mu = np.array([0, 2, 1])[layer] * 3 + (2 - turn)
    comp = lambda a, b: a[b]
    return np.stack([np.arange(9), rho, comp(rho, rho), mu, comp(mu, rho), comp(mu, comp(rho, rho))])


def build_symmetry(T, perm):
    """State map sigma with sigma(T[s, m]) = T[sigma(s), perm[m]] and sigma(solved) = solved,
    built by BFS from the solved state.  Raises ValueError if perm is not an automorphism."""
    N = T.shape[0]
    sigma = np.full(N, -1, dtype=np.int64)
    sigma[SOLVED_INDEX] = SOLVED_INDEX
    frontier = np.array([SOLVED_INDEX])
    while frontier.size:
        new = []
        for m in range(9):
            nxt = T[frontier, m]
            img = T[sigma[frontier], perm[m]]
            fresh = sigma[nxt] < 0
            if fresh.any():
                u, first = np.unique(nxt[fresh], return_index=True)
                sigma[u] = img[fresh][first]
                new.append(u)
            if not np.array_equal(sigma[nxt], img):
                raise ValueError('not a symmetry of the state graph')
        frontier = np.unique(np.concatenate(new)) if new else np.array([], dtype=np.int64)
    assert (sigma >= 0).all()
    return sigma


def symmetries(T=None, cache_dir='cache'):
    """(6, N_STATES) int32: state maps of the six symmetries (row 0 = identity), cached on disk.
    Every row preserves the BFS distance of every state."""
    path = os.path.join(cache_dir, 'symmetries.npy')
    if os.path.exists(path):
        return np.load(path, mmap_mode='r')
    if T is None:
        T = transitions(cache_dir)
    perms = symmetry_move_perms()
    sig = np.stack([build_symmetry(T, p) for p in perms]).astype(np.int32)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, sig)
    return sig
