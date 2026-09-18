"""gamescore.py — device-side engine for FeatureFold (numpy ONLY).

Invariants (do not break):
  * Scorer = pixels + products of ADJACENT placed pairs -> layout-sensitive.
    A pixels-only model is provably layout-INVARIANT (permuting features
    permutes the weight vector; accuracy cannot move). See __main__ check.
  * Deterministic everywhere (seeded data/split/bootstrap, full-batch fit):
    the same layout scores identically on every device — no server needed.
"""
import os
import numpy as np

LEVEL_TAGS = ("pairs_16", "xor_16")

# ------------------------------------------------------------------ data
def make_game_data(n=4000, F=16, n_syn=4, seed=42):
    """Warm-up level: label = linear part + 4 hidden synergy PAIRS."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, F))
    w = rng.normal(0, 0.6, F)
    perm = rng.permutation(F)
    pairs = [(int(perm[2 * k]), int(perm[2 * k + 1])) for k in range(n_syn)]
    syn = sum(rng.normal(1.3, 0.1) * X[:, a] * X[:, b] for a, b in pairs)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w + syn)))).astype(np.int64)
    return X, y, pairs

def make_xor_data(n=4000, F=16, seed=42):
    """Hard level: label = XOR of two hidden bits. Zero individual signal —
    the surrogate is blind BY DESIGN; only a Test credit reveals the pair."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n); b = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, F))
    X[:, 0] = a + rng.normal(0, .15, n)
    X[:, 1] = b + rng.normal(0, .15, n)
    return X, (a ^ b).astype(np.int64), [(0, 1)]

def split_scale(X, y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); n_val = int(len(X) * val_frac)
    va, tr = idx[:n_val], idx[n_val:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    return Z[tr], y[tr], Z[va], y[va]

# ------------------------------------------------------------------ layout
def adjacent_cells(H, W):
    out = []
    for r in range(H):
        for c in range(W):
            if c + 1 < W: out.append(((r, c), (r, c + 1)))
            if r + 1 < H: out.append(((r, c), (r + 1, c)))
    return out

import re

def _fid(key):
    """Feature id from a layout-map key. Accepts ints directly, or any label
    with a trailing number: 'F07' -> 7, 'feature_07' -> 7, 'F15' -> 15."""
    if isinstance(key, (int, np.integer)):
        return int(key)
    m = re.search(r"(\d+)\s*$", str(key))
    if m is None:
        raise ValueError(f"cannot read feature id from label {key!r}")
    return int(m.group(1))

def fac_from_map(layout_map, H, W):
    fac = -np.ones(H * W, int)
    for key, (r, c) in layout_map.items():
        fac[r * W + c] = _fid(key)
    return fac

# ------------------------------------------------------------------ scorer
def make_design(Z, fac, H, W, use_products=True):
    n = Z.shape[0]
    pix = np.zeros((n, H * W), np.float32)
    m = fac >= 0
    pix[:, m] = Z[:, fac[m]]
    if not use_products:
        return pix
    cols = []
    for (r1, c1), (r2, c2) in adjacent_cells(H, W):
        ia, ib = r1 * W + c1, r2 * W + c2
        if fac[ia] >= 0 and fac[ib] >= 0:
            cols.append((pix[:, ia] * pix[:, ib])[:, None])
    return np.hstack([pix] + cols) if cols else pix

def _fit_logistic(D, y, l2=1e-3, iters=500, lr=0.05):
    n, d = D.shape
    Db = np.hstack([D, np.ones((n, 1), np.float32)])
    w = np.zeros(d + 1); m = np.zeros(d + 1); v = np.zeros(d + 1)
    yf = y.astype(np.float32)
    for t in range(1, iters + 1):
        p = 1 / (1 + np.exp(-(Db @ w)))
        g = Db.T @ (p - yf) / n
        g[:-1] += l2 * w[:-1]
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.01 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w

def accuracy(trX, trY, vaX, vaY, fac, H, W, use_products=True):
    Dtr = make_design(trX, fac, H, W, use_products)
    w = _fit_logistic(Dtr, trY)
    Dva = make_design(vaX, fac, H, W, use_products)
    pred = (np.hstack([Dva, np.ones((len(Dva), 1), np.float32)]) @ w) > 0
    correct = (pred.astype(np.int64) == vaY)
    acc = float(correct.mean())
    rng = np.random.default_rng(0); n = len(correct)
    boots = [correct[rng.integers(0, n, n)].mean() for _ in range(300)]
    return acc, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

# ------------------------------------------------------------------ surrogate
class Surrogate:
    """Instant per-cell guide: imp = |corr(f, y)|, syn = |corr(f_i*f_j, y)|.
    Blind to XOR structure by design — that's what Test credits are for."""
    def __init__(self, trX, trY):
        y = trY.astype(np.float32); yc = y - y.mean()
        Xc = trX - trX.mean(0)
        self.imp = np.abs(Xc.T @ yc) / (len(trX) * trX.std(0) * yc.std() + 1e-9)
        F = trX.shape[1]
        self.syn = np.zeros((F, F), np.float32)
        for i in range(F):
            for j in range(i + 1, F):
                p = trX[:, i] * trX[:, j]; pc = p - p.mean()
                self.syn[i, j] = self.syn[j, i] = \
                    abs(pc @ yc) / (len(trX) * p.std() * yc.std() + 1e-9)

    @classmethod
    def from_stats(cls, imp, syn):
        s = object.__new__(cls); s.imp = imp; s.syn = syn
        return s

    def cell_scores(self, fac, H, W):
        g = fac.reshape(H, W); out = np.zeros(H * W)
        for r in range(H):
            for c in range(W):
                f = g[r, c]
                if f < 0: continue
                s = self.imp[f]
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < H and 0 <= c2 < W and g[r2, c2] >= 0:
                        s += 0.5 * self.syn[f, g[r2, c2]]
                out[r * W + c] = s
        mx = out.max()
        return out / mx if mx > 0 else out

# ------------------------------------------------------------------ levels
def load_level(tag):
    """Baked npz if present (fast, byte-identical across devices), else
    deterministic generation (the numbers are identical either way)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"level_{tag}.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["trX"], d["trY"], d["vaX"], d["vaY"], \
               Surrogate.from_stats(d["imp"], d["syn"])
    X, y, _ = (make_game_data if tag == "pairs_16" else make_xor_data)()
    trX, trY, vaX, vaY = split_scale(X, y)
    return trX, trY, vaX, vaY, Surrogate(trX, trY)

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    X, y, _ = make_game_data(seed=7)
    trX, trY, vaX, vaY = split_scale(X, y, seed=1)
    a0 = [accuracy(trX, trY, vaX, vaY, rng.permutation(16), 4, 4, use_products=False)[0]
          for _ in range(3)]
    print("pixels-only   :", "  ".join(f"{a:.4f}" for a in a0), " <- must be IDENTICAL")
    a1 = [accuracy(trX, trY, vaX, vaY, rng.permutation(16), 4, 4)[0] for _ in range(3)]
    print("with products :", "  ".join(f"{a:.4f}" for a in a1), " <- must spread")