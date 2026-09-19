"""gamescore.py — device-side engine for FeatureFold (numpy ONLY).  v2

Invariants (do not break):
  * Scorer = pixels + products of ADJACENT placed pairs -> layout-sensitive.
    A pixels-only model is provably layout-INVARIANT (permuting features
    permutes the weight vector; accuracy cannot move). See invariance_check().
  * Deterministic everywhere (seeded data/split/folds, full-batch fit):
    the same layout scores identically on every device — no server needed.
  * BLINDNESS CONTRACT: in 'blind' mode the surrogate is, by construction,
    uninformative about interaction structure — enforced at bake time by
    blindness_gate().  v1 LEAKED XOR: raw |corr(x_i*x_j, y)| detects any
    joint interaction (for fair bits corr(ab, a^b) = -1/sqrt(3) ~= -0.577),
    so the XOR level was solvable without spending a single Test credit.
    v2 gates the synergy matrix by individual importance:
        syn_blind = syn_raw * min(imp_i, imp_j)   (~= 0 when either is hidden)

v2 changes
  * Surrogate modes: 'reveal' (pairs/tutorial: raw product-corr heatmap)
    and 'blind' (XOR: imp-gated; heatmap carries NO interaction signal).
  * blindness_gate(): automated check that the surrogate cannot distinguish
    an interaction-optimal layout from random ones.  Baked levels must pass.
  * Referee moved on-device: stratified K-fold, per-fold scaling, product
    design (Chebyshev radius r), softmax fit — no sklearn, no torch.
  * paired_from_folds(): "new best" is judged by the PAIRED fold difference,
    not point estimates.
  * Level versioning: LEVELS[tag]['seed'] bumps per release (moves hidden
    pairs/weights); baked npz carries fmt/version/pairs/credits.
  * sigmoid overflow fix (clip before exp).
"""
import os
import json
import numpy as np

LEVEL_FMT = 2   # bump only when npz INTERPRETATION changes

# ------------------------------------------------------------------ levels
LEVEL_TAGS = ("pairs_16", "xor_16")

# Bump 'seed' to release a fresh puzzle (moves hidden pairs / weights);
# baked npz stores it as the level version.
LEVELS = {
    "pairs_16": dict(name="Pairs (tutorial)", F=16, H=4, W=4, n=4000,
                     n_syn=4, seed=101, gated=False, credits=10,
                     hint="Tutorial: the heatmap is trustworthy here — "
                          "place high-synergy pairs on adjacent cells."),
    "xor_16":   dict(name="XOR", F=16, H=4, W=4, n=4000,
                     n_syn=1, seed=704, gated=True, credits=10,
                     hint="The heatmap is blind BY DESIGN. Only a Test "
                          "credit reveals the hidden pair."),
}

# ------------------------------------------------------------------ data
def make_game_data(n=4000, F=16, n_syn=4, seed=42):
    """Tutorial level: label = linear part + n_syn hidden synergy PAIRS."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, F)).astype(np.float32)
    w = rng.normal(0, 0.6, F)
    perm = rng.permutation(F)
    pairs = [(int(perm[2 * k]), int(perm[2 * k + 1])) for k in range(n_syn)]
    syn = sum(rng.normal(1.3, 0.1) * X[:, a] * X[:, b] for a, b in pairs)
    y = (rng.random(n) < 1 / (1 + np.exp(-np.clip(X @ w + syn, -30, 30))
                              )).astype(np.int64)
    return X, y, pairs


def make_xor_data(n=4000, F=16, seed=704, noise=0.15):
    """Hard level: label = XOR of two hidden bits.  Zero individual signal —
    the blind surrogate is blind BY DESIGN; only a Test credit reveals the
    pair.  v2: the pair is drawn from the seeded rng, so bumping the level
    seed moves it (v1 hardcoded (0, 1) -> memorizable)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(F)
    pair = (int(perm[0]), int(perm[1]))
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, F)).astype(np.float32)
    X[:, pair[0]] = a + rng.normal(0, noise, n)
    X[:, pair[1]] = b + rng.normal(0, noise, n)
    return X, (a ^ b).astype(np.int64), [pair]


def split_scale(X, y, val_frac=0.2, seed=0):
    """Deterministic 80/20 split + train-stats scaling (used by the
    invariance self-test; the referee scales per fold instead)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    va, tr = idx[:n_val], idx[n_val:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    return Z[tr], y[tr], Z[va], y[va]


def _surrogate_indices(X, y, seed, frac=0.8):
    """Deterministic subset the surrogate is fit on (leakage hygiene)."""
    rng = np.random.default_rng(seed ^ 0xA5A5A5)
    idx = rng.permutation(len(X))
    return idx[:int(len(X) * frac)]


# ------------------------------------------------------------------ layout
def adjacent_cells(H, W):
    out = []
    for r in range(H):
        for c in range(W):
            if c + 1 < W:
                out.append(((r, c), (r, c + 1)))
            if r + 1 < H:
                out.append(((r, c), (r + 1, c)))
    return out


import re


def _fid(key):
    """Feature id from a layout-map key: int, or any label with a trailing
    number ('F07' -> 7, 'feature_07' -> 7)."""
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


def sequential_layout(F, H, W):
    fac = -np.ones(H * W, int)
    fac[:F] = np.arange(F)
    return fac


def random_layout(F, H, W, rng):
    fac = -np.ones(H * W, int)
    cells = rng.choice(H * W, size=F, replace=False)
    fac[cells] = rng.permutation(F)
    return fac


def oracle_fac(F, H, W, pairs):
    """Full placement with each interaction pair on adjacent cells
    (domino tiling; requires even W).  Used by the gates."""
    fac = -np.ones(H * W, int)
    used = set()
    for k, (a, b) in enumerate(pairs):
        fac[2 * k], fac[2 * k + 1] = a, b
        used |= {a, b}
    rest = iter(f for f in range(F) if f not in used)
    for i in range(H * W):
        if fac[i] < 0:
            fac[i] = next(rest)
    return fac

def worst_fac(F, H, W, pairs):
    """Anti-oracle: every interaction pair placed at (greedy) maximal
    Chebyshev distance; remaining features fill the rest in order.
    Deterministic.  On 4x4 all pairs land at distance 3."""
    fac = -np.ones(H * W, int)
    used = set()
    for a, b in pairs:
        free = [i for i in range(H * W) if i not in used]
        best = None
        for ii, i in enumerate(free):
            r1, c1 = divmod(i, W)
            for j in free[ii + 1:]:
                r2, c2 = divmod(j, W)
                d = max(abs(r1 - r2), abs(c1 - c2))
                if best is None or d > best[0]:
                    best = (d, i, j)
        _, i, j = best
        fac[i], fac[j] = a, b
        used |= {i, j}
    rest = iter(f for f in range(F) if f not in used)
    for i in range(H * W):
        if fac[i] < 0:
            fac[i] = next(rest)
    return fac
# ------------------------------------------------------------------ scorer
def make_design(Z, fac, H, W, use_products=True):
    """v1 scorer (4-neighborhood adjacent products); kept for the
    invariance self-test.  The referee uses _design_products(radius=1),
    which includes diagonals (Chebyshev radius 1)."""
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
    w = np.zeros(d + 1)
    m = np.zeros(d + 1)
    v = np.zeros(d + 1)
    yf = y.astype(np.float32)
    for t in range(1, iters + 1):
        p = 1 / (1 + np.exp(-np.clip(Db @ w, -30, 30)))   # [v2] overflow fix
        g = Db.T @ (p - yf) / n
        g[:-1] += l2 * w[:-1]
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.01 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w


def accuracy(trX, trY, vaX, vaY, fac, H, W, use_products=True):
    """v1 single-split binary scorer.  KEPT for the invariance self-test and
    cheap checks — the in-game Test credit uses referee_arrays() instead
    (5-fold paired design, ~4x better powered than one 80/20 split)."""
    Dtr = make_design(trX, fac, H, W, use_products)
    w = _fit_logistic(Dtr, trY)
    Dva = make_design(vaX, fac, H, W, use_products)
    pred = (np.hstack([Dva, np.ones((len(Dva), 1), np.float32)]) @ w) > 0
    correct = (pred.astype(np.int64) == vaY)
    acc = float(correct.mean())
    rng = np.random.default_rng(0)
    n = len(correct)
    boots = [correct[rng.integers(0, n, n)].mean() for _ in range(300)]
    return acc, (float(np.percentile(boots, 2.5)),
                 float(np.percentile(boots, 97.5)))


# ------------------------------------------------------------------ referee
# [v2] on-device referee: stratified K-fold CV of pixels + Chebyshev-radius-1
# product features, softmax fit, per-fold scaling (no leakage).  numpy only.
def stratified_folds(y, K, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % K
    return folds


def _design_products(Z, fac, H, W, radius=1):
    """pixels + products of every placed pair within Chebyshev radius r."""
    n = Z.shape[0]
    pix = np.zeros((n, H * W), np.float32)
    m = fac >= 0
    pix[:, m] = Z[:, fac[m]]
    cols = []
    for i in range(H * W):
        if fac[i] < 0:
            continue
        r1, c1 = divmod(i, W)
        for j in range(i + 1, H * W):
            if fac[j] < 0:
                continue
            r2, c2 = divmod(j, W)
            if max(abs(r1 - r2), abs(c1 - c2)) <= radius:
                cols.append((pix[:, i] * pix[:, j])[:, None])
    return np.hstack([pix] + cols) if cols else pix


def _fit_softmax(D, Y, l2=1e-3, iters=400, lr=0.05):
    n, d = D.shape
    k = Y.shape[1]
    Db = np.hstack([D, np.ones((n, 1), np.float32)])
    W = np.zeros((d + 1, k))
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    for t in range(1, iters + 1):
        S = Db @ W
        S -= S.max(1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(1, keepdims=True)
        G = Db.T @ (P - Y) / n
        G[:-1] += l2 * W[:-1]
        m = 0.9 * m + 0.1 * G
        v = 0.999 * v + 0.01 * G * G
        W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W


def _acc(D, y, Wm, classes):
    pred = (np.hstack([D, np.ones((len(D), 1), np.float32)]) @ Wm).argmax(1)
    return float((classes[pred] == y).mean())


def _fold_accs(X, y, folds, fac, H, W, radius, classes):
    accs = []
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Dtr = _design_products((X[tr] - mu) / sd, fac, H, W, radius)
        Dva = _design_products((X[va] - mu) / sd, fac, H, W, radius)
        Ytr = np.stack([(y[tr] == c).astype(np.float32)
                        for c in classes], 1)
        accs.append(_acc(Dva, y[va], _fit_softmax(Dtr, Ytr), classes))
    return np.array(accs)


def referee_arrays(X, y, fac, H, W, radius=1, K=5, seed=0):
    """THE Test-credit scorer.  Pass seed = level version so fold identity
    is stable across sessions and devices (determinism contract)."""
    folds = stratified_folds(np.asarray(y), K, seed)
    classes = np.unique(y)
    a = _fold_accs(np.asarray(X, np.float32), np.asarray(y), folds,
                   fac, H, W, radius, classes)
    hw = 2.776 * a.std(ddof=1) / np.sqrt(K)
    return {"acc": float(a.mean()), "folds": a.tolist(),
            "half_width": float(hw)}


def referee_paired_arrays(X, y, fac_a, fac_b, H, W, radius=1, K=5, seed=0):
    """Paired layout comparison: SAME folds for both layouts, so fold
    difficulty cancels — the difference is the well-powered quantity."""
    folds = stratified_folds(np.asarray(y), K, seed)
    classes = np.unique(y)
    X = np.asarray(X, np.float32)
    a = _fold_accs(X, np.asarray(y), folds, fac_a, H, W, radius, classes)
    b = _fold_accs(X, np.asarray(y), folds, fac_b, H, W, radius, classes)
    d = a - b
    hw = 2.776 * d.std(ddof=1) / np.sqrt(K)
    return {"mean_diff": float(d.mean()), "per_fold": d.tolist(),
            "ci": (float(d.mean() - hw), float(d.mean() + hw))}


def paired_from_folds(fa, fb):
    """Paired difference from two stored fold-acc vectors (main.py uses this
    to judge 'new best' without re-running the previous layout)."""
    d = np.asarray(fa, float) - np.asarray(fb, float)
    hw = 2.776 * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), (float(d.mean() - hw), float(d.mean() + hw))


# ------------------------------------------------------------------ surrogate
class Surrogate:
    """Instant per-cell guide: imp = |corr(f, y)| (normalized), plus an
    adjacent-pair synergy term.

    mode='reveal' (pairs/tutorial): syn = raw |corr(x_i*x_j, y)| — the
        heatmap is a genuine hint.
    mode='blind'  (XOR): syn = raw * min(imp_i, imp_j).  For XOR the hidden
        bits have imp ~= 0, so syn ~= 0 for EVERY pair — the heatmap carries
        no interaction signal, which is what makes Test credits valuable.
        [v1 LEAK: syn was ungated, and corr(ab, a^b) = -1/sqrt(3) is huge —
        the level was solvable for free.  Fixed here; enforced by
        blindness_gate() at bake time.]
    """
    def __init__(self, trX, trY, mode="blind"):
        y = trY.astype(np.float32)
        yc = y - y.mean()
        Xc = trX - trX.mean(0)
        sd = trX.std(0) + 1e-9
        imp = np.abs(Xc.T @ yc) / (len(trX) * sd * (yc.std() + 1e-9))
        self.imp = (imp / (imp.max() + 1e-9)).astype(np.float32)
        F = trX.shape[1]
        syn = np.zeros((F, F), np.float32)
        for i in range(F):
            for j in range(i + 1, F):
                p = trX[:, i] * trX[:, j]
                pc = p - p.mean()
                syn[i, j] = syn[j, i] = abs(
                    pc @ yc) / (len(trX) * (p.std() + 1e-9)
                                * (yc.std() + 1e-9))
        if mode == "blind":
            syn = syn * np.minimum.outer(self.imp, self.imp)  # THE gate
        self.syn = syn.astype(np.float32)   # NOT renormalized in blind mode
        self.mode = mode                    # (renormalizing would amplify
                                            #  noise into fake structure)

    @classmethod
    def from_stats(cls, imp, syn):
        s = object.__new__(cls)
        s.imp = np.asarray(imp, np.float32)
        s.syn = np.asarray(syn, np.float32)
        s.mode = "baked"
        return s

    def cell_scores(self, fac, H, W):
        g = np.asarray(fac).reshape(H, W)
        out = np.zeros(H * W)
        for r in range(H):
            for c in range(W):
                f = g[r, c]
                if f < 0:
                    continue
                s = self.imp[f]
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < H and 0 <= c2 < W and g[r2, c2] >= 0:
                        s += 0.5 * self.syn[f, g[r2, c2]]
                out[r * W + c] = s
        mx = out.max()
        # normalize for DISPLAY only when there is real structure to show;
        # in blind mode mx ~= max(imp) so this never amplifies syn noise.
        return out / mx if mx > 0 else out


# ------------------------------------------------------------------ gates
def invariance_check(trX, trY, vaX, vaY, F, H, W, trials=3, tol=1e-10):
    """Pixels-only scores must be IDENTICAL under feature permutation."""
    rng = np.random.default_rng(1)
    a0 = [accuracy(trX, trY, vaX, vaY, rng.permutation(F), H, W,
                   use_products=False)[0] for _ in range(trials)]
    return {"ok": (max(a0) - min(a0)) < tol, "accs": a0}


def scorer_gate(H=4, W=4, min_gap=0.10, K=5):
    """GATE 1 — the referee must separate adjacent from anti-adjacent
    placement of a planted XOR pair, AND the radius-0 control must be flat
    (the invariance assertion, at referee level)."""
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, H * W)).astype(np.float32)
    X[:, 0] = a + rng.normal(0, .15, n)
    X[:, 1] = b + rng.normal(0, .15, n)
    y = a ^ b

    def fill(forced):
        fac = -np.ones(H * W, int)
        for cell, f in forced.items():
            fac[cell] = f
        rest = iter(f for f in range(H * W) if f not in set(forced.values()))
        for i in range(H * W):
            if fac[i] < 0:
                fac[i] = next(rest)
        return fac

    adj, anti = fill({0: 0, 1: 1}), fill({0: 0, H * W - 1: 1})
    real = referee_paired_arrays(X, y, adj, anti, H, W, radius=1, K=K)
    ctrl = referee_paired_arrays(X, y, adj, anti, H, W, radius=0, K=K)
    ok = (real["mean_diff"] >= min_gap and real["ci"][0] > 0
          and abs(ctrl["mean_diff"]) < 0.02)
    return {"ok": ok, "referee": real, "pixels_only_control": ctrl}


def blindness_gate(sur, F, H, W, pairs, trials=200, seed=0, z_max=2.0):
    """GATE 2 (blind levels) — the surrogate must NOT distinguish an
    interaction-optimal layout (pairs adjacent) from random placements.
    v1 would have FAILED this gate loudly; that is the point."""
    rng = np.random.default_rng(seed)
    t_opt = float(sur.cell_scores(oracle_fac(F, H, W, pairs), H, W).sum())
    rand = np.array([float(sur.cell_scores(rng.permutation(H * W),
                                           H, W).sum())
                     for _ in range(trials)])
    mu, sd = float(rand.mean()), float(rand.std() + 1e-12)
    z = (t_opt - mu) / sd
    return {"ok": z <= z_max, "oracle": t_opt, "rand_mu": mu,
            "rand_sd": sd, "z": float(z)}


def reveal_gate(sur, F, H, W, pairs, trials=200, seed=0, z_min=3.0):
    """Mirror of blindness_gate for 'reveal' levels: the tutorial surrogate
    MUST point at the true pairs (otherwise the tutorial teaches nothing)."""
    g = blindness_gate(sur, F, H, W, pairs, trials=trials, seed=seed)
    g["ok"] = g["z"] >= z_min
    return g


def level_sensitivity_gate(lvl, min_gap=0.05, K=5):
    """GATE 3 (per level) — the level's full layout lever must move accuracy:
    referee(ORACLE: every hidden pair adjacent) vs referee(WORST: every pair
    maximally separated), paired folds on the level's own data.  Linear info
    is identical in both layouts (all F features placed), so the paired
    difference isolates interaction adjacency — exactly what the player buys
    with a good layout.

    v2.1 fix: the previous version contrasted ONE pair of a multi-pair level
    with a threshold calibrated on single-pair XOR.  pairs_16 failed with
    d=+0.034 (CI +0.021..+0.047) — real, but ~4x diluted by construction and
    sitting on a strong additive baseline.  The gate now measures the whole
    lever; 5pp is the bar for 'this level's layout matters'."""
    H, W, F = lvl["H"], lvl["W"], lvl["F"]
    orc = oracle_fac(F, H, W, lvl["pairs"])
    wrs = worst_fac(F, H, W, lvl["pairs"])
    real = referee_paired_arrays(lvl["X"], lvl["y"], orc, wrs, H, W,
                                 radius=1, K=K)
    ok = (real["mean_diff"] >= min_gap and real["ci"][0] > 0)
    return {"ok": ok, "diff": real, "oracle_fac": orc, "worst_fac": wrs,
            "min_gap": min_gap}


def interaction_budget(X, y, K=5, max_pairs=200, seed=0):
    """GATE 4 — upper bound on layout signal: CV gap between an additive
    model and one handed the top pairwise products regardless of layout."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    folds = stratified_folds(y, K, seed)
    classes = np.unique(y)
    F = X.shape[1]
    scored = []
    for i in range(F):
        for j in range(i + 1, F):
            p = X[:, i] * X[:, j]
            if p.std() < 1e-12:
                continue
            c = np.corrcoef(p, y)[0, 1]
            scored.append((0.0 if np.isnan(c) else abs(c), i, j))
    scored.sort(reverse=True)
    keep = scored[:max_pairs]

    def cv(with_products):
        accs = []
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Ztr, Zva = (X[tr] - mu) / sd, (X[va] - mu) / sd
            ctr, cva = [Ztr], [Zva]
            if with_products:
                for _, i, j in keep:
                    ctr.append((Ztr[:, i] * Ztr[:, j])[:, None])
                    cva.append((Zva[:, i] * Zva[:, j])[:, None])
            Ytr = np.stack([(y[tr] == c).astype(np.float32)
                            for c in classes], 1)
            accs.append(_acc(np.hstack(cva), y[va],
                             _fit_softmax(np.hstack(ctr), Ytr), classes))
        return float(np.mean(accs))

    a0, a1 = cv(False), cv(True)
    gap = a1 - a0
    verdict = ("game-worthy" if gap >= 0.03 else
               "weak — sandbox only" if gap >= 0.01 else
               "no layout signal — keep out of the rotation")
    return {"additive": a0, "top_products": a1, "gap": gap,
            "verdict": verdict}


# ------------------------------------------------------------------ levels
def generate_level(tag):
    """Deterministic level generation (identical to the baked npz)."""
    cfg = LEVELS[tag]
    if tag == "pairs_16":
        X, y, pairs = make_game_data(n=cfg["n"], F=cfg["F"],
                                     n_syn=cfg["n_syn"], seed=cfg["seed"])
    elif tag == "xor_16":
        X, y, pairs = make_xor_data(n=cfg["n"], F=cfg["F"],
                                    seed=cfg["seed"])
    else:
        raise KeyError(tag)
    idx = _surrogate_indices(X, y, cfg["seed"])
    sur = Surrogate(X[idx], y[idx],
                    mode="blind" if cfg["gated"] else "reveal")
    return {"X": X.astype(np.float32), "y": np.asarray(y, np.int64),
            "sur": sur, "H": cfg["H"], "W": cfg["W"], "F": cfg["F"],
            "version": cfg["seed"], "gated": cfg["gated"],
            "credits": cfg["credits"], "pairs": pairs,
            "name": cfg["name"], "hint": cfg["hint"]}


def load_level(tag):
    """Baked npz if present (fast, byte-identical across devices), else
    deterministic generation (the numbers are identical either way)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"level_{tag}.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=False)
        if int(d["fmt"]) == LEVEL_FMT:
            return {"X": d["X"], "y": d["y"],
                    "sur": Surrogate.from_stats(d["imp"], d["syn"]),
                    "H": int(d["H"]), "W": int(d["W"]), "F": int(d["F"]),
                    "version": int(d["version"]), "gated": bool(d["gated"]),
                    "credits": int(d["credits"]),
                    "pairs": json.loads(str(d["pairs"])),
                    "name": str(d["name"]), "hint": str(d["hint"])}
        # stale format -> regenerate (bake_assets.py re-bakes in CI)
    return generate_level(tag)


if __name__ == "__main__":
    print("=== invariance self-test (pixels-only must be IDENTICAL) ===")
    lvl = generate_level("pairs_16")
    trX, trY, vaX, vaY = split_scale(lvl["X"], lvl["y"], seed=1)
    inv = invariance_check(trX, trY, vaX, vaY, lvl["F"], lvl["H"], lvl["W"])
    print("pixels-only  :", "  ".join(f"{a:.6f}" for a in inv["accs"]),
          " <- must be identical", "OK" if inv["ok"] else "FAIL")
    rng = np.random.default_rng(1)
    a1 = [accuracy(trX, trY, vaX, vaY, rng.permutation(lvl["F"]),
                   lvl["H"], lvl["W"])[0] for _ in range(3)]
    print("with products:", "  ".join(f"{a:.4f}" for a in a1),
          " <- must spread")
    print("\n=== scorer gate ===")
    g = scorer_gate()
    print(f"  control d(acc) = {g['pixels_only_control']['mean_diff']:+.4f}"
          "  (must be ~0)")
    print(f"  referee  d(acc) = {g['referee']['mean_diff']:+.3f}"
          f"  CI {g['referee']['ci'][0]:+.3f}..{g['referee']['ci'][1]:+.3f}")
    print(f"  SCORER GATE: {'PASS' if g['ok'] else 'FAIL'}")
    print("\n=== blindness gate (xor_16: v1 would FAIL this) ===")
    x = generate_level("xor_16")
    bg = blindness_gate(x["sur"], x["F"], x["H"], x["W"], x["pairs"])
    print(f"  oracle z = {bg['z']:+.2f}  (must be <= 2)  "
          f"{'PASS' if bg['ok'] else 'FAIL — surrogate leaks!'}")
