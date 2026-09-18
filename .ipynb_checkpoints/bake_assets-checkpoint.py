"""bake_assets.py — bake the shipped level files, then gate the shipped artifact.
Run:  python bake_assets.py"""
import os
import numpy as np
import gamescore
from gamescore import (make_game_data, make_xor_data, split_scale,
                       Surrogate, accuracy, load_level)

HERE = os.path.dirname(os.path.abspath(gamescore.__file__))

def place_features(fixed):                       # fixed: {cell: feature_id}
    fac = -np.ones(16, int)
    for cell, f in fixed.items(): fac[cell] = f
    pool = [f for f in range(16) if f not in set(fixed.values())]
    it = iter(pool)
    for i in range(16):
        if fac[i] < 0: fac[i] = next(it)
    return fac

def domino_layouts(pairs):
    """best: every planted pair side-by-side (cells 4k, 4k+1)
    worst: every planted pair split across its row (cells 4k, 4k+3).
    Exactly 4 synergy pairs co-visible vs exactly 0 — the level's full skill range."""
    best, worst = -np.ones(16, int), -np.ones(16, int)
    pool = [f for f in range(16) if f not in {x for p in pairs for x in p}]
    for k, (a, b) in enumerate(pairs):
        best[4*k], best[4*k+1] = a, b
        worst[4*k], worst[4*k+3] = a, b
    bi = wi = 0
    for i in range(16):
        if best[i] < 0:  best[i]  = pool[bi]; bi += 1
        if worst[i] < 0: worst[i] = pool[wi]; wi += 1
    return best, worst

def bake(tag, X, y, pairs=None):
    trX, trY, vaX, vaY = split_scale(X, y)
    sur = Surrogate(trX, trY)
    payload = dict(trX=trX, trY=trY, vaX=vaX, vaY=vaY, imp=sur.imp, syn=sur.syn)
    if pairs is not None: payload["pairs"] = np.array(pairs)   # gate reads these back
    np.savez_compressed(os.path.join(HERE, f"level_{tag}.npz"), **payload)
    print(f"level_{tag}.npz  train={trX.shape}  val={vaX.shape}")

def make_icons():
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        raise SystemExit("matplotlib needed once for icon.png/presplash.png "
                         "(buildozer requires them); pip install matplotlib")
    for name, px in (("icon.png", 512), ("presplash.png", 1024)):
        fig, ax = plt.subplots(figsize=(px/100, px/100), dpi=100)
        fig.patch.set_facecolor("#0d0f16"); ax.set_facecolor("#0d0f16")
        ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.axis("off")
        hot = {(0, 0): "#3fb950", (0, 1): "#3fb950"}
        for r in range(4):
            for c in range(4):
                ax.add_patch(Rectangle((c+0.06, 3-r+0.06), 0.88, 0.88,
                             fc=hot.get((r, c), "#232735"), ec="#3a3f52"))
        fig.savefig(name, dpi=100); plt.close(fig); print(name)

if __name__ == "__main__":
    X, y, pairs = make_game_data()                 # seed 42 — the shipped level
    bake("pairs_16", X, y, pairs=pairs)
    bake("xor_16", *make_xor_data()[:2])
    make_icons()

    # Gate A — scorer sensitivity, certified on the shipped XOR artifact
    trX, trY, vaX, vaY, _ = load_level("xor_16")
    a_adj  = accuracy(trX, trY, vaX, vaY, place_features({0: 0, 1: 1}), 4, 4)[0]
    a_anti = accuracy(trX, trY, vaX, vaY, place_features({0: 0, 15: 1}), 4, 4)[0]
    print(f"\nGate A (XOR):  adjacent {a_adj:.3f} vs scattered {a_anti:.3f} "
          f"-> gap {a_adj - a_anti:+.3f}   [need >= 0.10]")
    assert a_adj - a_anti >= 0.10, "scorer cannot see an adjacent XOR pair — do not ship"

    # Gate B — the pairs level's skill gradient, certified on its own artifact
    trX, trY, vaX, vaY, _ = load_level("pairs_16")
    best, worst = domino_layouts(pairs)
    a_best  = accuracy(trX, trY, vaX, vaY, best, 4, 4)[0]
    a_worst = accuracy(trX, trY, vaX, vaY, worst, 4, 4)[0]
    print(f"Gate B (pairs): domino {a_best:.3f} vs split {a_worst:.3f} "
          f"-> gap {a_best - a_worst:+.3f}   [need >= 0.015]")
    assert a_best - a_worst >= 0.015, "pairs level has no skill gradient — rebalance"
    print("\nGATES PASS — assets certified to ship.")