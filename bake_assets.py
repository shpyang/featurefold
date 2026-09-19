"""bake_assets.py — deterministic level baking + ALL gates.  v2
CI USAGE: python bake_assets.py        (bake + run every gate; exit 1 on fail)
          python bake_assets.py --verify   (reload npz + re-run gates only)

Gates per level:
  G1 scorer_gate           referee separates adjacent/anti XOR pair (r=0 ctrl flat)
  G2 blindness_gate        blind surrogate cannot find interaction pairs
     reveal_gate           ... or, for 'reveal' levels, MUST find them
  G3 level_sensitivity_gate  paired referee diff on the LEVEL's own data
  G4 interaction_budget    upper bound on layout signal (gap >= 0.02)
  G5 invariance_check      pixels-only referee is layout-invariant
"""
import json
import sys

import numpy as np

from gamescore import (LEVEL_TAGS, LEVELS, LEVEL_FMT, generate_level,
                       load_level, scorer_gate, blindness_gate, reveal_gate,
                       level_sensitivity_gate, interaction_budget,
                       invariance_check, split_scale)

HARD = {"scorer", "blindness", "reveal", "sensitivity", "invariance"}


def _path(tag):
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"level_{tag}.npz")


def bake(tag, verbose=True):
    lvl = generate_level(tag)
    fails = []

    g1 = scorer_gate()
    if verbose:
        print(f"  G1 scorer_gate   : {'PASS' if g1['ok'] else 'FAIL'}"
              f"  (referee d={g1['referee']['mean_diff']:+.3f}, "
              f"control d={g1['pixels_only_control']['mean_diff']:+.4f})")
    if not g1["ok"]:
        fails.append("scorer")

    if lvl["gated"]:
        g2 = blindness_gate(lvl["sur"], lvl["F"], lvl["H"], lvl["W"],
                            lvl["pairs"])
        name = "G2 blindness"
    else:
        g2 = reveal_gate(lvl["sur"], lvl["F"], lvl["H"], lvl["W"],
                         lvl["pairs"])
        name = "G2 reveal"
    if verbose:
        print(f"  {name:<16s}: {'PASS' if g2['ok'] else 'FAIL'}"
              f"  (z={g2['z']:+.2f})")
    if not g2["ok"]:
        fails.append(name.split()[1])

    g3 = level_sensitivity_gate(lvl)
    if verbose:
        r = g3["diff"]
        print(f"  G3 sensitivity   : {'PASS' if g3['ok'] else 'FAIL'}"
              f"  (paired d={r['mean_diff']:+.3f}, "
              f"CI {r['ci'][0]:+.3f}..{r['ci'][1]:+.3f})")
    if not g3["ok"]:
        fails.append("sensitivity")

    g4 = interaction_budget(lvl["X"], lvl["y"])
    if verbose:
        print(f"  G4 inter-budget  : gap={g4['gap']:+.3f} -> {g4['verdict']}")
    if g4["gap"] < 0.02:
        fails.append("interaction_budget")

    trX, trY, vaX, vaY = split_scale(lvl["X"], lvl["y"],
                                     seed=lvl["version"])
    g5 = invariance_check(trX, trY, vaX, vaY, lvl["F"], lvl["H"], lvl["W"])
    if verbose:
        print(f"  G5 invariance    : {'PASS' if g5['ok'] else 'FAIL'}"
              f"  ({g5['accs']})")
    if not g5["ok"]:
        fails.append("invariance")

    # ---- bake ----
    cfg = LEVELS[tag]
    np.savez_compressed(
        _path(tag),
        fmt=np.int64(LEVEL_FMT), version=np.int64(lvl["version"]),
        X=lvl["X"], y=lvl["y"], imp=lvl["sur"].imp, syn=lvl["sur"].syn,
        H=np.int64(lvl["H"]), W=np.int64(lvl["W"]), F=np.int64(lvl["F"]),
        gated=np.bool_(lvl["gated"]), credits=np.int64(lvl["credits"]),
        pairs=np.array(json.dumps(lvl["pairs"])),
        name=np.array(cfg["name"]), hint=np.array(cfg["hint"]))
    print(f"  baked {_path(tag)}  "
          f"{'OK' if not fails else 'GATES FAILED: ' + ', '.join(fails)}")
    return fails


def verify(tag):
    """Reload the npz and re-run the cheap gates against what was baked."""
    lvl = load_level(tag)
    assert int(np.asarray(lvl["version"])) == LEVELS[tag]["seed"], \
        f"{tag}: baked version != LEVELS seed — bump or re-bake"
    fails = []
    if lvl["gated"]:
        g = blindness_gate(lvl["sur"], lvl["F"], lvl["H"], lvl["W"],
                           lvl["pairs"])
        if not g["ok"]:
            fails.append("blindness")
        print(f"  [verify] {tag}: blindness z={g['z']:+.2f} "
              f"{'OK' if not fails else 'FAIL'}")
    g3 = level_sensitivity_gate(lvl)
    if not g3["ok"]:
        fails.append("sensitivity")
    print(f"  [verify] {tag}: sensitivity d={g3['diff']['mean_diff']:+.3f} "
          f"{'OK' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return fails


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bake"
    all_fails = []
    for tag in LEVEL_TAGS:
        print(f"[{tag}]")
        all_fails += verify(tag) if mode == "--verify" else bake(tag)
    if all_fails:
        print(f"\nGATE FAILURES: {all_fails}")
        sys.exit(1)
    print("\nAll gates passed.")
