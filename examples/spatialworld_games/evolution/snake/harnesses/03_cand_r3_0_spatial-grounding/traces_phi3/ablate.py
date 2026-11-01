"""ablate.py — knock out one piece of the warrant at a time, same rig, same 20 seeds.

Each arm patches ONLY module-scope state in tools/vlm.py, so the components under test are the
shipped ones. Ask model is the live measured mix; the appeal is NOT heeded (the honest worst case)
except in the row that prices the appeal itself.
"""
import os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warrant_rig as R          # noqa: E402
from tools import vlm            # noqa: E402

BASE = {k: getattr(vlm, k) for k in
        ("_pellet_cell", "_FITS", "_OCCUPIED", "_WARRANT", "_SATISFIES", "_TOP", "_ROOM",
         "_SEAT", "_WARM", "_SPREAD")}


def arm(name, heed=False, **patch):
    for k, v in BASE.items():
        setattr(vlm, k, v)
    for k, v in patch.items():
        setattr(vlm, k, v)
    st = {"head_n": 0, "head_ok": 0, "hud_n": 0, "hud_ok": 0, "empty_warrant": 0,
          "food_n": Counter(), "food_ok": Counter()}
    rows = [R.run(s, "mix", heed, st) for s in range(1, 21)]
    prim = sum(min(r["score"] / 4.0, 1.0) for r in rows) / len(rows)
    dec = sum(r["decisions"] for r in rows)
    hon = sum(r["honoured"] for r in rows)
    print(f"{name:<34} primary={prim:.4f} deaths={sum(r['dead'] for r in rows):>2}/20 "
          f"budget_out={sum(r['budget'] for r in rows):>2}/20 "
          f"pellets={sum(r['score'] for r in rows):>3} decisions={dec:>4} "
          f"honour={hon/max(dec,1):.3f} appeals={sum(r['appeals'] for r in rows)} "
          f"won_back={sum(r['won_back'] for r in rows):>3} "
          f"decode_d1={st['food_ok'][1]}/{st['food_n'][1]} "
          f"decode_all={sum(st['food_ok'].values())}/{sum(st['food_n'].values())}")
    for k, v in BASE.items():
        setattr(vlm, k, v)


if __name__ == "__main__":
    arm("SHIPPED (appeal ignored)")
    arm("SHIPPED (appeal heeded)", heed=True)
    # no warrant at all: whatever the model names is pressed -- this is the parent's behaviour
    arm("-warrant  (press the ask)", _SATISFIES=lambda led, n, p, a, b: True)
    # clamp only: legality kept, the pellet reading thrown away
    arm("-pellet   (clamp only)", _pellet_cell=lambda img: None)
    # the refuted symmetric size gate, round 2's version
    arm("-onesided (symmetric gate)", _FITS=lambda sp: abs(sp - vlm.DRAWN_D) <= 4.0)
    # the siblings' guess: a fixed 6-cell body window instead of the HUD digit
    arm("-hud      (fixed 6 window)", _OCCUPIED=lambda trail, length: frozenset(trail[:6][:-1]))
    # drop the one-ply room tie-break, keep L1 + least-trodden + menu order
    arm("-room     (no continuations)", _ROOM=lambda cell, blocked: 0)
    # never re-ask: the text channel is closed, so the harness substitutes straight away
    arm("-appeal   (no second chance)", _SEAT=lambda talk: None)
    arm("-appeal   (heeded arm, closed)", heed=True, _SEAT=lambda talk: None)
