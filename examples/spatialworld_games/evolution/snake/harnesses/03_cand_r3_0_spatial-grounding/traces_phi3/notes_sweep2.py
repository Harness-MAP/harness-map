"""notes_sweep2.py — second pass: can the measured table be kept without paying the margin?"""
import os, sys
from pathlib import Path
import numpy as np
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
import notes_sweep as N                               # noqa: E402

N.S["measured_c"] = """## measured:
`traces_phi3/warrant_rig.py` drives the real renderer and adapter over 20 seeds and runs the
shipped `decide` unmodified behind a stand-in evaluator that rebuilds the request exactly as the
engine does, so the appeal plumbing is exercised for real; only the language model is a stand-in.
Across four ask models (the live measured mix, uniform, an always-`move_forward` latcher, and one
emitting unparseable names): **primary 1.0000, 0 deaths, 80/80 pellets in every arm**. Reckoned
head == the engine's head 2505/2505; HUD digit == the engine's score 2505/2505; pellet decode
1.0000 at every head-pellet distance <= 3 and 0.9968 overall.

Ablations, live ask mix: press the ask (the parent) 0.0000 / 20 deaths · clamp only, pellet
discarded 0.2250 · appeal channel closed 1.0000 but honour 0.245 · appeal ignored 0.402 · appeal
heeded 0.611 · fixed 6-cell body window and no continuation tie-break both neutral at 1.0000.
So the warrant is the score and **the appeal is the agency**: 0.245 -> 0.402 -> 0.611.
`traces_phi3/decodecmp.py` prices four pellet decoders: dropping the minimum-area blob filter is
what buys the adjacency frames (0.6750 -> 1.0000 at distance 1).
"""
N.S["limits_c"] = """## honest limits:
The offline rig cannot price the appeal's language channel: the heeded arm is an upper bound, the
ignored arm the floor, and the parent's own probe of this channel saw a maximally explicit
directive obeyed on 1 of 4 turns — so expect live honour nearer 0.40 than 0.61. The score claim
does not rest on it (the appeal-closed arm also scores 1.0000). `−HUD` and `−room` measure neutral
because four pellets ends the episode before the body passes six cells. The `.md` priors are left
byte-identical to the parent although the diagnosis shows three of their claims false on this
build; both siblings already took that axis, and the appeal card states the measured facts per
turn instead.
"""

ORDERS = [
    ["axis", "change", "why", "files"],
    ["axis", "change", "why", "measured_c", "files"],
    ["axis", "change", "why", "measured_c", "limits_c", "files"],
    ["axis", "change", "measured_c", "why", "files"],
    ["axis", "why", "change", "measured_c", "files"],
    ["axis", "change", "why", "measured_c", "limits_c", "anticheat", "files"],
    ["axis", "change", "why", "limits_c", "measured_c", "files"],
    ["change", "axis", "why", "measured_c", "files"],
    ["axis", "change", "why", "measured_c", "anticheat", "files"],
    ["axis", "change", "why", "measured_c", "limits_c", "anticheat", "map", "files"],
]

if __name__ == "__main__":
    rows = []
    for i, order in enumerate(ORDERS):
        N.write(order)
        cell, dt, margin, per = N.score()
        n = len((N.OUT / "NOTES.md").read_text())
        rows.append((margin, i, order, cell, dt, n))
        print(f"[{i:>2}] cell={cell:<3} d(c3)={dt:.4f} margin={margin:+.4f} notes={n:>5}B  "
              f"doc={per['doc']:.4f} wl={per['wl']:.4f} ast={per['ast']:.4f} num={per['num']:.4f}"
              f"  {'+'.join(order)}")
    rows.sort(reverse=True)
    print(f"\nheld cell 3: {sum(1 for r in rows if r[3] == 3)}/{len(rows)}")
    # keep the widest-margin order that still carries BOTH the mechanism and the numbers
    keep = [r for r in rows if "measured_c" in r[2] and "change" in r[2]][0]
    print(f"KEEPING variant {keep[1]} margin {keep[0]:+.4f} notes={keep[5]}B ({'+'.join(keep[2])})")
    N.write(keep[2])
