"""THE TAPE — turning the measured envelope into an ordered press list, and pricing a proposal
against it.

The plan is one sentence: **lay the proof first, then lift the volume slice by slice, and inside a
slice lay its core before its rim.** Everything else in this module is the arithmetic of that
sentence.

Why those keys and no others:

* The build is APPEND-ONLY, so the adapter re-tests the placed set for equality inside every
  executed press and the episode ends the instant they match. The episode therefore wins iff the
  tape puts every true cell before every surplus one — a PREFIX property — and each press after
  the first member of the family is a free test of one new structure.
* FORCED cells are a proof (`tools/optics.py::forced`), so fronting them costs nothing.
* z-ASCENDING is the slice order; the structure is built from the ground up because the panels
  constrain a layer's row and column support exactly and the coupling between layers is the one
  thing a slice-local read cannot see.
* CORE-BEFORE-RIM inside a slice is the only prior in the harness, and the rim is measured from
  the ADMITTED VOLUME'S OWN CENTROID rather than from the grid centre. That distinction is
  load-bearing: the surplus a three-panel read invents sits on the far side of the volume it
  admitted, which is not in general the far side of the grid.
* TRAVEL breaks ties and does nothing else. It is the only compressible cost in the episode — a
  move and a press are the same price — so a nearest-first walk inside a tier is free ordering.

`closes_gap` is the whole arbitration rule. A move that strictly reduces the L1 distance to the
current goal lies on SOME shortest route, so honouring it leaves the remaining tape exactly one
press shorter whichever closing axis is taken: free-or-cheaper rungs are honoured, and nothing
else is. `advance` dead-reckons under the adapter's own clamping, so a press into a grid face
leaves the cursor exactly where it was — which is what makes a frozen frame a position statement
rather than a bookkeeping error.
"""
from __future__ import annotations

from operator import sub

from .action import AXIS, STEP, TOGGLE, _CLAMP
from .optics import forced, trimmable

_RIM = lambda c, mid: abs(c[0] - mid[0]) + abs(c[1] - mid[1])
_MID = lambda part: (sum(c[0] for c in part) / len(part), sum(c[1] for c in part) / len(part))
_HEIGHTS = lambda cells: sorted({c[2] for c in cells})


def l1(a, b):
    """Manhattan distance — the exact press cost of a move leg, since every move is one cell."""
    return sum(map(abs, map(sub, a, b)))


def build_tape(cells, panels, cursor, rim=2.0):
    """The build order: proof first, then slice by slice upward, core before rim, nearest first.

    `rim` is the L1 radius, measured from the admitted volume's OWN centroid IN THAT LAYER, that
    separates a slice's core from its rim — the one tunable prior in the harness, and it only ever
    ORDERS presses; no cell is ever filtered out of the envelope by it. The layer centroid rather
    than the volume centroid or the grid centre is the load-bearing choice: where three panels
    admit more cells than the target contains, the surplus sits on the outside of the shape THAT
    SLICE admitted, which is not in general the outside of the whole volume."""
    need = forced(cells, panels)
    mids = {z: _MID([c for c in cells if c[2] == z]) for z in _HEIGHTS(cells)}
    tiers = {}
    for c in cells:
        lay = c[2]
        tiers.setdefault((c not in need, lay, _RIM(c, mids[lay]) >= rim), []).append(c)
    tape = []
    at = cursor
    for rank in sorted(tiers):
        pool = list(tiers[rank])
        while pool:
            nxt = min(pool, key=lambda c: (l1(at, c), c))
            pool.remove(nxt)
            tape.append(nxt)
            at = nxt
    return tape


def trim_tape(placed, panels, cursor):
    """The continuation once the envelope is laid and the episode is STILL running: remove cells
    whose removal leaves another member, nearest first. Each costs one press, needs no restore,
    and lands on a structure the picture still allows."""
    live = set(placed)
    out = []
    cand = trimmable(live, panels)
    at = cursor
    while cand:
        nxt = min(cand, key=lambda c: (l1(at, c), c))
        out.append(nxt)
        live.discard(nxt)
        at = nxt
        cand = trimmable(live, panels) - set(out)
    return out


def step_toward(cursor, goal, first=()):
    """One legal unit action that closes the gap, largest remaining axis first (deterministic).

    An axis named in `first` is taken earlier when it also closes the gap. Every axis that closes
    the gap drops the L1 distance by one, so they are all on a shortest route and the preference
    is FREE — it cannot lengthen the tape. The harness passes the axes the projection has not
    measured yet, which finishes calibrating the camera out of legs it was going to walk anyway
    instead of spending a press to go and look."""
    for _pref, _gap, i in sorted(((k in first, abs(goal[k] - cursor[k]), k)
                                  for (_p, _m, k) in AXIS), reverse=True):
        if goal[i] != cursor[i]:
            return AXIS[i][0] if goal[i] > cursor[i] else AXIS[i][1]
    return TOGGLE


def closes_gap(cursor, goal, name):
    """Is `name` a MOVE that strictly reduces the L1 distance to `goal` — i.e. a free leg?"""
    for plus, minus, i in AXIS:
        if name == plus:
            return goal[i] > cursor[i]
        if name == minus:
            return goal[i] < cursor[i]
    return False


def advance(cursor, name):
    """Dead-reckon one applied action under the adapter's own clamping at the grid faces."""
    out = list(cursor)
    for plus, minus, i in AXIS:
        if name == plus:
            out[i] = _CLAMP(out[i] + STEP)
        if name == minus:
            out[i] = _CLAMP(out[i] - STEP)
    return tuple(out)
