"""RECKON — where the cursor is, what has been placed, and which steps are dead.

Two independent accounts of the cursor are kept and compared every decision.

  * The BOOK. The spawn cell is fixed, every returned step shifts the cursor by one cell, and a
    step into a wall leaves it exactly where it was. Replaying the harness's own returned
    actions through that rule therefore reproduces the cursor without reading anything.
  * The PICTURE. The cursor cube is the only thing drawn in the highlight ink, and a cube's
    silhouette in this projection is a hexagon whose position is a bijection of the cell. Taking
    the highlight pixels and asking which cell's hexagon CONTAINS them all — and demanding that
    exactly one cell does — reads the cursor straight off the frame.

The picture wins when it speaks, because it is immune to a mis-modelled clamp; the book carries
the state when the highlight is occluded by the placed cubes and no unique hexagon covers it.
Disagreements are counted rather than hidden, so a drifting model shows up in the trace instead
of silently corrupting the route.

The placed set is booked, never counted off the picture: the isometric scene self-occludes, so a
new cube can hide more painted surface than it adds, and a pixel census of the placed ink is not
a sound witness of a placement. A toggle is its own inverse, so the book is exact: the grid
starts empty and only this harness's own returned toggles ever change it.
"""

from tools.keys import AXIS_NAMES, STEP, TOGGLE

GRID_ORDER = 6
SPAWN = (2, 2, 0)
WALLET = 50
HIGHLIGHT_INK = (100, 200, 250)
MASK_X = 300
ORIGIN_X = 640
ORIGIN_Y = 280
BLOCK = 40
SQUASH = 0.85
SLOPE = 0.5
SAMPLE_CAP = 48
SLACK_PX = 4
HEX_SIDES = 6
RECENT_KEEP = 8

_flat = lambda a, b: (ORIGIN_X + (a - SQUASH * b) * BLOCK, ORIGIN_Y + (a + SQUASH * b) * BLOCK * SLOPE)
_at = lambda c, i, j, k: (_flat(c[0] + i, c[1] + j)[0], _flat(c[0] + i, c[1] + j)[1] - (c[2] + k) * BLOCK)
_hexagon = lambda c: [_at(c, 0, 0, 1), _at(c, 1, 0, 1), _at(c, 1, 0, 0),
                      _at(c, 1, 1, 0), _at(c, 0, 1, 0), _at(c, 0, 1, 1)]
_side_of = lambda p, q, r: (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
_inside = lambda poly, r: all(_side_of(poly[i], poly[(i + 1) % HEX_SIDES], r) >= -SLACK_PX * BLOCK
                              for i in range(HEX_SIDES))
_penned = lambda c: tuple(min(GRID_ORDER - 1, max(0, v)) for v in c)
_gap = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
_shift = lambda c, d: _penned(tuple(a + b for a, b in zip(c, d)))
_cells = [(x, y, z) for x in range(GRID_ORDER) for y in range(GRID_ORDER) for z in range(GRID_ORDER)]
_lit_ink = lambda px, w, h: [(x, y) for x in range(MASK_X, w) for y in range(h)
                             if px[x, y] == HIGHLIGHT_INK]
_covering = lambda probes: [c for c in _cells if all(_inside(_hexagon(c), r) for r in probes)]
_leg = lambda src, dst, axis: [AXIS_NAMES[axis][0] if dst[axis] > src[axis] else AXIS_NAMES[axis][1]
                               ] * abs(dst[axis] - src[axis])
_walk = lambda src, dst: _leg(src, dst, 0) + _leg(src, dst, 1) + _leg(src, dst, 2)


def clamped_at(cell, name):
    """True when this step would leave the cursor where it is — a provably wasted decision."""
    shift = STEP.get(name)
    return shift is not None and _shift(cell, shift) == cell


def route_names(src, dst):
    """A shortest step sequence from one cell to another, axis by axis in a fixed order."""
    return _walk(src, dst)


def cursor_cell(px, w, h, cap=48):
    """Read the cursor off the frame by hexagon containment, or None when it is not unique.

    The highlight ink also paints one panel caption, so the census is restricted to the scene
    side of the screen. A sample of the lit pixels is enough: containment is a conjunction, so
    any point that falls outside a candidate hexagon already eliminates it. Demanding that the
    cover be UNIQUE is what turns the readout into a measurement: a highlight half-hidden behind
    a placed cube is covered by several cells, and the honest answer there is no answer at all.
    """
    if w <= MASK_X or h < 1:
        return None                                    # not the screen this readout was fitted to
    lit = _lit_ink(px, w, h)
    if not lit:
        return None
    found = _covering(lit[::len(lit) // cap + 1])
    return found[0] if len(found) == 1 else None


class Ledger:
    """The episode's self-derived account: cursor, placed set, budget and the step audit."""

    def __init__(self):
        self.cursor = SPAWN
        self.placed = set()
        self.lifted = set()
        self.spent = 0
        self.recent = []
        self.nulls = 0
        self.cancels = 0
        self.corrections = 0

    def left(self):
        return max(0, WALLET - self.spent)

    def dead_steps(self):
        return [n for n in sorted(STEP) if clamped_at(self.cursor, n)]

    def live_steps(self):
        return [n for n in sorted(STEP) if not clamped_at(self.cursor, n)]

    def closer(self, name, goal):
        """Does this step strictly shorten the walk to the goal cell?"""
        shift = STEP.get(name)
        if shift is None or goal is None:
            return False
        nxt = _penned(tuple(c + s for c, s in zip(self.cursor, shift)))
        return _gap(nxt, goal) < _gap(self.cursor, goal)

    def sight(self, seen):
        """Adopt a cursor the picture reports, booking the disagreement when there is one."""
        if seen is not None and seen != self.cursor:
            self.corrections += 1
            self.cursor = seen
        return self.cursor

    def book(self, name):
        """Advance the account by the action that is about to be returned to the engine."""
        if name in STEP:
            if clamped_at(self.cursor, name):
                self.nulls += 1
            else:
                self.cursor = _penned(tuple(c + s for c, s in zip(self.cursor, STEP[name])))
        elif name == TOGGLE:
            if self.cursor in self.placed:
                self.placed.discard(self.cursor)
                self.lifted.add(self.cursor)
                self.cancels += 1
            else:
                self.placed.add(self.cursor)
        self.spent += 1
        self.recent.append(name)
        del self.recent[:-RECENT_KEEP]
