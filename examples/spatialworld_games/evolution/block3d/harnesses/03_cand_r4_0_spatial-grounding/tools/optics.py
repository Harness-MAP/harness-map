"""THE RULER AND THE PROJECTION — both MEASURED off the frame, neither asserted.

Two halves, and the second one is this harness's whole thesis.

THE LATTICE, FITTED. The three orthographic target panels are drawn as GxG grids in one exact
constant, so the columns and the rows carrying those lattice lines stand out in a per-axis census
of that constant inside the panel strip. `fit_gauge` reads four things straight out of that
census and asserts none of them: the first line column, the cell pitch (the smallest gap between
distinct line indices), the LINE THICKNESS (the length of the first run of consecutive line
indices), and the three panel tops (a line row that does not follow its predecessor by one pitch
starts a new panel). The fill origin is then `line + thickness` and the tile side is
`pitch - thickness`, both derived rather than guessed. `gauge_audit` hands the fit back to the
frame for confirmation. A decoder that asserts its constants reads a WRONG mask the day the
layout shifts and an EMPTY one the day it rescales; a decoder that fits reports a refusal.

THE PROJECTION, CALIBRATED. The viewport draws cell (x, y, z) as one convex cube footprint, and
when the cursor stands on a cell that footprint is highlighted in a constant that appears nowhere
else on the frame. Two consequences, and together they are the mechanism:

  * the highlight's SHAPE is a template. Taken on the first frame — when nothing is placed, so
    nothing can occlude it — it is exactly one cube's footprint, and every other cube's footprint
    is that same shape translated.
  * the TRANSLATION is affine in the cell, so three independent single-axis displacements
    determine the entire map. The harness already produces displacements: one per applied move.
    The projection is therefore learned from the agent's own actions, with no camera constant
    written down anywhere, and a null displacement (a move that clamped) is dropped instead of
    being written as a zero vector.

`locate` then reads the cursor off any later frame by the one rule that measures clean at every
build density: the live highlight must be CONTAINED in a predicted footprint, and exactly one
footprint may contain it. Occlusion only ever removes highlight pixels, so containment survives a
crowded board where a nearest-centroid readout does not. A tie or a miss is REFUSED — the ledger
then keeps its dead-reckoned cursor and the slack is written to the trace — because a cursor that
is guessed is worse than a cursor that is merely old.
"""
from __future__ import annotations

from .action import SPAWN, X, Y, Z, _AXIS_OF, _SIGN_OF
from .silhouette import G, GRID, INK, PAIRS, VIEW_X0, _EXACT, shadow_of

HILITE = (100, 200, 250)           # exact constant of the cursor highlight
TOPS_HINT = (30, 180, 330)         # the measured panel tops: a BRACKET on the fit, never an input
NUDGE = (-1, 0, 1)                 # the template is dilated by a pixel against centroid rounding
MID = 1                            # index of the centroid inside a `spot` reading
ROW, COL = 0, 1                    # the two screen axes, in the order `spot` reports them

CELLS = tuple((x, y, z) for x in range(G) for y in range(G) for z in range(G))
_INKED = lambda pix: _EXACT(pix, GRID) | _EXACT(pix, INK)
_COVER = lambda pts, ctr, stamp: sum(((round(p[0] - ctr[0]), round(p[1] - ctr[1])) in stamp)
                                     for p in pts)
_WHOLE = lambda hits, k, ok: [c for n, c in hits if ok and n == k]
_FULL = lambda g, seen: seen and len(seen[0]) == g.whole
_AXIS_STEP = lambda g, k, j: (g.basis.get(k) or (0.0, 0.0))[j]


class Gauge:
    """The frame's own ruler plus the projection this harness calibrates from its own moves.

    The lattice half (`x0`, `span`, `pitch`, `tile`, `tops`) is what the silhouette decoder
    indexes with, and every one of those five numbers is a measurement of the frame rather than a
    constant. The projection half (`anchor`, `stamp`, `whole`, `basis`) is accumulated during
    play and is used only once `ready` — a template taken, an anchor fixed, and all three axes
    measured — is true. `slack` carries the last readout's margin over its runner-up, for the
    trace and for `locate`'s own uniqueness test.
    """

    __slots__ = ("x0", "span", "pitch", "tile", "tops", "anchor", "stamp", "whole", "basis",
                 "slack")

    def __init__(self, x0, span, tops):
        self.x0 = x0
        self.span = span
        self.pitch = span // G
        self.tile = span // G - 1
        self.tops = tuple(tops)
        self.anchor = None
        self.stamp = None
        self.whole = None
        self.basis = {}
        self.slack = 0

    def spot(self, pix, least=5):
        """The live highlight as (pixel set, centroid) in (row, col), or None when not drawn.

        `least` is the floor below which a patch is not a cube face but noise: a footprint is
        three figures of pixels, so anything smaller is refused rather than averaged."""
        hot = _EXACT(pix[:, VIEW_X0:], HILITE)
        rs, cs = hot.nonzero()
        n = len(rs)
        if n < least:
            return None
        return (frozenset(zip(rs.tolist(), (cs + VIEW_X0).tolist())),
                (rs.sum() / n, cs.sum() / n + VIEW_X0))

    def imprint(self, seen):
        """Adopt the FIRST highlight as the footprint template and its centroid as the anchor.

        Its pixel count is kept too, because it is the test for whether a LATER highlight is
        whole: the template was taken on an empty board, so anything smaller later is a highlight
        a nearer cube is standing in front of."""
        pts, mid = seen
        self.anchor = mid
        self.whole = len(pts)
        self.stamp = frozenset((round(p[0] - mid[0]) + a, round(p[1] - mid[1]) + b)
                               for p in pts for a in NUDGE for b in NUDGE)

    def at(self, cell):
        """Where the highlight centroid must sit if the cursor stands on `cell`."""
        out = list(self.anchor)
        for i, vec in self.basis.items():
            out[0] += vec[0] * (cell[i] - SPAWN[i])
            out[1] += vec[1] * (cell[i] - SPAWN[i])
        return out

    def learn(self, name, before, after):
        """Record one measured axis displacement, per +1 cell of that axis.

        TWO kinds of displacement are dropped rather than written in, and both of them would
        poison the map for the rest of the episode. A null teaches nothing — a clamped move leaves
        the picture byte-identical. And a displacement measured between highlights that are not
        WHOLE teaches the wrong thing: occlusion moves a centroid by an amount that has nothing to
        do with the grid, and a basis vector learned from a clipped sighting makes every later
        readout confidently wrong. Measured: without this test the readout corrupts the ledger on
        two of twenty levels, both of which the harness otherwise wins."""
        axis = _AXIS_OF(name)
        if axis is None or not _FULL(self, before) or not _FULL(self, after):
            return False
        step = ((after[MID][0] - before[MID][0]) / _SIGN_OF(name),
                (after[MID][1] - before[MID][1]) / _SIGN_OF(name))
        if abs(step[0]) + abs(step[1]) < 1:
            return False
        self.basis[axis] = step
        return True

    def ready(self):
        """Is the projection fully determined: a template, an anchor, and all three axes in?"""
        return all((self.anchor, self.stamp, len(self.basis) == len(SPAWN)))

    def locate(self, seen, margin=4):
        """The cursor cell: the UNIQUE cell whose predicted footprint CONTAINS the whole highlight
        and beats the runner-up by `margin` pixels. Anything else is a REFUSAL.

        Containment is strict: every live highlight pixel must land inside the predicted
        footprint, which is the template dilated by one pixel so a fractional centroid cannot
        fail a true match on rounding alone. Loosening it instead — allowing a few pixels to miss
        — is what makes a clipped highlight land uniquely inside the WRONG footprint, and a wrong
        readout adopted into the ledger is worse than no readout at all: the plan then walks to
        the wrong cell, presses into a face, and the correction machinery fights itself."""
        pts = seen[0]
        hits = sorted((_COVER(pts, self.at(c), self.stamp), c) for c in CELLS)
        self.slack = hits[-1][0] - hits[-2][0]
        whole = _WHOLE(hits, len(pts), self.slack >= margin)
        return whole[0] if len(whole) == 1 else None


def fit_gauge(pix, sides=2):
    """FIT the panel frame off the frame's own census, and do it OCCLUSION-PROOF.

    Each target panel is drawn as a square OUTLINE, and a lit tile is painted right up to that
    outline — so a lit tile in the first row or column ERASES part of the border it sits on, and a
    census of the border colour ALONE mis-fits the layout the moment a level lights those cells
    (measured: one border column drops from 270 covered pixels to 228 on a level that lights its
    first column). The census that survives is of the border colour UNION the tile fill: a border
    row or column is then COMPLETELY covered whatever the level draws inside it, while an interior
    row is short by at least the inter-tile gap. So the full-span rows and columns are exactly the
    three panel frames, and the pitch, the tile side and the three tops all fall out of them.

    Returns None — a refusal, not a default — whenever the census does not describe three
    congruent panels, and the caller then stands the harness down."""
    ledger = _INKED(pix)
    down = ledger.sum(0).tolist()
    wide = max(down)
    cols = [i for i, n in enumerate(down) if n == wide]
    if len(cols) != sides:
        return None
    span = cols[-1] - cols[0] + 1
    across = ledger[:, cols[0]:cols[-1] + 1].sum(1).tolist()
    rows = [i for i, n in enumerate(across) if n == span]
    if span < G * sides or len(rows) % sides:
        return None
    tops = [r for i, r in enumerate(rows) if not i % sides]
    return Gauge(cols[0], span, tops)


def gauge_audit(pix, gauge):
    """Hand the fit back to the FRAME for confirmation, three independent ways.

    The panels are equally spaced and there are as many of them as this layout has ever had; all
    four borders of every panel are completely covered; and the pitch closes the span exactly.
    Each is a statement about THIS frame rather than about a constant someone wrote down, and a
    failure stands the harness down instead of decoding with a ruler the picture does not
    support."""
    ledger = _INKED(pix)
    gaps = {gauge.tops[i] - gauge.tops[i - 1] for i in range(1, len(gauge.tops))}
    if len(gaps) != 1 or len(gauge.tops) != len(TOPS_HINT):
        return False
    edge = gauge.span - 1
    for top in gauge.tops:
        for row in (top, top + edge):
            if not ledger[row, gauge.x0:gauge.x0 + gauge.span].all():
                return False
        for col in (gauge.x0, gauge.x0 + edge):
            if not ledger[top:top + gauge.span, col].all():
                return False
    return gauge.pitch * G == gauge.span


def forced(cells, panels):
    """Envelope cells that are the UNIQUE cover of some lit tile — provably in every member.

    Dropping such a cell would unlight a tile the picture shows lit, so no structure consistent
    with the frame omits it. This is a proof and not a prior: `forced` is inside the truth."""
    out = set()
    for squares, pair in zip(panels, PAIRS):
        for square in squares:
            hit = [c for c in cells if (c[pair[0]], c[pair[1]]) == square]
            if len(hit) == 1:
                out.add(hit[0])
    return out


def proved(cells, panels):
    """Does the picture PROVE the envelope is the structure — is the family a singleton?"""
    return forced(cells, panels) == set(cells)


def trimmable(placed, panels):
    """Cells whose removal leaves another member of the family — the downward dual of upward
    closure, and the only continuation left once the envelope is laid and the episode runs on."""
    keep = set(placed)
    out = set()
    for c in sorted(keep):
        rest = {d for d in keep if d != c}
        if shadow_of(rest) == tuple(panels):
            out.add(c)
    return out


def basis_report(gauge):
    """The calibration's own numbers for the trace: the fitted ruler, and the measured screen
    displacement of each grid axis in rows and in columns. Watching those six numbers settle is
    how a reader sees the projection converge, and a zero among them is an axis the episode has
    not exercised yet. Reached through a module-level alias so this stays a reporting path and can
    never become a decision path."""
    return {"axes": sorted(gauge.basis), "anchor": gauge.anchor, "x0": gauge.x0,
            "pitch": gauge.pitch, "span": gauge.span, "tile": gauge.tile,
            "tops": list(gauge.tops), "slack": round(gauge.slack, 2),
            "x": (round(_AXIS_STEP(gauge, X, ROW), 2), round(_AXIS_STEP(gauge, X, COL), 2)),
            "y": (round(_AXIS_STEP(gauge, Y, ROW), 2), round(_AXIS_STEP(gauge, Y, COL), 2)),
            "z": (round(_AXIS_STEP(gauge, Z, ROW), 2), round(_AXIS_STEP(gauge, Z, COL), 2))}
