"""PANELS — read the three orthographic silhouettes back out of the rendered strip.

The strip the frozen model is shown IS the specification of the structure to build: three
square panels of GRID_ORDER x GRID_ORDER cells, a painted cell meaning "some block of the
structure projects here". The panels are a complete tomographic statement of the goal, so
reading them back exactly turns a vision problem the model demonstrably cannot solve into
arithmetic the harness can.

The readout is all-or-nothing by construction and can therefore REFUSE ITSELF:

  * a painted square is a flat block of one exact ink, so a cell either clears the area vote by
    a wide margin or misses it by a wide margin — there is no grey zone to tune;
  * every lattice axis appears on TWO panels (the first panel and the second share one axis, the
    first and the third share another, the second and the third share the last), so the three
    readings over-determine each other and a garbled panel breaks the cross-check;
  * the admitted volume must re-project onto all three silhouettes exactly, which is the final
    closure test.

If any of those fail the caller stands the whole wrapper down to the pristine pass-through
rather than acting on a guess. `admitted` is the MAXIMAL cell set the silhouettes allow: the
structure is always a subset of it, and a cell outside it is PROVED by the picture to be wrong.
"""

from tools.lattice import GRID_ORDER, fit_panels

FILL_INK = (80, 80, 80)
INSET = 3
VOTE = 60
LAST = GRID_ORDER - 1
TOP_PANEL = 0
FRONT_PANEL = 1
SIDE_PANEL = 2
PANEL_COUNT_SEEN = 3
PERCENT = 100

_grid = [(x, y, z) for x in range(GRID_ORDER) for y in range(GRID_ORDER) for z in range(GRID_ORDER)]
_squares = [(u, v) for u in range(GRID_ORDER) for v in range(GRID_ORDER)]
_hits = lambda px, x0, y0, reach: sum(1 for dy in range(reach) for dx in range(reach)
                                      if px[x0 + INSET + dx, y0 + INSET + dy] == FILL_INK)
_allowed = lambda xy, yz, xz: frozenset(c for c in _grid if (c[0], c[1]) in xy
                                        and (c[1], c[2]) in yz and (c[0], c[2]) in xz)
_cast = lambda cells, i, j: frozenset((c[i], c[j]) for c in cells)
_axis = lambda squares, i: {s[i] for s in squares}
_read_panel = lambda px, fit: {s for s in _squares if ink_ratio(px, fit, s[0], s[1]) >= VOTE}


def ink_ratio(px, fit, u, v):
    """Percentage of a panel cell's inset interior that is EXACTLY the fill ink. The painted
    square is a flat block of one ink, so this is near 0 or near 100 and never in between."""
    x0, y0 = fit.cell_box(u, v)
    reach = fit.pitch - INSET - INSET
    if reach < 1:
        return 0
    return _hits(px, x0, y0, reach) * PERCENT // (reach * reach)


def admitted_cells(xy, yz, xz):
    """Every grid cell the three silhouettes jointly allow — the visual hull of the structure."""
    return _allowed(xy, yz, xz)


class Shadows:
    """The three decoded silhouettes plus the volume they admit, with its own consistency verdict."""

    __slots__ = ("xy", "yz", "xz", "admitted", "ok", "why")

    def __init__(self, xy, yz, xz):
        self.xy = frozenset(xy)
        self.yz = frozenset(yz)
        self.xz = frozenset(xz)
        self.admitted = admitted_cells(self.xy, self.yz, self.xz)
        self.ok, self.why = self.verdict()

    def shadow_of(self, cells):
        """The three silhouettes a candidate cell set would itself cast."""
        return (_cast(cells, 0, 1), _cast(cells, 1, 2), _cast(cells, 0, 2))

    def explains(self, cells):
        """Does this cell set reproduce all three decoded silhouettes exactly?"""
        return self.shadow_of(cells) == (self.xy, self.yz, self.xz)

    def axis_agrees(self):
        """The three panels share their axes pairwise, so their marginals must agree."""
        return (_axis(self.xy, 0) == _axis(self.xz, 0)
                and _axis(self.xy, 1) == _axis(self.yz, 0)
                and _axis(self.yz, 1) == _axis(self.xz, 1))

    def droppable(self, cells):
        """Cells whose removal still reproduces all three silhouettes.

        When every admitted cell is filled and the structure was still not accepted, the real
        structure is a strict subset of what the panels admit, and these are the only removals
        that can reach it — lifting anything else would break a silhouette outright.
        """
        out = []
        for cell in sorted(cells):
            if self.explains(set(cells) - {cell}):
                out.append(cell)
        return out

    def verdict(self):
        if not (self.xy and self.yz and self.xz):
            return False, "a panel read empty"
        if not self.axis_agrees():
            return False, "the shared axes disagree across panels"
        if not self.admitted:
            return False, "the three panels admit no cell"
        if not self.explains(self.admitted):
            return False, "the admitted volume does not re-project"
        return True, "consistent"


def decode_panels(evaluator):
    """Frame -> Shadows. Returns None when the strip cannot be fitted or read."""
    from tools.frame import capture_frame, pixels_of

    grab = pixels_of(capture_frame(evaluator))
    if grab is None:
        return None
    w, h, px = grab
    fits = fit_panels(px, w, h)
    if fits is None or len(fits) != PANEL_COUNT_SEEN:
        return None
    lit = []
    for fit in fits:
        lit.append(_read_panel(px, fit))
    xy = {(v, LAST - u) for (u, v) in lit[TOP_PANEL]}
    yz = {(LAST - u, LAST - v) for (u, v) in lit[FRONT_PANEL]}
    xz = {(u, LAST - v) for (u, v) in lit[SIDE_PANEL]}
    return Shadows(xy, yz, xz)
