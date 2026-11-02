"""LATTICE — FIT the panel geometry off the frame instead of asserting a constant table.

The left strip of the block3d screen carries three congruent square outlines, each drawn as a
one-pixel ring in the grid ink. Those rings are the picture's own ruler: their span gives the
panel side, their span divided by the grid order gives the cell pitch, and their vertical runs
give the three panel origins. Reading them as MEASUREMENTS rather than hard-coding pixel offsets
is what keeps the decoder honest if the renderer ever moves, rescales or reorders the strip — a
frozen table silently returns a wrong mask in that case, while a fit returns a REFUSAL.

Two occlusion facts drive the shape of the fit, and both were read off the renderer rather than
assumed:

  * a filled silhouette square is painted after the ring and is one pixel smaller than the pitch,
    so the squares in the FIRST column and the FIRST row eat the ring's near borders;
  * the ring's FAR borders — its last column and its last row — are never painted over.

So the side and the left edge are measured from the ring rows that survive intact, and the three
panel tops are measured as vertical runs down the far column. Nothing here reads engine state:
the only input is the pixel accessor for the frame the frozen model is shown.
"""

GRID_ORDER = 6
GRID_INK = (70, 70, 75)
STRIP_X = 300
MIN_SIDE = 36
PANEL_COUNT = 3
SPACING_SLACK = 2
NOTHING = 0

_inked = lambda px, y, wide: [x for x in range(wide) if px[x, y] == GRID_INK]
_solid = lambda xs: xs and max(xs) - min(xs) + 1 == len(xs)
_origin = lambda px, wide, ys: max([(min(xs), len(xs)) for xs in
                                    (_inked(px, y, wide) for y in ys) if _solid(xs)]
                                   or [(NOTHING, NOTHING)], key=lambda r: r[1])


def grid_mask(px, w, h):
    """Row and column censuses of EXACT grid ink inside the panel strip.

    Returns (rows, cols) where rows[y] counts ink pixels in row y and cols[x] counts them in
    column x. Restricting to the strip removes the ground-plane lines of the isometric scene,
    which are painted in the same ink far to the right of every panel.
    """
    edge = STRIP_X if STRIP_X < w else w
    rows = [0] * h
    cols = [0] * edge
    for x in range(edge):
        column = cols
        for y in range(h):
            if px[x, y] == GRID_INK:
                rows[y] += 1
                column[x] += 1
    return rows, cols


class PanelFit:
    """One measured panel: left edge, top edge, square side and the derived cell pitch."""

    __slots__ = ("left", "top", "side", "pitch")

    def __init__(self, left, top, side):
        self.left = left
        self.top = top
        self.side = side
        self.pitch = side // GRID_ORDER

    def cell_box(self, u, v):
        """Pixel origin of panel cell (u, v) — the painted square starts exactly there."""
        return self.left + u * self.pitch, self.top + v * self.pitch

    def sane(self):
        return self.side >= MIN_SIDE and self.side % GRID_ORDER == 0 and self.pitch > SPACING_SLACK


def fit_panels(px, w, h, panels=3, slack=2):
    """Measure the three panel rings. Returns a list of PanelFit, or None when the strip does
    not look like `panels` congruent rings — which the caller must treat as a refusal, never as
    an empty reading."""
    rows, cols = grid_mask(px, w, h)
    span = max(rows)
    if span < MIN_SIDE:
        return None
    intact = [y for y in range(h) if rows[y] == span]
    if not intact or not cols:
        return None
    left, side = _origin(px, len(cols), intact)
    if side != span or side % GRID_ORDER:
        return None                                    # the ring is not a whole number of cells
    far = left + side - 1
    tops, run, y = [], NOTHING, NOTHING
    while y < h:
        run = run + 1 if px[far, y] == GRID_INK else 0
        if run == side:
            tops.append(y - side + 1)
            if len(tops) > panels:
                break                                  # a fourth ring is not this renderer
        y += 1
    if len(tops) != panels:
        return None
    gaps = [tops[i + 1] - tops[i] for i in range(panels - 1)]
    if max(gaps) - min(gaps) > slack:
        return None                                    # the rings are not evenly stacked
    out = [PanelFit(left, t, side) for t in tops]
    return out if all(p.sane() for p in out) else None
