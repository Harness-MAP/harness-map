"""The three orthographic TARGET panels, read off the frame by FITTING the lattice the frame draws.

The left strip carries three congruent 6x6 panels — TOP (XY), FRONT (YZ), SIDE (XZ) — in which the
engine paints one solid square per SHADOW of the target structure. The strip is drawn from the LEVEL and
never from the player's blocks, so it is STATIC for the whole episode: read it once, reuse it.

THE LATTICE IS MEASURED, NEVER ASSERTED. The empty grid is drawn in `RULE` as three square OUTLINES, so
the strip carries its own ruler: the two columns whose `RULE` census is tallest are the panels' left and
right edges (three stacked outlines, hence a tall census), and the rows whose census spans most of that
width are the six panel tops and bottoms. Left edge, square side, cell pitch and the three tops all
come back as MEASUREMENTS, which is what makes this survive a renderer that moves or rescales the strip
— a constant table would read a displaced mask in the first case and an empty one in the second.

The census must be occlusion-proof: a lit square at a panel border paints over part of the outline, so
the edge census drops (observed: a tall column losing a sixth of its ink). The thresholds below are
therefore stated as fractions of the measured side, with a wide gap between "edge" (>= 1.5 sides of ink
in a column, >= 0.5 side in a row) and "interior" (a handful of pixels), rather than as exact counts.

What the panels give is an exact but LOSSY channel: three silhouettes pin the SHADOWS, not the volume.
`hull_of` is the largest structure they admit and `squares_wanted` is the debt every admitted structure
must settle; tools/layers.py turns that debt into the SMALLEST admitted structure instead.
"""
GRID = 6
LAST = GRID - 1
STRIP_X1 = 300                 # the panel strip lives left of the 3-D viewport
RULE = (70, 70, 75)            # the empty-grid outline colour
FILL = (80, 80, 80)            # the target-shadow fill
PANELS = ("top", "front", "side")
PROBE = (5, 7, 9)              # interior patch offsets, well inside the fill rect
PROBE_VOTES = 5
# Module-level one-liners: the census arithmetic and the little set projections, written once.
_HIT = lambda m, ink: ((m[:, :, 0] == ink[0]) & (m[:, :, 1] == ink[1]) & (m[:, :, 2] == ink[2]))
_LINE = lambda m, ink: ((m[:, 0] == ink[0]) & (m[:, 1] == ink[1]) & (m[:, 2] == ink[2]))
_MARKS = lambda census, gate: [i for i in range(len(census)) if census[i] >= gate]
_STRIDES = lambda marks, floor: sorted({b - a for a in marks for b in marks if b - a >= floor})
_SPANS = lambda runs: sorted({run[1] for run in runs})
_LONG = lambda runs, floor: [run for run in runs if run[1] >= floor]
_CELLS = [(u, v) for u in range(GRID) for v in range(GRID)]
_AT = lambda pairs, z: {pair[0] for pair in pairs if pair[1] == z}
EDGE_ROW_SHARE = 0.5           # a panel top/bottom row carries >= this share of one side
SIDE_MIN = 24
RIGHT_MIN = 2                  # the right edge must out-ink an interior column by this factor


def ink_runs(image, colour):
    """(per-column census, per-row census) of an exact colour inside the panel strip."""
    try:
        import numpy as np
        strip = image[:, :STRIP_X1, :].astype("int16")
    except Exception:                               # noqa: BLE001 — not an array we can read
        return (None, None)
    hit = _HIT(strip, colour)
    return (hit.sum(axis=0).tolist(), hit.sum(axis=1).tolist())


def edge_runs(image, col):
    """The maximal runs of outline ink down one column — the panels' own row extents, measured.

    This is the occlusion-proof half of the fit. A lit shadow square painted at a panel border eats
    part of that border, so a census that counts ink per row can lose a panel's top row entirely; the
    LEFT edge column, by contrast, loses at most a few pixels per lit cell and its three runs still
    read as three runs. Returns [(first row, length), ...] in order.
    """
    try:
        import numpy as np
        line = image[:, col, :].astype("int16")
    except Exception:                               # noqa: BLE001
        return []
    lit = _LINE(line, RULE).tolist()
    runs = []
    start = None
    y = 0
    while y < len(lit):
        if lit[y] and start is None:
            start = y
        if start is not None and not lit[y]:
            runs.append((start, y - start))
            start = None
        y = y + 1
    if start is not None:
        runs.append((start, len(lit) - start))
    return runs


def fit_lattice(image):
    """The strip's own ruler: {left, side, pitch, tops} measured from the outline ink, or None.

    TWO INDEPENDENT READINGS, because either can be eroded by a lit square sitting on a border:

      * the LEFT EDGE's runs. The tallest column of outline ink is the panels' left edge; the maximal
        runs down it are the three panels' row extents. Three runs of one length settle side and tops
        at once, and the panels being squares then settles the right edge too.
      * the ROW CENSUS as an arithmetic progression. The three panels are congruent and equally
        spaced, so among the rows carrying at least half a side of ink there is a start and a stride
        with all three of t, t+stride, t+2*stride present. This answers when the left edge is eroded.

    Whichever answers, `side` must be a whole number of cells and the right edge must out-ink an
    interior column; nothing is accepted on one reading's word alone without those two checks.
    """
    cols, rows = ink_runs(image, RULE)
    if cols is None or rows is None:
        return None
    left = max(range(len(cols)), key=lambda i: cols[i])
    long_runs = _LONG(edge_runs(image, left), SIDE_MIN)
    spans = _SPANS(long_runs)
    firm = len(long_runs) == 3 and len(spans) == 1
    side = spans[0] if firm else 0
    tops = [run[0] for run in long_runs] if firm else []
    if not firm:
        marks = _MARKS(rows, EDGE_ROW_SHARE * SIDE_MIN)
        seen = set(marks)
        for stride in _STRIDES(marks, SIDE_MIN):
            starts = [t for t in marks if (t + stride) in seen and (t + 2 * stride) in seen]
            if starts:
                side = stride - (stride % GRID)
                tops = [starts[0] + k * stride for k in range(3)]
                break                                # the shortest workable stride is the right one
        aligned = [run for run in long_runs if tops and run[0] == tops[0]]
        side = aligned[0][1] if aligned else side
    if side < SIDE_MIN or side % GRID != 0:
        return None
    if len(tops) != 3:
        return None
    right = left + side - 1
    if right >= len(cols):
        return None
    if not firm and cols[right] < RIGHT_MIN * 3 * GRID:
        return None                                  # the progression alone needs the right edge too
    if max(tops) + side > len(rows):
        return None
    return {"left": left, "side": side, "pitch": side // GRID, "tops": sorted(tops)}


def probe_square(image, rule, top, u, v):
    """Is panel cell (column u, row v) painted with the shadow fill? Patch-voted, not one pixel."""
    pitch = rule["pitch"]
    if pitch < 2:
        return False
    votes = 0
    try:
        for dy in PROBE:
            line = image[top + v * pitch + dy]
            for dx in PROBE:
                px = line[rule["left"] + u * pitch + dx]
                if int(px[0]) == FILL[0] and int(px[1]) == FILL[1] and int(px[2]) == FILL[2]:
                    votes = votes + 1
    except Exception:                               # noqa: BLE001 — a short frame is a failed read
        return False
    return votes >= PROBE_VOTES


def read_strip(image):
    """The three shadow masks in GRID coordinates, or None when the strip is not readable.

    Panel (column u, row v) -> grid axes, in the renderer's own projections:
      TOP   (XY): (x, y) = (v, LAST - u)
      FRONT (YZ): (y, z) = (LAST - u, LAST - v)
      SIDE  (XZ): (x, z) = (u, LAST - v)
    """
    rule = fit_lattice(image)
    if rule is None:
        return None
    if len(rule["tops"]) != 3:
        return None
    seen = {name: set() for name in PANELS}
    for slot, name in enumerate(PANELS):
        top = rule["tops"][slot]
        for u, v in _CELLS:
            if not probe_square(image, rule, top, u, v):
                continue
            if name == "top":
                seen[name].add((v, LAST - u))
            elif name == "front":
                seen[name].add((LAST - u, LAST - v))
            else:
                seen[name].add((u, LAST - v))
    if not (seen["top"] and seen["front"] and seen["side"]):
        return None                                  # a blank strip is a failed read, not a target
    seen["rule"] = rule
    return seen


def marginals_agree(masks):
    """The 3-way self-check: each grid axis is drawn on TWO panels, so its support must match.

    x is carried by TOP and SIDE, y by TOP and FRONT, z by FRONT and SIDE. A decode that mis-orients
    one panel, or reads a displaced mask, breaks at least one of the three. Disagreement means stand
    the whole mechanism down — it is never a licence to guess which panel was right.
    """
    top, front, side = masks["top"], masks["front"], masks["side"]
    if {pair[0] for pair in top} != {pair[0] for pair in side}:
        return False                                 # x is carried by TOP and SIDE
    if {pair[1] for pair in top} != {pair[0] for pair in front}:
        return False                                 # y is carried by TOP and FRONT
    return {pair[1] for pair in front} == {pair[1] for pair in side}


def shadows_of(cell):
    """The three lit squares one cell is answerable for, as comparable ids."""
    x, y, z = cell
    return (("top", x, y), ("front", y, z), ("side", x, z))


def squares_wanted(masks):
    """The squares any admitted structure must light — the decoded masks as square ids."""
    want = {("top", x, y) for (x, y) in masks["top"]}
    want |= {("front", y, z) for (y, z) in masks["front"]}
    want |= {("side", x, z) for (x, z) in masks["side"]}
    return want


def hull_of(masks):
    """The ADMITTED HULL: every cell all three of whose shadows are lit. The largest admitted set."""
    front, side = masks["front"], masks["side"]
    return sorted((x, y, z) for (x, y) in masks["top"] for z in range(GRID)
                  if (y, z) in front and (x, z) in side)


def census(masks):
    """Per-height (columns, rows, top squares) — the layer factorisation, as three small counts."""
    deep = {}
    for z in range(GRID):
        wide = len(_AT(masks["side"], z))
        tall = len(_AT(masks["front"], z))
        if wide + tall > 0:
            deep[z] = (wide, tall)
    return deep
