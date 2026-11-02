"""THE SILHOUETTE READ — the three orthographic TARGET panels, the envelope they back-project to,
and the censuses that make the read a MEASUREMENT instead of an assertion.

Everything here reads the array `tools/frame.py::raster` returns and nothing else: no
`evaluator.game`, no re-render, no file. The panels are part of the picture the frozen agent is
itself looking at, so decoding them adds no channel the agent does not have.

HOW A PANEL IS READ. The fitted ruler (`tools/optics.py::Gauge`) supplies the panel strip's left
edge, the cell pitch, the tile side and the three panel tops; this module only indexes with them.
A target tile is filled with one exact constant that nothing else on the frame uses, so a tile is
lit iff its interior patch is SOLID in that constant. The on-screen (row, col) convention of the
three projections, and the only geometric prior in the file:

    TOP   (XY)   row = x        col = 5 - y
    FRONT (YZ)   row = 5 - z    col = 5 - y
    SIDE  (XZ)   row = 5 - z    col = x

WHY AN ENVELOPE AND NOT A GUESS. Three silhouettes do not determine a structure, they determine a
family of them. `admissible` back-projects the panels into the set of cells whose three shadows
all land on lit tiles. That set CONTAINS every structure consistent with the picture — hence the
true one — and `consistent` verifies it is itself a member, so it is simultaneously an upper
bound and a witness that the family is non-empty. The planner then orders it; it never filters it.

THREE CENSUSES, because a wrong read must announce itself rather than quietly becoming a wrong
plan:
  * per-tile    — a tile's interior is ALL ink or NO ink, `side*side` or zero, nothing between.
  * whole-frame — the frame's total exact-ink count equals `side*side` times the lit-tile count.
  * re-cast     — the envelope's OWN three shadows must reproduce the three panels exactly.
Any failure returns None from `read_scene`, and the harness then stands down to the pristine
per-step VLM fn for the rest of the episode rather than acting on a misread.
"""
from __future__ import annotations

G = 6                              # the grid is 6x6x6
INK = (80, 80, 80)                 # exact fill constant of a lit target tile
GRID = (70, 70, 75)                # exact constant of the empty panel lattice lines
VIEW_X0 = 300                      # the 3D viewport starts here; the panel strip is left of it
NOTILE = -1                        # tile_ink's refusal: the patch ran off the frame

_EXACT = lambda a, rgb: ((a[:, :, 0] == rgb[0]) & (a[:, :, 1] == rgb[1]) & (a[:, :, 2] == rgb[2]))
_TOP = lambda r, c: (r, G - 1 - c)
_FRONT = lambda r, c: (G - 1 - c, G - 1 - r)
_SIDE = lambda r, c: (c, G - 1 - r)
PROJ = (_TOP, _FRONT, _SIDE)
# which two cell coordinates each panel is a shadow of, in panel order
PAIRS = ((0, 1), (1, 2), (0, 2))


def tile_ink(raster, top, left, side):
    """Exact-INK pixel count inside one panel tile's interior, or NOTILE when the patch is clipped."""
    patch = raster[top:top + side, left:left + side]
    if len(patch.shape) < 2 or patch.shape[:2] != (side, side):
        return NOTILE
    return int(_EXACT(patch, INK).sum())


def read_masks(raster, gauge):
    """The three panels as GxG boolean row-major tuples plus the lit-tile count, or None.

    Two distinct refusals, because they mean different things. A NEGATIVE count means the patch
    ran off the frame, so the fitted span does not fit the picture at all. A count strictly
    between zero and a full tile means the tile is partially inked, so the lattice is a pixel or
    two out of step with the picture. Either way the read is not trustworthy and the caller stands
    the harness down; neither is rounded off into a guess."""
    full = gauge.tile * gauge.tile
    out = []
    lit = 0
    for top in gauge.tops:
        rows = []
        for row in range(G):
            cells = []
            for col in range(G):
                n = tile_ink(raster, top + row * gauge.pitch,
                             gauge.x0 + col * gauge.pitch, gauge.tile)
                if n < 0:
                    return None
                if n not in (0, full):
                    return None
                cells.append(n)
            rows.append(tuple(n == full for n in cells))
            lit += cells.count(full)
        out.append(tuple(rows))
    return tuple(out), lit


def frame_census(raster, inks=(INK,)):
    """Total exact-match pixels on the whole frame, summed over the constants asked for.

    The denominator of the tile census. The target panels are the only thing on the frame drawn
    in the target constant, so the lit-tile count times the tile area must account for ALL of it;
    stray ink anywhere means the constant is not unique here and the read is refused."""
    return sum(int(_EXACT(raster, rgb).sum()) for rgb in inks)


def mask_to_cells(masks):
    """The three panel masks as coordinate sets: TOP {(x,y)}, FRONT {(y,z)}, SIDE {(x,z)}."""
    out = []
    for m, proj in zip(masks, PROJ):
        out.append({proj(r, c) for r in range(G) for c in range(G) if m[r][c]})
    return tuple(out)


def shadow_of(cells):
    """The three shadows a candidate voxel set would cast — the forward model, and the only one."""
    return ({(x, y) for (x, y, _z) in cells},
            {(y, z) for (_x, y, z) in cells},
            {(x, z) for (x, _y, z) in cells})


def admissible(panels):
    """The ENVELOPE: every cell whose three shadows all land on lit tiles.

    Adding an envelope cell can only light tiles that are already lit and unlights nothing, so the
    family is upward closed inside the envelope — which is why an append-only build tests one NEW
    structure per press once it is inside the family, the most a single press can buy."""
    top, front, side = panels
    return {(x, y, z) for (x, y) in top for z in range(G)
            if (y, z) in front and (x, z) in side}


def consistent(cells, panels):
    """Is `cells` a member of the family — do its own shadows reproduce the three panels?"""
    return shadow_of(cells) == tuple(panels)


def read_scene(raster, gauge):
    """Measure the episode's static geometry from ONE frame, or REFUSE.

    Returns `(panels, envelope)` on a clean read; None when any census or the re-cast fails."""
    got = read_masks(raster, gauge)
    if got is None:
        return None
    masks, lit = got
    if not all(ok for ok in (lit, frame_census(raster) == lit * gauge.tile * gauge.tile)):
        return None
    panels = mask_to_cells(masks)
    hull = admissible(panels)
    if not all(ok for ok in (hull, consistent(hull, panels))):
        return None
    return panels, hull
