"""tools/frame.py — the frame, and the one frame READ EXACTLY.

The harness reads `frame_data.image` — the very frame the frozen model is shown — and never any
engine state. Everything here is the inverse of the painter that drew it.

WHY THE READING IS A MEASUREMENT AND NOT A HEURISTIC. `draw_3d_view` is a flat-shaded painter with
a closed-form geometry, so it inverts. Three facts, all measured in this chain against the engine's
own level data over 25 levels x both floors x every reachable cell x 4 headings:

  * FRONT FACES. The cell `d` steps down the forward ray is painted as a rect of size
    (600/d, 400/d) centred on (400, 300), FAR TO NEAR, so the nearest non-empty cell completely
    covers every farther one. The 21x21 patch at the frame centre is inside rects[12] — the
    smallest face the painter draws — so that patch is ALWAYS owned by the nearest front face when
    one exists within 12 cells. Forward ray `(kind, distance)`: 5624/5624 exact.
  * SHADE NAMES DEPTH. Every face is `darken(base, fog)` with `fog(d) = max(0.4, 1 - 0.08*d)` and
    per-channel int() truncation, so one packed RGB value names BOTH the material and its distance.
    The stair border is the one exception — its fog is floored at 0.8 — and the exit is painted
    flat, with no fog at all; for those two the painted run's width supplies the distance instead.
    Fog has hit its floor by d = 8, where the wall shade stops resolving and the width takes over.
  * THE FLANKS. The d=1 side quads reach x = 100 on the left and x = 700 on the right, and no other
    quad does, so columns 0..89 and 710..799 are painted by NOTHING except the cell immediately
    beside the agent — one cell, no occlusion, no mixing. Lateral cells: 11248/11248 exact.

The cell under the agent's own feet is never drawn, so "am I on a stair" is not a pixel question at
all; it is answered by the chart in tools/action.py, from a stair seen ahead or beside and then
stood on.

TOKEN-LEVEL NOTE. The lineage's decoders carry one module-level look-up dict per material family
and vote across four of them. This one carries ONE flat table, built by a single comprehension over
the painter's own formulas: packed RGB -> (tag, depth, depth_is_exact). Classification is then one
census of the patch and one `min` over the winning tag's exact depths, with the run width as the
stated fallback for the two families whose shade cannot name a depth.
"""
from __future__ import annotations

import hashlib

try:                                      # the evaluator hands over a numpy frame; never hard-fail
    import numpy as _np
except Exception:                          # pragma: no cover
    _np = None

WIDTH, HEIGHT = 800, 600
CX, CY = 400, 300
DEPTH_MAX = 12                  # the painter's loop bound: cells 1..12 ahead are drawn
FRONT_W = 600                   # rects[d] spans FRONT_W / d pixels across, centred on CX
FOG_FLOOR, FOG_RATE = 0.4, 0.08
EDGE_FOG_FLOOR = 0.8            # the stair border's fog is floored here, so it loses its depth
FOG_PLATEAU = 8                 # the first d at which fog(d) == FOG_FLOOR and shade stops naming d

WALL_AHEAD = (80, 40, 30)
WALL_BESIDE = (160, 85, 60)
STAIR_FILL = (0, 100, 120)
STAIR_EDGE = (0, 255, 255)
EXIT_FACE = (255, 200, 0)
EXIT_EDGE = (255, 255, 255)
EMPTY_BASES = ((50, 60, 65), (15, 15, 35))        # floor, ceiling: the two "nothing here" colours

PATCH = 10                      # half-width of the centre patch; 21x21 fits inside rects[12]
FLANK_TOP, FLANK_BOTTOM = 170, 431               # rows safely inside the d=1 side quad
FLANK_WIDTH = 90                                 # columns 0..89 and 710..799

WALL, STAIR, EXIT, OPEN, BESIDE = "wall", "stair", "exit", "open", "beside"
_TABLE = {}
_EMPTY = ()


def fog(depth: int) -> float:
    """The painter's own fog at `depth` cells."""
    return max(FOG_FLOOR, 1.0 - FOG_RATE * depth)


def shade(base, factor: float) -> int:
    """The painter's `darken` (per-channel int() truncation), packed into one integer."""
    r, g, b = (max(0, int(channel * factor)) for channel in base)
    return (r << 16) | (g << 8) | b


def face_table() -> dict:
    """packed RGB -> (tag, depth, depth_is_exact) for every face the painter can draw.

    One comprehension over the painter's formulas. A shade claimed by two depths keeps the NEAREST,
    which is the only one that can be visible — a nearer face covers every farther one — and that is
    what collapses the d >= 8 fog plateau into a single entry. `depth_is_exact` is False for the
    stair border (its fog is floored, so several depths share one shade) and for the exit (painted
    flat); for those two the run width carries the distance.
    """
    global _EMPTY
    if not _TABLE:
        depths = range(1, DEPTH_MAX + 1)
        rows = [(shade(WALL_AHEAD, fog(d)), (WALL, d, True)) for d in depths]
        rows += [(shade(STAIR_FILL, fog(d)), (STAIR, d, True)) for d in depths]
        rows += [(shade(STAIR_EDGE, max(fog(d), EDGE_FOG_FLOOR)), (STAIR, d, False)) for d in depths]
        rows += [(shade(WALL_BESIDE, fog(d)), (BESIDE, d, True)) for d in depths]
        rows += [(shade(EXIT_FACE, 1.0), (EXIT, 1, False)), (shade(EXIT_EDGE, 1.0), (EXIT, 1, False))]
        for code, row in rows:
            _TABLE.setdefault(code, row)
        _EMPTY = tuple(shade(base, 1.0) for base in EMPTY_BASES)
    return _TABLE


def capture(frame_data):
    """The current frame as an (H, W, >=3) pixel array, or None when it is not readable.

    `frame_data.image` IS the array the evaluator just rendered and handed to the model (measured
    fresh every step in this chain, 0 conflicts against the model's own view), so this is a plain
    accessor with a shape check, not a second rendering path. A bare array is accepted too, which is
    what lets a stub rig drive the harness with no evaluator at all.
    """
    if _np is None or frame_data is None:
        return None
    image = getattr(frame_data, "image", frame_data)
    if image is None:
        return None
    try:
        arr = _np.asarray(image)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] < HEIGHT or arr.shape[1] < WIDTH:
        return None
    return arr


def packed(block):
    """One RGB block as a 2-D array of packed integers, which is what the table is keyed on."""
    a = block[:, :, :3].astype(_np.int32)
    return (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]


def census(block) -> dict:
    """packed colour -> pixel count over a block."""
    codes, counts = _np.unique(packed(block), return_counts=True)
    return dict(zip(codes.tolist(), counts.tolist()))


def run_depth(image, kind) -> int:
    """The front face's depth from the WIDTH of its painted run through the frame centre.

    rects[d] is FRONT_W / d wide and those widths (600, 300, 200, 150, 120, 100, 86, 75, 67, 60, 55,
    50) stay further apart than the rasteriser's rounding, so the nearest width is the depth. This
    is the stated fallback for the two families whose shade cannot name a depth and for a wall out
    on the fog plateau.
    """
    family = [code for code, row in face_table().items() if row[0] == kind]
    row = packed(image[CY:CY + 1, :, :])[0]
    hit = _np.isin(row, family)
    if not bool(hit[CX]):
        return 1
    left = right = CX
    while left > 0 and bool(hit[left - 1]):
        left -= 1
    while right < WIDTH - 1 and bool(hit[right + 1]):
        right += 1
    width = right - left + 1
    return min(range(1, DEPTH_MAX + 1), key=lambda d: abs(FRONT_W / d - width))


def read_ray(image):
    """(kind, depth) of the first non-empty cell straight ahead.

    `OPEN` means the painter drew nothing within DEPTH_MAX, and carries depth DEPTH_MAX + 1. The
    patch sits inside the smallest front face drawn, so whichever material owns the most of it IS
    the nearest non-empty cell ahead.
    """
    table = face_table()
    patch = image[CY - PATCH:CY + PATCH + 1, CX - PATCH:CX + PATCH + 1, :]
    votes, exact = {}, {}
    for code, count in census(patch).items():
        row = table.get(code)
        if row is None or row[0] == BESIDE:
            continue                       # background, or a side quad that cannot reach here
        tag, depth, precise = row
        votes[tag] = votes.get(tag, 0) + count
        if precise:
            exact[tag] = min(exact.get(tag, depth), depth)
    if not votes:
        return OPEN, DEPTH_MAX + 1
    kind = max(votes, key=lambda tag: votes[tag])
    depth = exact.get(kind)
    if depth is None or (kind == WALL and depth >= FOG_PLATEAU):
        depth = run_depth(image, kind)
    return kind, depth


def read_flank(image, column: int) -> str:
    """`wall` / `stair` / `open` for the single cell immediately beside the agent.

    Only the d=1 side quad paints these columns, so there is exactly one cell to classify. An EXIT
    beside the agent is drawn as nothing at all — the painter draws the exit only as a front face —
    and correctly reads `open`; it is enterable either way, and the chart keeps "open because a ray
    went through it" apart from "open because a flank said so" for exactly this reason.
    """
    table = face_table()
    counts = census(image[FLANK_TOP:FLANK_BOTTOM, column:column + FLANK_WIDTH, :])
    wall = counts.get(shade(WALL_BESIDE, fog(1)), 0)
    stair = sum(n for code, n in counts.items() if table.get(code, (None,))[0] == STAIR)
    empty = sum(counts.get(code, 0) for code in _EMPTY)
    tag, votes = max((WALL, wall), (STAIR, stair), (OPEN, empty), key=lambda pair: pair[1])
    return tag if votes else OPEN


class Scene:
    """One decoded frame: the forward ray, both flanks, the exact frame identity, and the travel a
    forward translation would actually get. Constructing it never raises — an unreadable frame
    yields `ok = False` and the harness falls back to the model's own proposal for that step."""

    __slots__ = ("ok", "digest", "kind", "depth", "left", "right", "walkable")

    def __init__(self, image=None):
        self.ok, self.digest = False, ""
        self.kind, self.depth = OPEN, DEPTH_MAX + 1
        self.left = self.right = OPEN
        self.walkable = 0
        if image is None:
            return
        try:
            face_table()
            self.digest = hashlib.md5(_np.ascontiguousarray(image[:, :, :3]).tobytes()).hexdigest()
            self.kind, self.depth = read_ray(image)
            self.left = read_flank(image, 0)
            self.right = read_flank(image, WIDTH - FLANK_WIDTH)
            # the engine walks INTO an empty cell, a stair tile and the exit alike, and only a wall
            # costs the last cell of the ray, so this is the exact travel of a forward translation.
            reach = self.depth if self.kind in (STAIR, EXIT) else self.depth - 1
            self.walkable = max(0, min(reach, DEPTH_MAX))
            self.ok = True
        except Exception:
            self.ok = False

    def ahead_line(self) -> str:
        """The forward ray, in the words the agent is given it in."""
        if not self.ok:
            return "not readable this step."
        if self.kind == OPEN:
            return "open floor for at least %d cells." % DEPTH_MAX
        what = {WALL: "a wall", STAIR: "a STAIR tile", EXIT: "the EXIT"}[self.kind]
        where = "in the very next cell" if self.depth == 1 else "%d cells away" % self.depth
        return "%s %s; one forward move travels %d cell(s)." % (what, where, self.walkable)

    def flank_line(self, side: str) -> str:
        """One lateral neighbour, in the words the agent is given it in."""
        if not self.ok:
            return "not readable this step."
        return {WALL: "a wall.", STAIR: "a STAIR tile, one turn and one step away.",
                OPEN: "open floor."}[getattr(self, side)]
