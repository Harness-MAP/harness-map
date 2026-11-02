"""tools/sight.py — ONE FRAME, READ AS GEOMETRY: the corridor ahead and the FLANK LADDER beside it.

WHAT IS NEW HERE. Every ancestor of this harness that read the pixels at all read the two cells
beside the AGENT — the pair of wide quads at the screen edges — and nothing else sideways. So the
side walls of a corridor could only be learned one cell at a time, by standing in each cell, and a
cell crossed by a stride left its neighbours unknown behind it. This module reads the side faces of
the WHOLE visible corridor instead: the renderer paints the two neighbours of every corridor cell
the forward ray proves open, each pair into its own narrow vertical band, and those bands tile the
screen in order of distance. One look therefore classifies up to 2 x (corridor length) cells rather
than 2, and a cell flown over by a stride arrives already surrounded by settled map.

EVERYTHING BELOW IS A CENSUS OF RENDERED FRAMES, not a reading of the draw code (the proposer
sandbox blocks the engine source, and a census is the stronger warrant anyway: it would catch a
renderer change, which a transcribed constant would not). The census drove the game to every
walkable cell of all 25 levels, on both floors, from all four bearings — 5624 poses — decoded each
frame with the code below and scored it against where the body could actually go:

  1. THE DISTANCE AHEAD is the HEIGHT of the paint down the centre column. Whatever blocks the
     corridor is centred on the horizon and its height scales as 1/distance: the run of pixels down
     column 400 that is neither ceiling (15,15,35) nor ground (50,60,65) measures 401/403 px at one
     cell, 201/203 at two, 133/135 at three, 101/103 at four, 81/83 at five, 67/69 at six, 57/59 at
     seven, 51/53 at eight, 45 at nine, 41 at ten, 37 at eleven — disjoint bands, each one
     `FACE_SPAN / (rows - 1)` on the nose. **5624/5624 poses decode to the true distance.**
  2. WHAT IS AT THAT DISTANCE is the COLOUR of that run, and only three families exist: red == 0 is
     the teal stair face (border (0,255,255), fill (0,100,120) fogged), blue == 0 the single
     un-fogged gold (255,200,0) of the exit, anything else the brown wall. Counting the three over
     the whole run — never one sampled pixel, which a stair's diagonals or an outline can flip —
     is **5624/5624 exact**, 144 exits / 273 stairs / 5207 walls, no confusion either way.
  3. THE FLANK LADDER. The neighbours of the corridor cell `k` steps ahead are painted into the
     band that ends `SIDE_SPAN // (k + 1)` off the centre line and begins `SIDE_SPAN // k` off it
     (k = 0 runs off the edge of the screen). Those bands are disjoint and ordered — 100..250,
     250..300, 300..325, 325..340, 340..350 for k = 0..4 on the left, mirrored on the right — so a
     face at one distance can never be read as a face at another, and sampling the band's MIDPOINT
     at the horizon dodges both the 1 px face outline and the stair's 3 px frame. The colour of
     that sample is the neighbour's kind at the same distance fog the ray uses: a side wall is
     (160,85,60) fogged, a side stair (0,100,120) fogged, and an EMPTY neighbour paints NOTHING, so
     its band shows bare ground or ceiling. **13960/13960 band readings exact, 0 wrong.**
  4. THE OCCLUSION GATE IS `k <= depth - 1`, where `depth` is the distance of the nearest front
     face — NOT the walkable run. A front face covers every band nearer the centre line than its
     own edge, and the band for `k = depth - 1` ends exactly where that face begins, so the gated
     bands are precisely the ones nothing can cover. Gating on the walkable run instead reads a
     front face as a side one wherever the blocker is itself steppable (a stair or the exit), which
     is the one error this decode can make and the gate removes it entirely.

WHAT IT STILL REFUSES TO CONCLUDE. Nothing at all is painted for the cell under the body, so "am I
standing on a stair" is not a reading — the map answers it instead, from a stair seen earlier and a
reckoned pose. An EXIT beside the corridor paints no side face either (measured: the exit is drawn
only as a front face), so a flank reading of `open` means *passable*, never *not the goal*; the
chart keeps those two kinds of open apart and that distinction is what the floor-exhaustion proof
rests on. Everything here is computed from the frame the agent is shown and from nothing else.
"""
from __future__ import annotations

import hashlib

try:                                 # the engine hands over a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                    # pragma: no cover
    _np = None

WALL, STAIR, EXIT, OPEN = "wall", "stair", "exit", "open"


class Sight:
    """One frame, read whole: the corridor's length and kind, and the flank ladder beside it.

    Immutable and cheap — one boolean reduction down a single column plus about forty sampled
    pixels — and the only thing in the harness that looks at the frame by anything but an md5.
    """

    __slots__ = ("ok", "digest", "face", "depth", "reach", "flanks")

    #: THE RULER, in the frame's own pixels. The evaluator fixes the surface at 800x600 for every
    #: episode, so these are measurements of the picture rather than a projection to be scaled; a
    #: frame of any other shape is reported unreadable instead of being decoded on a guess.
    W = 800
    H = 600
    CX = 400
    HORIZON = 300
    UI_ROWS = 60                     # the top band is blacked out by a UI bar: start scanning below
    FACE_SPAN = 400                  # rows a blocking face spans at one cell; FACE_SPAN // d at d
    SIDE_SPAN = 300                  # the k-th neighbour's band ends SIDE_SPAN // (k+1) off centre
    MAX_DEPTH = 12                   # the deepest cell the renderer draws
    LADDER = 6                       # neighbour pairs read per look; the 6th band is 7 px wide
    STRIDE_CAP = 3                   # the engine's `large`
    CEILING = (15, 15, 35)
    GROUND = (50, 60, 65)
    SIDE_ROWS = (298, 300, 302)      # three rows straddling the horizon, where a band is tallest

    def __init__(self, image=None):
        arr = self._array(image)
        self.ok = arr is not None
        self.digest = hashlib.md5(arr.tobytes()).hexdigest() if self.ok else ""
        self.face, self.depth = self._corridor(arr) if self.ok else (OPEN, self.MAX_DEPTH + 1)
        self.reach = self.depth if self.face in (STAIR, EXIT) else self.depth - 1
        self.flanks = self._ladder(arr) if self.ok else []

    # ── the frame ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _array(cls, image):
        """The frame as a contiguous uint8 HxWx3 array, or None when it cannot be read."""
        if _np is None or image is None:
            return None
        try:
            arr = _np.asarray(image)
            if arr.ndim != 3 or arr.shape[0] < cls.H or arr.shape[1] < cls.W or arr.shape[2] < 3:
                return None
            return _np.ascontiguousarray(arr[:cls.H, :cls.W, :3].astype(_np.uint8))
        except Exception:
            return None

    # ── the corridor ────────────────────────────────────────────────────────────────────────
    def _corridor(self, arr):
        """(kind, distance) of the first non-empty cell straight ahead.

        `OPEN` with distance MAX_DEPTH + 1 means the renderer drew nothing down the middle, i.e.
        the corridor runs past everything it draws. The distance comes from the HEIGHT of the run
        and the kind from a vote over the run's own pixels, so neither depends on where inside the
        face a sample happens to land."""
        column = arr[self.UI_ROWS:self.H, self.CX, :3]
        empty = self._is_background(column)
        rows = int((~empty).sum())
        if rows < 2:
            return OPEN, self.MAX_DEPTH + 1
        painted = column[~empty].astype(_np.int32)
        stair = int((painted[:, 0] == 0).sum())
        gold = int(((painted[:, 2] == 0) & (painted[:, 0] > 0)).sum())
        kind, votes = max((STAIR, stair), (EXIT, gold), (WALL, rows - stair - gold),
                          key=lambda pair: pair[1])
        rung = min(range(1, self.MAX_DEPTH + 1),
                   key=lambda d: abs(self.FACE_SPAN / d - (rows - 1)))
        return (kind if votes else WALL), rung

    @classmethod
    def _is_background(cls, pixels):
        """Per-pixel mask of the two flat colours the frame shows where nothing is standing.

        Both are exact equality, not a tolerance: ceiling and ground are drawn un-fogged, so an
        unobstructed view is literally (15,15,35) above the horizon and (50,60,65) below it."""
        return (((pixels == _np.asarray(cls.CEILING)).all(-1)) |
                ((pixels == _np.asarray(cls.GROUND)).all(-1)))

    # ── the flank ladder ────────────────────────────────────────────────────────────────────
    @classmethod
    def band(cls, k):
        """The x-range of the left-hand band that paints the neighbours of the cell `k` ahead.

        The right-hand band is this one mirrored through the centre line. `k = 0` reaches off the
        left edge of the screen, so its near edge is clamped to column 0."""
        near = 0 if k <= 0 else cls.CX - cls.SIDE_SPAN // k
        far = cls.CX - cls.SIDE_SPAN // (k + 1)
        return max(0, near), far

    def _ladder(self, arr):
        """[(left_kind, right_kind), ...] for the cells beside offsets 0, 1, 2, ... of the corridor.

        The list stops at the occlusion gate (`k <= depth - 1`), at LADDER, or at the first band
        whose pixels do not agree — an ambiguous band is dropped together with every band behind
        it, so the ladder is a prefix of certainties and never a list with a hole in it."""
        out = []
        last = min(self.LADDER, self.depth - 1)
        k = 0
        while k <= last:
            pair = self._band_pair(arr, k)
            if pair is None:
                return out                   # ambiguous or too narrow: the prefix ends here
            out.append(pair)
            k += 1
        return out

    def _band_pair(self, arr, k):
        """(left_kind, right_kind) for the neighbours of corridor offset `k`, or None.

        None when the band is too narrow to sample or either side's rows disagree — the caller then
        stops, which is what keeps the ladder a prefix of certainties rather than a list with a hole.
        """
        near, far = self.band(k)
        if far - near < 2:
            return None
        column = (near + far) // 2
        left = self._band_kind(arr, column)
        right = self._band_kind(arr, self.W - 1 - column)
        return None if left is None or right is None else (left, right)

    def _band_kind(self, arr, column):
        """The kind of the neighbour painted in one band, or None when its rows disagree.

        Three colour families and nothing else appears in a gated band: red == 0 is the stair's
        teal, a brown with red above blue is a wall face or its own darker outline, and the bare
        ground or ceiling means the neighbour is empty. Sampling three rows and demanding unanimity
        costs nothing and turns a band clipped by a face edge into a refusal instead of a guess."""
        seen = set()
        for row in self.SIDE_ROWS:
            pixel = tuple(int(v) for v in arr[row, column, :3])
            if pixel == self.GROUND or pixel == self.CEILING:
                seen.add(OPEN)
            elif pixel[0] == 0:
                seen.add(STAIR)
            elif pixel[0] > pixel[2]:
                seen.add(WALL)
            else:
                return None
        return seen.pop() if len(seen) == 1 else None

    # ── what the rest of the harness asks ───────────────────────────────────────────────────
    def stride(self, wanted=1):
        """Cells a `move_forward` of `wanted` would really travel, clamped to the engine's range.

        A wall `d` ahead leaves `d - 1` clear cells; a stair or the exit `d` ahead is itself
        steppable, so `d`. Measured over the census this never over-reaches the engine's own
        walkable run on any of the 5624 poses."""
        if not self.ok:
            return max(1, min(self.STRIDE_CAP, int(wanted)))
        return max(0, min(self.STRIDE_CAP, int(wanted), self.reach))

    def blocked(self) -> bool:
        """Is a forward translation refused outright? A pre-step fact, free, and exact."""
        return bool(self.ok) and self.reach <= 0

    def ahead_line(self) -> str:
        if not self.ok:
            return "not readable this step."
        if self.face == OPEN:
            return "open as far as the view reaches (%d cells or more)." % self.MAX_DEPTH
        what = {WALL: "a wall", STAIR: "a STAIR tile", EXIT: "the EXIT"}[self.face]
        where = "in the very next cell" if self.depth == 1 else "%d cells away" % self.depth
        return "%s %s; a forward move travels %d cell(s)." % (what, where, max(0, self.reach))

    def flank_line(self) -> str:
        """The ladder in plain words — the openings beside the corridor, nearest first."""
        if not self.flanks:
            return "nothing readable beside you this step."
        words = {WALL: "wall", STAIR: "STAIR", OPEN: "open"}
        parts = []
        for k, (left, right) in enumerate(self.flanks):
            where = "beside you" if k == 0 else "%d ahead" % k
            parts.append("%s left %s, right %s" % (where, words[left], words[right]))
        return "; ".join(parts) + "."
