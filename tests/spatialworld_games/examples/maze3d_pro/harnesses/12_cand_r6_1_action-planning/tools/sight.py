"""ONE FRAME, READ AS GEOMETRY — the corridor straight ahead and the openings beside it.

This is the planner's only instrument. `tools/plan.py` plans a route over a chart; this module is
what fills the chart. It reads `frame_data.image` — the frame the agent itself is shown — and
nothing else: no engine state, no layout, no coordinates, no floor count.

WHAT THE PIXELS CARRY (a census of rendered frames; the proposer sandbox blocks the renderer source
and a census is the stronger warrant anyway, because it would catch a renderer change that a
transcribed constant would not):

  1. NO TEXT IS DRAWN. There is no coordinate readout, no bearing readout, and no `[Q] UP` /
     `[E] DOWN` stair prompt anywhere in the render. Every prior that gates a climb on reading that
     prompt gates it on nothing, which is why the pass-through ancestor of this harness pressed a
     climb zero times in 2340 decisions.
  2. THE DISTANCE AHEAD IS THE HEIGHT OF THE PAINT DOWN THE CENTRE COLUMN. Whatever blocks the
     corridor is centred on the horizon and its height scales as 1/distance: the run of pixels down
     column 400 that is neither ceiling (15,15,35) nor ground (50,60,65) spans about 401 px at one
     cell, 201 at two, 133 at three, 101 at four, 81 at five, 67 at six — `FACE_SPAN / d` on the
     nose, and the bands are disjoint, so rounding to the nearest rung is exact rather than fitted.
  3. WHAT IS AT THAT DISTANCE IS THE COLOUR OF THAT RUN, and only three families exist: red == 0 is
     the teal stair face, blue == 0 the un-fogged gold of the exit, anything else a brown wall.
     The vote is taken over the WHOLE run, never one sampled pixel, which a stair's diagonals or an
     outline can flip.
  4. THE OPENINGS BESIDE THE CORRIDOR tile the screen in order of distance: the two neighbours of
     the corridor cell `k` steps ahead are painted into the band that ends `SIDE_SPAN // (k + 1)`
     off the centre line and starts `SIDE_SPAN // k` off it. Those bands are disjoint and ordered,
     so a face at one distance can never be read as a face at another, and sampling the band's
     midpoint at three rows straddling the horizon dodges both the 1 px face outline and the
     stair's own frame. An EMPTY neighbour paints nothing at all, so its band shows bare ground or
     ceiling.
  5. THE OCCLUSION GATE IS `k <= depth - 1`, where `depth` is the distance of the nearest blocking
     face — not the walkable run. A face covers every band nearer the centre line than its own
     edge, and the band for `k = depth - 1` ends exactly where the face begins, so the gated bands
     are precisely the ones nothing can cover.

WHAT IT REFUSES TO CONCLUDE. Nothing is painted for the cell under the body, so "am I standing on a
stair" is NOT a reading — the chart answers it, from a stair seen earlier and a reckoned pose. An
exit beside the corridor paints no side face either, so a flank reading of `open` means PASSABLE and
never NOT-THE-GOAL; `tools/chart.py` keeps those two kinds of open strictly apart.
"""
from __future__ import annotations

import hashlib

try:                                 # the engine hands over a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                    # pragma: no cover
    _np = None

WALL, STAIR, EXIT, OPEN = "wall", "stair", "exit", "open"


class Sight:
    """One frame, read whole: how far the corridor runs, what closes it, and what lies beside it.

    Immutable and cheap — one boolean reduction down a single column plus about forty sampled
    pixels — and the only thing in the harness that looks at the frame by anything but a digest.
    """

    __slots__ = ("ok", "digest", "face", "depth", "reach", "flanks")

    #: THE RULER, in the frame's own pixels. The evaluator fixes the surface for every episode, so
    #: these are measurements of the picture rather than a projection to be scaled; a frame of any
    #: other shape is reported unreadable instead of being decoded on a guess.
    WIDE = 800
    TALL = 600
    MIDDLE = 400
    HORIZON = 300
    UI_ROWS = 60                     # the top band is blacked out by a bar: start scanning below it
    FACE_SPAN = 400                  # rows a blocking face spans at one cell; FACE_SPAN // d at d
    SIDE_SPAN = 300                  # the k-th neighbour's band ends SIDE_SPAN // (k+1) off centre
    FAR = 12                         # the deepest cell the renderer draws
    RUNGS = 6                        # neighbour pairs read per look; the 6th band is 7 px wide
    CAP = 3                          # the engine's `large`
    CEILING = (15, 15, 35)
    GROUND = (50, 60, 65)
    SIDE_ROWS = (298, 300, 302)      # three rows straddling the horizon, where a band is tallest
    #: the fraction of the frame that must be EXACTLY one of the two background colours before this
    #: decode is willing to speak. Every number above is an equality against this renderer's palette,
    #: so a frame from a different renderer — or a blank one — would otherwise decode to confident
    #: nonsense and fill the chart with it. A real corridor view is mostly ceiling and ground; an
    #: eighth is a floor so loose that no view of this build can fail it, and so tight that nothing
    #: else can pass it.
    GROUNDED = 0.125

    def __init__(self, image=None):
        arr = self.as_array(image)
        self.ok = arr is not None and self.grounded(arr)
        self.digest = hashlib.md5(arr.tobytes()).hexdigest() if self.ok else ""
        self.face, self.depth = self.corridor(arr) if self.ok else (OPEN, self.FAR + 1)
        self.reach = self.depth if self.face in (STAIR, EXIT) else self.depth - 1
        self.flanks = self.ladder(arr) if self.ok else []

    @classmethod
    def as_array(cls, image):
        """The frame as a contiguous uint8 HxWx3 array, or None when it cannot be read."""
        if _np is None or image is None:
            return None
        try:
            arr = _np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return None
            if arr.shape[0] < cls.TALL or arr.shape[1] < cls.WIDE:
                return None
            return _np.ascontiguousarray(arr[:cls.TALL, :cls.WIDE, :3].astype(_np.uint8))
        except Exception:
            return None

    @classmethod
    def grounded(cls, arr):
        """Does this frame come from the renderer every constant in this module was measured on?

        One cheap reduction, and it is the gate on the whole mechanism: if the two flat background
        colours are not actually there, nothing below is a measurement and the harness must say so
        rather than decode a guess. `tools/plan.py:honours` stands the governor down to a
        pass-through on an unreadable frame, so the cost of refusing is the parent's behaviour and
        the cost of NOT refusing is a chart full of confident fiction."""
        try:
            return float(cls.is_background(arr).mean()) >= cls.GROUNDED
        except Exception:
            return False

    @classmethod
    def is_background(cls, pixels):
        """Per-pixel mask of the two flat colours the frame shows where nothing is standing.

        Exact equality, not a tolerance: ceiling and ground are drawn un-fogged, so an unobstructed
        view is literally (15,15,35) above the horizon and (50,60,65) below it."""
        return (((pixels == _np.asarray(cls.CEILING)).all(-1)) |
                ((pixels == _np.asarray(cls.GROUND)).all(-1)))

    def corridor(self, arr):
        """`(kind, distance)` of the first non-empty cell straight ahead.

        `OPEN` at distance `FAR + 1` means the renderer drew nothing down the middle, i.e. the
        corridor runs past everything it draws. The distance comes from the HEIGHT of the run and
        the kind from a vote over the run's own pixels, so neither depends on where inside the face
        a sample happens to land."""
        column = arr[self.UI_ROWS:self.TALL, self.MIDDLE, :3]
        painted = column[~self.is_background(column)].astype(_np.int32)
        rows = int(painted.shape[0])
        if rows < 2:
            return OPEN, self.FAR + 1
        teal = int((painted[:, 0] == 0).sum())
        gold = int(((painted[:, 2] == 0) & (painted[:, 0] > 0)).sum())
        kind, votes = max((STAIR, teal), (EXIT, gold), (WALL, rows - teal - gold),
                          key=lambda pair: pair[1])
        rung = min(range(1, self.FAR + 1), key=lambda d: abs(self.FACE_SPAN / d - (rows - 1)))
        return (kind if votes else WALL), rung

    @classmethod
    def band(cls, k):
        """The x-range of the left-hand band painting the neighbours of the cell `k` ahead.

        The right-hand band is this one mirrored through the centre line. `k = 0` reaches off the
        left edge of the screen, so its near edge is clamped to column 0."""
        near = 0 if k <= 0 else cls.MIDDLE - cls.SIDE_SPAN // k
        far = cls.MIDDLE - cls.SIDE_SPAN // (k + 1)
        return max(0, near), far

    def ladder(self, arr):
        """`[(left_kind, right_kind), ...]` for the cells beside offsets 0, 1, 2, ... ahead.

        The list stops at the occlusion gate, at `RUNGS`, or at the first band whose rows disagree —
        an ambiguous band is dropped together with every band behind it, so the ladder is a PREFIX
        OF CERTAINTIES and never a list with a hole in it."""
        out = []
        for k in range(0, min(self.RUNGS, self.depth - 1) + 1):
            near, far = self.band(k)
            if far - near < 2:
                break
            column = (near + far) // 2
            left = self.band_kind(arr, column)
            right = self.band_kind(arr, self.WIDE - 1 - column)
            if left is None or right is None:
                break
            out.append((left, right))
        return out

    def band_kind(self, arr, column):
        """The kind of the neighbour painted in one band, or None when its rows disagree.

        Three colour families and nothing else appears in a gated band: red == 0 is the stair's
        teal, a brown with red above blue is a wall face or its own darker outline, and bare ground
        or ceiling means the neighbour is empty. Demanding unanimity over three rows costs nothing
        and turns a band clipped by a face edge into a refusal instead of a guess."""
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

    def stride(self, wanted=1):
        """Cells a `move_forward` of `wanted` would really travel, clamped to the engine's range.

        A wall `d` ahead leaves `d - 1` clear cells; a stair or the exit `d` ahead is itself
        steppable, so `d`. Never over-reaches the engine's own walkable run."""
        if not self.ok:
            return max(1, min(self.CAP, int(wanted)))
        return max(0, min(self.CAP, int(wanted), self.reach))

    def blocked(self):
        """Is a forward translation refused outright? A PRE-step fact, free and exact — which is
        the whole point: the ancestor harness had to spend a decision to learn it, 1829 times."""
        return bool(self.ok) and self.reach <= 0

    def ahead_line(self):
        """The corridor in the plain words the working-memory block states back to the agent."""
        if not self.ok:
            return "not readable this step."
        if self.face == OPEN:
            return "open for at least %d cells." % self.FAR
        what = {WALL: "a WALL", STAIR: "a CYAN STAIR tile", EXIT: "the GOLD EXIT"}[self.face]
        where = "in the very next cell" if self.depth == 1 else "%d cells away" % self.depth
        return "%s %s; a forward move travels %d cell(s)." % (what, where, max(0, self.reach))

    def flank_line(self):
        """The openings beside the corridor, nearest first."""
        if not self.flanks:
            return "nothing readable to either side this step."
        words = {WALL: "wall", STAIR: "STAIR", OPEN: "open"}
        parts = []
        for k, pair in enumerate(self.flanks):
            where = "beside you" if k == 0 else "%d ahead" % k
            parts.append("%s left %s / right %s" % (where, words[pair[0]], words[pair[1]]))
        return "; ".join(parts) + "."


def look(image):
    """The census of the frame the agent is being shown. Never raises."""
    try:
        return Sight(image)
    except Exception:
        return Sight(None)
