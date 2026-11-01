"""tools/sight.py — the FRAME, DECODED: the local corridor the agent is standing in.

WHY THIS EXISTS. tools/plan.py used to read the frame as one bit (an md5, equal or not) and then,
from the iteration before this one, as two exact colour masks whose painted AREA is a distance
ladder. Both readings answer "is something there"; neither answers the question the clock actually
turns on — HOW FAR CAN I WALK, and WHICH WAY IS OPEN. The diagnosis of the parent is a clock loss:
the win set spends 2.3-5.7 decisions per cell of shortest path, 5 of 16 losses need 32-54 cells of
route out of a 100-decision budget, 36% of turns land on a wall, and 83 translations were spent
proving by execution what the picture already shows.

This module turns the picture into those two answers, and it does it by INVERTING THE RENDERER
rather than by fitting a threshold. The maze3d_pro renderer draws, for depth d = 12 .. 1, three
flat polygons: the FRONT face of the cell d cells straight ahead as the rect of size
(600/d, 400/d) centred on the frame, and the LEFT/RIGHT faces of the cell (d-1) cells ahead as
trapezoids strictly outside that rect. Painter order is far-to-near, so at the frame's centre
column only front faces can ever be painted, and the NEAREST one wins. Two consequences, and they
are the whole module:

  * THE CENTRE COLUMN IS A RANGEFINDER. The contiguous run of non-background pixels through the
    frame's centre is exactly one front face, so its VERTICAL EXTENT is 2*floor((400/d)/2)+1 and
    its COLOUR says which family it belongs to. Extent 401 -> 1 cell, 201 -> 2, 133 -> 3, 101 -> 4,
    81 -> 5, 67 -> 6, 57 -> 7, 51 -> 8, 45 -> 9 (a stair or an exit face reads +2 because its
    3px border is drawn outside the fill). The wall faces are additionally fog-shaded by
    max(0.4, 1 - 0.08d), so (73,36,27) / (67,33,25) / (60,30,22) / ... corroborate the same number.
  * THE FLANKS ARE A MAP STRIP. The side face of the cell k cells ahead occupies the band between
    x = cx - 300/k and x = cx - 300/(k+1) (mirrored on the right), and nothing nearer can overdraw
    it while the front face stands farther away than k. So one pixel per band reads whether the
    cell beside me, and beside each cell down the corridor I can see, is a wall, a stair or open.
    k = 0 is my OWN left and right neighbour: the two bearings a turn would take me to.

MEASURED, not assumed (this build, 1820 decisions over all 25 levels, scripted policy driving the
real evaluator, decode scored against the engine's own tower_data):

    front-face family from the centre pixel   1820 / 1820 exact  (1732 wall, 72 stair, 16 exit)
    front-face distance from the extent       1820 / 1820 exact  (0 off-by-one, 0 unreadable)
    `reach == 0`  <=>  a forward move cannot move the body   1820 / 1820 exact
    `reach` never exceeds the true walkable run              0 violations in 1820
    left / right neighbour (k = 0)            3640 / 3640 exact  (2047 wall, 1558 open, 35 stair)
    side cells at k = 1, 2, 3 while the front face stands farther than k   all exact

WHAT IT STILL REFUSES TO CONCLUDE. The cell BEHIND is never drawn, so a backward move is not
predictable from the frame and is still only knowable by executing it. Nothing past the first front
face is visible, so `reach` is a LOWER bound whenever that face is a stair or the exit. And the tile
UNDER THE BODY is never drawn at all: that fact is still the arrival transition in `StairSense`,
whose premise this module merely makes exact (the stair must be the front face at one cell, not
merely a large patch of cyan somewhere in the view).

Everything here is computed from `frame_data.image`, the frame the agent is shown. No engine state,
no layout, no coordinates, no floor count, no distance to anything.
"""
from __future__ import annotations

try:                                 # the engine hands us a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                    # pragma: no cover
    _np = None


class Scene:
    """ONE frame, decoded. Immutable, ~1 ms, and the only thing in the harness that reads pixels.

    Three readings, all from the same array: the exact-colour CENSUS of the two special tile
    families (what is in view at all), the centre-column RANGEFINDER (what is dead ahead and how
    far), and the flank BANDS (which of the cells beside the corridor are open). The renderer
    constants below are the inversion of maze3d_pro's draw_3d_view and are derived from the frame's
    own shape, so the decode survives a resolution change instead of silently reading a wrong mask.
    """

    __slots__ = ("ok", "stair", "stair_band", "exit", "exit_band",
                 "face", "cells", "reach", "left", "right", "flanks")

    #: the two palette entries draw_3d_view lays down BEFORE any geometry: ceiling over the top half
    #: of the frame, ground over the bottom. Unfogged and exact, so "nothing was drawn here" is an
    #: equality test and not a tolerance.
    BACKGROUND = ((50, 60, 65), (15, 15, 35))
    #: the renderer's base rect, as a fraction of the frame it was authored for (600x400 of 800x600).
    #: Every distance in this module comes out of these two numbers.
    BASE_W, BASE_H = 0.75, 0.6667
    #: draw_3d_view's MAX_DEPTH: nothing beyond this is drawn at all.
    MAX_DEPTH = 12
    #: how many flank bands to read. Band k spans (cx - hw/k, cx - hw/(k+1)), i.e. 100 px wide at
    #: k = 1 and 10 px at k = 5 — past that the band is too thin to sample safely.
    MAX_FLANK = 5
    #: index of the channel that is exactly 0 in each family's palette entries. These two numbers
    #: ARE the census decode: no other colour in the render has a zero channel.
    STAIR_ZERO_CHANNEL = 0
    EXIT_ZERO_CHANNEL = 2
    #: fraction of the frame a special tile paints when it is the ADJACENT cell dead ahead
    #: (0.49..0.51 measured) against 0.13 at two cells. Kept for the census half; the rangefinder
    #: answers the same question exactly and without a cut.
    ADJACENT = 0.30

    def __init__(self, image=None):
        arr = self.to_array(image)
        self.ok = arr is not None
        self.stair, self.stair_band = self.paint(arr, self.STAIR_ZERO_CHANNEL)
        self.exit, self.exit_band = self.paint(arr, self.EXIT_ZERO_CHANNEL)
        self.face, self.cells = self.rangefind(arr)
        self.flanks = self.flank_strip(arr, self.cells)
        here = self.flanks[0] if self.flanks else ("wall", "wall")
        self.left, self.right = here
        if self.face == "wall":
            self.reach = max(self.cells - 1, 0)
        elif self.face == "none":
            self.reach = self.MAX_DEPTH
        else:
            self.reach = self.cells          # a stair or the exit is itself walkable

    # ── the array ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def to_array(image):
        """`frame_data.image` as an HxWx3 int16 array, or None if it is not a colour frame."""
        if image is None or _np is None:
            return None
        arr = _np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        return arr[:, :, :3].astype(_np.int16)

    @classmethod
    def classify(cls, pixel) -> str:
        """The tile family one pixel belongs to: a palette lookup, not a nearest-colour guess.

        The order matters. Both special families are identified by an exactly-zero channel, which
        no wall, ground or ceiling colour has; the two background colours are matched exactly; and
        a wall — front faces (80,40,30) and flank faces (160,85,60), both fog-scaled — is the only
        remaining family with red > green > blue. Anything else is reported as a wall, because the
        one direction in which this decode must never be wrong is claiming a way is open.
        """
        r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
        if r == 0 and (g > 0 or b > 0):
            return "stair"
        if b == 0 and (r > 0 or g > 0):
            return "exit"
        if (r, g, b) in cls.BACKGROUND:
            return "open"
        if r > g > b:
            return "wall"
        return "wall"

    # ── the census (whole-frame colour areas; the previous iteration's reading) ────────────
    @classmethod
    def family_mask(cls, arr, zero_channel: int, painted: int = 1):
        """The exact mask of one tile family: the named channel exactly dark, the other two
        painted. `painted` is 1, not 0, and that single unit is what keeps pure black — which
        satisfies "channel == 0" on every channel — out of both masks."""
        m = arr[:, :, zero_channel] == 0
        for c in (0, 1, 2):
            if c != zero_channel:
                m = m & (arr[:, :, c] >= painted)
        return m

    @classmethod
    def paint(cls, arr, zero_channel: int):
        """(fraction of the whole frame, fraction of the central strip) painted by one family."""
        if arr is None:
            return 0.0, 0.0
        mask = cls.family_mask(arr, zero_channel)
        w = arr.shape[1]
        strip = mask[:, int(w * 0.35):int(w * 0.65)]
        whole = float(mask.mean()) if mask.size else 0.0
        return whole, (float(strip.mean()) if strip.size else 0.0)

    # ── the rangefinder (centre column) ────────────────────────────────────────────────────
    @classmethod
    def rungs(cls, height: int) -> dict:
        """extent in pixels -> distance in cells, GENERATED from the renderer's own arithmetic.

        draw_3d_view gives the front face of the cell d cells ahead a height of BASE_H*height/d and
        centres it, so it covers rows cy +- floor((BASE_H*height/d)/2) and its extent through the
        centre column is 2*that + 1. Reading the ladder out of the frame's own shape rather than
        carrying a table of pixel counts is what makes the decode a measurement: the nine rungs
        this produces for a 600-row frame (401, 201, 133, 101, 81, 67, 57, 51, 45) are exactly the
        nine distinct extents observed over 1820 live frames, one per true distance, no collisions.
        """
        base = cls.BASE_H * height
        out = {}
        for d in range(1, cls.MAX_DEPTH + 1):
            out[2 * int((base / d) // 2) + 1] = d
        return out

    @classmethod
    def rangefind(cls, arr, tol: int = 3):
        """(family, cells) of the nearest front face dead ahead; ("none", 0) if nothing is drawn.

        The run of non-background pixels through the centre of the centre column IS one front face
        — the painter draws far to near and no flank polygon ever reaches the centre column — so
        its extent matches exactly one rung of the ladder. `tol` absorbs the +2 a stair or exit
        face carries from the 3px border drawn outside its fill; every rung is at least 4 px from
        its neighbour at every distance this renderer draws, so the match stays unambiguous.
        """
        if arr is None:
            return "none", 0
        h, w = arr.shape[0], arr.shape[1]
        cy, cx = h // 2, w // 2
        col = arr[:, cx]
        bg = None
        for colour in cls.BACKGROUND:
            hit = (col[:, 0] == colour[0]) & (col[:, 1] == colour[1]) & (col[:, 2] == colour[2])
            bg = hit if bg is None else (bg | hit)
        if bg is None or bool(bg[cy]):
            return "none", 0
        top = cy
        while top > 0 and not bg[top - 1]:
            top -= 1
        bot = cy
        while bot + 1 < h and not bg[bot + 1]:
            bot += 1
        extent = bot - top + 1
        best, cells = tol + 1, 0
        for rung, d in cls.rungs(h).items():
            gap = abs(extent - rung)
            if gap < best:
                best, cells = gap, d
        if not cells:
            return "none", 0
        return cls.classify(col[cy]), cells

    # ── the flanks (one band per visible cell down the corridor) ──────────────────────────
    @classmethod
    def flank_strip(cls, arr, blocked_at: int):
        """[(left, right)] for the cell I stand in and each cell ahead of it that is in sight.

        The side face of the cell k ahead is drawn in the band between x = cx - hw/k and
        x = cx - hw/(k+1) (k = 0 reaching the frame edge), and the only polygon that can overdraw
        that band is a front face standing nearer than k+1. So the strip is read out to
        `blocked_at - 1` and no further: beyond that the band carries the blocking face's own
        colour, which is exactly the garbage this bound excludes.
        """
        if arr is None:
            return [("wall", "wall")]
        h, w = arr.shape[0], arr.shape[1]
        cy, cx = h // 2, w // 2
        hw = cls.BASE_W * w / 2.0
        depth = cls.MAX_DEPTH if blocked_at <= 0 else blocked_at
        out = []
        for k in range(min(depth, cls.MAX_FLANK + 1)):
            near = 0.0 if k == 0 else cx - hw / k
            far = cx - hw / (k + 1)
            x = int(max(near, 0.0) + (far - max(near, 0.0)) / 2.0)
            if x < 0 or x >= cx:
                break
            out.append((cls.classify(arr[cy, x]), cls.classify(arr[cy, w - 1 - x])))
        return out or [("wall", "wall")]

    # ── what the planner asks of a frame ──────────────────────────────────────────────────
    @property
    def stair_ahead(self) -> bool:
        """A stair tile is the VERY NEXT cell dead ahead: one small step lands the body on it."""
        return self.face == "stair" and self.cells == 1

    @property
    def exit_ahead(self) -> bool:
        return self.face == "exit" and self.cells == 1

    @property
    def junction(self) -> int:
        """Cells to the nearest side opening ahead of me, or 0 if the corridor in sight has none.

        A stride that walks past a branch costs the decisions it takes to come back, so this is the
        other cap on the stride: stop ON the junction, where both of its bearings are still a
        single turn away.
        """
        for k, (lf, rt) in enumerate(self.flanks):
            if k and (lf != "wall" or rt != "wall"):
                return k
        return 0

    def stride(self, limit: int = 3) -> int:
        """Cells a forward move should ask for: 0 when the way is provably blocked.

        Four caps, in order of how much they cost to get wrong. The exit is walked ONTO, because
        entering it ends the episode. A stair is walked UP TO and no further: the arrival latch is
        set by a stair that was the front face at one cell and then vanished, so a stride that
        crosses the tile from two cells away lands on it with the fact unmeasured — and the engine
        walks straight through a stair tile without stopping. A branch is stopped at. Everything
        else is the clear run, bounded by the engine's own three-cell maximum.
        """
        if self.face == "exit":
            cap = min(limit, max(self.cells, 1))
        elif self.face == "stair":
            cap = 1 if self.cells <= 1 else min(limit, self.cells - 1)
        else:
            cap = min(limit, self.reach)
        branch = self.junction
        if branch:
            cap = min(cap, branch)
        return max(cap, 0)

    def bearings(self) -> tuple:
        """(left_is_open, right_is_open) for the cell the body is standing in."""
        return self.left != "wall", self.right != "wall"

    def ahead_line(self) -> str:
        """The rangefinder, in the plain English the working-memory block states back."""
        what = {"wall": "a WALL", "stair": "a CYAN STAIR tile", "exit": "the GOLD EXIT",
                "none": "open space"}.get(self.face, "a WALL")
        if self.face == "none":
            return "the way you are facing is clear as far as the view reaches"
        if self.cells <= 1:
            return "%s is IN THE VERY NEXT CELL, directly against you" % what
        return "%s is %d cells away down the way you are facing, with %d clear cell(s) in between" \
            % (what, self.cells, self.cells - 1)

    def open_line(self) -> str:
        """Which bearings a turn can actually take the body to."""
        lf, rt = self.bearings()
        sides = ([("your LEFT", lf), ("your RIGHT", rt)])
        openings = [n for n, v in sides if v]
        closed = [n for n, v in sides if not v]
        said = []
        if openings:
            said.append("%s %s open" % (" and ".join(openings), "is" if len(openings) == 1 else "are"))
        if closed:
            said.append("%s %s walled" % (" and ".join(closed), "is" if len(closed) == 1 else "are"))
        return "; ".join(said)

    def describe(self) -> str:
        """The CENSUS, in the plain English the working-memory block states back to the agent.

        Deliberately only the half the rangefinder does not already state: what is in view at all
        but NOT dead ahead. A tile painted off the centre line is beside the corridor, which means
        it is reached by turning rather than by walking, and that is the only thing the whole-frame
        colour areas can say that the centre column cannot.
        """
        said = []
        if self.exit > 0.0 and self.face != "exit":
            said.append("the GOLD EXIT is painted somewhere in this view but NOT dead ahead, so it "
                        "is beside the way you are facing")
        if self.stair > 0.0 and self.face != "stair":
            said.append("a CYAN STAIR is painted somewhere in this view but NOT dead ahead, so it "
                        "is beside the way you are facing")
        return "; ".join(said) or "no stair and no exit anywhere in this view except what is listed above"

    @classmethod
    def of(cls, image):
        """The decode of the frame the agent is being shown. Never raises."""
        try:
            return cls(image)
        except Exception:
            try:
                return cls(None)
            except Exception:                # pragma: no cover
                return None


class StairSense:
    """The floor sense: a latch saying a stair is underfoot, and the price of spending it.

    Deliberately NOT a map and NOT a pose. It knows three things — whether the body is standing on
    a stair (an exact frame transition), how many presses that latch is still worth (a budget,
    because an unchanged frame cannot refute a climb), and which transition to try (down only once
    a climb has been spent going up, so a probe never immediately undoes the last one).

    The transition itself is the one fact the picture cannot state directly, because the tile under
    the body is never drawn: a stair that was THE FRONT FACE AT ONE CELL, a translation, a frame
    that changed, and the stair gone from in front => the body is standing on it. Each clause is
    load-bearing, and the first one is now the rangefinder's exact reading rather than a share of
    the frame's cyan: measured on the parent's live tape the area version scored 110 true positives,
    1 false positive and 0 false negatives over 1880 decisions, and the one false positive is the
    case the exact version removes (cyan on a flank counted toward the area).
    """

    __slots__ = ("underfoot", "spent", "level", "arrivals", "presses")

    #: THE PRICE OF LEAVING A FLOOR, in two parts. Climbing at the FIRST stair the body lands on is
    #: a measured trap — the exit sits on the STARTING floor on 13 of 25 levels, so leaving early
    #: throws away levels that were already being won — but a steep price never gets paid at all,
    #: and an episode that never changes floor cannot win the 12 whose exit is upstairs.
    #: `STALE_FLOOR`    consecutive decisions this floor must show no view it had not already shown.
    #: `FLOOR_PATIENCE` decisions this floor must have been given at all, counted from arriving.
    STALE_FLOOR = 1
    FLOOR_PATIENCE = 24
    #: grounded climbs per latch. A bound, not a knob: one press settles it, and a frame that did
    #: not change is not proof the press failed.
    CLIMB_BUDGET = 1

    def __init__(self):
        self.underfoot = False      # a stair is under the body right now
        self.spent = 0              # grounded climbs pressed since the latch was set
        self.level = 0              # floors climbed, net, as believed from our own presses
        self.arrivals = 0           # stair tiles stepped onto this episode
        self.presses = 0            # grounded climbs pressed this episode

    @staticmethod
    def arrived(before, after, moved: bool) -> bool:
        """The exact on-stair transition. The tile underfoot is never drawn, so the stair's
        disappearance from the cell in front IS the arrival."""
        if before is None or after is None or not moved:
            return False
        return before.stair_ahead and not after.stair_ahead

    def update(self, before, after, moved: bool, translated: bool) -> None:
        """Fold one measured outcome in. A translation that moved the body either lands it on a
        stair (latch set, budget refilled) or carries it off one (latch cleared)."""
        if not translated or not moved:
            return
        arrived = self.arrived(before, after, moved)
        self.underfoot, self.spent = arrived, 0
        self.arrivals += 1 if arrived else 0

    def patience_owed(self, floor_age: int) -> int:
        """Decisions this floor is still owed in CLOCK alone — the half of the price that does not
        depend on what has been seen. This is the whole price of a climb the MODEL asks for."""
        return max(self.FLOOR_PATIENCE - floor_age, 0)

    def owed(self, stale: int, floor_age: int) -> int:
        """Decisions still owed to this floor before the HARNESS will spend its stair unasked: the
        clock half above, and the staleness half — this floor must also have stopped producing views
        it had not already shown."""
        return max(self.STALE_FLOOR - stale, self.patience_owed(floor_age))

    def wanted(self, stale: int, floor_age: int, seen) -> str:
        """The climb worth pressing right now, or "".

        Four conditions, each measured: a stair is underfoot, the latch's budget is unspent, this
        floor has been given its patience AND has stopped producing new views, and there is no exit
        in sight — because an exit anywhere in this view is reachable on this floor and ends the
        episode, which no floor change can improve on."""
        if not self.underfoot or self.spent >= self.CLIMB_BUDGET:
            return ""
        if seen is not None and seen.exit > 0.0:
            return ""
        if self.owed(stale, floor_age) > 0:
            return ""
        return "climb_down" if self.level > 0 else "climb_up"

    def pressed(self, name: str) -> None:
        """Record a grounded climb: the budget drops and the believed floor moves.

        The latch is deliberately LEFT SET. The body is still standing on the stair tile (a stair
        runs between the two floors), so clearing it would be a false statement about the pixels;
        what stops a second press is the BUDGET, and what re-arms it is stepping off the tile and
        arriving on one again — the only event that can prove the fact afresh."""
        self.spent += 1
        self.presses += 1
        self.level += 1 if name == "climb_up" else -1

    def line(self) -> str:
        """The latch, as the working-memory block states it."""
        if not self.underfoot:
            return "you are NOT standing on a stair (no stair paint vanished from under you)"
        return ("you ARE STANDING ON A STAIR right now - the tile under your feet is not drawn, "
                "which is exactly why the screen shows none")
