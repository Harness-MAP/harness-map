"""tools/sight.py — the FRAME, READ AS GEOMETRY: the corridor ahead, the openings beside it, and
the two special tiles — plus the floor sense that all of it supports.

WHY THIS FILE GREW. The previous build read the frame by COLOUR only: an exact mask for the cyan
stair and the gold exit, and the painted area as a distance ladder. That solved the floor
transition (the stair latch measured 110 TP / 1 FP / 0 FN over 1880 live decisions, 15/15 grounded
climbs effective) and it left the other half of the maze untouched: every one of the 16 losses ran
out of its 100-decision clock, the agent gained 1.02 cells per forward move against routes 32-54
cells long, and it visited 36% of the cells it could reach. The frame was being asked where the
STAIR is and never asked where the CORRIDOR goes.

It goes somewhere very specific, and the FRAMES say so exactly. Everything below is a census of
rendered frames — every walkable cell of all 25 levels seen from all four bearings, 5624 frames,
each one decoded and then checked against where the body actually could and could not go. Nothing
here is reasoned from how the picture is produced; it is all read off the picture:

  1. THE DISTANCE AHEAD is the HEIGHT of the paint down the centre column. The shape that blocks the
     corridor is centred on the horizon and its height scales as 1/distance, so the run of pixels
     down the middle column that is neither ceiling nor ground measures 401/403 px at 1 cell,
     201/203 at 2, 133/135 at 3, 101/103 at 4, 81/83 at 5, 67/69 at 6, 57/59 at 7, 51/53 at 8,
     45 at 9, 41 at 10, 37 at 11 — disjoint bands, every one of them `round(400 / (rows - 1))` on
     the nose. **5624/5624 frames decode to the true distance.**
  2. WHAT IS AT THAT DISTANCE is the COLOUR of that run: red == 0 is the teal stair face, blue == 0
     the single gold (255,200,0) of the exit, anything else the brown wall. **5624/5624 exact**,
     144 exits / 273 stairs / 5207 walls, zero confusion either way.
  3. THE OPENINGS BESIDE THE CORRIDOR are the ABSENCE of paint in a side band. A neighbouring cell
     shows a face only when something solid (a wall) or a stair tile is standing there, so an EMPTY
     neighbour leaves its band showing bare ceiling/ground. The k-th neighbour's band was located by
     scanning the frame for where the paint boundaries fall and lands at `SIDE_SPAN/k` off the
     centre line; sampling its midpoint at the horizon classifies each neighbour wall / stair /
     open at **23688/23688 exact over the same frames, five depths, both sides.**

WHAT THAT BUYS, all of it a MEASUREMENT and none of it a guess:

  * THE LEGAL STRIDE, for free, before the move is spent. A wall `d` cells ahead means `d-1` clear
    cells; a stair or an exit `d` cells ahead is itself steppable, so `d`. Clamped to the engine's
    3-cell maximum this NEVER over-reaches: 0 of 5624 poses read a stride longer than the real
    walkable run, and the under-reads are the 64 poses where a stair or exit occludes the corridor
    behind it, which is precisely where stopping is wanted anyway.
  * AN EXACT PRE-STEP "the way ahead is blocked". A zero stride and a wall in the very next cell
    are the same event, and it decodes on 2882 of 5624 poses with no false positive and no false
    negative. The fingerprint ledger learns this only by SPENDING a decision on the refused move;
    the frame gives it away for nothing, from poses never visited before.
  * WHERE TO TURN WHEN THE WAY AHEAD IS SHUT, instead of rotating blind. A side the frame shows
    solid cannot be worth a probe, so the escape rotates onto the open one — which walks the body
    out of a sealed pocket in three decisions instead of five. (It must still ROTATE: see the trap
    recorded in `plan.ActionPlanner._reaim`.)
  * A JUNCTION LINE. The first opening within reach is where a long stride would carry the body
    PAST a turning, so the stride stops there. It costs almost nothing: 34.4% of poses can stride
    2+ cells and 32.3% still can after every junction has been honoured.

WHAT IT STILL REFUSES TO CONCLUDE. Nothing is painted for the cell UNDER the body, so "am I on a
stair" remains a transition and not a reading (see `StairSense`). An EXIT beside the body paints no
side band either — only walls and stairs do, measured — so an adjacent exit off the facing axis
reads as an opening, which is what it is, and nothing here claims more. And a
climb that SUCCEEDS can leave the frame byte-identical, so an unchanged frame is never read as a
failed climb: a grounded climb is spent from a budget instead.

Everything in this module is computed from `frame_data.image`, the frame the agent is shown. No
engine state, no layout, no coordinates, no floor count, no distance to the exit.
"""
from __future__ import annotations

try:                                 # the engine hands us a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                    # pragma: no cover
    _np = None

#: index of the channel that is exactly 0 in each family's palette entries. These two numbers ARE
#: the colour decode: no other colour in the render has a zero channel.
STAIR_ZERO_CHANNEL = 0
EXIT_ZERO_CHANNEL = 2

#: fraction of the frame a special tile paints when it is the ADJACENT cell dead ahead (0.49 .. 0.51
#: measured) against 0.13 at two cells — one cut, a factor of 4 of headroom either side.
ADJACENT = 0.30

#: THE PRICE OF LEAVING A FLOOR, in two parts. Climbing at the FIRST stair the body lands on is a
#: measured trap — the exit sits on the STARTING floor on 13 of the 25 levels, so leaving early
#: throws away levels that were already being won — but a steep price never gets paid at all, and an
#: episode that never changes floor cannot win the 12 levels whose exit is upstairs.
#: `STALE_FLOOR`     consecutive decisions this floor must show no view it had not already shown.
#: `FLOOR_PATIENCE`  decisions this floor must have been given, counted from arriving on it.
#: Both were swept on two independent offline rigs and are the only tuned constants in this file;
#: every other number here is a MEASUREMENT of the render.
STALE_FLOOR = 1
FLOOR_PATIENCE = 24

#: grounded climbs per latch. A bound, not a knob: one press settles it, and a frame that did not
#: change is not proof the press failed.
CLIMB_BUDGET = 1


class Sight:
    """One frame, read whole: the colour census, the corridor, and the neighbours beside it.

    Immutable and cheap (two boolean reductions over the frame plus about twenty sampled pixels),
    and the only thing in the harness that looks at pixels by anything other than an md5.
    """

    __slots__ = ("ok", "stair", "stair_band", "exit", "exit_band",
                 "depth", "face", "reach", "stride", "junction",
                 "left", "right", "side_stair")

    #: the two flat colours the frame shows where nothing is standing: everything else on the centre
    #: column or in a side band is paint. Measured across the 5624-frame census, the upper half of an
    #: unobstructed view is exactly (15,15,35) and the lower half exactly (50,60,65) — never shaded,
    #: so both comparisons are exact equality and not a tolerance.
    CEILING_RGB = (15, 15, 35)
    GROUND_RGB = (50, 60, 65)

    #: THE RULER the frames themselves fix, as fractions of the frame so the readout survives a
    #: resolution change. A blocking shape one cell ahead is `FACE_SPAN` of the frame height
    #: (400 of 600) and one `d` cells ahead is that over `d`; the paint beside a cell ends
    #: `SIDE_SPAN` of the width off the centre line (300 of 800) and the k-th neighbour's band ends
    #: at that over k. Both were read off the census, not assumed.
    FACE_SPAN = 400.0 / 600.0
    SIDE_SPAN = 300.0 / 800.0

    #: rows sampled in a side band, as fractions of the height: three straddling the horizon (where
    #: a side band's paint is tallest at every depth) and one well inside the ground half.
    SIDE_ROWS = (0.4967, 0.5000, 0.5033, 0.5333)

    #: the top 60 rows of every frame are blacked out by a UI bar, so the centre-column scan starts
    #: below them. The tallest blocking shape measured spans rows 100..500, so nothing is lost.
    SCAN_TOP = 60.0 / 600.0

    #: the furthest distance any frame in the census resolved (12), the engine's 3-cell granularity
    #: ceiling, and how many neighbours out the side bands stay wide enough to sample (the 5th is
    #: 15 px across).
    MAX_DEPTH = 12
    STRIDE_CAP = 3
    SIDE_SCAN = 4

    def __init__(self, image=None):
        arr = self._array(image)
        self.ok = arr is not None
        self.stair, self.stair_band = self._paint(arr, STAIR_ZERO_CHANNEL)
        self.exit, self.exit_band = self._paint(arr, EXIT_ZERO_CHANNEL)
        self.depth, self.face = self._corridor(arr)
        self.left, self.right, self.side_stair = self._neighbours(arr, self.depth)
        # THE STRIDE THE ENGINE WOULD HONOUR. A wall `d` cells off leaves `d-1` clear cells. The
        # EXIT is different in both directions: stepping onto it ends the episode, so the stride
        # goes all the way to it. And a STAIR is different again — it is steppable, but the
        # arrival latch in `StairSense` is a TRANSITION that needs the saturated one-cell approach
        # to have been on screen first, so a stride that lands on a stair from two cells away
        # reaches the tile and does not RECOGNISE it. Stopping one cell short keeps the latch
        # exact at a cost of nothing: the next step lands on it with the approach frame seen.
        # (Measured: striding over stairs took effective floor changes from 16 to 3 on the stub
        # rig, which is the whole of the previous build's gain.) 0 of 5624 poses read `reach`
        # longer than the engine's own walkable run.
        if not self.depth:
            far = 0
        elif self.face == "exit":
            far = self.depth
        elif self.face == "stair":
            far = max(1, self.depth - 1)
        else:
            far = self.depth - 1
        self.reach = max(0, min(far, self.STRIDE_CAP))
        # ... and the stride actually worth asking for: never past the first WALKABLE neighbour,
        # because a turning walked past costs more than the cell it saved, and a stair walked past
        # may be the only one on the floor.
        self.junction = next((k for k in range(1, self.reach + 1)
                              if self.left[k:k + 1] in (("open",), ("stair",))
                              or self.right[k:k + 1] in (("open",), ("stair",))), 0)
        self.stride = min(self.reach, self.junction) if self.junction else self.reach

    # ── pixels ─────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _array(image):
        """`frame_data.image` as an HxWx3 int16 array, or None if it is not a colour frame."""
        if image is None or _np is None:
            return None
        arr = _np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        return arr[:, :, :3].astype(_np.int16)

    def _paint(self, arr, zero_channel: int):
        """(fraction of the whole frame, fraction of the central third) painted by one tile family.

        The mask is the named channel exactly dark with the other two painted; the floor of 1 on
        those two is what keeps pure black — which satisfies "channel == 0" everywhere — out of both
        families. No other colour in the 30-entry palette has a zero channel."""
        if arr is None:
            return 0.0, 0.0
        mask = arr[:, :, zero_channel] == 0
        for c in (0, 1, 2):
            if c != zero_channel:
                mask = mask & (arr[:, :, c] >= 1)
        width = arr.shape[1]
        strip = mask[:, int(width * 0.35):int(width * 0.65)]
        whole = float(mask.mean()) if mask.size else 0.0
        return whole, (float(strip.mean()) if strip.size else 0.0)

    def _corridor(self, arr):
        """(cells to the first drawn face, what that face is) — the CENTRE-COLUMN read.

        Only the NEAREST thing in the corridor reaches the centre column — side bands never do, and
        anything further away is hidden behind what is in front of it — so the height of the
        non-ceiling, non-ground run down that column is `FACE_SPAN / d` of the frame and its colour
        says what is standing there. Measured 5624/5624 for the distance and 5624/5624 for the kind
        over every walkable cell of all 25 levels from all four bearings."""
        if arr is None:
            return 0, "none"
        height, width = arr.shape[0], arr.shape[1]
        column = arr[int(height * self.SCAN_TOP):height, width // 2, :]
        bare = ((column[:, 0] == self.CEILING_RGB[0]) & (column[:, 1] == self.CEILING_RGB[1])
                & (column[:, 2] == self.CEILING_RGB[2])) | \
               ((column[:, 0] == self.GROUND_RGB[0]) & (column[:, 1] == self.GROUND_RGB[1])
                & (column[:, 2] == self.GROUND_RGB[2]))
        painted = ~bare
        rows = int(painted.sum())
        if rows < 2:
            return 0, "none"
        lit = column[painted]
        teal = int(((lit[:, 0] == 0) & (lit[:, 1] >= 1) & (lit[:, 2] >= 1)).sum())
        gold = int(((lit[:, 2] == 0) & (lit[:, 0] >= 1) & (lit[:, 1] >= 1)).sum())
        if teal > gold and teal * 2 > rows:
            face = "stair"
        elif gold * 2 > rows:
            face = "exit"
        else:
            face = "wall"
        depth = int(round(height * self.FACE_SPAN / (rows - 1)))
        return max(1, min(depth, self.MAX_DEPTH)), face

    def _neighbours(self, arr, depth: int):
        """(left kinds, right kinds, the nearest side stair) — the SIDE-BAND read.

        The k-th neighbour's band lies between `SIDE_SPAN/k` and `SIDE_SPAN/(k+1)` off the centre
        line — located by finding where the paint boundaries fall in the census frames — and its
        midpoint at the horizon is inside that band at every depth. A band showing bare ceiling or
        ground carries no paint at all, which means that neighbour is EMPTY. Bands at or beyond the
        first thing in the corridor are hidden behind it and are not reported — that is an
        occlusion, not an opening. 23688/23688 exact, three ways, both sides."""
        if arr is None:
            return (), (), None
        height, width = arr.shape[0], arr.shape[1]
        rows = [min(height - 1, int(height * f)) for f in self.SIDE_ROWS]
        reachable = self.MAX_DEPTH if not depth else depth
        half = self.SIDE_SPAN * width
        left, right, stair_at = [], [], None
        for k in range(0, min(self.SIDE_SCAN, reachable - 1) + 1):
            inner = width // 2 - half / (k + 1)
            outer = 0.0 if k == 0 else width // 2 - half / k
            column = int((inner + outer) / 2.0)
            for side, bucket in (("left", left), ("right", right)):
                col = column if side == "left" else width - 1 - column
                open_votes, teal_votes = 0, 0
                for row in rows:
                    pixel = (int(arr[row, col, 0]), int(arr[row, col, 1]), int(arr[row, col, 2]))
                    if pixel == self.CEILING_RGB or pixel == self.GROUND_RGB:
                        open_votes += 1
                    elif pixel[0] == 0 and pixel[1] >= 1 and pixel[2] >= 1:
                        teal_votes += 1
                kind = "open" if open_votes >= len(rows) - 1 else \
                    ("stair" if teal_votes >= 2 else "wall")
                bucket.append(kind)
                if kind == "stair" and stair_at is None:
                    stair_at = (k, side)
        return tuple(left), tuple(right), stair_at

    # ── what the rest of the harness asks it ───────────────────────────────────────────────
    @property
    def stair_adjacent(self) -> bool:
        """A stair tile is the VERY NEXT cell dead ahead: one `small` step lands on it."""
        return self.stair >= ADJACENT

    @property
    def exit_adjacent(self) -> bool:
        return self.exit >= ADJACENT

    @property
    def blocked(self) -> bool:
        """The very next cell ahead is solid, so `move_forward` cannot move the body AT ALL.

        Exact and free: it agreed with the engine on 2882 of 2882 blocked poses and never fired on
        a pose that could move. The fingerprint ledger can only learn this by spending a decision."""
        return self.ok and self.reach == 0

    def open_sides(self):
        """The neighbours of the cell the body is IN that the frame shows WALKABLE, in the order a
        right-handed sweep wants to try them. `("right",)` means only the right-hand cell is free.

        A side STAIR counts as walkable, because it is: a cyan side band and a brown one are both
        paint, but only the brown one is an obstacle — the cyan is a cell the body can step into.
        An EXIT beside the body paints no side band at all and therefore reads as plain open, which
        is also exactly right."""
        out = []
        for side, kinds in (("right", self.right), ("left", self.left)):
            if kinds[0:1] in (("open",), ("stair",)):
                out.append(side)
        return tuple(out)

    def cells_away(self, fraction: float) -> int:
        """Cells to a tile painting `fraction` of the frame down the corridor; 0 = not in view.

        The area ladder, kept for the two special tiles because it reads them even when they are
        OFF the centre column (where the geometric corridor read cannot see them). A stair down a
        clear corridor paints 0.5063 / 0.1281 / 0.0571 / 0.0328 / 0.0213 / 0.0148 of the frame at
        1 .. 6 cells and the exit 0.4938 / 0.1219 / 0.0529 / 0.0297 at 1 .. 4, the same values byte
        for byte on every level; each cut below sits at the geometric mean of the rungs it parts."""
        if fraction <= 0.0:
            return 0
        for cut, cells in ((ADJACENT, 1), (0.09, 2), (0.045, 3), (0.026, 4), (0.018, 5)):
            if fraction >= cut:
                return cells
        return 6

    def describe(self) -> str:
        """The census in the plain English the working-memory block states back to the agent."""
        said = []
        if self.exit:
            said.append("the GOLD EXIT is in sight, %s" % (
                "the VERY NEXT CELL dead ahead" if self.exit_adjacent
                else "about %d cells down the way you are facing" % self.cells_away(self.exit)))
        if self.stair:
            said.append("a CYAN STAIR tile is in sight, %s" % (
                "the VERY NEXT CELL dead ahead" if self.stair_adjacent
                else "about %d cells down the way you are facing" % self.cells_away(self.stair)))
        if self.side_stair is not None:
            k, side = self.side_stair
            said.append("a CYAN STAIR is in the cell to your %s%s - turn that way to reach it" % (
                side, "" if not k else ", %d cell(s) further down this corridor" % k))
        return "; ".join(said) or "no stair and no exit anywhere in this view"

    def corridor_line(self) -> str:
        """The geometry, as the working-memory block states it: how far the way ahead is open, what
        closes it, and where the openings beside it are."""
        if not self.ok:
            return "the screen could not be measured this step."
        what = {"wall": "a WALL", "stair": "a CYAN STAIR tile", "exit": "the GOLD EXIT",
                "none": "nothing measurable"}[self.face]
        if self.reach <= 0:
            head = "%s fills the VERY NEXT CELL ahead - `move_forward` cannot move you at all " \
                   "from here, at any granularity." % what
        else:
            head = "%d cell(s) of the way ahead are CLEAR (%s stands %d cell(s) off), so the " \
                   "longest step worth asking for here is `move_forward` with `%s`." % (
                       self.reach, what, self.depth, GRANULARITY[self.stride])
        gaps = []
        for side, kinds in (("left", self.left), ("right", self.right)):
            for k, kind in enumerate(kinds):
                if kind == "open":
                    gaps.append("%s at %s" % (side, "your own cell" if not k
                                              else "%d cell(s) ahead" % k))
        if not gaps:
            if self.reach <= 0:
                return head + " Both cells beside you are solid too, so this is a DEAD END: TURN " \
                              "AROUND and walk out facing the way you came. Stepping backward " \
                              "would leave you facing the dead end and you would walk back into it."
            return head + " Both walls beside you run solid as far as you can see."
        tail = " OPENINGS off this corridor: %s." % ", ".join(gaps)
        if self.junction:
            tail += " The first one is %d cell(s) on, so a stride longer than `%s` would carry " \
                    "you straight past it." % (self.junction, GRANULARITY[self.stride])
        return head + tail


#: cells -> the granularity label that asks for exactly that many. The engine clamps to [1, 3] and
#: stops a stride early at a wall, so a label is never worse than a shorter one.
GRANULARITY = {0: "small", 1: "small", 2: "medium", 3: "large"}


def look(image) -> Sight:
    """The census of the frame the agent is being shown. Never raises."""
    try:
        return Sight(image)
    except Exception:
        return Sight(None)


class StairSense:
    """The floor sense: a latch saying a stair is underfoot, and the budget for spending it.

    Deliberately NOT a map and NOT a pose. It knows three things — whether the body is standing on a
    stair (an exact frame TRANSITION, since the tile underfoot is never drawn), how many presses
    that latch is still worth (a budget, because an unchanged frame cannot refute a climb), and
    which transition to try (down only once a climb has been spent going up, so a probe never
    immediately undoes the last one).
    """

    __slots__ = ("underfoot", "spent", "level", "arrivals", "presses")

    def __init__(self):
        self.underfoot = False      # a stair is under the body right now
        self.spent = 0              # grounded climbs pressed since the latch was set
        self.level = 0              # floors climbed, net, as believed from our own presses
        self.arrivals = 0           # stair tiles stepped onto this episode
        self.presses = 0            # grounded climbs pressed this episode

    def update(self, before: Sight, after: Sight, moved: bool, translated: bool) -> None:
        """Fold one measured outcome in. A translation that moved the body either lands it on a
        stair (latch set, budget refilled) or carries it off one (latch cleared).

        THE ARRIVAL TEST, each clause load-bearing: the previous frame painted a stair at saturation
        (so it was the very next cell and not four cells off), the action was a translation (a turn
        would have swung the tile out of view), the frame changed (otherwise a wall refused the move
        and the body never left), and the paint is gone (the tile under the feet is not drawn)."""
        if not translated or not moved or before is None or after is None:
            return
        arrived = before.stair_adjacent and not after.stair_adjacent
        self.underfoot, self.spent = arrived, 0
        self.arrivals += 1 if arrived else 0

    def wanted(self, stale: int, floor_age: int, seen: Sight) -> str:
        """The climb worth pressing right now, or "".

        Four conditions, each measured: a stair is underfoot, the latch's budget is unspent, this
        floor has been given its patience AND has stopped producing new views, and there is no exit
        in sight — because an exit anywhere in this view is reachable on this floor and ends the
        episode, which no floor change can improve on."""
        if not self.underfoot or self.spent >= CLIMB_BUDGET:
            return ""
        if seen is not None and seen.exit > 0.0:
            return ""
        if stale < STALE_FLOOR or floor_age < FLOOR_PATIENCE:
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
