"""tools/sight.py — the EXACT-COLOUR CENSUS of the frame, and the floor sense it supports.

WHY THIS EXISTS. tools/plan.py reads the frame as ONE BIT: an md5 fingerprint, equal or not equal.
That bit is enough to learn "something solid is ahead", and it took the anchor from 2/25 to 7/25.
It is not enough for the other half of this maze. The exit is on the upper floor on about half the
levels, and a floor change needs a keystroke pressed while standing on a particular tile — a fact
the fingerprint cannot express, because the tile a body stands on is NEVER DRAWN. So the harness
walked over stairs and kept going, and finished every one of 25 episodes on the floor it started on.

WHAT THE PIXELS ACTUALLY CARRY (measured 2026-10-05 on live frames of this build, 800x600, via the
real evaluator driven by a scripted stub — the same frames `frame_data.image` hands the agent):

  * the renderer is flat-shaded and its whole palette is 30 RGB values, so both special tiles decode
    with an EXACT mask, no tolerance and no tuning:
        a STAIR pixel  <=>  red == 0   and green, blue both painted   (the teal family: borders
                            (0,234,234)/(0,214,214)/(0,204,204), faces (0,92,110) ... (0,52,62))
        an EXIT pixel  <=>  blue == 0  and red, green both painted    (the single gold (255,200,0))
    No other colour in the palette has a zero channel: the walls are browns with red > green > blue
    down to (21,10,8), the ground is (50,60,65), the ceiling (15,15,35)/(4,4,10), and the exit's
    white outline (255,255,255) has no zero. The two masks cannot be confused with each other or
    with anything else in the scene.
  * there is NO TEXT ANYWHERE in the render. No POS/FLOOR/DIR readout, and no `[Q] UP`/`[E] DOWN`
    stair prompt. The priors that gate a climb on seeing that prompt gate it on nothing.
  * the painted AREA is a quantized DISTANCE LADDER, and the same rung values recur byte-identically
    on every level, because the view is a pure function of the local geometry. As a fraction of the
    frame, a tile straight down the corridor ahead paints
        stair  0.0148 / 0.0213 / 0.0328 / 0.0571 / 0.1281 / 0.5063   at 6, 5, 4, 3, 2, 1 cells
        exit            0.0297 / 0.0529 / 0.1219 / 0.4938            at 4, 3, 2, 1 cells
    The step from 2 cells to 1 is a factor of FOUR, so a single cut at `ADJACENT` = 0.30 of the frame
    separates "the tile is the very next cell" from everything further away with nothing to tune.
    An adjacent tile that is OFF the centre line paints far less than that, so the cut also means
    "dead ahead" and not merely "nearby".
  * standing ON the tile paints ZERO of it. That is the whole trick, and it makes the one fact the
    fingerprint could never supply an EXACT frame fact rather than a guess:

        the previous frame painted a stair at >= ADJACENT  and  the action executed was a
        translation  and  the frame CHANGED  =>  THE STAIR IS NOW UNDERFOOT.

    Each clause is load-bearing. Without the saturation the tile could have been four cells off;
    without a translation a turn would have swung it out of view; without the change the move was
    refused by a wall and the body never left the cell it was in.

WHAT IT REFUSES TO CONCLUDE. A climb that SUCCEEDS can leave the frame byte-identical (the floor
above can render identically to the floor below), so an unchanged frame is NOT evidence that a climb
failed, and nothing here ever reads it that way: a grounded climb is spent from a BUDGET and the
latch is then released, rather than retried until the frame agrees. "Am I on a stair" is also not a
property of the view — two poses that render identically can disagree about it — so it is never
filed against a view fingerprint the way "is the way ahead blocked" legitimately is.

Everything in this module is computed from `frame_data.image`, the frame the agent is shown. No
engine state, no layout, no coordinates, no floor count, no distance.
"""
from __future__ import annotations

try:                                 # the engine hands us a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                    # pragma: no cover
    _np = None

#: index of the channel that is exactly 0 in each family's palette entries. These two numbers ARE
#: the decode: no other colour in the render has a zero channel.
STAIR_ZERO_CHANNEL = 0
EXIT_ZERO_CHANNEL = 2

#: fraction of the frame a special tile paints when it is the ADJACENT cell dead ahead (0.49 .. 0.51
#: measured) against 0.13 at two cells — one cut, a factor of 4 of headroom either side.
ADJACENT = 0.30

#: THE PRICE OF LEAVING A FLOOR, in two parts. Climbing at the FIRST stair the body lands on is a
#: measured trap — the exit sits on the STARTING floor on a large minority of levels, so leaving
#: early throws away levels that were already being won — but a steep price never gets paid at all,
#: and an episode that never changes floor cannot win the levels whose exit is upstairs.
#: `STALE_FLOOR`     consecutive decisions this floor must show no view it had not already shown.
#:                   At 1 this is simply "not on the very decision you arrived": the frame after the
#:                   arrival has to be one this floor has already produced.
#: `FLOOR_PATIENCE`  decisions this floor must have been given at all, counted from arriving on it.
#:                   24 of a 100-decision budget: the floor you start on gets the first quarter of
#:                   the clock before any stair on it is worth spending.
#: Both numbers were swept on two independent offline rigs (see NOTES.md). They are the only tuned
#: constants here; every other number in this file is a MEASUREMENT of the render.
STALE_FLOOR = 1
FLOOR_PATIENCE = 24

#: grounded climbs per latch. A bound, not a knob: one press settles it, and a frame that did not
#: change is not proof the press failed.
CLIMB_BUDGET = 1


def to_array(image):
    """`frame_data.image` as an HxWx3 int16 array, or None if it is not a colour frame."""
    if image is None or _np is None:
        return None
    arr = _np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    return arr[:, :, :3].astype(_np.int16)


def family_mask(arr, zero_channel: int, painted: int = 1):
    """The exact mask of one tile family: the named channel exactly dark, the other two painted.

    `painted` is the floor the two live channels must clear. It is 1, not 0, and that single unit is
    what keeps pure black — which satisfies "channel == 0" on every channel — out of both masks."""
    m = arr[:, :, zero_channel] == 0
    for c in (0, 1, 2):
        if c != zero_channel:
            m = m & (arr[:, :, c] >= painted)
    return m


def band_columns(width: int, left: float = 0.35, right: float = 0.65):
    """(first, last) column of the central strip — the part of the view that is dead ahead rather
    than beside. Fractions of the frame width, so the readout is resolution-independent."""
    return int(width * left), int(width * right)


def paint_fractions(arr, zero_channel: int):
    """(fraction of the whole frame, fraction of the central strip) painted by one family."""
    if arr is None:
        return 0.0, 0.0
    mask = family_mask(arr, zero_channel)
    lo, hi = band_columns(arr.shape[1])
    strip = mask[:, lo:hi]
    whole = float(mask.mean()) if mask.size else 0.0
    return whole, (float(strip.mean()) if strip.size else 0.0)


def cells_away(fraction: float, ladder=((0.30, 1), (0.09, 2), (0.045, 3),
                                        (0.026, 4), (0.018, 5), (0.0, 6))) -> int:
    """Cells to a tile painting `fraction` of the frame down the corridor ahead; 0 = not in view.

    `ladder` is THE MEASUREMENT, carried here as the default so there is exactly one copy of it:
    (cut, cells) read top down, each cut placed at the geometric mean of the two measured rungs it
    separates. A stair down a
    clear corridor paints 0.5063 / 0.1281 / 0.0571 / 0.0328 / 0.0213 / 0.0148 of the frame at 1, 2,
    3, 4, 5, 6 cells and the exit 0.4938 / 0.1219 / 0.0529 / 0.0297 at 1, 2, 3, 4 — the same values
    byte for byte on every level, because the view is a pure function of the local geometry. Every
    one of those ten values reads back at its own rung through these cuts."""
    if fraction <= 0.0:
        return 0
    for cut, cells in ladder:
        if fraction >= cut:
            return cells
    return ladder[-1][1]


class Sight:
    """One frame's census. Immutable, cheap, and the only thing that reads pixels by colour."""

    __slots__ = ("stair", "stair_band", "exit", "exit_band", "ok")

    def __init__(self, image=None):
        arr = to_array(image)
        self.ok = arr is not None
        self.stair, self.stair_band = paint_fractions(arr, STAIR_ZERO_CHANNEL)
        self.exit, self.exit_band = paint_fractions(arr, EXIT_ZERO_CHANNEL)

    @property
    def stair_adjacent(self) -> bool:
        """A stair tile is the VERY NEXT cell dead ahead: one `small` step lands on it."""
        return self.stair >= ADJACENT

    @property
    def exit_adjacent(self) -> bool:
        return self.exit >= ADJACENT

    def describe(self) -> str:
        """The census in the plain English the working-memory block states back to the agent."""
        said = []
        if self.exit:
            near = "the VERY NEXT CELL dead ahead" if self.exit_adjacent \
                else "about %d cells down the way you are facing" % cells_away(self.exit)
            said.append("the GOLD EXIT is in sight, %s" % near)
        if self.stair:
            near = "the VERY NEXT CELL dead ahead" if self.stair_adjacent \
                else "about %d cells down the way you are facing" % cells_away(self.stair)
            said.append("a CYAN STAIR tile is in sight, %s" % near)
        return "; ".join(said) or "no stair and no exit anywhere in this view"


def look(image) -> Sight:
    """The census of the frame the agent is being shown. Never raises."""
    try:
        return Sight(image)
    except Exception:
        return Sight(None)


def stepped_onto_stair(before: Sight, after: Sight, moved: bool) -> bool:
    """The exact on-stair transition: a saturated stair ahead, a translation, the frame changed, and
    the paint gone. The tile underfoot is never drawn, so its disappearance IS the arrival."""
    if before is None or after is None or not moved:
        return False
    return before.stair_adjacent and not after.stair_adjacent


class StairSense:
    """The floor sense: a latch saying a stair is underfoot, and the budget for spending it.

    Deliberately NOT a map and NOT a pose. It knows three things — whether the body is standing on a
    stair (an exact frame transition), how many presses that latch is still worth (a budget, because
    an unchanged frame cannot refute a climb), and which transition to try (down only once a climb
    has been spent going up, so a probe never immediately undoes the last one).
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
        stair (latch set, budget refilled) or carries it off one (latch cleared)."""
        if not translated or not moved:
            return
        arrived = stepped_onto_stair(before, after, moved)
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
        arriving on one again — which is the only event that can prove the fact afresh."""
        self.spent += 1
        self.presses += 1
        self.level += 1 if name == "climb_up" else -1

    def line(self) -> str:
        """The latch, as the working-memory block states it."""
        if not self.underfoot:
            return "you are NOT standing on a stair (no stair paint vanished from under you)"
        return ("you ARE STANDING ON A STAIR right now - the tile under your feet is not drawn, "
                "which is exactly why the screen shows none")
