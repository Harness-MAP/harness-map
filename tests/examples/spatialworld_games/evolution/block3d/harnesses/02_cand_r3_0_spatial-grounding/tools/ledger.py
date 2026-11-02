"""The self-derived episode ledger: where the cursor is, what carries a block, what each press bought.

Dead-reckoned. The cursor starts at the documented spawn and every key this harness RETURNS is replayed
through the engine's own clamped update (tools/press.shifted), so the ledger is exact without reading one
byte of engine state. The measured basis for trusting it: replaying the emitted key list reproduces the
engine's own matched-block count on every episode of the parent's sweep.

Two independent pixel audits keep the reckoning honest, and each is used only where it is sound:

  * the frame DIGEST. The frame is a pure function of (cursor, placed), so a digest that HELD across a
    move press proves the press was clamped. If the ledger predicted a move there, the reckoning and
    the picture disagree and `suspect` goes up. The converse is unavailable — a new cube can be hidden
    behind a standing one — so placements are booked from the toggles, never from the pixels.
  * the highlight INK (tools/frame.cursor_from_ink), which answers only when the footprint is whole.
    When it answers and disagrees, the picture wins and the ledger is re-seated: the ink is an exact
    readout under its own precondition, while reckoning drifts permanently once it is wrong.

`refused` counts the presses the plan declined to spend: a move into a grid face, and a toggle that
would cancel work already correct. Those two were 42.5% and 21.0% of the parent's episode.
"""
from tools import press

_NAMES = lambda keys: [press.action_name(k) for k in keys]


class Ledger:
    """Cursor, placed set and the press account. One instance per episode."""

    def __init__(self, start=None):
        self.cursor = tuple(start or press.SPAWN)
        self.placed = set()
        self.presses = 0
        self.toggles = 0
        self.travel = 0
        self.wasted = 0
        self.suspect = 0
        self.reseated = 0
        self.refused = 0
        self.history = []
        self.digest = ""
        self.stirred = False
        self.drift = 0

    def spend(self, key):
        """Replay one returned key. Returns the cell a toggle hit, else None."""
        self.presses = self.presses + 1
        self.history.append(key)
        if key == press.TOGGLE:
            self.stirred = False
            self.toggles = self.toggles + 1
            seat = self.cursor
            if seat in self.placed:
                self.placed.discard(seat)
                self.wasted = self.wasted + 1
            else:
                self.placed.add(seat)
            return seat
        was = self.cursor
        self.cursor = press.shifted(self.cursor, key)
        self.stirred = self.cursor != was
        if self.stirred:
            self.travel = self.travel + 1
        else:
            self.wasted = self.wasted + 1
        return None

    def audit(self, digest, seen, inked):
        """Fold the two pixel audits in. `digest` is this frame's, `seen` the ink cell or None, and
        `inked` says the highlight was actually found — the digest audit is only valid when it was,
        because a frame with no highlight in it does not depend on the cursor at all and would report
        every single move as clamped."""
        held = bool(self.digest) and digest == self.digest
        if held and inked and self.stirred:
            self.suspect = self.suspect + 1       # the picture says clamped, the reckoning says moved
        self.digest = digest
        if seen is None:
            return self.cursor
        if tuple(seen) != self.cursor:
            self.drift = press.l1(self.cursor, tuple(seen))
            self.cursor = tuple(seen)
            self.reseated = self.reseated + 1
            if self.drift > 2:
                self.history = []                 # a gap this wide means the replay is not recoverable
        return self.cursor

    def holds(self, cell):
        return tuple(cell) in self.placed

    def left(self):
        """Presses still affordable this episode."""
        return press.WALLET - self.presses

    def recent(self, span=3):
        return _NAMES(self.history[-span:])

    def snapshot(self):
        return {"cursor": list(self.cursor), "placed": len(self.placed), "presses": self.presses,
                "toggles": self.toggles, "travel": self.travel, "wasted": self.wasted,
                "refused": self.refused, "suspect": self.suspect, "reseated": self.reseated,
                "drift": self.drift, "steps": press.engine_steps(self.presses)}


def drift_of(reckoned, seen):
    """L1 gap between the reckoned cursor and an ink reading — 0 while the two agree."""
    if seen is None:
        return 0
    return press.l1(tuple(reckoned), tuple(seen))
