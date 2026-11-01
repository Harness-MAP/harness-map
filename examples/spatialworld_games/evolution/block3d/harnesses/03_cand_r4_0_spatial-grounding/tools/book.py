"""SELF-DERIVED EPISODE STATE — the cursor, the placed set, the book of refuted structures, and
the GRADE of every press the harness applied.

Nothing here is read from the engine. The cursor starts at the published spawn (a public rule,
printed in the game's own instructions and in AGENT.md) and is dead-reckoned forward from the
action the harness APPLIED, under the adapter's clamping; the placed set is maintained from the
harness's own toggles, which are involutive at a cell.

THE GRADE, which is why this module is not just a ledger. Dead reckoning is a PRIOR, and a prior
that is never checked is a lie waiting to happen. Two statements close the loop:

  * a MOVE that left the picture byte-identical is a CLAMP, and a clamp is the one gadget in this
    game that FORCES a coordinate instead of reporting it: whatever the ledger believed, the
    cursor's coordinate on that axis is now exactly that face. `grade` writes it in.
  * a TOGGLE that left the picture byte-identical is NOT a null. A newly placed cube can be
    entirely hidden behind a nearer one, so a held digest refutes a MOVE and never a press.
    Treating the two alike is the plausible version of this mechanism, and it is wrong.

`correct` takes the cursor the optics actually measured off the frame and counts the correction,
so a drifting ledger is visible in the trace rather than silently steering the tape.

THE REFUTATION BOOK, and why it is free. The adapter compares the placed set with the goal inside
every executed action and the engine stops the episode the instant they are equal, so BEING ASKED
FOR ANOTHER DECISION PROVES the structure the last press produced was not the answer. The book
records every structure the episode has passed through, keyed on the order-free frozenset (the win
test is set equality, and a permuted insertion order renders byte-identically), so a press that
would re-enter a booked structure is re-asking an answered question.
"""
from __future__ import annotations

from .action import FACE, SPAWN, TOGGLE, _SETAX
from .route import advance

COUNTS = ("turn", "presses", "honoured", "frozen", "fixes")


class BuildState:
    """Everything this harness knows about the episode, all of it derived by the harness itself."""

    def __init__(self, cursor=SPAWN):
        self.cursor = cursor
        self.placed = set()
        self.book = set()
        self.last = None
        self.notes = []
        self.turn = 0
        self.presses = 0
        self.honoured = 0
        self.frozen = 0
        self.fixes = 0
        self.down = False

    def apply(self, name):
        """Record the press the harness APPLIED — never the one the model merely proposed."""
        self.presses += 1
        self.last = name
        if name == TOGGLE:
            self.placed ^= {self.cursor}               # a toggle is an involution at a cell
            self.book.add(frozenset(self.placed))
        else:
            self.cursor = advance(self.cursor, name)

    def grade(self, same):
        """Grade the last applied press against THIS frame; a frozen move pins one coordinate."""
        if not same or self.last not in FACE:
            return None
        self.frozen += 1
        pin = FACE[self.last]
        self.cursor = _SETAX(self.cursor, pin[0], pin[1])
        self.notes.append((self.turn, self.last))
        return self.last

    def correct(self, cell):
        """Adopt a MEASURED cursor, and count the correction so drift is visible in the trace."""
        if cell is None or cell == self.cursor:
            return False
        self.fixes += 1
        self.cursor = cell
        return True

    def booked(self, cell):
        """Would toggling at `cell` re-enter a structure the engine has already refused?"""
        return frozenset(set(self.placed) ^ {cell}) in self.book

    def summary(self):
        """The row this module contributes to every line of the trace."""
        return dict({k: getattr(self, k) for k in COUNTS}, placed=len(self.placed),
                    book=len(self.book), down=self.down,
                    notes=[list(n) for n in self.notes[-2:]])
