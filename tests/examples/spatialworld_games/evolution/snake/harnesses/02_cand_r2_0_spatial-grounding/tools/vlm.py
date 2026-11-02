"""The perceive -> VLM -> parse step, rewritten as a LATTICE BALLOT.

The anchor delegated the whole decision to `evaluator.baseline_step` and pressed whatever came
back. This module keeps that single pristine model call -- one request per step, the evaluator
still owns screenshot capture, the API call, JSON parsing, its own fallback, logging and engine
`Action` construction -- but interposes on the two surfaces either side of it:

* REQUEST SIDE. `openai_ai_model` resolves `build_conversation_history` as a module global of its
  own module, so swapping that name for the duration of one call puts a block of text beside the
  newest screenshot, inside the very request the model is about to answer. Unlike mutating
  `evaluator.current_prompt`, which is re-read only on the first turn, this lands on EVERY turn.
  The block is stripped from every older turn each time it is rebuilt, so a stale belief can never
  ride along in the retained history.
* KEY SIDE. Only `Action.key` decides what the engine presses. The model's named action is kept
  whenever it is legal; when it is not, it is PROJECTED onto the legal set rather than discarded --
  first onto the same axis it named, and only if that axis is blocked end to end onto the legal
  move the model has asked for LEAST this episode.

Why a projection and not advice. A maximally explicit per-turn directive on the text channel was
obeyed about a quarter of the time on this build, so prose alone cannot keep the snake inside the
cube; and a blind substitution throws away the only food evidence in the loop, which is the model's
own look at the frame. Projecting keeps the model's axis whenever the axis is survivable.

Why least-asked as the last resort. The frozen model's answer distribution on this build collapses
onto one axis: it almost never names `move_left` even when the food sits on the -X side. So when
its named axis is wholly blocked, the legal move it has asked for least is precisely the one its
own habit is suppressing -- a de-biasing rule read off its own asks, never off the game.

Anti-cheat. Everything in `LatticeLedger` is integrated from the keys THIS MODULE handed the
engine plus the two documented engine laws (a 4x4x4 cube, and a 180 that the engine refuses). It
never touches `evaluator.game`, the engine, the score, the seed, the food, the true body or a
pixel. The ledger therefore does not know where the food is, says so in the block it writes, and
leaves target selection entirely to the model's own reading of the screenshot.

Contract. `decide(evaluator, frame_data) -> Action` is unchanged, `evaluator.baseline_step` is
still awaited exactly once per step, and every unfamiliar call surface falls through to baseline
behaviour instead of raising. See NOTES.md for the module map and the lever table.
"""

from __future__ import annotations

from collections import Counter, deque

from tools.action import DIRECTION_VECTORS, KEY_MAPPING

# The evaluator's module-global conversation builder -- the only request-side seam that is live on
# every turn (`current_prompt` is read once, a replaced `frame_data.image` is re-captured anyway).
_BUILDER = "build_conversation_history"
_SLOT = "_snake_lattice_ledger"
_MARK = "\n\n[LATTICE BALLOT -- harness bookkeeping from its own keys, not a decoded game state]"
_TEXT_TYPES = ("input_text", "text")

# Engine laws, both documented and seed-independent: the cube is 4 cells on a side and the opening
# pose is fixed, so arithmetic on the pressed keys is enough to track the head.
_SIDE = 4
_SPAWN = (2, 3, 2)
_NECK = (1, 3, 2)
_ORDER = ("move_right", "move_left", "move_up", "move_down", "move_forward", "move_backward")
_AXIS = {"move_right": "X", "move_left": "X", "move_up": "Y", "move_down": "Y",
         "move_forward": "Z", "move_backward": "Z"}
_OPPOSITE = {"move_right": "move_left", "move_left": "move_right",
             "move_up": "move_down", "move_down": "move_up",
             "move_forward": "move_backward", "move_backward": "move_forward"}
_FACE = {"move_right": "+X", "move_left": "-X", "move_up": "-Y (towards the top of the screen)",
         "move_down": "+Y (towards the bottom of the screen)",
         "move_forward": "-Z (into the screen)", "move_backward": "+Z (out of the screen)"}
_OPENING = ("", "", "nothing pressed yet")


def _function_globals(step):
    """The namespace the pristine step resolves its conversation builder in.

    Returns a THROWAWAY dict carrying the same key when the call surface is unfamiliar, so the
    caller installs and restores unconditionally and an evaluator that renames or hides its builder
    simply gets baseline behaviour instead of an exception.
    """
    function = getattr(step, "__func__", step)
    namespace = getattr(function, "__globals__", None)
    if isinstance(namespace, dict) and _BUILDER in namespace:
        return namespace
    return {_BUILDER: None}


class LatticeLedger:
    """Where this harness believes the snake is, integrated only from the keys it pressed itself.

    The opening pose is fixed on every seed -- head (2,3,2), neck (1,3,2), facing +X -- and each
    later cell follows by adding the pressed direction, except for a 180, which the engine refuses
    and which therefore leaves the belief untouched.

    `trail` is an OCCUPANCY WINDOW, not the true body. The snake's length is unobservable without
    reading pixels, so the window is deliberately a short under-approximation: it covers the two
    opening segments plus a couple of meals and then stops modelling the tail, which makes the
    ledger refuse slightly fewer cells than the real body occupies rather than inventing walls.
    Nothing here is privileged: no food, no score, no seed, no engine state.
    """

    def __init__(self):
        self.head = _SPAWN
        self.facing = "move_right"
        self.trail = deque([_NECK, _SPAWN], maxlen=4)
        self.asked = Counter()
        self.turns = 0
        self.overrides = 0
        self.refusals = 0
        self.honoured = 0
        self.last = _OPENING

    def arrival(self, name, origin=None):
        """The cell `origin` (default: the reckoned head) would move to if `name` were pressed."""
        base = origin if origin else self.head
        return tuple(a + b for a, b in zip(base, DIRECTION_VECTORS[name]))

    def in_cube(self, cell):
        """Inside the 4-cell cube. Leaving it is always fatal, so this bound outranks the window."""
        return min(cell) >= 0 and max(cell) < _SIDE

    def open_cell(self, cell):
        """Inside the cube and off the occupancy window."""
        return self.in_cube(cell) and cell not in self.trail

    def refused(self, name):
        """The engine rejects the exact negation of the heading; the snake does not move at all."""
        return _OPPOSITE[name] == self.facing

    def room(self, cell):
        """A one-ply breathing test: how many of `cell`'s six neighbours are themselves open."""
        return sum(self.open_cell(self.arrival(n, cell)) for n in _ORDER)

    def menu(self):
        """The ballot, in two tiers. Tier one is every action that is neither an engine-refused 180
        nor a step into a face of the cube nor a step into a cell the window still lists; tier two
        drops only the window, because leaving the cube is always fatal while re-entering a cell the
        window over-counts may not be. Any cell of a 4-cube has at least three in-cube neighbours
        and only one of the six is the refused 180, so tier two is never empty."""
        inside = [n for n in _ORDER if self.in_cube(self.arrival(n)) and not self.refused(n)]
        free = [n for n in inside if self.arrival(n) not in self.trail]
        return free if free else inside

    def fallback(self, menu):
        """De-bias the model's revealed preference: of the legal moves take the one it has asked for
        least this episode, breaking ties towards the roomiest arrival cell."""
        return min(menu, key=lambda n: (self.asked[n], -self.room(self.arrival(n)),
                                        _ORDER.index(n)))

    def record(self, asked, pressed):
        """Integrate the key actually handed to the engine, and keep a disclosure for next turn."""
        self.turns += 1
        if asked and asked in DIRECTION_VECTORS:
            self.asked[asked] += 1
        verdict = "honoured" if asked == pressed else "projected onto a legal move"
        if pressed not in DIRECTION_VECTORS:
            self.last = (asked, pressed, "unrecognised key -- belief frozen for this turn")
            return
        if self.refused(pressed):
            self.refusals += 1
            self.last = (asked, pressed, verdict + ", then refused as a 180 -- nothing moved")
            return
        self.head = self.arrival(pressed)
        self.facing = pressed
        self.trail.append(self.head)
        self.last = (asked, pressed, verdict)

    def ballot(self):
        """The text appended to the newest user turn: belief, last turn's disclosure, legal menu."""
        menu = self.menu()
        asked, pressed, verdict = self.last
        note = "This is the opening frame, so nothing has been pressed yet."
        if asked:
            note = f"Last turn you answered {asked}; the engine was handed {pressed} ({verdict})."
        if asked and _AXIS.get(asked) != _AXIS.get(pressed):
            note += (" Your axis was blocked end to end, so the move pressed was the legal one you"
                     " have asked for least this episode, not the one you named.")
        if self.overrides or self.refusals:
            note += (f" Tally: {self.honoured} answers pressed as given, {self.overrides}"
                     f" projected, {self.refusals} refused as 180s.")
        x, y, z = self.head
        shown = ", ".join(menu) if menu else "none"
        return (
            f"{_MARK}\n"
            f"Turn {self.turns}. Reckoned head cell (x,y,z)={self.head} in a 0..3 cube, facing "
            f"{_FACE[self.facing]}. Cells still held by the snake, newest last: {list(self.trail)}."
            f" Distance to the faces -- x {x} behind / {_SIDE - 1 - x} ahead, y {y} / "
            f"{_SIDE - 1 - y}, z {z} / {_SIDE - 1 - z}.\n"
            f"{note}\n"
            f"LEGAL MOVES NOW: {shown}. Everything else is a face of the cube, a cell this snake "
            "is still holding, or the 180 the engine refuses outright.\n"
            "These cells are bookkeeping from the keys this harness pressed; the FOOD IS NOT IN "
            "THEM and the harness cannot see it. Find the yellow food sphere in the screenshot, "
            "work out which of the listed moves shortens the head-to-food gap along x, y or z, and "
            "answer with that move. An answer outside the list is projected onto it before the key "
            "is pressed, so naming an unlisted move spends the turn without steering."
        )


def _ballot_card(ledger, conversation):
    """Strip every stale ballot already in the rebuilt request, then append a fresh one.

    The evaluator stores the list it gets back as `conversation_history`, so the parts are mutated
    in place: that is what keeps exactly one ballot -- the current one -- in a 29-turn window. An
    unfamiliar message or part shape is skipped rather than rejected, and when no text part is
    found at all the card lands on a throwaway dict and the request goes out untouched.
    """
    if not isinstance(conversation, list) or not conversation:
        return conversation
    slots = [{"text": ""}]
    for message in conversation:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in _TEXT_TYPES:
                part["text"] = str(part.get("text", "")).split(_MARK)[0]
                slots.append(part)
    slots[-1]["text"] += ledger.ballot()
    return conversation


def _resolve_move(ledger, asked):
    """Project the model's named action onto the legal set, preserving its axis where possible."""
    menu = ledger.menu()
    if asked in menu:
        ledger.honoured += 1
        return asked
    ledger.overrides += 1
    axis = _AXIS.get(asked)
    keep = ledger.fallback(menu)
    for name in menu:
        if axis and _AXIS[name] == axis:
            keep = name
            break
    return keep


async def decide(evaluator, frame_data):
    """One decision step -> engine Action: one pristine model call, a ballot in its request, and a
    key projected onto what the reckoned lattice says is actually pressable."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    ledger = getattr(evaluator, _SLOT, None)
    if not isinstance(ledger, LatticeLedger):
        ledger = LatticeLedger()
        setattr(evaluator, _SLOT, ledger)
    namespace = _function_globals(step)
    stock = namespace[_BUILDER]
    namespace[_BUILDER] = lambda *a, **k: _ballot_card(ledger, stock(*a, **k))
    try:
        act = await step(frame_data)
    finally:
        namespace[_BUILDER] = stock
    book = getattr(act, "metadata", None)
    if not isinstance(book, dict):
        book = {}
    chosen = _resolve_move(ledger, book.get("function_called"))
    act.key = KEY_MAPPING[chosen]
    ledger.record(book.get("function_called"), chosen)
    book["lattice_head"] = ledger.head
    book["lattice_pressed"] = chosen
    book["lattice_projections"] = ledger.overrides
    return act
