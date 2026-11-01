"""THE ACTION ALGEBRA — the legal action set, the engine key table, and the grid's own arithmetic.

Every decision this harness returns is an engine `Action(type="key_press", key=...)`. The
SpatialWorld Block3D adapter drives a cursor over a 6x6x6 grid: six 1-unit cursor moves along
+/-X, +/-Y, +/-Z, plus a place/remove TOGGLE at the cursor. There is no movement granularity —
every move is exactly one cell — and every move CLAMPS at the grid face, which is why a press
into a face is money burnt rather than an error, and why `place_block` is an involution at a cell.

Three properties of the key layer that the rest of the harness is built on:

* **A move that clamps is an ABSOLUTE POSITION STATEMENT.** The clamp is not a failure mode to be
  avoided; it is the one gadget in the game that FORCES a coordinate instead of reading it. If a
  move leaves the picture byte-identical, the cursor stood on that axis's face, whatever the
  ledger believed — so `FACE` turns the null into a correction (see tools/book.py::grade).
* **An unknown action name is silently mapped to `move_right`** by the evaluator, with the bogus
  name left in `metadata["function_called"]`. tools/vlm.py reads that field, so an unmapped
  proposal is detected rather than played as a stray `right`.
* **`metadata["repeat"]` replays a key inside ONE callback** with an engine update between
  presses. It divides the number of model calls, not the number of engine steps, so it buys
  latency and never budget; this harness does not use it.

This table is the same mapping `rules.md` states to the model, and it is the only place it lives.
"""
from __future__ import annotations

# action_name -> engine key (the adapter's own ActionMapping list, unchanged)
KEY_MAPPING = {
    "move_left": "left",       # -X : move cursor left
    "move_right": "right",     # +X : move cursor right
    "move_up": "up",           # -Y : move cursor up
    "move_down": "down",       # +Y : move cursor down
    "move_forward": "w",       # +Z : move cursor forward
    "move_backward": "s",      # -Z : move cursor backward
    "place_block": "space",    # toggle a block at the cursor (place or remove)
}
LEGAL = frozenset(KEY_MAPPING)
TOGGLE = "place_block"

X, Y, Z = 0, 1, 2                  # the three grid axes, by index into a cell tuple
SPAWN = (2, 2, 0)                  # the published spawn cell, printed in the game's own rules
LIMIT = 5                          # highest index on any axis of the 6-wide grid
STEPS_PER_PRESS = 2                # one mapped press costs two of the episode's engine steps
STEP = 1                           # a move is exactly one cell: Block3D has no granularity

# axis table: (+ action, - action, axis index)
AXIS = (("move_right", "move_left", 0),
        ("move_down", "move_up", 1),
        ("move_forward", "move_backward", 2))
# a move that clamps proves the cursor's coordinate on ONE axis: action -> (axis, that face)
FACE = {"move_left": (0, 0), "move_right": (0, LIMIT),
        "move_up": (1, 0), "move_down": (1, LIMIT),
        "move_backward": (2, 0), "move_forward": (2, LIMIT)}

_CLAMP = lambda v: max(0, min(LIMIT, v))
_SETAX = lambda c, i, v: tuple(v if j == i else c[j] for j in range(len(c)))
_AXIS_OF = lambda nm: FACE[nm][0] if nm in FACE else None
_SIGN_OF = lambda nm: 1 if nm in ("move_right", "move_down", "move_forward") else -1


def key_for(name):
    """The engine key for a legal Block3D action, or None when the name is not a legal action."""
    return KEY_MAPPING.get(name)
