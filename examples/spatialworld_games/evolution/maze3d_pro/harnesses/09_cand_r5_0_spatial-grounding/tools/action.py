"""tools/action.py — the maze3d_pro key table, the engine `Action` contract, and the BEARING algebra.

Three things live here, all of them pure table-and-integer work with no engine import:

1. THE KEY TABLE. Every decision returns an `Action(type="key_press", key=...)` and the adapter
   dispatches on that key: `w`/`s` translate (both carry a `granularity` of small/medium/large =
   1/2/3 cells and stop early at a wall), `a`/`d` turn 90 degrees, `q`/`e` change floor from a stair
   tile. Read the KEY, never `metadata["function_called"]` — the evaluator's own parse fallback
   stamps `default_move_forward` there while the key is what actually runs.

2. THE METADATA ENVELOPE. `PygameInputSource` reads `metadata["granularity"]` (falling back to
   `metadata["json_data"]["granularity"]`) and `metadata["repeat"]`, and it forces `repeat` to 1
   only when the action carries a granularity — so a stale `repeat: 3` inherited from a movement
   proposal fires THREE 90-degree turns. `retarget` strips all three keys before rebuilding and
   re-adds them only for the two translations, exactly as the evaluator does.

3. THE BEARING ALGEBRA. The harness dead-reckons its pose, so it needs the engine's own heading
   arithmetic. Measured off the live adapter: `turn_right` walks NORTH->EAST->SOUTH->WEST and the
   translation vectors are N:(0,-1) E:(1,0) S:(0,1) W:(-1,0) in (x, y) cells. Encoding a heading as
   the index 0,1,2,3 = N,E,S,W therefore makes `turn_right` exactly `+1 (mod 4)`, `turn_left`
   exactly `-1 (mod 4)`, and `STEP[h]` the engine's own vector. A turn is never refused, which is
   what makes the heading exactly reckonable; the SIDE vectors below are the ones the flank ladder
   needs, and the screen's left-hand neighbour of heading `h` is `STEP[(h + LEFT) % 4]`.
"""
from __future__ import annotations

KEY_MAPPING = {
    "move_forward": "w",
    "move_backward": "s",
    "turn_left": "a",
    "turn_right": "d",
    "climb_up": "q",
    "climb_down": "e",
}
NAME_OF_KEY = {k: n for n, k in KEY_MAPPING.items()}
LEGAL_ACTIONS = tuple(KEY_MAPPING)

MOVEMENT_ACTIONS = frozenset(("move_forward", "move_backward"))
TURN_ACTIONS = frozenset(("turn_left", "turn_right"))
CLIMB_ACTIONS = frozenset(("climb_up", "climb_down"))

GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}
STEPS_GRANULARITY = {1: "small", 2: "medium", 3: "large"}

BEARING_NAMES = ("north", "east", "south", "west")
STEP = ((0, -1), (1, 0), (0, 1), (-1, 0))
AHEAD, RIGHT, BEHIND, LEFT = 0, 1, 2, 3     # bearing offsets relative to the current heading
_CLIMB_INVERSE = {"climb_up": "climb_down", "climb_down": "climb_up"}
_STRIPPED = ("granularity", "repeat", "json_data")


def key_for(action_name):
    """The engine key a legal action name executes, or None when the name is not legal."""
    return KEY_MAPPING.get(action_name)


def name_for(key):
    """The legal action name an engine key will execute, or None."""
    if not isinstance(key, str):
        return None
    return NAME_OF_KEY.get(key.strip().lower())


def granularity_steps(name) -> int:
    """Cells a granularity label asks for; mirrors the adapter's clamp into [1, 3]."""
    return GRANULARITY_STEPS.get(str(name or "small").strip().lower(), 1)


def steps_granularity(cells) -> str:
    """The granularity label that asks for `cells` cells, clamped into the engine's 1..3 range."""
    return STEPS_GRANULARITY[max(1, min(3, int(cells)))]


def inverse_climb(action_name):
    """The floor transition that undoes `action_name`, or None when it is not a climb."""
    return _CLIMB_INVERSE.get(action_name)


def step_from(cell, bearing, cells=1):
    """The cell `cells` steps along `bearing` from `cell`."""
    dx, dy = STEP[int(bearing) % 4]
    return (cell[0] + dx * int(cells), cell[1] + dy * int(cells))


def side_of(heading, offset):
    """The bearing `offset` quarter-turns clockwise of `heading` (LEFT / RIGHT / BEHIND)."""
    return (int(heading) + int(offset)) % 4


def turn_toward(heading, bearing):
    """The one turn that rotates `heading` toward `bearing`, or None when it already faces it.

    The 180-degree case needs two turns and is reported as `turn_right`, which keeps the handedness
    consistent through a reversal instead of alternating between the same two bearings."""
    delta = (int(bearing) - int(heading)) % 4
    if delta == 0:
        return None
    return "turn_left" if delta == 3 else "turn_right"


def turns_to(heading, bearing) -> int:
    """How many 90-degree turns separate a heading from a bearing (0, 1 or 2)."""
    delta = (int(bearing) - int(heading)) % 4
    return 2 if delta == 2 else (0 if delta == 0 else 1)


def bearing_to(cell_from, cell_to):
    """The bearing from one grid cell to an orthogonally adjacent one, or None."""
    delta = (cell_to[0] - cell_from[0], cell_to[1] - cell_from[1])
    return STEP.index(delta) if delta in STEP else None


def retarget(proposal, action_name, cells=1, reason="", proposed=""):
    """A NEW engine Action for `action_name`, rebuilt from `proposal`'s metadata envelope.

    `type(proposal)` is reused so this module never imports the engine, and the displacement is
    stamped under `harness_override` so a trace keeps both what the model proposed and what ran."""
    key = KEY_MAPPING.get(action_name)
    if key is None:
        return proposal
    meta = dict(getattr(proposal, "metadata", None) or {})
    for dead in _STRIPPED:
        meta.pop(dead, None)
    meta["harness_override"] = {"proposed": proposed or meta.get("function_called"),
                                "executed": action_name, "reason": reason}
    meta["function_called"] = action_name
    if action_name in MOVEMENT_ACTIONS:
        label = steps_granularity(cells)
        meta["granularity"] = label
        meta["repeat"] = GRANULARITY_STEPS[label]
    try:
        return type(proposal)(type="key_press", key=key, metadata=meta)
    except Exception:
        return proposal                  # an unexpected Action shape must never cost the step
