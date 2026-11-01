"""The maze3d_pro action / key algebra + the engine `Action` contract + the BEARING algebra.

Two things live here, both pure table-and-integer work with no engine imports:

1. THE ENGINE CONTRACT. Every decision returns an `Action(type="key_press", key=...)`. The
   SpatialWorld maze3d_pro adapter (`maze3d_pro_adapter.py::execute_mapped_action`) dispatches on
   that key: `w`/`s` translate (both take a `granularity` of small/medium/large = 1/2/3 grid cells
   and stop early at a wall), `a`/`d` turn 90 degrees, `q`/`e` change floor on a stair tile. The
   evaluator's `PygameInputSource` reads `metadata["granularity"]` (falling back to
   `metadata["json_data"]["granularity"]`) and `metadata["repeat"]`, so `retarget` strips ALL THREE
   before rebuilding a turn: `_get_action_repeat` clamps `repeat` to [1, 10] and a stale `repeat: 3`
   inherited from a movement proposal would fire THREE 90-degree turns instead of one.

2. THE BEARING ALGEBRA. The harness dead-reckons a pose, so it needs the engine's own heading
   arithmetic rather than a re-derivation. The adapter's turn tables are
   `turn_right: N->E->S->W->N` and `turn_left` its inverse, and its translation vectors are
   `N:(0,-1) E:(1,0) S:(0,1) W:(-1,0)` in (x, y) grid cells. Encoding a heading as the index
   0,1,2,3 = N,E,S,W makes `turn_right` exactly `+1 (mod 4)` and `turn_left` exactly `-1 (mod 4)`,
   so `STEP[h]` is the engine's own vector and `turn_toward` is integer arithmetic on bearings.
   A turn is never refused by the engine, which is what makes the heading exactly dead-reckonable.
"""
from __future__ import annotations

# action_name -> engine key (the adapter's execute_mapped_action dispatch table)
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

# bearings: 0=N 1=E 2=S 3=W, matching the adapter's own translation vectors
BEARING_NAMES = ("north", "east", "south", "west")
STEP = ((0, -1), (1, 0), (0, 1), (-1, 0))
AHEAD, RIGHT, BEHIND, LEFT = 0, 1, 2, 3          # bearing offsets relative to the heading
_CLIMB_INVERSE = {"climb_up": "climb_down", "climb_down": "climb_up"}
_CLIMB_DZ = {"climb_up": 1, "climb_down": -1}
# metadata keys the input source reads for distance; all three must go before a turn is rebuilt
_STRIPPED = ("granularity", "repeat", "json_data")


def key_for(action_name: str):
    """The engine key a legal action name executes, or None if the name is not legal."""
    return KEY_MAPPING.get(action_name)


def name_for(key):
    """The legal action name an engine key will actually execute, or None.

    Read the KEY, never `metadata["function_called"]`: the evaluator also writes non-action labels
    there (its parse fallback stamps `default_move_forward`), while the key is what runs."""
    if not isinstance(key, str):
        return None
    return NAME_OF_KEY.get(key.strip().lower())


def granularity_steps(name) -> int:
    """Grid cells a granularity label asks for; mirrors the adapter's clamp to [1, 3]."""
    return GRANULARITY_STEPS.get(str(name or "small").strip().lower(), 1)


def steps_granularity(cells: int) -> str:
    """The granularity label that asks for `cells` cells, clamped into the engine's 1..3 range."""
    return STEPS_GRANULARITY[max(1, min(3, int(cells)))]


def inverse_climb(action_name: str):
    """The floor transition that undoes `action_name`, or None when it is not a climb."""
    return _CLIMB_INVERSE.get(action_name)


def climb_dz(action_name: str) -> int:
    """+1 for `climb_up`, -1 for `climb_down`, 0 for everything else."""
    return _CLIMB_DZ.get(action_name, 0)


def turn_toward(heading: int, bearing: int):
    """The single turn that rotates `heading` toward `bearing`, or None when already facing it.

    Offsets are mod 4, so +1 is one `turn_right` and +3 is one `turn_left`; the 180-degree case
    (+2) needs two turns and is reported as `turn_right`, which keeps the handedness consistent
    across a reversal instead of alternating and re-treading the same two bearings."""
    delta = (int(bearing) - int(heading)) % 4
    if delta == 0:
        return None
    return "turn_left" if delta == 3 else "turn_right"


def turns_to(heading: int, bearing: int) -> int:
    """How many 90-degree turns separate a heading from a bearing (0, 1 or 2)."""
    delta = (int(bearing) - int(heading)) % 4
    return 2 if delta == 2 else (0 if delta == 0 else 1)


def bearing_to(cell_from, cell_to):
    """The bearing from one grid cell to an orthogonally adjacent one, or None."""
    dx = cell_to[0] - cell_from[0]
    dy = cell_to[1] - cell_from[1]
    try:
        return STEP.index((dx, dy))
    except ValueError:
        return None


def retarget(proposal, action_name: str, cells: int = 1, reason: str = "", proposed: str = ""):
    """A NEW engine Action for `action_name`, rebuilt from `proposal`'s metadata envelope.

    `type(proposal)` is reused so this module never imports the engine. The three distance keys are
    dropped first (see the module docstring) and re-added only for the two translations, where
    `repeat` is set to the granularity's own cell count exactly as the evaluator does. The
    displacement is stamped under `harness_override` so a trace records both what the frozen model
    proposed and what actually ran."""
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
        gran = steps_granularity(cells)
        meta["granularity"] = gran
        meta["repeat"] = GRANULARITY_STEPS[gran]
    try:
        return type(proposal)(type="key_press", key=key, metadata=meta)
    except Exception:
        return proposal                 # an unexpected Action shape must never cost the step
