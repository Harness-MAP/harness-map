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

4. THE ROTATION, PRICED LIKE THE STRIDE (new this iteration). A translation already buys up to
   three cells for one decision through `granularity`. A ROTATION has the same knob and this
   lineage never used it: `repeat` is the quarter-turn count, and the adapter FORCES it to 1 only
   when the action also carries a `granularity`. Measured on the real `PygameInputSource`:
   `turn_right` with `repeat` 1/2/3 and no granularity walks NORTH -> EAST / SOUTH / WEST in ONE
   `execute_action`, while `move_forward` with `granularity: large` and `repeat: 3` still advances a
   single stride. The episode budget is the engine loop's own `step_count` — one iteration per
   `ai_model_func` call (evaluation_engine.py: `if max_steps and step_count >= max_steps: break`) —
   so a rotation spent in one call costs ONE decision whatever its arc. `pivot_to` / `pivot_path`
   below are that arithmetic; tools/plan.py decides when the arc is safe to spend (never across a
   heading that still has something to show). NB the game's own `steps_taken` counter DOES advance
   once per quarter, so a compressed rotation makes the reported `steps_taken` exceed `vlm_calls`;
   `success` is the scored field and the budget is the loop, so that divergence is telemetry only.
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
#: THE ORDER MATTERS AND IS NOT ALPHABETICAL. On a tie the harness spends `climb_up` first, so the
#: first floor change of an episode goes away from the starting floor rather than back into the
#: ground. Measured: taking these alphabetically instead (climb_down first) costs SEVEN WINS and
#: 6.3 decisions an episode on the six-tape panel. Iterate this tuple, never `sorted(CLIMB_ACTIONS)`.
CLIMB_ORDER = ("climb_up", "climb_down")

GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}
STEPS_GRANULARITY = {1: "small", 2: "medium", 3: "large"}

BEARING_NAMES = ("north", "east", "south", "west")
STEP = ((0, -1), (1, 0), (0, 1), (-1, 0))
AHEAD, RIGHT, BEHIND, LEFT = 0, 1, 2, 3     # bearing offsets relative to the current heading
QUARTERS = 4                                # quarter turns in a full revolution
PIVOT_CAP = 3                               # a 4-quarter rotation is the identity; 3 is the cap
_CLIMB_INVERSE = {"climb_up": "climb_down", "climb_down": "climb_up"}
_STRIPPED = ("granularity", "repeat", "json_data")
_HANDED = {"turn_right": RIGHT, "turn_left": LEFT}


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
    delta = (int(bearing) - int(heading)) % QUARTERS
    return 2 if delta == 2 else (0 if delta == 0 else 1)


def quarter_of(action_name) -> int:
    """The signed quarter-turn offset a turn action applies, or 0 when it is not a turn."""
    return _HANDED.get(action_name, 0)


def heading_after(heading, action_name, quarters=1) -> int:
    """The heading left by `quarters` repetitions of a turn action."""
    return (int(heading) + quarter_of(action_name) * int(quarters)) % QUARTERS


def pivot_to(heading, bearing):
    """(turn action, quarter turns) — the ONE rotation that lands `heading` on `bearing`.

    `(None, 0)` when it already faces it. The 180-degree case is reported as two `turn_right`
    quarters, which keeps the handedness consistent through a reversal rather than alternating
    between the same two bearings; `repeat` lets both quarters ride one decision."""
    delta = (int(bearing) - int(heading)) % QUARTERS
    if delta == 0:
        return (None, 0)
    if delta == 3:
        return ("turn_left", 1)
    return ("turn_right", delta)


def pivot_path(heading, action_name, quarters):
    """The headings a rotation passes THROUGH — start and end excluded.

    This is what makes an arc safe to compress into one decision: the harness sees a frame only at
    the heading it STOPS on, so every heading in here is one it would have looked from and will not."""
    return tuple(heading_after(heading, action_name, q) for q in range(1, int(quarters)))


def sweep_arc(heading, action_name, bearings):
    """Quarters needed to pass through every bearing in `bearings`, rotating one way only.

    0 when there is nothing to visit. Rotating consistently visits all four headings in three
    quarters, where picking the nearest unvisited bearing each time can double back and spend more."""
    wanted = frozenset(int(b) % QUARTERS for b in bearings)
    arc = 0
    for q in range(1, QUARTERS):
        if heading_after(heading, action_name, q) in wanted:
            arc = q
    return arc


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
    elif action_name in TURN_ACTIONS:
        # A rotation's `repeat` is its quarter-turn count, and it MUST ride without a granularity:
        # the adapter forces `repeat` to 1 whenever one is present (measured). No granularity key is
        # written here, so the response contract ("granularity iff a translation") still holds.
        meta["repeat"] = max(1, min(PIVOT_CAP, int(cells)))
    try:
        return type(proposal)(type="key_press", key=key, metadata=meta)
    except Exception:
        return proposal                  # an unexpected Action shape must never cost the step
