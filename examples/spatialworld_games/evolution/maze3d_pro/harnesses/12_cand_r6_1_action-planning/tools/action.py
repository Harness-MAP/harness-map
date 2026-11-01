"""The maze3d_pro action / key algebra, the engine `Action` envelope, and the BEARING arithmetic
the route planner needs.

Each step returns an engine Action mapping a legal maze action to an engine key. The adapter
recognizes six actions: forward/backward translation (both carry a `granularity` of
small/medium/large = 1/2/3 grid cells, stopping early at a wall), a 90-degree left/right turn, and
climb up/down between floors on a stair tile. The evaluator's PygameInputSource dispatches on
`action.key` and reads `metadata["granularity"]`, so this table must stay in step with rules.md's
legal action set.

This module is the planner's ARITHMETIC as well as its table. `tools/plan.py` searches over poses
`(x, y, heading)` and prices every edge at ONE DECISION, so it needs to know, in closed form:

  * `heading_after(h, name)` — what a turn does to a bearing (and nothing else);
  * `advance(cell, h, cells)` — the cell a translation of `cells` lands on, forward or backward;
  * `turn_toward(h, want)` — which single turn shortens the arc onto a wanted bearing, and how
    many quarter-turns the arc still is. The planner pays one decision per quarter-turn, which is
    why a route that avoids turning is cheaper than a route that is shorter in cells.
  * `retarget(...)` — a NEW Action for a different legal action, reusing the proposal's metadata
    envelope. CRITICAL: `granularity`, `repeat` and `json_data` are stripped first. The input
    source forces repeat=1 only when a granularity is present, so a stale `repeat: 3` left on a
    re-targeted TURN would fire three 90-degree turns instead of one.

Bearings are integers 0..3 and the frame of reference is the harness's OWN: the pose the episode
starts in is called `(0, 0)` facing 0. Nothing here is read from the engine, so the labels are
arbitrary and only their arithmetic matters.
"""
from __future__ import annotations

# action_name -> engine key (from the adapter's execute_mapped_action key handling)
KEY_MAPPING = {
    "move_forward": "w",
    "move_backward": "s",
    "turn_left": "a",
    "turn_right": "d",
    "climb_up": "q",
    "climb_down": "e",
}
MOVEMENT_ACTIONS = {"move_forward", "move_backward"}
TURN_ACTIONS = {"turn_left", "turn_right"}
CLIMB_ACTIONS = {"climb_up", "climb_down"}
LEGAL_ACTIONS = tuple(KEY_MAPPING)
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}
STEPS_GRANULARITY = {1: "small", 2: "medium", 3: "large"}

REVERSE_KEY_MAPPING = {v: k for k, v in KEY_MAPPING.items()}
_CLIMB_INVERSE = {"climb_up": "climb_down", "climb_down": "climb_up"}
_TURN_DELTA = {"turn_left": -1, "turn_right": 1}

#: unit step per bearing, in the harness's own (x, y) frame. Four bearings, one quarter-turn apart.
BEARING_STEP = ((0, -1), (1, 0), (0, 1), (-1, 0))


def key_for(action_name):
    """The engine key for a legal maze3d_pro action, or None if the name is not a legal action."""
    return KEY_MAPPING.get(action_name)


def name_for(key):
    """The legal action name an engine key will actually execute, or None.

    Preferred over `metadata["function_called"]`, which also carries non-action labels such as the
    evaluator's own `default_move_forward` parse fallback. The engine executes the KEY, so the key
    is the truth about what ran."""
    if not isinstance(key, str):
        return None
    return REVERSE_KEY_MAPPING.get(key.strip().lower())


def granularity_steps(name):
    """Cells a granularity label asks for (defaults to 1 = `small`); mirrors the adapter's own
    clamp of the label into the range 1..3."""
    return GRANULARITY_STEPS.get(str(name or "small").strip().lower(), 1)


def granularity_for(cells):
    """The label that asks for exactly `cells` cells, clamped into the engine's range 1..3."""
    return STEPS_GRANULARITY[max(1, min(3, int(cells)))]


def heading_after(heading, action_name):
    """The bearing a turn leaves behind. Any non-turn leaves the bearing alone."""
    return (int(heading) + _TURN_DELTA.get(action_name, 0)) % 4


def advance(cell, heading, cells=1, backward=False):
    """The cell `cells` steps along `heading`, or against it when `backward`."""
    dx, dy = BEARING_STEP[int(heading) % 4]
    sign = -1 if backward else 1
    return (cell[0] + sign * dx * int(cells), cell[1] + sign * dy * int(cells))


def bearing_between(src, dst):
    """The bearing that points from `src` at its orthogonal neighbour `dst`, or None."""
    step = (dst[0] - src[0], dst[1] - src[1])
    return BEARING_STEP.index(step) if step in BEARING_STEP else None


def turn_toward(heading, wanted):
    """`(turn_name, quarter_turns_remaining)` for swinging `heading` onto `wanted`.

    The arc is at most two quarter-turns, and at exactly two the handedness is a free choice — this
    returns `turn_right` there, so the sweep a route of two turns performs is always the same way
    round. `(None, 0)` when the bearing already matches."""
    arc = (int(wanted) - int(heading)) % 4
    if arc == 0:
        return None, 0
    if arc == 3:
        return "turn_left", 1
    return "turn_right", arc


def inverse_climb(action_name):
    """The floor transition that undoes `action_name`, or None if it is not a climb."""
    return _CLIMB_INVERSE.get(action_name)


def is_undo(previous, action_name):
    """Is `action_name` the exact undo of `previous`? A left after a right (and the reverse) is a
    two-cycle that returns the body to the pose it was already refused in, and a climb straight
    back down undoes a floor transition before the new floor has shown anything."""
    if previous in TURN_ACTIONS and action_name in TURN_ACTIONS:
        return previous != action_name
    return bool(previous) and _CLIMB_INVERSE.get(previous) == action_name


def retarget(proposal, action_name, granularity="small", reason="", proposed_name=""):
    """A NEW engine Action for `action_name`, built from `proposal`'s metadata envelope.

    Reuses `type(proposal)` so no engine import is needed. `granularity` / `repeat` / `json_data`
    are dropped and re-added only for the two translations (json_data because the input source ALSO
    reads `metadata["json_data"]["granularity"]` as a fallback). The override is stamped into
    metadata as `harness_override`, so the trace records both what the frozen model proposed and
    what actually ran."""
    key = KEY_MAPPING.get(action_name)
    if key is None:
        return proposal
    meta = dict(getattr(proposal, "metadata", None) or {})
    for dead in ("granularity", "repeat", "json_data"):
        meta.pop(dead, None)
    was = proposed_name or meta.get("function_called") or ""
    meta["function_called"] = action_name
    meta["harness_override"] = {"proposed": was, "executed": action_name, "reason": reason}
    if action_name in MOVEMENT_ACTIONS:
        label = str(granularity or "small").strip().lower()
        if label not in GRANULARITY_STEPS:
            label = "small"
        meta["granularity"] = label
        meta["repeat"] = granularity_steps(label)
    try:
        return type(proposal)(type="key_press", key=key, metadata=meta)
    except Exception:
        return proposal           # an unexpected Action shape must never cost the step


def read_proposal(proposal):
    """`(action_name, granularity_label)` the engine will really execute for `proposal`.

    The granularity is read from `metadata["granularity"]` first and from the parsed
    `metadata["json_data"]` second, because the input source consults them in that order."""
    name = name_for(getattr(proposal, "key", None))
    meta = getattr(proposal, "metadata", None) or {}
    label = meta.get("granularity")
    if not label:
        blob = meta.get("json_data")
        label = blob.get("granularity") if isinstance(blob, dict) else None
    return name, (str(label).strip().lower() if label else "small")
