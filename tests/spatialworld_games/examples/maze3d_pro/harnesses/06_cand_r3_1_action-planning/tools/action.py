"""The multi-floor 3D-maze (maze3d_pro) action / key algebra + the engine Action contract.

Each step returns an engine Action mapping a legal maze action to an engine key. The SpatialWorld
maze3d_pro adapter recognizes six actions: forward/backward translation (both carry a `granularity`
of small/medium/large = 1/2/3 grid cells, stopping early at a wall), a 90-degree left/right turn,
and climb up/down between floors on a stair tile. The evaluator's PygameInputSource dispatches on
`action.key` and reads `metadata["granularity"]`, so this table must stay in step with rules.md's
legal action set.

This module is now an ACTIVE lever: besides the table it owns the algebra the planner
(tools/plan.py) needs to reason about and RE-TARGET a proposal —

  * `name_for(key)`      — the inverse table; the engine executes `action.key`, so the key (not the
                           trace-only `metadata["function_called"]`) is the truth about what ran.
  * `inverse_climb(name)`— the undo of a floor transition, for the post-climb lock.
  * `retarget(...)`      — build a NEW Action for a different legal action, reusing the proposal's
                           metadata envelope. CRITICAL: `granularity` AND `repeat` are stripped
                           first. PygameInputSource only forces repeat=1 when a granularity is
                           present, so a stale `repeat: 3` left on a re-targeted TURN would fire
                           three 90-degree turns (270 degrees) instead of one.
"""
from __future__ import annotations

# action_name -> engine key (from the adapter's execute_mapped_action key handling)
KEY_MAPPING = {
    "move_forward": "w",
    "move_backward": "s",
    "turn_left": "a",
    "turn_right": "d",
    "climb_up": "q",       # only valid on a stair with "[Q] UP" shown
    "climb_down": "e",     # only valid on a stair with "[E] DOWN" shown
}
# Only the two translations take a granularity; turns and climbs never do.
MOVEMENT_ACTIONS = {"move_forward", "move_backward"}
TURN_ACTIONS = {"turn_left", "turn_right"}
CLIMB_ACTIONS = {"climb_up", "climb_down"}
LEGAL_ACTIONS = tuple(KEY_MAPPING)
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}

REVERSE_KEY_MAPPING = {v: k for k, v in KEY_MAPPING.items()}
_CLIMB_INVERSE = {"climb_up": "climb_down", "climb_down": "climb_up"}


def key_for(action_name: str):
    """The engine key for a legal maze3d_pro action, or None if the name is not a legal action."""
    return KEY_MAPPING.get(action_name)


def name_for(key) -> str | None:
    """The legal action name an engine key will actually execute, or None. Preferred over
    `metadata["function_called"]`, which also carries non-action labels (e.g. the evaluator's
    `default_move_forward` parse fallback)."""
    if not isinstance(key, str):
        return None
    return REVERSE_KEY_MAPPING.get(key.strip().lower())


def inverse_climb(action_name: str):
    """The floor transition that undoes `action_name`, or None if it is not a climb."""
    return _CLIMB_INVERSE.get(action_name)


def granularity_steps(name: str) -> int:
    """Cells moved for a granularity label (defaults to 1 = `small`); mirrors the adapter's
    _granularity_to_steps clamp to [1, 3]."""
    return GRANULARITY_STEPS.get((name or "small").strip().lower(), 1)


def retarget(proposal, action_name: str, granularity: str = "small",
             reason: str = "", proposed_name: str = ""):
    """A NEW engine Action for `action_name`, built from `proposal`'s metadata envelope.

    Reuses `type(proposal)` so no engine import is needed. `granularity`/`repeat`/`json_data` are
    dropped and re-added only for the two translations (json_data is dropped because the input
    source ALSO reads `metadata["json_data"]["granularity"]` as a fallback). The override is
    stamped into metadata as `harness_override` so the trace records both what the frozen model
    proposed and what actually ran.
    """
    key = KEY_MAPPING.get(action_name)
    if key is None:
        return proposal
    meta = dict(getattr(proposal, "metadata", None) or {})
    for dead in ("granularity", "repeat", "json_data"):
        meta.pop(dead, None)
    meta["function_called"] = action_name
    meta["harness_override"] = {
        "proposed": proposed_name or meta.get("function_called"),
        "executed": action_name,
        "reason": reason,
    }
    if action_name in MOVEMENT_ACTIONS:
        gran = (granularity or "small").strip().lower()
        if gran not in GRANULARITY_STEPS:
            gran = "small"
        meta["granularity"] = gran
        meta["repeat"] = granularity_steps(gran)
    try:
        return type(proposal)(type="key_press", key=key, metadata=meta)
    except Exception:
        return proposal        # an unexpected Action shape must never cost the step
