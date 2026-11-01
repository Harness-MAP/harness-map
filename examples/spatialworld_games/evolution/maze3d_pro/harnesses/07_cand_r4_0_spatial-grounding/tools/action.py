"""The multi-floor 3D-maze (maze3d_pro) action / key algebra + the engine Action contract.

Each step returns an engine Action mapping a legal maze action to an engine key. The SpatialWorld
maze3d_pro adapter (maze3d_pro_adapter.py::execute_mapped_action / get_action_mapping) recognizes
six actions: forward/backward translation (both carry a `granularity` of small/medium/large =
1/2/3 grid cells, stopping early at a wall), a 90-degree left/right turn, and climb up/down
between floors while standing on a stair tile. The evaluator's PygameInputSource maps the key ->
execute_mapped_action(key, granularity), so this table must stay in step with rules.md's legal
action set.

`retarget` is the half of this module that the governor in tools/plan.py needs. Two details in
PygameInputSource make a naive rebuild of an Action wrong:

  * `_action_uses_granularity` forces repeat=1 ONLY when a granularity is present. A turn or a
    climb that inherits a stale `repeat: 3` from a replaced `move_forward` fires THREE times —
    three quarter-turns, i.e. a reversal the planner never asked for. `repeat` must be stripped.
  * `_get_action_granularity` falls back to `metadata["json_data"]["granularity"]`, so a stale
    `json_data` blob re-injects the granularity that was just removed. It must be stripped too.

Everything else in the metadata envelope (frame_number, response_text, the parse provenance
flags) is left verbatim so the evaluator's own logging and the anti-cheat VLM-call accounting see
exactly what they saw before.
"""

# action_name -> engine key (from maze3d_pro_adapter.py execute_mapped_action key handling)
KEY_MAPPING = {
    "move_forward": "w",
    "move_backward": "s",
    "turn_left": "a",
    "turn_right": "d",
    "climb_up": "q",       # fires only while the tile under your feet is a stair
    "climb_down": "e",     # fires only while the tile under your feet is a stair
}
NAME_MAPPING = {v: k for k, v in KEY_MAPPING.items()}

# Only the two translations take a granularity; turns and climbs never do.
MOVEMENT_ACTIONS = {"move_forward", "move_backward"}
TURN_ACTIONS = {"turn_left", "turn_right"}
CLIMB_ACTIONS = {"climb_up", "climb_down"}
LEGAL_ACTIONS = set(KEY_MAPPING)
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}
STEPS_GRANULARITY = {1: "small", 2: "medium", 3: "large"}


def key_for(action_name: str):
    """The engine key for a legal maze3d_pro action, or None if the name is not a legal action."""
    return KEY_MAPPING.get(action_name)


def name_for(key: str):
    """The inverse table: the engine dispatches on `action.key`, so this is how a proposal is
    read back out of an Action the evaluator's own parser built."""
    return NAME_MAPPING.get((key or "").strip().lower())


def granularity_steps(name: str) -> int:
    """Cells moved for a granularity label (defaults to 1 = `small`); mirrors the adapter's
    _granularity_to_steps clamp to [1, 3]."""
    return GRANULARITY_STEPS.get((name or "small").strip().lower(), 1)


def read_proposal(action):
    """(action_name, granularity_label_or_None) for an Action built by the evaluator's parser."""
    if action is None:
        return None, None
    name = name_for(getattr(action, "key", None))
    meta = getattr(action, "metadata", None) or {}
    gran = meta.get("granularity")
    if gran is None and isinstance(meta.get("json_data"), dict):
        gran = meta["json_data"].get("granularity")
    if name in MOVEMENT_ACTIONS:
        gran = STEPS_GRANULARITY[granularity_steps(gran)]
    else:
        gran = None
    return name, gran


def retarget(action, name, granularity=None):
    """A new engine Action for `name`, reusing `action`'s metadata envelope with every field that
    could re-inject the OLD action's distance removed. Returns `action` unchanged on anything
    unexpected, so the governor can never turn a legal proposal into an illegal one."""
    key = key_for(name)
    if key is None or action is None:
        return action
    cls = type(action)
    meta = dict(getattr(action, "metadata", None) or {})
    for stale in ("granularity", "repeat", "json_data"):
        meta.pop(stale, None)
    meta["function_called"] = name
    meta["harness_override"] = True
    if name in MOVEMENT_ACTIONS:
        steps = granularity_steps(granularity)
        meta["granularity"] = STEPS_GRANULARITY[steps]
        meta["repeat"] = steps
    try:
        return cls(type="key_press", key=key, metadata=meta)
    except Exception:
        return action
