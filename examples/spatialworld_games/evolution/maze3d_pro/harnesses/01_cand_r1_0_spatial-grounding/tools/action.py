"""The multi-floor 3D-maze (maze3d_pro) action / key algebra + the engine Action contract.

Each step returns an engine Action mapping a legal maze action to an engine key. The SpatialWorld
maze3d_pro adapter (maze3d_pro/maze3d_pro_adapter.py::execute_mapped_action / get_action_mapping)
recognizes six actions: forward/backward translation (both carry a `granularity` of small/medium/large
= 1/2/3 grid cells, stopping early at a wall), a 90-degree left/right turn, and climb up/down between
floors on a stair tile. The evaluator's PygameInputSource maps the key -> execute_mapped_action(key,
granularity), so this table must stay in step with rules.md's legal action set. This is a dormant,
evolvable lever: the anchor delegates the decision to `baseline_step`, so a proposer can add richer
action shaping here without touching the engine.
"""

# action_name -> engine key (from maze3d_pro_adapter.py execute_mapped_action key handling)
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
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}


def key_for(action_name: str):
    """The engine key for a legal maze3d_pro action, or None if the name is not a legal action."""
    return KEY_MAPPING.get(action_name)


def granularity_steps(name: str) -> int:
    """Cells moved for a granularity label (defaults to 1 = `small`); mirrors the adapter's
    _granularity_to_steps clamp to [1, 3]."""
    return GRANULARITY_STEPS.get((name or "small").strip().lower(), 1)
