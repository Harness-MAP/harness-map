"""The Block3D action / key algebra + the engine Action contract (reused, unchanged).

Each step returns an engine Action mapping a legal block-builder action to an engine key. The
SpatialWorld Block3D adapter (Block3D/block3d_adapter.py::get_action_profile / execute_mapped_action)
drives a cursor over a 6x6x6 grid: six 1-unit cursor moves along +/-X, +/-Y, +/-Z, plus a
place/remove toggle at the cursor. There is NO movement granularity — every move is exactly one cell.
The evaluator's PygameInputSource maps the key -> game.execute_mapped_action, so this table must stay
in step with rules.md's control mapping. This is a dormant, evolvable lever: the anchor delegates the
decision to `baseline_step`, so a proposer can add richer action shaping here without touching the
engine.
"""

# action_name -> engine key (from Block3D/block3d_adapter.py get_action_profile ActionMapping list)
KEY_MAPPING = {
    "move_left": "left",       # -X : move cursor left
    "move_right": "right",     # +X : move cursor right
    "move_up": "up",           # -Y : move cursor up
    "move_down": "down",       # +Y : move cursor down
    "move_forward": "w",       # +Z : move cursor forward
    "move_backward": "s",      # -Z : move cursor backward
    "place_block": "space",    # toggle a block at the cursor (place or remove)
}
MOVEMENT_ACTIONS = {"move_left", "move_right", "move_up", "move_down",
                    "move_forward", "move_backward"}
TOGGLE_ACTIONS = {"place_block"}
# Block3D has no granularity — each move changes the cursor by exactly 1 cell.
CELLS_PER_MOVE = 1


def key_for(action_name: str):
    """The engine key for a legal Block3D action, or None if the name is not a legal action."""
    return KEY_MAPPING.get(action_name)
