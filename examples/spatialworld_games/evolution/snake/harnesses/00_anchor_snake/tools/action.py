"""The 3D-Snake action / key algebra + the engine Action contract (reused, unchanged).

Each step returns an engine Action mapping a legal snake action to an engine key. The SpatialWorld
Snake adapter (Snake/snake_adapter.py::get_action_profile / execute_mapped_action) steers the snake
one cell per move through a 4x4x4 grid along six directions (+/-X, +/-Y, +/-Z). There is NO movement
granularity — every move advances exactly one cell — and a 180-degree reversal of the current heading
is rejected by the engine (`_try_move` refuses `last_move_dir + delta == 0`). The evaluator's
PygameInputSource maps the key -> game.execute_mapped_action, so this table must stay in step with
rules.md's control mapping. This is a dormant, evolvable lever: the anchor delegates the decision to
`baseline_step`, so a proposer can add richer action shaping here without touching the engine.
"""

# action_name -> engine key (from Snake/snake_adapter.py get_action_profile ActionMapping list)
KEY_MAPPING = {
    "move_right": "d",         # +X
    "move_left": "a",          # -X
    "move_up": "w",            # -Y
    "move_down": "s",          # +Y
    "move_forward": "q",       # -Z  (into the screen)
    "move_backward": "e",      # +Z  (out of the screen)
}
MOVEMENT_ACTIONS = set(KEY_MAPPING)     # all snake actions are 1-cell moves
# The 6 unit direction vectors (dx, dy, dz) each action applies, mirroring the adapter's _try_move.
DIRECTION_VECTORS = {
    "move_right": (1, 0, 0),
    "move_left": (-1, 0, 0),
    "move_up": (0, -1, 0),
    "move_down": (0, 1, 0),
    "move_forward": (0, 0, -1),
    "move_backward": (0, 0, 1),
}
# Snake has no granularity — each move advances the snake by exactly 1 cell.
CELLS_PER_MOVE = 1


def key_for(action_name: str):
    """The engine key for a legal snake action, or None if the name is not a legal action."""
    return KEY_MAPPING.get(action_name)


def is_reversal(action_name: str, last_dir) -> bool:
    """True if `action_name` reverses `last_dir` (dx,dy,dz) — the engine rejects such 180 turns."""
    v = DIRECTION_VECTORS.get(action_name)
    if not v or not last_dir:
        return False
    return all(a + b == 0 for a, b in zip(v, last_dir))
