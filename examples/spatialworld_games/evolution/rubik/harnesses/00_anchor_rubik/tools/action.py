"""The Rubik's-Cube action / key algebra + the engine Action contract (reused, unchanged).

Each step returns an engine Action naming a legal cube move. The SpatialWorld rubik evaluator
(scripts/game/rubik_openai_evaluation.py RUBIK_ACTIONS) exposes two families of move, executed by
rubik_cube/rubik_cube_adapter.py::rotate_face(face, direction) and rotate_cube(axis, direction):

  * FACE ROTATIONS (90 degrees, animated, CHANGE the cube state): one per face x {cw, acw}, i.e.
    rotate_<front|back|left|right|up|down>_<cw|acw>. Adapter face keys: f/b/l/r/u/d (MOVE_KEY_MAP).
  * VIEW ROTATIONS (45 degrees, immediate, DO NOT change the cube state): rotate the whole cube
    about an axis x {cw, acw}, i.e. rotate_cube_<x|y|z>_<cw|acw>. Adapter axis keys map x->R (RIGHT),
    y->U (UP), z->F (FRONT) (ROTATE_KEY_MAP), each dispatched through rotate_cube(axis, direction).

The model emits {"action": <name>, "face": <letter|axis>, "direction": "CW"|"ACW"}. This is a dormant,
evolvable lever: the anchor delegates the decision to `baseline_step`, so a proposer can add richer
action shaping here without touching the engine. The tables below mirror the shipped action set.
"""

# The 6 cube faces (rotate_face targets) -> adapter face key (MOVE_KEY_MAP in constants.py)
FACE_KEYS = {
    "F": "f",  # FRONT (Red)
    "B": "b",  # BACK  (Orange)
    "L": "l",  # LEFT  (Blue)
    "R": "r",  # RIGHT (Green)
    "U": "u",  # UP    (Yellow)
    "D": "d",  # DOWN  (White)
}

# The 3 view-rotation axes -> adapter axis face + key (ROTATE_KEY_MAP in constants.py)
VIEW_AXES = {
    "x": {"axis_face": "R", "key": "x"},   # rotate about RIGHT face (X-axis)
    "y": {"axis_face": "U", "key": "y"},   # rotate about UP face    (Y-axis)
    "z": {"axis_face": "F", "key": "z"},   # rotate about FRONT face (Z-axis)
}

DIRECTIONS = {"CW": "CLOCKWISE", "ACW": "ANTICLOCKWISE"}   # adapter CW/ACW constants

# The full model-facing action vocabulary (RUBIK_ACTIONS names in rubik_openai_evaluation.py).
FACE_ROTATION_ACTIONS = {
    "rotate_front_cw", "rotate_front_acw",
    "rotate_back_cw", "rotate_back_acw",
    "rotate_left_cw", "rotate_left_acw",
    "rotate_right_cw", "rotate_right_acw",
    "rotate_up_cw", "rotate_up_acw",
    "rotate_down_cw", "rotate_down_acw",
}
VIEW_ROTATION_ACTIONS = {
    "rotate_cube_x_cw", "rotate_cube_x_acw",
    "rotate_cube_y_cw", "rotate_cube_y_acw",
    "rotate_cube_z_cw", "rotate_cube_z_acw",
}
LEGAL_ACTIONS = FACE_ROTATION_ACTIONS | VIEW_ROTATION_ACTIONS


def is_legal(action_name: str) -> bool:
    """True if `action_name` is one of the shipped face- or view-rotation actions."""
    return action_name in LEGAL_ACTIONS


def is_view_rotation(action_name: str) -> bool:
    """True for the immediate 45-degree whole-cube view rotations (state-preserving)."""
    return action_name in VIEW_ROTATION_ACTIONS


def direction_const(direction: str) -> str:
    """Map a model 'CW'/'ACW' to the adapter's CLOCKWISE/ANTICLOCKWISE constant."""
    return DIRECTIONS.get((direction or "CW").strip().upper(), "CLOCKWISE")
