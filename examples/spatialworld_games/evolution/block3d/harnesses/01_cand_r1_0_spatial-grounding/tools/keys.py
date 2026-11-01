"""KEYS — the block3d action algebra, and the only place a name becomes an engine keystroke.

The engine exposes seven actions: six one-cell cursor steps along the three axes and a toggle
that places a block on an empty cell or lifts one off a filled cell. There is no step
granularity — every move is exactly one cell — and the toggle is its own inverse, which is the
single fact that makes an un-booked second visit to a finished cell destructive.

This table is the inverse map as well as the forward one: the arbiter in tools/vlm.py reads the
model's chosen key back into a name so that a proposal can be compared with the harness's own
route before anything is applied. Keeping both directions in one module is what makes the two
agree by construction.
"""

KEY_MAPPING = {
    "move_left": "left",
    "move_right": "right",
    "move_up": "up",
    "move_down": "down",
    "move_forward": "w",
    "move_backward": "s",
    "place_block": "space",
}

STEP = {
    "move_left": (-1, 0, 0),
    "move_right": (1, 0, 0),
    "move_up": (0, -1, 0),
    "move_down": (0, 1, 0),
    "move_forward": (0, 0, 1),
    "move_backward": (0, 0, -1),
}

AXIS_NAMES = (("move_right", "move_left"), ("move_down", "move_up"),
              ("move_forward", "move_backward"))
TOGGLE = "place_block"
BY_KEY = {v: k for k, v in KEY_MAPPING.items()}


def key_for(name):
    """The engine keystroke for a legal action name, or None when the name is not legal."""
    return KEY_MAPPING.get(name)


def name_for(key):
    """The legal action name behind an engine keystroke, or None for an unknown key."""
    return BY_KEY.get(key)


def move_names():
    """The six cursor steps, in a fixed order, so every ranking over them is reproducible."""
    return sorted(STEP)
