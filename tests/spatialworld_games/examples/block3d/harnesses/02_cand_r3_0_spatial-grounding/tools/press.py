"""The press algebra: what one press costs, where it leaves the cursor, and which presses are FREE.

Block3D has no movement granularity — each of the six moves shifts the cursor by exactly one cell and
is CLAMPED at the grid face, so a press into a face is a no-op that still costs the full press. The
adapter bills TWO `steps_taken` per executed action, so `max_steps=100` is a budget of PRESSES (see
`WALLET`), not of engine steps, and a clamped press is charged at the same rate as a productive one.

The one arithmetic fact the schedule rests on: travelling from the cursor to a cell along an L1 route
lets the axes be closed in ANY order for the SAME number of presses. Every key that strictly reduces
the L1 gap is therefore interchangeable, which is exactly the choice tools/vlm.py hands back to the
frozen model for free. Everything here is pure arithmetic over the harness's own emitted keys; nothing
in this module reads engine state.
"""

KEY_BY_NAME = {
    "move_left": "left",       # -X
    "move_right": "right",     # +X
    "move_up": "up",           # -Y
    "move_down": "down",       # +Y
    "move_forward": "w",       # +Z
    "move_backward": "s",      # -Z
    "place_block": "space",    # toggle at the cursor
}
NAME_BY_KEY = {v: k for k, v in KEY_BY_NAME.items()}
DELTA_BY_KEY = {"left": (-1, 0, 0), "right": (1, 0, 0), "up": (0, -1, 0),
                "down": (0, 1, 0), "w": (0, 0, 1), "s": (0, 0, -1)}
TOGGLE = "space"
MOVE_KEYS = tuple(DELTA_BY_KEY)
GRID = 6
LAST = GRID - 1
SPAWN = (2, 2, 0)              # the documented initial cursor, stated in AGENT.md
WALLET = 50                    # max_steps 100 at 2 engine steps per executed press
FLOOR = 0
CELLS_PER_MOVE = 1             # no granularity: every move is exactly one cell
_SHUT = {"left": "right", "right": "left", "up": "down",
         "down": "up", "w": "s", "s": "w"}


def clamp(v):
    """One axis of the engine's own cursor update: the grid faces absorb, they do not wrap."""
    if v < FLOOR:
        return FLOOR
    return v if v <= LAST else LAST


def shifted(cursor, key):
    """The cell a move key leaves the cursor in. A clamped press returns the SAME cell."""
    step = DELTA_BY_KEY.get(key)
    if step is None:
        return tuple(cursor)
    x, y, z = cursor
    dx, dy, dz = step
    return (clamp(x + dx), clamp(y + dy), clamp(z + dz))


def l1(here, there):
    """The travel metric: the six moves are unit steps along the axes, so distance is Manhattan."""
    ax, ay, az = here
    bx, by, bz = there
    return abs(ax - bx) + abs(ay - by) + abs(az - bz)


def inside(cell):
    """Is this a cell of the grid at all? The faces are at 0 and 5 on each of the three axes."""
    for axis in cell:
        if axis < 0 or axis > LAST:
            return False
    return True


def key_for(name):
    """The engine key for a legal action name, or None when the name is not one of the seven."""
    return KEY_BY_NAME.get(name)


def action_name(key):
    """The legal action name for an engine key; the toggle is the safe default for anything else."""
    return NAME_BY_KEY.get(key, "place_block")


def moves_cursor(cursor, key):
    """Would this press actually move? False for the toggle and for every press into a face."""
    if key not in DELTA_BY_KEY:
        return False
    return shifted(cursor, key) != tuple(cursor)


def closing_keys(cursor, goal):
    """The move keys that strictly close the L1 gap — interchangeable, hence free to hand away."""
    if goal is None:
        return []
    gap = l1(cursor, goal)
    if gap < 1:
        return []
    return [k for k in MOVE_KEYS if l1(shifted(cursor, k), goal) < gap]


def opposite(key):
    """The key that undoes `key` — the one press a travelling route never needs."""
    return _SHUT.get(key)


def engine_steps(presses):
    """Presses converted to the `steps_taken` the adapter actually bills.

    One executed action advances the counter TWICE — the mapped method updates and the input source
    updates again — which is why `max_steps` of 100 is a wallet of 50 presses and not of 100.
    """
    return 2 * presses


def toll(start, order):
    """Presses to walk `order` and toggle each cell: travel plus one press per cell."""
    spent = 0
    here = tuple(start)
    step = 0
    while step < len(order):
        cell = tuple(order[step])
        spent = spent + l1(here, cell) + 1
        here = cell
        step = step + 1
    return spent
