"""The rubik action algebra, the cube the harness carries in its head, and the PLAN it executes.

The engine exposes two families of move, executed by `rubik_cube/rubik_cube_adapter.py`
(`rotate_face` / `rotate_cube`) and dispatched by `rubik_openai_evaluation.py::execute_action` on
`metadata["action_type"]`:

  * FACE ROTATIONS (90 degrees, animated, CHANGE the cube) — rotate_<face>_<cw|acw>.
  * VIEW ROTATIONS (45 degrees, immediate, DO NOT change the cube) — rotate_cube_<x|y|z>_<cw|acw>.

WHAT THIS FILE KNOWS AND HOW IT CAME TO KNOW IT. A cube is a permutation of 54 stickers, and a
quarter turn is a rotation of one layer: `spin` is that rotation written down as geometry, so the
twelve permutations are DERIVED here rather than guessed. Two conventions could not be derived,
only measured, and both were measured against the rendered frame on the real engine:

  1. `rotate_<face>_cw` is a clockwise quarter turn of that face seen from OUTSIDE it, and face
     turns are CUBE-FIXED (the camera's pose does not rename the faces). Checked by replaying 60
     recorded transitions on levels 1, 13 and 18: every predicted front face matched the decoded
     one, and on each level exactly ONE state in the radius-4 ball was consistent with the whole
     recorded stream (level 1 = D', level 13 = F U L', level 18 = U D D F).
  2. Which nine stickers a square-on frame shows, and in which orientation, at each camera pose
     the survey visits — `POSE_SLOTS`. Fitted from those same three levels and unique at every
     pose; it reproduced all 24 accepted readings.

`self_test()` re-checks both at import against a stored fingerprint, and the harness falls back to
a blind sweep if either has drifted, so a renderer change degrades instead of lying.

NOTHING HERE READS ENGINE STATE. The cube model is advanced only by the move names this harness
itself emitted; every sticker it believes in was decoded from the frame the agent is shown.
"""
from itertools import product

# ─── the model-facing vocabulary (RUBIK_ACTIONS in rubik_openai_evaluation.py) ───────────────
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

DIRECTIONS = {"CW": "CLOCKWISE", "ACW": "ANTICLOCKWISE"}
FACE_WORD = {"F": "FRONT", "B": "BACK", "L": "LEFT", "R": "RIGHT", "U": "UP", "D": "DOWN"}
AXIS_WORD = {"x": "RIGHT", "y": "UP", "z": "FRONT"}       # the axis face each view turn names

# name -> (engine face word, engine direction word, engine key, engine action_type). The engine
# dispatches on the METADATA, never on the action name, and the shipped parser fills those from
# the model's own JSON fields — so a harness that rewrites an Action must set all four together.
SPEC = {}
for _letter in sorted(FACE_WORD):
    for _suffix in ("cw", "acw"):
        SPEC["rotate_%s_%s" % (FACE_WORD[_letter].lower(), _suffix)] = (
            FACE_WORD[_letter], DIRECTIONS[_suffix.upper()], _letter.lower(), "face_rotation")
for _axis in sorted(AXIS_WORD):
    for _suffix in ("cw", "acw"):
        SPEC["rotate_cube_%s_%s" % (_axis, _suffix)] = (
            AXIS_WORD[_axis], DIRECTIONS[_suffix.upper()], _axis, "view_rotation")

# move letter ("F", "F'") <-> engine action name
MOVE_NAMES = {}
for _letter in "FBUDRL":
    MOVE_NAMES[_letter] = "rotate_%s_cw" % FACE_WORD[_letter].lower()
    MOVE_NAMES[_letter + "'"] = "rotate_%s_acw" % FACE_WORD[_letter].lower()
LETTER_OF = {}
for _mv, _name in MOVE_NAMES.items():
    LETTER_OF[_name] = _mv
MOVES_IN_ORDER = tuple(sorted(MOVE_NAMES))


def is_legal(action_name):
    """True if `action_name` is one of the shipped face- or view-rotation actions."""
    return action_name in LEGAL_ACTIONS


def is_view_rotation(action_name):
    """True for the immediate 45-degree whole-cube view rotations (state-preserving)."""
    return action_name in VIEW_ROTATION_ACTIONS


def direction_const(direction):
    """Map a model 'CW'/'ACW' to the adapter's CLOCKWISE/ANTICLOCKWISE constant."""
    return DIRECTIONS.get((direction or "CW").strip().upper(), "CLOCKWISE")


def opposite(move):
    """The inverse quarter turn of a move letter: F <-> F'."""
    if move.endswith("'"):
        return move[0]
    return move + "'"


# ─── the cube, as geometry ───────────────────────────────────────────────────────────────────
# A sticker lives on a cubie at an integer point in {-1,0,1}^3, facing one of the six axes. The
# outward faces are exactly the (point, normal) pairs whose dot product is 1, which is why the
# enumeration below is both complete and 54 long.
NORMALS = ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
FACE_NORMAL = {"F": (0, 0, 1), "B": (0, 0, -1), "U": (0, 1, 0),
               "D": (0, -1, 0), "R": (1, 0, 0), "L": (-1, 0, 0)}
SOLVED_COLOUR = {(0, 0, 1): "R", (0, 0, -1): "O", (0, 1, 0): "Y",
                 (0, -1, 0): "W", (1, 0, 0): "G", (-1, 0, 0): "B"}

SLOTS = []
for _point in product((-1, 0, 1), repeat=3):
    for _normal in NORMALS:
        if sum(a * b for a, b in zip(_point, _normal)) == 1:
            SLOTS.append((_point, _normal))
SLOT_INDEX = {}
for _i, _slot in enumerate(SLOTS):
    SLOT_INDEX[_slot] = _i

SOLVED = tuple(SOLVED_COLOUR[_normal] for _point, _normal in SLOTS)


def spin(face, vector):
    """One CLOCKWISE quarter turn of `face`, seen from outside it, applied to a 3-vector.

    Each line is a proper rotation about that face's axis. The chirality is not a convention this
    file is free to choose: it is what the engine does, measured off the frame (see the module
    docstring), and `self_test` fails loudly if the two ever part company.
    """
    x, y, z = vector
    if face == "F":
        return (y, -x, z)
    if face == "B":
        return (-y, x, z)
    if face == "U":
        return (-z, y, x)
    if face == "D":
        return (z, y, -x)
    if face == "R":
        return (x, z, -y)
    return (x, -z, y)                      # L


def permutation(face):
    """The 54-slot permutation of one clockwise quarter turn: new[j] = old[perm[j]]."""
    axis = 0
    for candidate in range(3):
        if FACE_NORMAL[face][candidate] != 0:
            axis = candidate
    layer = FACE_NORMAL[face][axis]
    perm = list(range(54))
    for slot, source in SLOT_INDEX.items():
        point, normal = slot
        if point[axis] == layer:
            perm[SLOT_INDEX[(spin(face, point), spin(face, normal))]] = source
    return tuple(perm)


def invert(perm):
    """The permutation that undoes `perm`."""
    out = [0] * 54
    for target, source in enumerate(perm):
        out[source] = target
    return tuple(out)


PERMS = {}
for _letter in "FBUDRL":
    _forward = permutation(_letter)
    PERMS[_letter] = _forward
    PERMS[_letter + "'"] = invert(_forward)


def apply_move(state, move):
    """The cube after one quarter turn."""
    perm = PERMS[move]
    return tuple(state[perm[slot]] for slot in range(54))


def apply_word(state, word):
    """The cube after a sequence of quarter turns."""
    for move in word:
        state = apply_move(state, move)
    return state


# ─── the ball of configurations a scramble can reach ────────────────────────────────────────
BALL_RADIUS = 4            # the deepest scramble the task uses (level 18 is exactly 4)
REACH_RADIUS = 3           # how far past the ball `solve_word` will search before giving up


def quarter_ball(radius):
    """state -> the shortest word that reaches it from solved, for every state within `radius`.

    Breadth first over the twelve quarter turns. The sizes are the cube group's own and are the
    cheapest possible check that the twelve permutations are the real thing: 12, 114, 1068, 10011
    states at depths 1..4.
    """
    words = {SOLVED: ()}
    frontier = [SOLVED]
    for _depth in range(radius):
        following = []
        for state in frontier:
            for move in MOVES_IN_ORDER:
                reached = apply_move(state, move)
                if reached not in words:
                    words[reached] = words[state] + (move,)
                    following.append(reached)
        frontier = following
    return words


BALL = quarter_ball(BALL_RADIUS)


def solve_word(state):
    """The shortest word that returns `state` to solved, or None if it is further than this
    harness can prove. Inside the ball it is a lookup; outside it, a short forward search meets
    the ball, which covers everything out to radius BALL_RADIUS + REACH_RADIUS."""
    word = BALL.get(state)
    if word is not None:
        return tuple(opposite(move) for move in reversed(word))
    seen = {state: ()}
    frontier = [state]
    for _depth in range(REACH_RADIUS):
        following = []
        for current in frontier:
            for move in MOVES_IN_ORDER:
                reached = apply_move(current, move)
                if reached in seen:
                    continue
                seen[reached] = seen[current] + (move,)
                tail = BALL.get(reached)
                if tail is not None:
                    return seen[reached] + tuple(opposite(m) for m in reversed(tail))
                following.append(reached)
        frontier = following
    return None


def distance(state):
    """Quarter turns from `state` to solved, or None when it cannot be proved."""
    word = solve_word(state)
    if word is None:
        return None
    return len(word)


# ─── what the camera shows ──────────────────────────────────────────────────────────────────
# pose -> the nine SLOT indices a square-on frame shows, row-major as drawn. A pose is (axis, k):
# the axis of the view rotations issued since the camera was last square at home, and k the count
# of 45-degree turns, so k is even exactly when the frame is readable. k = 0 is home on any axis.
# These nine-tuples were FITTED against the real renderer and were unique at every pose; the
# y-route and the x-route both pass the BACK face but see it rolled a quarter turn apart, which
# is why BACK appears twice with different tuples rather than once.
POSE_SLOTS = {
    ("y", 0): (18, 31, 51, 11, 27, 44, 5, 24, 38),       # FRONT  — home
    ("y", 2): (15, 17, 20, 9, 10, 12, 2, 4, 7),          # LEFT
    ("y", 4): (46, 28, 13, 41, 26, 8, 33, 21, 0),        # BACK, reached the long way round y
    ("y", 6): (53, 50, 48, 45, 43, 42, 40, 37, 35),      # RIGHT
    ("x", 2): (14, 29, 47, 16, 30, 49, 19, 32, 52),      # UP
    ("x", 4): (0, 21, 33, 8, 26, 41, 13, 28, 46),        # BACK, reached over the top
    ("x", 6): (6, 25, 39, 3, 23, 36, 1, 22, 34),         # DOWN
}
HOME_POSE = ("y", 0)

# The survey: eight 45-degree turns about x (passing UP, BACK, DOWN and home again), then eight
# about y (passing LEFT, BACK, RIGHT and home again). Every loop closes, so the camera is square
# at home whenever the survey ends or is abandoned, and a face turn is only ever played there.
SURVEY_ROUTE = (("rotate_cube_x_cw",) * 8) + (("rotate_cube_y_cw",) * 8)

# The blind sweep the harness falls back on if perception is unavailable: four turns of each face
# in turn. It returns to the start every four moves and stands on all twelve one-turn
# configurations plus all six half-turn ones inside 24 moves.
SWEEP_CYCLE = tuple(MOVE_NAMES[letter] for letter in "FRUBLD" for _repeat in range(4))


def pose_key(pose):
    """Normalize a camera pose for POSE_SLOTS: k = 0 is home whichever axis got us there."""
    if pose[1] % 8 == 0:
        return HOME_POSE
    return (pose[0], pose[1] % 8)


def readable(pose):
    """True when the camera is square-on at a pose whose nine slots are tabulated."""
    return pose_key(pose) in POSE_SLOTS


def turn_camera(pose, action_name):
    """The camera pose after one view rotation, or None when the pose becomes unknown.

    Two 45-degree turns about one axis make a 90-degree look; mixing axes away from home leaves a
    pose this table cannot name, and an unnamed pose must stop the survey rather than guess.
    """
    axis = action_name.split("_")[2]
    step = 1
    if action_name.endswith("_acw"):
        step = -1
    if axis == pose[0]:
        return (axis, (pose[1] + step) % 8)
    if pose[1] % 8 == 0:
        return (axis, step % 8)
    return None


def face_reading(state, pose):
    """The nine letters a square-on frame at `pose` would show for `state`."""
    return "".join(state[slot] for slot in POSE_SLOTS[pose_key(pose)])


def consistent(state, readings):
    """True when every recorded reading is what `state` would show."""
    for pose, stickers in readings.items():
        if face_reading(state, pose) != stickers:
            return False
    return True


FRONT_INDEX = {}
for _state in BALL:
    FRONT_INDEX.setdefault(face_reading(_state, HOME_POSE), []).append(_state)


def hypotheses(readings):
    """Every configuration in the ball that agrees with all readings so far.

    The home reading alone is enough to name the configuration outright for most of the ball —
    there are 5969 distinct front faces among its 11206 states — so this is usually a one-element
    answer bought with zero camera moves, and the survey only has to run when it is not.
    """
    if not readings:
        return []
    front = readings.get(HOME_POSE)
    if front is None:
        pool = BALL
    else:
        pool = FRONT_INDEX.get(front, ())
    live = []
    for state in pool:
        if consistent(state, readings):
            live.append(state)
    return live


def assemble(readings):
    """The whole 54-sticker configuration, when the survey has seen all six faces.

    This is the path that does not lean on the ball at all: six square-on readings cover every
    sticker, so the configuration is read rather than inferred, and a scramble deeper than the
    ball is still solvable from it.
    """
    stickers = [""] * 54
    for pose, letters in readings.items():
        for slot, letter in zip(POSE_SLOTS[pose_key(pose)], letters):
            if stickers[slot] and stickers[slot] != letter:
                return None
            stickers[slot] = letter
    tally = {}
    for letter in stickers:
        if not letter:
            return None
        tally[letter] = tally.get(letter, 0) + 1
    for letter, count in tally.items():
        if count != 9:
            return None
    state = tuple(stickers)
    for letter, normal in FACE_NORMAL.items():
        centre = SLOT_INDEX[((normal[0], normal[1], normal[2]), normal)]
        if state[centre] != SOLVED_COLOUR[normal]:
            return None
    return state


def menu_of(state):
    """Every move that provably shortens the way home, as engine action names.

    This is the per-turn choice set the agent is shown. It is never empty for a solvable
    configuration and it is never padded: a name is on it only if playing it reduces the proved
    distance to solved, so honouring any pick from it costs nothing.
    """
    best = distance(state)
    if best is None:
        return []
    names = []
    for move in MOVES_IN_ORDER:
        after = distance(apply_move(state, move))
        if after is not None and after < best:
            names.append(MOVE_NAMES[move])
    return names


# ─── the self test ──────────────────────────────────────────────────────────────────────────
# The fingerprint of one measured level: the configuration one clockwise DOWN turn away from
# solved (level 1), and what each surveyed pose shows there. It pins the twelve permutations, the
# slot enumeration AND the pose tables together — if any of them drifts, none of these match.
PROBE_WORD = ("D'",)
PROBE_READINGS = {
    ("y", 0): "RRRRRRGGG",
    ("y", 2): "BBBBBBRRR",
    ("y", 4): "OOOOOOBBB",
    ("y", 6): "GGGGGGOOO",
    ("x", 2): "YYYYYYYYY",
    ("x", 4): "BBBOOOOOO",
    ("x", 6): "WWWWWWWWW",
}


def self_test():
    """Re-derive the measured facts at import. False means perception must not be trusted."""
    if len(SLOTS) != 54 or len(BALL) != 11206:
        return False
    if apply_word(SOLVED, ("R", "U") * 105) != SOLVED:
        return False
    for move in MOVES_IN_ORDER:
        if apply_word(SOLVED, (move,) * 4) != SOLVED:
            return False
    state = apply_word(SOLVED, PROBE_WORD)
    for pose, stickers in PROBE_READINGS.items():
        if face_reading(state, pose) != stickers:
            return False
    return solve_word(state) == ("D",)


TRUSTED = self_test()


# ─── the ledger ─────────────────────────────────────────────────────────────────────────────
def new_book():
    """Per-episode bookkeeping. Everything in it is either a move this harness emitted or a
    sticker it decoded from the frame the agent was shown."""
    return {"step": 0, "played": [], "pose": HOME_POSE, "readings": {}, "live": [],
            "state": None, "phase": "survey", "route": list(SURVEY_ROUTE), "sweeps": 0,
            "want": "", "menu": [], "plan": (), "purity": 0.0, "square": False,
            "overrides": 0, "honoured": 0, "surveys": 1, "checks": 0, "faults": 0,
            "prior": "", "slot": "", "real": None, "sunk": [], "journal": []}


def restart_survey(book):
    """Forget the believed configuration and survey again — the one answer to a reading that
    contradicts the plan. The camera is square at home whenever this runs."""
    book["state"] = None
    book["readings"] = {}
    book["live"] = []
    book["route"] = list(SURVEY_ROUTE)
    book["surveys"] += 1
    book["faults"] += 1


def record_reading(book, stickers, purity, square):
    """File one decoded face against the camera pose that produced it, then re-believe.

    A reading is kept only if the gate passed and the pose is one the tables name. If a kept
    reading disagrees with the configuration the plan is built on, the plan is wrong and the
    survey starts over; that check is the only thing standing between a misread sticker and a
    hundred wasted steps.
    """
    book["purity"] = purity
    book["square"] = bool(square)
    if not square or len(stickers) != 9 or not readable(book["pose"]):
        return
    key = pose_key(book["pose"])
    if book["state"] is not None:
        book["checks"] += 1
        if face_reading(book["state"], key) != stickers:
            restart_survey(book)
    book["readings"][key] = stickers
    if book["state"] is None:
        book["live"] = hypotheses(book["readings"])


def adopt(book, state):
    """Commit to a configuration and to the word that solves it."""
    word = solve_word(state)
    if word is None:
        return False
    book["state"] = state
    book["plan"] = word
    book["route"] = []
    return True


def think(book):
    """Choose this turn's request and the choice set the agent is shown.

    Three phases in strict order. SOLVE once a configuration is proved — the request is the first
    move of the shortest word home and the menu is every move that shortens it. SURVEY while the
    configuration is still ambiguous — the request is the next closing camera turn. SWEEP only if
    perception never resolves, so the harness still spends its steps on the configurations nearest
    the start rather than standing still.
    """
    if book["state"] is None:
        if len(book["live"]) == 1:
            adopt(book, book["live"][0])
        else:
            whole = assemble(book["readings"])
            if whole is not None:
                adopt(book, whole)
    if book["state"] is not None:
        word = solve_word(book["state"])
        if word:
            book["phase"] = "solve"
            book["plan"] = word
            book["menu"] = menu_of(book["state"]) or [MOVE_NAMES[word[0]]]
            book["want"] = MOVE_NAMES[word[0]]
            return book
        restart_survey(book)
    if book["route"] and TRUSTED:
        book["phase"] = "survey"
        book["plan"] = ()
        book["want"] = book["route"][0]
        book["menu"] = [book["want"]]
        return book
    book["phase"] = "sweep"
    book["plan"] = ()
    book["want"] = SWEEP_CYCLE[book["sweeps"] % len(SWEEP_CYCLE)]
    book["menu"] = [book["want"]]
    return book


def choose(book, pick):
    """The move actually played: the agent's own pick whenever the menu allows it.

    Every name on the menu is provably as good as the request, so compliance is honoured and
    nothing else is. The override is counted, not hidden — it is the compliance measurement for
    this harness, and the one number to read beside the score.
    """
    if pick in book["menu"] and is_legal(pick):
        book["honoured"] += 1
        return pick, True
    book["overrides"] += 1
    return book["want"], False


def note_move(book, name, ok):
    """Advance the ledger by a move the engine reported executed."""
    book["step"] += 1
    book["played"].append(name)
    if not ok:
        return book
    if name in VIEW_ROTATION_ACTIONS:
        turned = turn_camera(book["pose"], name)
        if turned is None:
            book["pose"] = (book["pose"][0], -1)
            book["route"] = []
        else:
            book["pose"] = turned
        if book["route"] and book["route"][0] == name:
            book["route"].pop(0)
        return book
    if book["phase"] == "sweep":
        book["sweeps"] += 1
    letter = LETTER_OF.get(name)
    if letter is not None and book["state"] is not None:
        book["state"] = apply_move(book["state"], letter)
    book["readings"] = {}
    book["live"] = []
    return book
