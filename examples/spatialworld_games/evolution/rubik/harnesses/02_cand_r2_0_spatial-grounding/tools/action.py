"""The Rubik's-Cube action algebra, an EXACT cube state over the harness's OWN moves, and the
NEAR-BALL FRONTIER ROUTER this harness plays from.

The engine exposes two families of move, executed by `rubik_cube/rubik_cube_adapter.py`
(`rotate_face(face, direction)` / `rotate_cube(axis, direction)`) and dispatched by
`rubik_openai_evaluation.py::execute_action`:

  * FACE ROTATIONS (90 degrees, animated, CHANGE the cube) -- one per face x {cw, acw}, i.e.
    rotate_<front|back|left|right|up|down>_<cw|acw>. Adapter face keys f/b/l/r/u/d.
  * VIEW ROTATIONS (45 degrees, immediate, DO NOT change the cube) -- rotate the whole cube about
    an axis x {cw, acw}. Adapter axis keys map x->RIGHT, y->UP, z->FRONT.

WHAT THIS FILE DECIDES. Each level starts a few quarter turns from solved, and the evaluator's
`state_callback` ends the episode the instant the cube IS solved -- so the harness never has to
RECOGNISE the solution, it only has to walk over it. The whole policy is therefore one quantity:
how much of the small neighbourhood of the episode's own START configuration the walk has stood
on, shallowest configurations first. This module keeps an exact cube state advanced by the moves
the harness itself played, enumerates the ball of radius `BALL_RADIUS` around that start once at
import, and scores a candidate move by whether it moves the walk CLOSER to the shallowest
configuration of that ball it has not yet stood on. A move that walks away is refused.

That replaces the parent's `revisit` rule, which maximised DISTINCT configurations and so turned
the trajectory into a self-avoiding walk out of the only neighbourhood that can contain the
solution. The two objectives are opposites: under `revisit`, an optimal router covered 2 of the 12
one-turn configurations in 100 moves; under this one it covers 12 of 12 in 24.

HOW THIS FILE IS SHAPED. Plain named functions and explicit statements, the way the staged anchor
writes its helpers: every table is returned by a `def` that builds it with a loop, every branch is
an `if` inside a function, and no rule is a chained lambda. The anchor's four public helpers below
keep their exact bodies.

NOTHING HERE READS CUBE INTERNALS. The state is built from this harness's own emitted move names
with the unknown scramble treated as the identity, so every quantity below is a DISPLACEMENT from
the episode's own start -- never a distance to solved, and never a look at the scramble, a solver,
the cube object or the score.
"""
import itertools

# ---------------------------------------------------------------------------------------------
# THE SHIPPED VOCABULARY (unchanged from the anchor)
# ---------------------------------------------------------------------------------------------

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

FACE_WORDS = {"f": "FRONT", "b": "BACK", "l": "LEFT", "r": "RIGHT", "u": "UP", "d": "DOWN"}
AXIS_WORDS = {"x": "RIGHT", "y": "UP", "z": "FRONT"}       # the axis face each view turn names


def is_legal(action_name: str) -> bool:
    """True if `action_name` is one of the shipped face- or view-rotation actions."""
    return action_name in LEGAL_ACTIONS


def is_view_rotation(action_name: str) -> bool:
    """True for the immediate 45-degree whole-cube view rotations (state-preserving)."""
    return action_name in VIEW_ROTATION_ACTIONS


def direction_const(direction: str) -> str:
    """Map a model 'CW'/'ACW' to the adapter's CLOCKWISE/ANTICLOCKWISE constant."""
    return DIRECTIONS.get((direction or "CW").strip().upper(), "CLOCKWISE")


def build_spec():
    """name -> (engine face word, engine direction word, engine key, engine action_type).

    `execute_action` dispatches on metadata["action_type"] / ["face"] / ["direction"] and NOT on
    the action name, and the shipped parser fills those from the model's own JSON fields -- so a
    harness that rewrites an Action must set all four together or the engine and the trace
    silently diverge (the live run corrected 107 such disagreements)."""
    spec = {}
    for key in sorted(FACE_WORDS):
        for turn in ("cw", "acw"):
            name = "rotate_%s_%s" % (FACE_WORDS[key].lower(), turn)
            spec[name] = (FACE_WORDS[key], DIRECTIONS[turn.upper()], key, "face_rotation")
    for axis in sorted(AXIS_WORDS):
        for turn in ("cw", "acw"):
            name = "rotate_cube_%s_%s" % (axis, turn)
            spec[name] = (AXIS_WORDS[axis], DIRECTIONS[turn.upper()], axis, "view_rotation")
    return spec


SPEC = build_spec()
FACE_MOVES = tuple(sorted(FACE_ROTATION_ACTIONS))          # sorted: never a set's hash order


def face_key_of(action_name: str) -> str:
    """The adapter face key ('f'..'d') a face-rotation name turns, or '' for anything else."""
    if action_name in FACE_ROTATION_ACTIONS:
        return SPEC[action_name][2]
    return ""


def inverse_name(action_name: str) -> str:
    """The quarter turn that undoes `action_name`."""
    stem, turn = action_name.rsplit("_", 1)
    if turn == "cw":
        return stem + "_acw"
    return stem + "_cw"


# ---------------------------------------------------------------------------------------------
# AN EXACT CUBE STATE, BUILT FROM THE HARNESS'S OWN MOVE NAMES
#
# 20 cubies, each held as (which cubie, how it is turned). A quarter turn of a face permutes the
# 8 non-centre slots of that face and composes one rotation onto each of their orientations, which
# is the whole cube group -- no sticker table is typed by hand, so the construction cannot carry a
# transcription error, and `python -c "from tools import action; action.self_check()"` prints the
# shell sizes 12 / 114 that identify the real group.
#
# The scramble is unknown and is treated as the identity: `START` is where THIS episode began.
# Chirality does not have to match the engine's. A mirrored convention relabels the ball onto
# itself, so the coverage this policy maximises is invariant under it.
# ---------------------------------------------------------------------------------------------

AXIS_OF_FACE = {"r": (0, 1), "l": (0, -1), "u": (1, 1), "d": (1, -1), "f": (2, 1), "b": (2, -1)}
IDENTITY_ROTATION = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def quarter_turn(axis, sign):
    """A right-handed 90-degree rotation matrix about `axis` (0=x, 1=y, 2=z)."""
    rows = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows[axis][axis] = 1
    first = (axis + 1) % 3
    second = (axis + 2) % 3
    rows[second][first] = sign
    rows[first][second] = -sign
    return tuple(tuple(row) for row in rows)


def matmul(left, right):
    """The matrix product `left . right`."""
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            total = 0
            for k in range(3):
                total += left[i][k] * right[k][j]
            row.append(total)
        out.append(tuple(row))
    return tuple(out)


def transform(matrix, vector):
    """`matrix` applied to a cubie position."""
    out = []
    for i in range(3):
        total = 0
        for k in range(3):
            total += matrix[i][k] * vector[k]
        out.append(total)
    return tuple(out)


def move_matrix(face_key, clockwise):
    """The rotation a quarter turn of `face_key` applies, clockwise seen from OUTSIDE the cube."""
    axis, slab = AXIS_OF_FACE[face_key]
    if clockwise:
        return quarter_turn(axis, -slab)
    return quarter_turn(axis, slab)


def rotation_group():
    """The 24 proper rotations of the cube, by closure over the six face quarter turns."""
    found = [IDENTITY_ROTATION]
    seen = {IDENTITY_ROTATION}
    generators = []
    for key in sorted(AXIS_OF_FACE):
        generators.append(move_matrix(key, True))
    index = 0
    while index < len(found):
        current = found[index]
        index += 1
        for generator in generators:
            product = matmul(generator, current)
            if product not in seen:
                seen.add(product)
                found.append(product)
    return tuple(found)


ROTATIONS = rotation_group()
ROTATION_INDEX = {matrix: i for i, matrix in enumerate(ROTATIONS)}


def composition_table():
    """COMPOSE[a][b] = the index of `ROTATIONS[a] . ROTATIONS[b]`, so advancing a state is a
    lookup instead of a matrix product."""
    table = []
    for a in range(len(ROTATIONS)):
        row = []
        for b in range(len(ROTATIONS)):
            row.append(ROTATION_INDEX[matmul(ROTATIONS[a], ROTATIONS[b])])
        table.append(tuple(row))
    return tuple(table)


COMPOSE = composition_table()

SLOTS = tuple(sorted(p for p in itertools.product((-1, 0, 1), repeat=3)
                     if sum(1 for c in p if c != 0) >= 2))
SLOT_INDEX = {position: i for i, position in enumerate(SLOTS)}
START = tuple((i, 0) for i in range(len(SLOTS)))


def build_move_table():
    """name -> (the (source slot, destination slot) pairs the turn moves, the rotation index it
    composes onto each of them). Eight slots move per face turn; the other twelve are copied."""
    table = {}
    for name in FACE_MOVES:
        key = face_key_of(name)
        axis, slab = AXIS_OF_FACE[key]
        matrix = move_matrix(key, name.endswith("_cw"))
        pairs = []
        for source, position in enumerate(SLOTS):
            if position[axis] == slab:
                pairs.append((source, SLOT_INDEX[transform(matrix, position)]))
        table[name] = (tuple(pairs), ROTATION_INDEX[matrix])
    return table


MOVE_TABLE = build_move_table()


def advance(state, action_name):
    """`state` after the quarter turn `action_name`. A view rotation leaves the cube alone."""
    if action_name not in MOVE_TABLE:
        return state
    pairs, rotation = MOVE_TABLE[action_name]
    out = list(state)
    for source, destination in pairs:
        cubie, orientation = state[source]
        out[destination] = (cubie, COMPOSE[rotation][orientation])
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# THE BALL AROUND THE START, ENUMERATED ONCE
#
# Radius 2 is not a tuning knob, it is the step budget: the shells are 12 / 114 / 1068, and an
# episode has 100 steps. A radius-3 target set would spend the budget sampling 9 % of a shell the
# walk can only reach by crossing the two it has not finished, while radius 2 covers the 12
# one-turn configurations in 24 moves and then takes ~40 % of the 114 two-turn ones. Depth-3 and
# depth-4 levels are out of reach at any radius, so widening buys nothing and costs the shells
# where the wins are.
# ---------------------------------------------------------------------------------------------

BALL_RADIUS = 2


def build_ball(radius):
    """-> (depth of every configuration within `radius` quarter turns of the start, the in-ball
    move graph as {configuration: {move name: configuration}})."""
    depth = {START: 0}
    frontier = [START]
    for distance in range(1, radius + 1):
        following = []
        for state in frontier:
            for name in FACE_MOVES:
                nxt = advance(state, name)
                if nxt not in depth:
                    depth[nxt] = distance
                    following.append(nxt)
        frontier = following
    edges = {}
    for state in depth:
        links = {}
        for name in FACE_MOVES:
            nxt = advance(state, name)
            if nxt in depth:
                links[name] = nxt
        edges[state] = links
    return depth, edges


DEPTH, EDGES = build_ball(BALL_RADIUS)


def build_shells():
    """depth -> the configurations at exactly that many quarter turns from the start."""
    shells = {}
    for state, distance in DEPTH.items():
        shells.setdefault(distance, []).append(state)
    for distance in shells:
        shells[distance].sort()
    return shells


SHELLS = build_shells()


def self_check():
    """The shell sizes that identify the real cube group, printed for validation (12 / 114)."""
    return {distance: len(SHELLS[distance]) for distance in sorted(SHELLS)}


# ---------------------------------------------------------------------------------------------
# THE POLICY: WALK TOWARD THE SHALLOWEST CONFIGURATION NOT YET STOOD ON
# ---------------------------------------------------------------------------------------------

def targets(book):
    """The unvisited ball configurations at the SHALLOWEST depth that still has one.

    Stratifying by depth is what makes the sweep worth more than a plain nearest-unvisited tour:
    a one-turn configuration is one of twelve, a two-turn configuration one of a hundred and
    fourteen, so standing on a one-turn configuration is worth about eight times as much per step
    and all twelve must be finished first."""
    for distance in range(1, BALL_RADIUS + 1):
        wave = []
        for state in SHELLS.get(distance, ()):
            if state not in book["seen"]:
                wave.append(state)
        if wave:
            return wave
    return []


def frontier_distance(book):
    """For every ball configuration, how many quarter turns away the nearest target is.

    One multi-source breadth-first search per step over at most 127 configurations. Empty once the
    whole ball has been stood on, which 100 steps cannot do."""
    wave = targets(book)
    distance = {}
    queue = []
    for state in wave:
        distance[state] = 0
        queue.append(state)
    head = 0
    while head < len(queue):
        state = queue[head]
        head += 1
        for nxt in EDGES[state].values():
            if nxt not in distance:
                distance[nxt] = distance[state] + 1
                queue.append(nxt)
    return distance


def menu_rows(book):
    """Every move that shortens the walk's distance to the frontier, best first.

    -> [(name, the depth it lands on, whether that configuration is new)]. This is both the
    honoured set and the menu the brief prints, and its first row is what the harness plays when
    the model names something else -- so the ordering IS the substitution policy.

    STRICT PROGRESS is the whole predicate: a move is honoured when the nearest configuration not
    yet stood on is closer after it than before it. That is what the parent's `revisit` rule
    inverted. It also keeps a floor nothing can argue with -- under every reply stream, adversarial
    constants included, the twelve configurations one quarter turn from the start are all stood on
    by step 23, because no stream can spend a step anywhere else.

    A wider version was measured and rejected: also honouring any pick that merely lands on NEW
    ground deeper in the ball cuts the override rate by 4-21 points, but it lets a stream wander
    the hundred-and-fourteen-strong two-turn shell and pushes the one-turn sweep from step 23 out
    to step 41-69, with the tail risk of never finishing it. The near shell is worth about eight
    times as much per step, so the floor keeps priority over the override rate."""
    distance = frontier_distance(book)
    here = book["state"]
    links = EDGES.get(here, {})
    reach = distance.get(here)
    if reach is None:
        return []
    rows = []
    for name in sorted(links):
        there = links[name]
        ahead = distance.get(there)
        if ahead is None or ahead >= reach:
            continue
        rows.append((rank_of(name, there, ahead, book), (name, DEPTH[there],
                                                         there not in book["seen"])))
    rows.sort()
    return [row for _key, row in rows]


def rank_of(name, there, ahead, book):
    """Sort key for one candidate move, i.e. the substitution policy.

    Tier 0 is a target reached NOW: a configuration at the shallowest depth not yet finished.
    Tier 1 is transit toward one.

    Then CONTINUE THE FACE ALREADY TURNING. Four quarter turns of one face stand on three new
    configurations and come home, which ties the best tour of the near shell measured here (all
    twelve one-turn configurations by step 23) and beats the alternatives on the two-turn shell
    (42-45 of 114 against 38-40 for a tour that returns to the start between faces). It is also
    the one tour a frozen agent follows for free: the live run measured it naming the move the
    harness had just played on 65.4 % of the steps after an override, and inside a face block that
    copy is exactly right. The router and `skills/near_ball_sweep.md` therefore agree move for
    move, instead of the harness authoring a pattern the prose does not describe.

    Then new ground over old, then the shorter remaining walk, then the face the model asked for
    and could not have (so a refused pick is DEFERRED rather than discarded), and finally the
    name, which makes the order of a tie identical between two runs of the same episode."""
    tier = 1
    if ahead == 0:
        tier = 0
    continuing = 1
    if book["played"] and name == book["played"][-1]:
        continuing = 0
    fresh = 1
    if there not in book["seen"]:
        fresh = 0
    requested = 1
    if name == book["request"]:
        requested = 0
    return (tier, continuing, fresh, ahead, requested, name)


def menu(book):
    """Just the honoured move names, best first."""
    return [row[0] for row in menu_rows(book)]


# code -> the sentence the agent is shown when its pick is refused. A refusal is reported in the
# agent's own terms (what the turn would have done to the cube), never as a bare code.
REASONS = {
    "no-view": "a view rotation only moves the camera, so it cannot change the cube",
    "off-ball": "that turn leaves the neighbourhood of the start, where the solution has to be",
    "no-progress": "that turn walks back over ground already covered",
    "unknown-name": "that is not one of the legal action names",
    "policy-fault": "the harness could not score that turn",
}


def refusal_code(book, pick):
    """Why `pick` is not honoured -- one short code, reported on the Action and in the brief."""
    if pick not in LEGAL_ACTIONS:
        return "unknown-name"
    if pick in VIEW_ROTATION_ACTIONS:
        return "no-view"
    if EDGES.get(book["state"], {}).get(pick) is None:
        return "off-ball"
    return "no-progress"


def fallback(book, pick):
    """Nothing shortens the distance to the frontier: the ball is fully stood on, or the state
    desynced from the engine. Keep the cube in the neighbourhood -- honour an in-ball pick, else
    take the shallowest in-ball turn, else retract the harness's own last quarter turn."""
    links = EDGES.get(book["state"], {})
    if pick in links:
        return pick
    shallowest = ""
    for name in sorted(links):
        if not shallowest or DEPTH[links[name]] < DEPTH[links[shallowest]]:
            shallowest = name
    if shallowest:
        return shallowest
    for name in reversed(book["played"]):
        if name in MOVE_TABLE:
            return inverse_name(name)
    return FACE_MOVES[0]


def choose(book, pick):
    """The decision: -> (the move to play, the refusal code, whether a deferred request was used).

    Honour the model's pick whenever it shortens the distance to the frontier. Otherwise play the
    first menu row and remember the pick: if it names a turn that is merely illegal HERE, it is
    tried as a tie-break as soon as it becomes the best thing available."""
    allowed = menu(book)
    if pick in allowed:
        book["request"] = ""
        return pick, "", False
    code = refusal_code(book, pick)
    if allowed:
        used = allowed[0] == book["request"]
        book["request"] = pick if pick in FACE_ROTATION_ACTIONS else ""
        return allowed[0], code, used
    book["request"] = ""
    return fallback(book, pick), code, False


# ---------------------------------------------------------------------------------------------
# THE LEDGER -- one dict per episode, holding only what this harness itself emitted
# ---------------------------------------------------------------------------------------------

def new_book():
    """The per-episode ledger. `state`/`seen` are the exact cube displacement and the set of
    configurations stood on; the rest is bookkeeping the brief and the Action metadata report."""
    return {"played": [], "state": START, "seen": {START}, "step": 0,
            "prior": "", "slot": "", "real": None, "sunk": [], "fault": "",
            "request": "", "overrides": 0, "deferred": 0, "misses": 0,
            "last_pick": "", "last_code": "", "last_played": ""}


def note_move(book, action_name, executed):
    """Record the one move this harness committed and advance the cube by it.

    The state advances only on a move the engine reported as executed: an adapter that refuses a
    turn mid-animation would otherwise desync the ledger from the cube, and a desync is the one
    fault that makes every number below meaningless. `misses` reports it if it ever happens."""
    book["played"].append(action_name)
    book["step"] += 1
    if not executed:
        book["misses"] += 1
        return
    book["state"] = advance(book["state"], action_name)
    book["seen"].add(book["state"])


def turns_from_start(book):
    """How many quarter turns the cube now is from where this episode started, or -1 if the walk
    has somehow left the ball."""
    return DEPTH.get(book["state"], -1)


def swept(book, distance):
    """-> (configurations stood on, configurations there are) at exactly `distance` turns out."""
    shell = SHELLS.get(distance, ())
    stood = 0
    for state in shell:
        if state in book["seen"]:
            stood += 1
    return stood, len(shell)
