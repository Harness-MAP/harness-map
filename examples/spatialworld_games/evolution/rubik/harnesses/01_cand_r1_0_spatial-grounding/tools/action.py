"""The Rubik's-Cube action algebra AND the DECISION TABLE this harness plays from.

The engine exposes two families of move, executed by `rubik_cube/rubik_cube_adapter.py`
(`rotate_face(face, direction)` / `rotate_cube(axis, direction)`) and dispatched by
`rubik_openai_evaluation.py::execute_action`:

  * FACE ROTATIONS (90 degrees, animated, CHANGE the cube) — one per face x {cw, acw}, i.e.
    rotate_<front|back|left|right|up|down>_<cw|acw>. Adapter face keys f/b/l/r/u/d.
  * VIEW ROTATIONS (45 degrees, immediate, DO NOT change the cube) — rotate the whole cube about
    an axis x {cw, acw}. Adapter axis keys map x->RIGHT, y->UP, z->FRONT.

HOW THIS FILE IS SHAPED. The three public helpers below stay one-liners; every branch, table and
threshold of the policy lives in the module-level block after them. That is deliberate and it is
the axis of this candidate: a RULE is data (a code, a sentence the agent is shown, and a predicate),
not a nest of `if`s inside a function. Adding, removing or re-ordering a rule is editing one tuple,
and `tools/vlm.py` imports the table by name — which keeps the component graph exactly as the
anchor left it while the content of the components moves.

NOTHING HERE READS CUBE INTERNALS. Every predicate is a statement about the harness's OWN emitted
action word; `seen`/`word` are derived from the moves this harness itself played.
"""
import functools
import itertools

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


# ===========================================================================================
# THE DECISION TABLE.  Everything below is module-level data and module-level combinators.
# ===========================================================================================

_FACE_NAME = {"f": "FRONT", "b": "BACK", "l": "LEFT", "r": "RIGHT", "u": "UP", "d": "DOWN"}
_AXIS_NAME = {"x": "RIGHT", "y": "UP", "z": "FRONT"}     # the axis face each view turn names
_TURN = {"cw": 1, "acw": -1}                             # signed quarter turns

# name -> (engine face word, engine direction word, engine key, engine action_type).
# `execute_action` dispatches on metadata["action_type"] / ["face"] / ["direction"] and NOT on the
# action name, and the shipped parser fills those from the model's own JSON fields — so a harness
# that rewrites an Action must set all four together or the engine and the trace silently diverge.
SPEC = {"rotate_%s_%s" % (_FACE_NAME[k].lower(), s):
            (_FACE_NAME[k], DIRECTIONS[s.upper()], k, "face_rotation")
        for k in sorted(_FACE_NAME) for s in ("cw", "acw")}
SPEC.update({"rotate_cube_%s_%s" % (a, s):
                 (_AXIS_NAME[a], DIRECTIONS[s.upper()], a, "view_rotation")
             for a in sorted(_AXIS_NAME) for s in ("cw", "acw")})

# The camera's own viewing axis. A z view turn re-draws the same face set at every one of its eight
# poses, so it cannot reveal a facelet the frame is not already showing.
BLIND_VIEWS = frozenset(n for n in sorted(VIEW_ROTATION_ACTIONS) if "_z_" in n)

# A face turn as a letter of a free word: (adapter face key, signed quarter turns).
MOVE_OF = {n: (SPEC[n][2], _TURN[n.rsplit("_", 1)[1]]) for n in sorted(FACE_ROTATION_ACTIONS)}

# Quarter turns on ONE face add mod 4, and that is an EXACT identity of the cube group: a residue
# of 0 is the identity, 3 clockwise turns ARE one anticlockwise turn. Reducing the played word with
# only these two relations therefore never claims two prefixes are equal unless the cube really is
# in the same configuration — the refusal built on it has no false positives.
_RESIDUE = {0: (), 1: (1,), 2: (2,), 3: (-1,)}
_PUSH = lambda stack, mv: (
    stack[:-1] + tuple((mv[0], q) for q in _RESIDUE[(stack[-1][1] + mv[1]) % 4])
    if stack and stack[-1][0] == mv[0] else stack + (mv,))
reduce_word = lambda moves: functools.reduce(_PUSH, moves, ())
word_after = lambda book, name: reduce_word(book["word"] + (MOVE_OF[name],))

_MARK = {1: "", 2: "2", -1: "'"}
render_word = lambda word: " ".join(m[0].upper() + _MARK[m[1]] for m in word)
_view_run = lambda book: sum(1 for _ in itertools.takewhile(
    lambda a: a in VIEW_ROTATION_ACTIONS, reversed(book["played"])))

# (code, the sentence the agent is shown, predicate(action_name, book)). Order is the order the
# brief lists them in; a rule fires on the name the model proposed, never on the cube.
RULES = (
    ("blind-axis",
     "a z-axis view turn re-draws the face the camera already shows, so it cannot reveal anything",
     lambda n, b: n in BLIND_VIEWS),
    ("view-streak",
     "two 45-degree view turns in a row already complete a 90-degree look; a third only spins",
     lambda n, b: n in VIEW_ROTATION_ACTIONS and _view_run(b) >= 2),
    ("revisit",
     "this turn provably returns the cube to a configuration already stood on in this episode",
     lambda n, b: n in FACE_ROTATION_ACTIONS and word_after(b, n) in b["seen"]),
)

refusals = lambda name, book: tuple((c, why) for c, why, p in RULES if p(name, book))
honoured = lambda name, book: name in LEGAL_ACTIONS and not refusals(name, book)

# The substitution order. Face turns first, rotated by the step count so a refused pick is not
# answered with the same constant every time, then the four view turns that can reveal a new face.
# Every table here is built over `sorted(...)`: a comprehension over a SET takes its order from the
# process hash seed, which would make the substitution order differ between runs of one episode.
_FACE_RING = tuple(sorted(FACE_ROTATION_ACTIONS))
_VIEW_RING = tuple(n for n in sorted(VIEW_ROTATION_ACTIONS) if n not in BLIND_VIEWS)
_offset = lambda book: len(book["played"]) % len(_FACE_RING)
preference = lambda book: (_FACE_RING[_offset(book):] + _FACE_RING[:_offset(book)] + _VIEW_RING)
# Never empty: once every legal turn is refused the harness must still play one, and the first
# preference is what it plays. That is the only point at which the no-revisit promise lapses.
survivors = lambda book: (tuple(n for n in preference(book) if honoured(n, book))
                          or preference(book)[:1])

new_book = lambda: {"played": [], "word": (), "seen": {()}, "frames": [], "still": False,
                    "step": 0, "prior": "", "slot": "", "real": None, "sunk": []}
note_move = lambda book, name: (
    book["played"].append(name),
    book.__setitem__("word", word_after(book, name) if name in MOVE_OF else book["word"]),
    book["seen"].add(book["word"]),
    book.__setitem__("step", book["step"] + 1),
)
