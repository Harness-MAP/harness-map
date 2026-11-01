"""Perceive -> decide, as a warrant the frozen model is given a chance to satisfy twice.

The pristine evaluator still owns capture, the model request, JSON parsing, logging and engine
``Action`` construction; this module owns only what happens around one of its calls. Three readings
meet here, and all three are public:

* RECKONING. The head cell, the heading and the trail of cells left behind are integrated from the
  engine's published opening position and the keys THIS module has pressed. Nothing else; the
  engine's own head is never consulted.
* THE HUD. The ``Score: N`` glyph the renderer blits into the top-left of the very screenshot the
  model is shown decodes to the number of pellets eaten, hence to the exact body length. The
  harness therefore stops guessing how long the snake is with a fixed window.
* THE PELLET. The one warm blob on the board snaps to the cube cell whose drawn diameter matches
  the visible fragment. The gate is ONE-SIDED on purpose: when the snake is adjacent to the pellet
  its own disc covers part of it, so the fragment is routinely SMALLER than the cell's disc and a
  symmetric size test snaps to a farther cell exactly on the frames that decide whether it eats.

From those three a warrant follows: the moves the engine would actually execute, and which of them
shorten the lattice gap to the pellet. An answer that satisfies the warrant is pressed untouched.
An answer that does not is NOT silently overwritten -- the unusable turn is rolled out of
model-visible history, the warrant is spliced into the live per-turn text channel, and the frozen
model answers the SAME screenshot once more. Only if that second answer also fails does the
harness press its own key. The appeal is free: the episode budget counts calls to ``decide``, not
calls inside one.

Numeric tables, pixel windows and the lattice algebra live at module scope beside the constants
they read, so each component below is the decision it names and little else.
"""
from copy import deepcopy

import numpy as np

from tools.action import DIRECTION_VECTORS, KEY_MAPPING, is_reversal

# ---------------------------------------------------------------- the cube and its algebra
SIDE = 4
RANK = 3
ZERO = 0
EMPTY = ""
SPAN = range(SIDE)
CELLS = tuple((i, j, k) for i in SPAN for j in SPAN for k in SPAN)
MENU = ("move_right", "move_left", "move_up", "move_down", "move_forward", "move_backward")
# The engine's published opening: head, the neck behind it, and the +X heading already set.
OPENING_HEAD = (2, 3, 2)
OPENING_NECK = (1, 3, 2)
OPENING_HEADING = (1, 0, 0)
START_LEN = 2
TRAIL_CAP = 32

_STEP = lambda cell, name: tuple(a + b for a, b in zip(cell, DIRECTION_VECTORS[name]))
_INSIDE = lambda cell: all(c in SPAN for c in cell)
_L1 = lambda a, b: sum(abs(p - q) for p, q in zip(a, b))
_PUSH = lambda trail, cell: trail.insert(ZERO, cell)
# Which single name the engine would refuse from each heading -- stated once, from the engine's own
# refusal test in tools/action.py, so no second copy of the rule exists.
BARRED_BY = {DIRECTION_VECTORS[a]: b for a in MENU for b in MENU
             if is_reversal(b, DIRECTION_VECTORS[a])}
# The body test is against every cell but the tail: the engine vacates the tail on the same step.
_OCCUPIED = lambda trail, length: frozenset(trail[:max(length, START_LEN)][:-1])
_OPEN = lambda head, barred, blocked: tuple(
    n for n in MENU
    if n != barred and _INSIDE(_STEP(head, n)) and _STEP(head, n) not in blocked)
_ROOM = lambda cell, blocked: sum(
    n in MENU for n in MENU if _INSIDE(_STEP(cell, n)) and _STEP(cell, n) not in blocked)
# Rank: close the gap first, then keep the most continuations, then prefer the least-trodden cell,
# then menu order, so the ranking is total and deterministic.
_WARRANT = lambda led, blocked, pellet, n: (
    _L1(_STEP(led["head"], n), pellet) if pellet is not None else ZERO,
    -_ROOM(_STEP(led["head"], n), blocked),
    led["seen"].get(_STEP(led["head"], n), ZERO),
    MENU.index(n))
_CLOSERS = lambda led, pellet, names: tuple(
    n for n in names
    if pellet is not None and _L1(_STEP(led["head"], n), pellet) < _L1(led["head"], pellet))
_TOP = lambda names, rank: min(names, key=rank) if names else None
# A HUD reading is accepted only when it does not fall: the engine's counter never goes backwards,
# so a lower reading is a misread and the carried count stands.
_ACCEPT = lambda read, held: read if read is not None and read >= held else held
_LATEST = lambda read, held: read if read is not None else held
_SHOWN = lambda pellet: pellet if pellet is not None else "not readable on this frame"
_SATISFIES = lambda led, names, pellet, asked, best: bool(
    asked in names and (pellet is None or best is None
                        or _L1(_STEP(led["head"], asked), pellet)
                        == _L1(_STEP(led["head"], best), pellet)))

# ---------------------------------------------------------------- the ledger
LEDGER_ATTR = "_pellet_warrant_ledger"
_FRESH = {"head": OPENING_HEAD, "trail": [OPENING_HEAD, OPENING_NECK], "last": OPENING_HEADING,
          "seen": {OPENING_HEAD: 1, OPENING_NECK: 1}, "eaten": 0, "pellet": None,
          "turns": 0, "pressed": 0, "honoured": 0, "appeals": 0, "won_back": 0}

# ---------------------------------------------------------------- the pixel windows
# `Score: N` is blitted at (20, 20) in Font(None, 24); the digit sits in these rows/cols and its
# per-column white-pixel profile separates 0-7 (pixel COUNT alone does not separate 0 from 5).
GLYPH_ROWS = slice(20, 36)
GLYPH_COLS = slice(68, 82)
GLYPH_HI = 150
GLYPH_LO = 140
GLYPH_PROFILES = np.array([
    (4, 0, 0, 0, 0, 0, 0, 8, 10, 4, 4, 4, 10, 8),
    (4, 0, 0, 0, 0, 0, 0, 1, 1, 1, 11, 11, 0, 0),
    (4, 0, 0, 0, 0, 0, 0, 6, 7, 7, 6, 6, 8, 6),
    (4, 0, 0, 0, 0, 0, 0, 6, 5, 4, 5, 8, 10, 5),
    (4, 0, 0, 0, 0, 0, 0, 3, 4, 3, 4, 11, 11, 2),
    (4, 0, 0, 0, 0, 0, 0, 6, 10, 6, 6, 6, 8, 6),
    (4, 0, 0, 0, 0, 0, 0, 8, 10, 5, 6, 6, 8, 6),
    (4, 0, 0, 0, 0, 0, 0, 2, 3, 7, 7, 5, 4, 3)], dtype=float)
_LIT = lambda band: ((band[..., 0] > GLYPH_HI) & (band[..., 1] > GLYPH_HI)
                     & (band[..., 2] > GLYPH_LO))
_COLS = lambda mask: mask.sum(axis=ZERO)
_NEAREST = lambda profile: int(np.argmin(np.abs(GLYPH_PROFILES - profile).sum(axis=1)))

# The board, and the renderer's camera, re-derived once so every cell's drawn centre and diameter
# are a table lookup rather than per-frame trigonometry.
BOARD_TOP = 60
BOARD_FLOOR = 514
WARM_R = 150
WARM_G = 120
COOL_B = 90
PELLET_R = 14.0
_LATTICE = (np.array(CELLS, dtype=float) - 1.5) * 80.0
_TILT = _LATTICE[:, 1] * np.cos(.5) - _LATTICE[:, 2] * np.sin(.5)
_DEPTH = _LATTICE[:, 1] * np.sin(.5) + _LATTICE[:, 2] * np.cos(.5)
_SWING = _LATTICE[:, 0] * np.cos(.6) + _DEPTH * np.sin(.6)
_AHEAD = _DEPTH * np.cos(.6) - _LATTICE[:, 0] * np.sin(.6)
_SHRINK = 600.0 / (600.0 + _AHEAD)
SCREEN_X = _SWING * _SHRINK + 400.0
SCREEN_Y = _TILT * _SHRINK + 300.0
DRAWN_D = 2 * (PELLET_R * _SHRINK).astype(int)
# One-sided: the fragment may be far smaller than the disc (the head covers it) but barely larger.
COVERED = -14.0
SWOLLEN = 4.0
_ROWBAND = lambda a: ((np.arange(a.shape[ZERO])[:, None] >= BOARD_TOP)
                      & (np.arange(a.shape[ZERO])[:, None] < BOARD_FLOOR))
_WARM = lambda a: (_ROWBAND(a) & (a[..., 0].astype(int) > WARM_R)
                   & (a[..., 1].astype(int) > WARM_G) & (a[..., 2].astype(int) < COOL_B))
_SPREAD = lambda rows, cols: (np.ptp(rows) + np.ptp(cols) + 2) / 2.0
_FITS = lambda spread: (COVERED <= spread - DRAWN_D) & (spread - DRAWN_D <= SWOLLEN)
_OFFSET = lambda cc, rr: (SCREEN_X - cc) ** 2 + (SCREEN_Y - rr) ** 2

# ---------------------------------------------------------------- the live text channel
MARK = "\n<<<harness warrant>>>\n"
# One model call leaves a user turn and an assistant turn behind; rolling an unusable answer out of
# model-visible history means dropping exactly that pair off the end. The engine REBUILDS the
# history list on every call, so the list object must be re-read inside the appeal, never cached
# across the call, and the pair must be addressed from the end rather than by a saved length.
PAIR = 2
_ASKED = lambda act: str((getattr(act, "metadata", None) or {}).get("function_called") or EMPTY)
_SEAT = lambda talk: (talk[ZERO]["content"][ZERO]
                      if isinstance(talk, list) and len(talk) > PAIR
                      and isinstance(talk[ZERO].get("content"), list) and talk[ZERO]["content"]
                      else None)
_AUDIT = lambda act, note: (getattr(act, "metadata", None) or {}).update(note)
_TURNS = lambda talk: len(talk) if isinstance(talk, list) else ZERO
_BASE = lambda seat: str(seat.get("text") or EMPTY).split(MARK)[ZERO]
_WHY = lambda refused, asked: (WHY_REFUSED if refused else WHY_WIDE).format(asked=asked)
WHY_REFUSED = ("- `{asked}` is the exact reverse of the last press. The engine refuses a 180 "
               "outright, so pressing it would move nothing at all.\n")
WHY_WIDE = "- `{asked}` can be pressed, but it moves AWAY from the pellet.\n"
CARD = (
    "### harness appeal {no}: the previous answer could not be used\n"
    "Reckoned from the published opening cell and the keys this harness has pressed, plus this "
    "same screenshot. Cube axes each run 0-3.\n"
    "- head cell now: {head}, last press `{last}`\n"
    "- body length {length} (the `Score: {eaten}` counter on this screenshot says {eaten} eaten)\n"
    "- pellet cell on this screenshot: {pellet}\n"
    "{why}"
    "- moves the engine would actually execute from here: {names}\n"
    "- of those, the ones that shorten the gap to the pellet: {closers}\n"
    "Decide again from THIS screenshot. Reply with exactly one JSON object and nothing else, "
    'for example {{"action": "{best}"}}.\n')


def _recall_trail(evaluator):
    """The episode-local reckoning ledger, created on first use and carried on the evaluator.

    Everything in it is self-derived: the opening cell the engine publishes, plus the keys this
    module has pressed since. It is deliberately NOT module state -- a fresh evaluator per episode
    must start from a fresh ledger, and module state would leak one episode's head into the next.
    """
    led = getattr(evaluator, LEDGER_ATTR, None)
    if led is None or not isinstance(led, dict):
        led = deepcopy(_FRESH)
        setattr(evaluator, LEDGER_ATTR, led)
    return led


def _tally_glyph(img, carried):
    """Pellets eaten, read off the engine's own `Score:` digit in the screenshot the model sees.

    Pure, and cheap enough (a 16x14 window) to ask wherever the body length is needed rather than
    threading it through signatures. Returns `carried` unchanged when there is no frame to read.
    """
    if np.ndim(img) != RANK or np.shape(img)[ZERO] < BOARD_FLOOR:
        return carried
    return _NEAREST(_COLS(_LIT(np.asarray(img)[GLYPH_ROWS, GLYPH_COLS])))


def _pellet_cell(img):
    """The pellet's cube cell from the warm blob on the board, or None when nothing survives.

    The size gate is one-sided (see the module docstring): a fragment much smaller than a cell's
    drawn disc is still that cell, because the snake's own disc is painted over it.
    """
    if np.ndim(img) != RANK or np.shape(img)[ZERO] < BOARD_FLOOR:
        return None
    rows, cols = np.nonzero(_WARM(np.asarray(img)))
    if not rows.size:
        return None
    gap = np.where(_FITS(_SPREAD(rows, cols)), _OFFSET(cols.mean(), rows.mean()), np.inf)
    nearest = int(np.argmin(gap))
    return CELLS[nearest] if np.isfinite(gap[nearest]) else None


def _moves_open(img, led):
    """The moves the engine would actually execute from the reckoned head, in menu order.

    In the cube, not the refused 180, and not into a body cell -- the body sized by the HUD count
    rather than by a fixed window, which is what stops a 2-cell snake being treated as a 6-cell one.
    The carried pellet count is refreshed here, under `_ACCEPT`'s no-falling rule; this is the only
    place the ledger's count is written, so every other reader sees one agreed number.
    """
    led["eaten"] = _ACCEPT(_tally_glyph(img, led["eaten"]), led["eaten"])
    return _OPEN(led["head"], BARRED_BY.get(led["last"]),
                 _OCCUPIED(led["trail"], START_LEN + led["eaten"]))


def _pick_press(img, open_names, pellet, led):
    """The open move this harness would press on its own: the warrant's top-ranked member.

    None when the warrant is empty -- every move from here is a wall, the refused 180 or the body.
    """
    blocked = _OCCUPIED(led["trail"], START_LEN + _tally_glyph(img, led["eaten"]))
    return _TOP(open_names, lambda n: _WARRANT(led, blocked, pellet, n))


def _correction_card(img, open_names, pellet, led, refused):
    """The appeal text: what the harness reckons, what it reads, and why the answer was unusable.

    Every field is public -- reckoned position, the HUD digit, the decoded pellet, the open moves.
    It states the harness's own preference last, as an example of the reply format, so the model
    can satisfy the warrant with a different closing move and still be obeyed.
    """
    eaten = _tally_glyph(img, led["eaten"])
    return CARD.format(no=led["appeals"] + 1, head=led["head"], last=led["last"],
                       length=START_LEN + eaten, eaten=eaten, pellet=_SHOWN(pellet),
                       why=_WHY(refused, led["asked"]), names=", ".join(open_names),
                       closers=", ".join(_CLOSERS(led, pellet, open_names)) or "none",
                       best=open_names[ZERO])


async def _retry_once(evaluator, frame_data, card):
    """Splice the appeal onto the live per-turn text channel and let the model answer once more.

    The unusable turn is dropped from model-visible history first, so the model is never asked to
    continue from an answer that was not pressed. The system text is written as
    `base + MARK + card`, never appended to, and is always restored -- the card is one turn's
    evidence, not an accumulating preamble.
    """
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    talk = getattr(evaluator, "conversation_history", None)
    seat = _SEAT(talk)
    if not card or seat is None or getattr(evaluator, "api_fatal_failure", False):
        return None
    del talk[-PAIR:]
    base = _BASE(seat)
    try:
        seat["text"] = base + MARK + card
        return await step(frame_data)
    finally:
        seat["text"] = base


async def decide(evaluator, frame_data):
    """One engine step: the frozen model answers, the warrant judges, and it may answer again."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    led = _recall_trail(evaluator)
    talk = getattr(evaluator, "conversation_history", None)
    keep = _TURNS(talk)
    act = await step(frame_data)                   # the frozen model decides first, every turn
    img = getattr(frame_data, "image", None)
    read = _pellet_cell(img)
    led["pellet"] = _LATEST(read, led["pellet"])
    names = _moves_open(img, led)
    best = _pick_press(img, names, led["pellet"], led)
    led["asked"] = _ASKED(act)
    refused = is_reversal(led["asked"], led["last"])
    if not _SATISFIES(led, names, led["pellet"], led["asked"], best) and keep and best:
        led["appeals"] += 1
        again = await _retry_once(
            evaluator, frame_data, _correction_card(img, names, led["pellet"], led, refused))
        if again is not None:
            act = again
            led["asked"] = _ASKED(act)
            led["won_back"] += int(_SATISFIES(led, names, led["pellet"], led["asked"], best))
    satisfied = _SATISFIES(led, names, led["pellet"], led["asked"], best)
    press = led["asked"] if satisfied else best
    _AUDIT(act, {"harness_open_moves": list(names), "harness_reckoned_head": led["head"],
                 "harness_hud_eaten": led["eaten"], "harness_pellet_cell": led["pellet"],
                 "harness_model_satisfied_warrant": satisfied, "harness_pressed": press})
    led["turns"] += 1
    if press in KEY_MAPPING:
        act.key = KEY_MAPPING[press]
        led["last"] = DIRECTION_VECTORS[press]
        led["head"] = _STEP(led["head"], press)
        _PUSH(led["trail"], led["head"])
        led["seen"][led["head"]] = led["seen"].get(led["head"], ZERO) + 1
        led["trail"][:] = led["trail"][:TRAIL_CAP]
        led["pressed"] += int(press in MENU)
        led["honoured"] += int(satisfied)
    return act
