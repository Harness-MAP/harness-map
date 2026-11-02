"""The perceive -> decide -> act step: measured local occupancy, a self-derived map, one VLM call.

The anchor passed this step straight through to `evaluator.baseline_step`, so the frozen agent met
every turn with one screenshot and no memory of the turns before it. This step keeps that single
delegation and wraps it in a measurement: what the frame proves about the three sides of the cell
the agent can see, what its own last action did, and which side has never been tested.

Everything is read from `frame_data.image`, the array the agent itself is shown; `evaluator.game`
is never touched and no layout file is opened. The pixel derivation, its census and the helper
table live in `traces_notes/`.
"""
from functools import reduce
from hashlib import md5
from pathlib import Path

import numpy as np

# constants
HEADINGS = ("NORTH", "EAST", "SOUTH", "WEST")          # clockwise; +1 is one turn_right
STEP = {"NORTH": (0, -1), "EAST": (1, 0), "SOUTH": (0, 1), "WEST": (-1, 0)}
SIDES = ("FRONT", "RIGHT", "BACK", "LEFT")             # egocentric names, in turn order
SPIN = ("turn_left", "turn_right")                     # [0] one left, [1] one right
WALK = "move_forward"
KEYS = {"move_forward": "w", "turn_left": "a", "turn_right": "d"}
ARROW = (255, 50, 50)                                  # the compass arrow's exact ink
VIEW = (40, 570)                                       # rows below the HUD bar, above the foot bar
BANDS = {"FRONT": (380, 421), "LEFT": (60, 141), "RIGHT": (660, 741)}
SHUT = 401                                             # FRONT extent when the next cell is solid
BLIND = 455                                            # side extent when that side is solid
RUNGS = (200, 132)                                     # FRONT extent at clear depth 1 / 2
TINT = 12                                              # red over green and blue, in the brick ink
LIT = 40                                               # the brick ink's red floor
GRAIN = "small"                                        # one cell: the only granularity we emit

HEAD = "\n\n--- MEASURED MAP (read off your own frame; nothing privileged) ---"
FOOT = "--- end map ---"
SAID = '{"action": "%s"%s}'
GRANT = ', "granularity": "%s"'
AGAIN = "  <-- you keep coming back here; spend the turn on an UNTESTED side"
PRIMER = Path(__file__).resolve().parent.parent / "memory.md"
PREFACE = PRIMER.read_text(errors="replace").split("-->")[-1].strip() if PRIMER.exists() else ""

# the frame
_ink = lambda reg, c: (reg[:, :, 0] == c[0]) & (reg[:, :, 1] == c[1]) & (reg[:, :, 2] == c[2])

_box = lambda m: (lambda r, c: (int(r.min()), int(r.max()), int(c.min()), int(c.max()),
                                float(r.mean()), float(c.mean())))(*np.nonzero(m))

_point = lambda b: ((HEADINGS[0] if b[4] > (b[0] + b[1]) / 2 else HEADINGS[2])
                    if (b[1] - b[0]) > (b[3] - b[2]) else
                    (HEADINGS[1] if b[5] < (b[2] + b[3]) / 2 else HEADINGS[3]))

_facing = lambda img: (lambda m: _point(_box(m)) if m.any() else None)(_ink(img[:VIEW[0]], ARROW))

_brick = lambda v: ((v[:, :, 0].astype(int) > v[:, :, 1].astype(int) + TINT)
                    & (v[:, :, 0].astype(int) > v[:, :, 2].astype(int) + TINT)
                    & (v[:, :, 0] > LIT))

_tall = lambda w, b: int(w[:, b[0]:b[1]].any(axis=1).sum())

_depth = lambda e: 0 if e >= SHUT else (1 if e >= RUNGS[0] else (2 if e >= RUNGS[1] else 3))

_look = lambda img: None if img is None else (lambda v: (lambda w, f: None if f is None else {
    "face": f, "FRONT": _depth(_tall(w, BANDS["FRONT"])),
    "LEFT": _tall(w, BANDS["LEFT"]) < BLIND, "RIGHT": _tall(w, BANDS["RIGHT"]) < BLIND,
    "view": md5(v.tobytes()).hexdigest()})(_brick(v), _facing(img)))(img[VIEW[0]:VIEW[1]])

# the map
_turned = lambda face, k: HEADINGS[(HEADINGS.index(face) + k) % len(HEADINGS)]

_ahead = lambda spot, face: (spot[0] + STEP[face][0], spot[1] + STEP[face][1])

_rot = lambda face, goal: (HEADINGS.index(goal) - HEADINGS.index(face)) % len(HEADINGS)

_fence = lambda spot, grid: (max(-grid[0], min(grid[0], spot[0])),
                             max(-grid[1], min(grid[1], spot[1])))

_stirred = lambda book, look: book["view"] != look["view"]

_settle = lambda book, look, grid: (
    book["edge"].update({(book["from"], book["bear"]): _stirred(book, look)}),
    book.update({"at": _fence(_ahead(book["from"], book["bear"]), grid)
                 if _stirred(book, look) else book["from"]})
) if (look and book["pending"] == WALK) else None

_mark = lambda book, look: book["edge"].update({
    (book["at"], look["face"]): look["FRONT"] > 0,
    (book["at"], _turned(look["face"], 3)): look["LEFT"],
    (book["at"], _turned(look["face"], 1)): look["RIGHT"]}) if look else None

_mirror = lambda book: book["edge"].update(
    {(_ahead(c, h), _turned(h, 2)): True for (c, h), v in list(book["edge"].items()) if v})

_side = lambda book, spot, h: book["edge"].get((spot, h))

_untested = lambda book, spot: [h for h in HEADINGS if _side(book, spot, h) is None]

_ways = lambda book, spot: [h for h in HEADINGS if _side(book, spot, h)]

_near = lambda face, hs: sorted(hs, key=lambda h: (min(_rot(face, h), -_rot(face, h) % len(HEADINGS)),
                                                   HEADINGS.index(h)))

# routing
_spread = lambda book, acc: {**acc, **{_ahead(c, h): (acc[c] or h) for c in list(acc)
                                       for h in _ways(book, c) if _ahead(c, h) not in acc}}

_flood = lambda book, spot, rounds: reduce(lambda a, _: _spread(book, a), range(rounds), {spot: None})

_quest = lambda book, spot, rounds, want: (lambda rt: next(
    (c for c in rt if rt[c] and want(c)), None))(_flood(book, spot, rounds))

_wanted = lambda book, c: c is not None and (not book["seen"].get(c) or bool(_untested(book, c)))

_pick = lambda book, rounds: _quest(book, book["at"], rounds, lambda c: not book["seen"].get(c)) \
    or _quest(book, book["at"], rounds, lambda c: _untested(book, c))

_target = lambda book, look, rounds: book.update({"goal": book["goal"] if (
    _wanted(book, book["goal"]) and book["goal"] != book["at"]) else _pick(book, rounds)}) if look \
    else None

_route = lambda book, rounds: _flood(book, book["at"], rounds).get(book["goal"])

_aim = lambda book, look, rounds: (
    next(iter(_near(look["face"], _untested(book, book["at"]))), None)
    or _route(book, rounds)
    or next(iter(_near(look["face"], _ways(book, book["at"]))), look["face"]))

_swing = lambda book, face, goal: (lambda r: WALK if not r else
                                   (SPIN[book["spin"]] if r == len(SPIN) else SPIN[int(r == 1)]))(
    _rot(face, goal))

_plan = lambda book, look, rounds: _swing(book, look["face"], _aim(book, look, rounds))

_futile = lambda book, look, name, plan: (
    (name == WALK and (not look["FRONT"] or _side(book, book["at"], look["face"]) is False))
    or (name in SPIN and book["pending"] in SPIN and name != book["pending"])
    or (not _untested(book, book["at"]) and name != plan))

_keep = lambda name: name if name in KEYS else WALK

# writing back
_dress = lambda old, plan, swap: dict(
    {k: v for k, v in (old or {}).items() if k not in ("granularity", "repeat")},
    function_called=plan, harness_override=swap,
    **({"granularity": GRAIN, "repeat": 1} if plan == WALK else {}))

_apply = lambda act, plan, swap: (setattr(act, "key", KEYS[plan]),
                                  setattr(act, "metadata",
                                          _dress(getattr(act, "metadata", None), plan, swap)))

_said = lambda plan: SAID % (plan, GRANT % GRAIN if plan == WALK else "")

_echo = lambda ev, plan: (lambda turns: turns[-1]["content"][0].update({"text": _said(plan)})
                          if (turns and turns[-1].get("role") == "assistant") else None)(
    getattr(ev, "conversation_history", None))

# the block
_verdict = lambda book, spot, h: ("OPEN, entered %dx" % book["seen"][_ahead(spot, h)]
                                  if book["seen"].get(_ahead(spot, h)) else
                                  {True: "OPEN, never entered", False: "WALL",
                                   None: "UNTESTED"}[_side(book, spot, h)])

_around = lambda book, look: ["  %-5s (%-5s): %s" % (SIDES[k], _turned(look["face"], k),
                                                     _verdict(book, book["at"], _turned(look["face"], k)))
                              for k in range(len(SIDES))]

_sketch = lambda book, look, tail, worn: "" if not look else "\n".join(
    [HEAD, PREFACE, "cell %s (counted from where you started, so not the HUD's POS) facing %s"
     "   decisions: %d   entered this cell %dx%s"
     % (book["at"], look["face"], book["tick"], book["seen"].get(book["at"], len(SPIN) - 1),
        AGAIN if book["seen"].get(book["at"], len(SPIN) - 1) > worn else "")]
    + _around(book, look)
    + ["  cells entered: %d   sides measured: %d   corrections so far: %d"
       % (len(book["seen"]), len(book["edge"]), book["swap"]),
       "  recent: " + "; ".join(book["trail"][-tail:]), FOOT])

_speak = lambda ev, book, look, tail, worn: (lambda turns: (lambda slot: (
    book.update({"base": book["base"] or slot["text"]}),
    slot.update({"text": book["base"] + _sketch(book, look, tail, worn)})))(
        turns[0]["content"][0]) if turns else None)(getattr(ev, "conversation_history", None))

_note = lambda book, look, plan, name, hold: (
    book["trail"].append("%s at %s facing %s%s" % (name, book["at"], look["face"],
                                                   "" if plan == name else " -> corrected to " + plan)),
    book["trail"].__delitem__(slice(None, -hold)))

_carry = lambda book, look, plan: book.update(
    {"from": book["at"], "bear": look["face"], "view": look["view"], "pending": plan,
     "spin": SPIN.index(plan) if plan in SPIN else book["spin"]})

FRESH = lambda: {"at": (0, 0), "from": (0, 0), "bear": HEADINGS[0], "view": None, "pending": None,
                 "base": None, "goal": None, "tick": 0, "swap": 0, "spin": 1, "blind": 0,
                 "seen": {}, "edge": {}, "trail": []}


async def decide(evaluator, frame_data, grid=(25, 25), rounds=40, hold=30, tail=4, worn=1):
    """One decision step -> engine Action. Measure the heading and the three visible sides off the
    frame, settle what the last action proved, show the agent its own map, take the frozen model's
    proposal, and correct it only where the measurement says it cannot help."""
    book = getattr(evaluator, "sg_book", None)
    if book is None:
        book = FRESH()
        evaluator.sg_book = book
    look = _look(getattr(frame_data, "image", None))
    book["tick"] = book["tick"] + 1
    _settle(book, look, grid)
    book["seen"][book["at"]] = book["seen"].get(book["at"], 0) + 1
    _mark(book, look)
    _mirror(book)
    _target(book, look, rounds)
    _speak(evaluator, book, look, tail, worn)
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    evaluator.sg_calls = getattr(evaluator, "sg_calls", 0) + 1
    act = await step(frame_data)
    meta = getattr(act, "metadata", None) or {}
    name = meta.get("function_called") or ""
    if look is None:
        book["blind"] = book["blind"] + 1
        return act
    plan = _plan(book, look, rounds)
    move = plan if _futile(book, look, name, plan) else _keep(name)
    _apply(act, move, move != name)
    if move != name:
        _echo(evaluator, move)
        book["swap"] = book["swap"] + 1
    _note(book, look, move, name, hold)
    _carry(book, look, move)
    return act
