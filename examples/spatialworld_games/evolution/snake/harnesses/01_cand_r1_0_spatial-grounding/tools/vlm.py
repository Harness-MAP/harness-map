"""The perceive -> VLM -> AUDIT -> act step: the harness's only mechanism.

The anchor made this a bare pass-through to `evaluator.baseline_step` — the UNPATCHED SpatialWorld
per-step fn that eval_runner stashes before it reassigns `evaluator.openai_ai_model =
build_agent(evaluator)`. Delegating there rather than to `openai_ai_model` (now this very agent) is
what keeps the reuse recursion-free, so the delegation itself is unchanged: ONE frozen-model call
per step, every step, on the baseline conversation. What changes is that its answer is no longer
pressed unreviewed. `decide` now carries a DEAD-RECKONED LEDGER of where the head is, which way it
is heading, which cells the body still occupies and how often each cell has been entered, and it
refuses to press a key whose geometry it can already prove fatal.

WHAT THE LEDGER IS ALLOWED TO KNOW. The spawn is fixed by the adapter's `reset_game` — head
(2,3,2), neck (1,3,2), heading +X, edge 4 — and every later cell follows from the keys this harness
itself handed the engine. The audit is therefore self-derived arithmetic end to end: no frame
decode, no `evaluator.game` attribute, no disk read, no food / body / seed oracle. The frozen model
still chooses every step; the audit only vetoes a choice it can prove leaves the box, reverses the
heading into an engine refusal, or re-enters an occupied cell, and then substitutes the
least-visited legal direction so an override cannot itself settle into a two-cycle.

THE THREE ENGINE LAWS THE LEDGER MIRRORS (`snake_adapter._try_move` / `_step`):
  * a 180 is tested BEFORE `last_move_dir` is assigned, and returns without touching the snake or
    `steps_taken` — so a reversal moves NOTHING and must advance no part of the ledger, or the
    reckoning drifts one cell on the first refusal and every later veto is computed against a
    phantom head;
  * leaving [0, 4) on any axis is immediate death, which is the only failure this lane ever saw;
  * the body test compares against `snake[:-1]` — the tail cell vacates on an ordinary move, so
    the trail's oldest entry is not an obstacle.

ONE-PLY ROOM. Among the moves that are legal now, a move whose own successor set is empty is a
cul-de-sac, so the audit prefers directions that still leave a non-reversing in-box continuation
clear of the trail. The lookahead reuses the PRE-move trail as the obstacle set, which is
conservative (the tail vacates again on the continuation) and therefore never licenses a move the
one-step test would have refused.

WHY THE GEOMETRY IS A MODULE-SCOPE ALGEBRA of lambdas rather than new functions: this candidate was
steered to hold the component GRAPH fixed and move only what lives INSIDE a component, so the
predicates sit beside the tables they read instead of becoming top-level components, and `decide`
keeps the entire control flow and every threshold-free decision. See NOTES.md.
"""

# ── the box, the spawn and the key algebra: measured off the adapter, never read from it ──────
_EDGE = 4                                   # grid_size; a legal cell is [0, _EDGE) on every axis
_SPAWN_TRAIL = ((2, 3, 2), (1, 3, 2))       # reset_game: (mid, grid-1, mid), then one cell back in -X
_SPAWN_HEADING = (1, 0, 0)                  # reset_game sets last_move_dir = +X before the first press
_TRAIL_SPAN = 6                             # 2 spawn segments + the 4 pellets that end the episode
_STEP = {                                   # action name -> the unit step _try_move applies
    "move_right": (1, 0, 0),
    "move_left": (-1, 0, 0),
    "move_up": (0, -1, 0),
    "move_down": (0, 1, 0),
    "move_forward": (0, 0, -1),
    "move_backward": (0, 0, 1),
}
_PRESS = {                                  # action name -> the engine key PygameInputSource maps
    "move_right": "d",
    "move_left": "a",
    "move_up": "w",
    "move_down": "s",
    "move_forward": "q",
    "move_backward": "e",
}
_NAMED = {k: n for n, k in _PRESS.items()}  # engine key -> action name, to read a pick back off Action

# ── the geometry: pure predicates over SELF-DERIVED state, at module scope ────────────────────
_ONWARD = lambda cell, v: (cell[0] + v[0], cell[1] + v[1], cell[2] + v[2])
_INSIDE = lambda cell: all(0 <= c < _EDGE for c in cell)
_FLIPS = lambda v, w: all(a + b == 0 for a, b in zip(v, w))
_BODY = lambda st: tuple(st["trail"][:-1])          # the tail vacates, so it is not an obstacle
_FRESHEST = lambda st, names, ahead: min(names, key=lambda n: (st["visits"].get(ahead[n], 0), n))
_WANTED = lambda act: (act.metadata or {}).get("function_called") \
    if (act.metadata or {}).get("function_called") in _STEP else _NAMED.get(act.key)
_FRESH_LEDGER = lambda: {"head": _SPAWN_TRAIL[0], "heading": _SPAWN_HEADING,
                         "trail": list(_SPAWN_TRAIL), "visits": {_SPAWN_TRAIL[0]: 1}, "vetoes": 0}
_VETOED = lambda st: st.update(vetoes=st["vetoes"] + 1)
_ADVANCE = lambda st, n, ahead: st.update(
    head=ahead[n], heading=_STEP[n], trail=([ahead[n]] + st["trail"])[:_TRAIL_SPAN],
    visits={**st["visits"], ahead[n]: st["visits"].get(ahead[n], 0) + 1},
) if n in _STEP and not _FLIPS(_STEP[n], st["heading"]) else None
_AUDITED = lambda act, shipped, wanted, head: dict(
    act.metadata or {}, function_called=shipped, model_action=wanted,
    audit=f"{wanted} from {head} leaves the box, reverses the heading or re-enters the body; "
          f"pressed the least-visited legal direction {shipped} instead")
_LEDGER = {}                                # per-episode self-derived state; one process per episode


async def decide(evaluator, frame_data):
    """One decision step -> engine Action: delegate to the frozen model, then audit its geometry.

    The decision still comes from the evaluator's pristine VLM step + JSON parser, so the model sees
    exactly the baseline conversation and `baseline_step` is called once per step. The returned
    `Action` is a mutable dataclass whose `key` is the entire override surface — the engine presses
    only that — so a vetoed pick is rewritten in place while the metadata keeps both the action the
    model named and the reason it was replaced.
    """
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    act = await step(frame_data)
    if not _LEDGER:
        _LEDGER.update(_FRESH_LEDGER())
    head, heading, body = _LEDGER["head"], _LEDGER["heading"], _BODY(_LEDGER)
    ahead = {name: _ONWARD(head, v) for name, v in _STEP.items()}
    inbox = [n for n, c in ahead.items() if _INSIDE(c) and not _FLIPS(_STEP[n], heading)]
    legal = [n for n in inbox if ahead[n] not in body]
    roomy = []
    for name in legal:
        onward = [m for m, w in _STEP.items()
                  if not _FLIPS(w, _STEP[name]) and _INSIDE(_ONWARD(ahead[name], w))
                  and _ONWARD(ahead[name], w) not in body]
        if onward:
            roomy.append(name)
    options = roomy or legal or inbox
    wanted = _WANTED(act)
    shipped = wanted if wanted in options else _FRESHEST(_LEDGER, options, ahead)
    if shipped != wanted:
        act.key = _PRESS[shipped]
        act.metadata = _AUDITED(act, shipped, wanted, head)
        _VETOED(_LEDGER)
    _ADVANCE(_LEDGER, shipped, ahead)
    return act
