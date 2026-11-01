"""The perceive -> VLM -> COMMIT step.

The anchor delegated the whole turn to `evaluator.baseline_step` (the pristine SpatialWorld
per-step fn that eval_runner stashes before it reassigns `evaluator.openai_ai_model`) and returned
whatever came back. This candidate still makes exactly ONE model call per step through that same
pristine path — same parser, same JSON contract, same legal action set — but it now owns the two
ends of the turn:

  BEFORE the call it splices a harness-authored BRIEF into the one durable prompt slot, so the
  turn carries the harness's own bookkeeping instead of only 29 replayed copies of the agent's
  previous answers.
  AFTER the call it reads the proposed action word off the returned Action, runs it past the
  decision table in `tools/action.py`, and plays either that move or the first move the table
  still allows.

`baseline_step` EXECUTES what it parses, so the veto cannot come afterwards: `_arm` swaps
`evaluator.execute_action` for a sink that only records the attempt, `_disarm` puts the real one
back, and `_after` plays the one move this harness actually commits to. One decision, one engine
action, one model call — the step accounting the evaluator scores is untouched.

Every branch is a module-level combinator; `decide` is a straight line through them. The brief's
wording lives in `memory.md`, which the template loader below reads at import, so the prose of the
working memory is an ordinary prose lever and not a string buried in code.
"""
import functools
from pathlib import Path

from tools.action import (LEGAL_ACTIONS, RULES, SPEC, honoured, new_book, note_move, refusals,
                          render_word, survivors)
from tools.frame import fingerprint

_HOME = Path(__file__).resolve().parent.parent
_TPL = _HOME / "memory.md"
_FALLBACK = ("HARNESS STATE (self-derived): step {{step}}; played {{played}}; "
             "reduced word {{word}}; refused {{refused}}; allowed {{allowed}}.")
TEMPLATE = _TPL.read_text() if _TPL.exists() else _FALLBACK

# --- the ledger -----------------------------------------------------------------------------
# One dict per episode, cached on the evaluator (one process per episode, so it cannot leak
# between levels). It holds only what this harness itself emitted.
_attach = lambda ev: (setattr(ev, "rubik_book", new_book()), ev.rubik_book)[1]
_book = lambda ev: getattr(ev, "rubik_book", None) or _attach(ev)

# --- the brief ------------------------------------------------------------------------------
_TAIL = 8                     # how many executed action words the brief echoes back
_hits = lambda code, b: tuple(n for n in sorted(LEGAL_ACTIONS)
                              if code in [c for c, _w in refusals(n, b)])
_refused_line = lambda b: " | ".join(
    "%s (%s): %s" % (code, why, ", ".join(_hits(code, b)))
    for code, why, _p in RULES if _hits(code, b)) or "nothing is refused this turn"
_allowed_line = lambda b: ", ".join(survivors(b))
_played_line = lambda b: ", ".join(b["played"][-_TAIL:]) or "nothing yet"
_word_line = lambda b: render_word(b["word"]) or "(the cube is back at its scrambled start)"
_still_line = lambda b: ("no comparison yet" if len(b["frames"]) < 2 else
                         "the last action changed NOTHING on screen" if b["still"] else
                         "the last action changed the picture")
_fields = lambda b: {"step": str(b["step"]), "played": _played_line(b), "word": _word_line(b),
                     "stood": str(len(b["seen"])), "screen": _still_line(b),
                     "refused": _refused_line(b), "allowed": _allowed_line(b)}
_fill = lambda text, fields: functools.reduce(
    lambda s, kv: s.replace("{{%s}}" % kv[0], kv[1]), sorted(fields.items()), text)
brief = lambda b: "\n\n" + _fill(TEMPLATE, _fields(b)).strip() + "\n"

# --- the prompt slot ------------------------------------------------------------------------
# `openai_utils.build_conversation_history` copies `previous_history[0]` verbatim forever, so slot
# 0 is the only durable channel a harness owns. It is captured ONCE, pristine, on the first turn
# that has a history; after that the brief is re-spliced onto that pristine copy each turn, which
# appends it AFTER the engine's AVAILABLE ACTIONS + RESPONSE FORMAT tail and never inside it. On
# step 0 there is no history and nothing to report, so the composed prior is left exactly as it is.
_slot = lambda ev: ev.conversation_history[0]["content"][0]
_remember = lambda ev, b: (
    b.__setitem__("prior", b["prior"] or getattr(ev, "current_prompt", "")),
    b.__setitem__("slot", b["slot"] or (_slot(ev)["text"] if ev.conversation_history else "")),
)
_splice = lambda ev, b: (_slot(ev).__setitem__("text", b["slot"] + brief(b))
                         if ev.conversation_history else None)
_inform = lambda ev, b: (_remember(ev, b), _splice(ev, b))

# --- muting the delegate's own execution ------------------------------------------------------
_sink = lambda b: (lambda action: (b["sunk"].append(_name_of(action)), True)[1])
_arm = lambda ev, b: (b.__setitem__("real", ev.execute_action),
                      setattr(ev, "execute_action", _sink(b)))
_disarm = lambda ev, b: setattr(ev, "execute_action", b["real"])

# --- committing -------------------------------------------------------------------------------
_meta = lambda action: (action.metadata if action.metadata is not None
                        else (setattr(action, "metadata", {}), action.metadata)[1])
_name_of = lambda action: _meta(action).get("function_called", "")
_choice = lambda b, pick: pick if honoured(pick, b) else survivors(b)[0]
_rewrite = lambda b, action, name, pick: (
    setattr(action, "key", SPEC[name][2]),
    _meta(action).update({"function_called": name, "face": SPEC[name][0],
                          "direction": SPEC[name][1], "action_type": SPEC[name][3],
                          "model_pick": pick, "harness_override": name != pick,
                          "refused": [c for c, _w in refusals(pick, b)]}),
    action)[2]
# A face turn is animated and the adapter REFUSES both families while one is in flight, returning
# False. A capture renders, and rendering is what advances the animation — so the one honest cure
# for a refused move is to capture and try once more. On a clean run this never fires.
_retry = lambda ev, action: (fingerprint(ev), ev.execute_action(action))[1]
_play = lambda ev, action: ev.execute_action(action) or _retry(ev, action)
_shutter = lambda ev, b: (b["frames"].append(fingerprint(ev)),
                          b.__setitem__("still", len(b["frames"]) > 1
                                        and b["frames"][-1] == b["frames"][-2]))
_after = lambda ev, b, action: (_play(ev, action), note_move(b, _name_of(action)),
                                _shutter(ev, b), action)[3]
_commit = lambda ev, b, action: _after(
    ev, b, _rewrite(b, action, _choice(b, _name_of(action)), _name_of(action)))


async def decide(evaluator, frame_data):
    """One decision step -> engine Action: inform, delegate, then commit through the table."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    book = _book(evaluator)
    _inform(evaluator, book)
    _arm(evaluator, book)
    proposed = await step(frame_data)
    _disarm(evaluator, book)
    return _commit(evaluator, book, proposed)
