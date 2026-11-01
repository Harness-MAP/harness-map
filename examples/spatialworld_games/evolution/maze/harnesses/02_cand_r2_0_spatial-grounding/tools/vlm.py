"""The perceive -> VLM -> parse step, bracketed by a frame-digest settlement ledger.

`decide` still delegates the decision to the evaluator's pristine per-step VLM fn
(`evaluator.baseline_step`) — one call per step, the same 29-turn image+text history, the same JSON
parser, the same engine `Action`. New is what brackets that call:

  MARK    a drift-tolerant digest of `frame_data.image`: every `grain`-th pixel, values bucketed by
          `qbin`, md5'd. Bucketing makes it a measurement rather than a hash of renderer noise.
  SETTLE  last step this harness returned a key. An unchanged mark means the key bought nothing — a
          forward the wall refused, or a turn the engine swallowed. `stall` is the streak of such
          findings, capped at `stall_cap`, reset the instant the frame moves.
  SHEET   the streak picks a cue from `_CUES`; step / last key / streak / cue render through
          ../memory.md and are grafted onto message 0.

Message 0 is the only per-turn channel: build_full_prompt is consulted once, and
build_conversation_history then copies `previous_history[0]` by reference, so re-rendering
`evaluator.current_prompt` after step 1 reaches nobody. The graft writes `stashed base + sheet`,
never truncate-then-append, which is what keeps the AVAILABLE ACTIONS list and the JSON RESPONSE
FORMAT in front of the model all episode. Nothing is said on step 1.

Everything the sheet carries is self-derived, from the frame shown and the actions returned.
`evaluator.game` is never read and the level is never read off disk. The frame-and-prompt path sits
in one `try`: a surprise increments `slips` and the step proceeds as the anchor's would.

The helpers are module-level lambdas on purpose: the component graph is fixed this iteration, so
the mechanism grows inside `decide` rather than as new nodes.

NB `evaluator.baseline_step` is the UNPATCHED SpatialWorld per-step fn that eval_runner stashes
before it reassigns `evaluator.openai_ai_model = build_agent(evaluator)`. Delegating here rather
than to `openai_ai_model` (now this very agent) keeps the reuse recursion-free.
"""
import hashlib
from pathlib import Path

_SLOT = "_maze_frame_ledger"             # where the per-episode book hangs off the evaluator
_GAP = "\n\n"

_PIX = lambda fd: getattr(fd, "image", None)
_MARK = lambda fd, grain, qbin: hashlib.md5(
    (_PIX(fd)[::grain, ::grain] // qbin).tobytes()).hexdigest() if _PIX(fd) is not None else ""
_FROZEN = lambda mark, book: int(mark == book["mark"])
_BET = lambda act: str(getattr(act, "key", None)) + ":" + str(
    (getattr(act, "metadata", None) or {}).get("granularity", "-"))
_STASH = lambda turns: turns[0]["content"][0]["text"]
_GRAFT = lambda turns, book, sheet: turns[0]["content"][0].update(
    text=book["base"] + _GAP + sheet.format(step=book["step"], bet=book["bet"],
                                            stall=book["stall"], advice=book["advice"]))

# Streak -> what the agent is told. Ascending, so the longest satisfied streak wins the loop.
_CUES = {
    1: "Your last key changed the screen in no way at all. Whatever you pressed did not happen: "
       "if it was move_forward the cell ahead is a wall, so turn before you try to move again.",
    3: "Three keys in a row have changed nothing. You are pinned on this bearing — turn, move one "
       "cell, and re-read POS before planning further; do not press forward into it a fourth time.",
    5: "Five keys in a row have changed nothing and you are burning the step budget in place. "
       "Leave this cell: turn, take one small step, then re-read the HUD from scratch.",
}

try:
    _SHEET = (Path(__file__).resolve().parent.parent / "memory.md").read_text()
except OSError:                           # the template is a lever; its absence must not be fatal
    _SHEET = "WORKING MEMORY (self-derived)\n- step: {step}\n- last key: {bet}\n" \
             "- unchanged-frame streak: {stall}\n- {advice}"


async def decide(evaluator, frame_data, *, grain=2, qbin=24, stall_cap=6):
    """One decision step -> engine Action. Settles the previous key against the frame digest, grafts
    the self-derived sheet onto the system turn, then reuses the evaluator's pristine VLM step and
    JSON parser for the decision itself."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    book = getattr(evaluator, _SLOT, None)
    if book is None:
        book = {"mark": None, "bet": None, "base": None, "stall": 0, "slips": 0,
                "step": 1, "advice": ""}
        setattr(evaluator, _SLOT, book)
    try:
        mark = _MARK(frame_data, grain, qbin)
        if book["mark"] is not None and book["bet"]:
            book["stall"] = min(book["stall"] + 1, stall_cap) * _FROZEN(mark, book)
        book["mark"] = mark
        book["advice"] = ""
        for cue in _CUES:
            if book["stall"] >= cue:
                book["advice"] = _CUES[cue]
        turns = getattr(evaluator, "conversation_history", None) or ()
        if turns:
            if book["base"] is None:
                book["base"] = _STASH(turns)
            _GRAFT(turns, book, _SHEET)
    except Exception:                      # noqa: BLE001 — the ledger must never cost a step
        book["slips"] = book["slips"] + 1
    act = await step(frame_data)
    book["bet"] = _BET(act)
    book["step"] = book["step"] + 1
    return act
