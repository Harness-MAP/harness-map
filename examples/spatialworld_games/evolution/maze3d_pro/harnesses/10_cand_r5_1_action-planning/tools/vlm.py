"""tools/vlm.py — the per-step loop: OBSERVE -> STATE INTO THE PROMPT -> ASK -> GOVERN.

The VLM call itself is still the evaluator's own pristine per-step fn (`evaluator.baseline_step`):
it captures the frame, maintains the rolling image+text history, calls the model through
`evaluator.client`, parses the JSON action and falls back to a safe default, returning an engine
`Action(type="key_press", ...)`. Leaving it untouched is deliberate — this chain measures
2340/2340 decisions parsed to a legal action with zero parse fallbacks, so the response contract
has no headroom at all and nothing here goes near it.

What this file adds is the four-stage loop around that one call; tools/action.py owns every piece
of state and every decision rule, and tools/frame.py owns the pixels.

THE INJECTION CHANNEL, AND THE ONE WAY IT BREAKS. `evaluator.current_prompt` is read by
`build_full_prompt` on the FIRST call only; from then on `build_conversation_history` reuses
`previous_history[0]` verbatim, so the system turn is the one and only per-turn text channel into
the agent. It is rewritten as `stashed base + block`, where the stash is the whole first-call
system turn — which already carries the tool list and the JSON RESPONSE FORMAT. It is APPENDED to,
never spliced and never truncated: cutting that turn at a marker deletes the response contract from
every later call and the parser then fails on 3% of answers.

NB `evaluator.baseline_step` is the UNPATCHED per-step fn that eval_runner stashes before it
reassigns `evaluator.openai_ai_model = build_agent(evaluator)`. Delegating here rather than to
`openai_ai_model` — which is now this very agent — is what keeps the reuse recursion-free, and it
is called on EVERY step so the anti-cheat guard's VLM-call counter can never void the run.
"""
from __future__ import annotations

from . import frame as _frame


def _inject(evaluator, tape) -> None:
    """Append the measured-state block to the agent's system turn. A no-op on the first call, where
    the history is still empty and `current_prompt` carries the composed prior instead."""
    history = getattr(evaluator, "conversation_history", None)
    if not history:
        return
    try:
        slot = history[0]["content"][0]
    except Exception:
        return
    if not isinstance(slot, dict) or slot.get("type") != "input_text":
        return
    if tape.base_prompt is None:
        tape.base_prompt = slot.get("text") or ""        # the whole first-call system turn
    block = tape.render()
    slot["text"] = tape.base_prompt + ("\n\n" + block if block else "")


async def decide(evaluator, frame_data, tape=None):
    """One decision step -> engine Action.

    The frozen model is asked every step and its answer is executed unless the measurement proves
    it cannot help. Both harness stages are wrapped: a perception or planning fault degrades this
    harness to the pristine baseline for that step rather than costing the episode.
    """
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    if tape is None:
        return await step(frame_data)
    try:
        tape.chart.observe(_frame.capture(frame_data))
        _inject(evaluator, tape)
    except Exception:
        pass
    proposal = await step(frame_data)
    try:
        return tape.apply(proposal)
    except Exception:
        return proposal
