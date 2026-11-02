"""The perceive -> inject -> ask -> govern step.

`evaluator.baseline_step` is the UNPATCHED SpatialWorld per-step fn that eval_runner stashes
before it reassigns `evaluator.openai_ai_model = build_agent(evaluator)`. It captures the frame,
maintains the rolling 29-turn image+text history, calls the VLM via `evaluator.client`, parses the
JSON action and falls back to a safe default, returning an engine `Action(type="key_press", ...)`.
It is still called on EVERY step: the parse path, the legal action set, the fallback and the
anti-cheat VLM-call accounting are untouched, and the frozen model still makes every decision.

What this file adds around it:

  1. OBSERVE. `planner.observe(tools.frame.pixels(frame_data))` inverts the renderer's projection on the frame
     the model is about to be shown (tools/plan.py), resolves the PREVIOUS decision's outcome
     against it, and writes the newly proven cells into the per-floor lattice.

  2. INJECT. The decoded state is appended to the system turn. `conversation_history[0]` is the
     only text the engine re-reads every turn — `current_prompt` is consumed by
     `build_full_prompt` on the FIRST call only, and `build_conversation_history` carries
     `previous_history[0]` forward verbatim thereafter. The pristine base is stashed whole and the
     block APPENDED to it every step; truncate-then-append would delete the JSON RESPONSE FORMAT
     contract from every later turn and the parser depends on it.

  3. GOVERN. The returned Action is overridden only where the pixels PROVE the proposal cannot
     accomplish anything (a translation into a touching wall, a climb with no stair underfoot, a
     retreat, a fourth consecutive turn). The replacement is the lattice's own routed move.

Every one of the three is inside one try/except that falls back to the pristine Action, so the
mechanism can never cost an episode.
"""
from tools import action as A
from tools import frame as F
from tools import plan as P

_MEMORY_SLOT = "_harness_memory_base"
_MARKER = "--- MEASURED STATE"


def _inject_memory(evaluator, planner, template):
    """Append the measured-state block to a stashed pristine system turn (never splice it)."""
    history = getattr(evaluator, "conversation_history", None)
    if not history:
        return
    head = history[0]
    content = head.get("content") if isinstance(head, dict) else None
    if not content:
        return
    slot = content[0]
    if not isinstance(slot, dict) or "text" not in slot:
        return
    base = getattr(evaluator, _MEMORY_SLOT, None)
    if base is None:
        # Stash the PRISTINE turn, once. The system turn is laid out as
        # <priors> + AVAILABLE ACTIONS + RESPONSE FORMAT, in that order, and the block is
        # appended after all of it, so the stash is only ever taken before the first append and
        # the text cannot already contain a block. If it somehow does, decline to stash and
        # leave the turn untouched: splitting it at the marker would truncate away the RESPONSE
        # FORMAT contract that the evaluator's parser depends on, which is a far worse failure
        # than skipping the injection.
        if _MARKER in slot["text"]:
            return
        base = slot["text"]
        setattr(evaluator, _MEMORY_SLOT, base)
    slot["text"] = base + "\n" + planner.render_memory(template)


async def decide(evaluator, frame_data, planner=None, template=None):
    """One decision step -> engine Action. The frozen model decides; the frame audits it."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    if planner is None:
        return await step(frame_data)

    sighted = False
    try:
        img = F.pixels(frame_data)
        if img is not None:
            planner.observe(img)
            _inject_memory(evaluator, planner, template or P.DEFAULT_TEMPLATE)
            sighted = True
    except Exception:
        sighted = False

    proposal = await step(frame_data)

    if not sighted:
        return proposal
    try:
        name, gran = A.read_proposal(proposal)
        if name is None:
            return proposal
        new_name, new_gran, overridden = planner.govern(name, gran)
        planner.record(new_name, new_gran)
        if not overridden and new_gran == gran:
            return proposal
        return A.retarget(proposal, new_name, new_gran)
    except Exception:
        return proposal
