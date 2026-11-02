"""The perceive -> GROUND -> VLM -> verify step.

The anchor delegated straight to `evaluator.baseline_step` (SpatialWorld's pristine per-step VLM
fn: capture the frame, build the 29-turn image+text history, call the model, parse the JSON, fall
back to a default). This file keeps that delegation — the frozen model still makes every decision
and the engine `Action` contract is untouched — and wraps it in the grounding loop:

  1. diff this frame against the last one (tools/grounding.Grounding.observe) to learn what the
     previous action actually DID, and scan it for the gold exit / cyan stair;
  2. write the resulting working-memory block into the live system message, which is the ONLY
     per-turn text channel the engine has (build_conversation_history re-uses
     `conversation_history[0]` verbatim every turn and ignores `current_prompt` after step 1 —
     so the block is APPENDED to a stashed pristine base, never spliced into it, or the JSON
     RESPONSE FORMAT contract the parser depends on would be destroyed);
  3. take the model's answer; if it is an action the frame has already PROVEN is a no-op from the
     unchanged state the body is in, tell the model so and ask again (at most RETRY_LIMIT times
     per step, MAX_EXTRA_CALLS per episode);
  4. if the model still insists on a provable no-op, substitute the grounding tool's rescue
     action — the most productive thing the frames have not already refuted, falling back to a
     turn, which is the one action the frame can never refuse, so this cannot loop.

The substitution is a last resort and it is counted: `grounding.substitutions`. Every fact used
here comes from the rendered frame plus the agent's own action history — nothing reads
`evaluator.game`, the layout, or the true pose.
"""
from tools import action as action_algebra
from tools import frame as frame_tool

# MEASURED on two live episodes (levels 10 and 18, Qwen3-VL-30B-A3B-Instruct): a FIRST re-ask
# changed the model's answer on 2 of 44 refused steps; a SECOND re-ask changed it on 0 of 42.
# So one re-ask, then substitute — the second was pure latency.
RETRY_LIMIT = 1            # re-asks per step when the answer is a proven no-op
MAX_EXTRA_CALLS = 60       # episode-wide ceiling on re-asks (keeps us far inside the 1800 s wall)


def _log(evaluator, message):
    """Leave the mechanism's decisions in the episode session log, where the diagnostician
    reads them. Never let logging break a step."""
    try:
        evaluator.logger.log_info(message)
    except Exception:                                        # noqa: BLE001
        pass


def _name_of(act):
    """The canonical maze action behind an engine Action, robust to the evaluator's
    `default_move_*` fallback label (the engine key is the ground truth of what will run)."""
    meta = getattr(act, "metadata", None) or {}
    called = str(meta.get("function_called") or "")
    for legal in action_algebra.KEY_MAPPING:
        if called == legal or called.endswith(legal):
            return legal
    for legal, key in action_algebra.KEY_MAPPING.items():
        if getattr(act, "key", None) == key:
            return legal
    return called or "unknown"


def _granularity_of(act):
    meta = getattr(act, "metadata", None) or {}
    return str(meta.get("granularity") or "small").strip().lower()


def _inject(evaluator, grounding, refused=None):
    """Append the grounded block to the pristine system message. Returns False on step 0, where
    no history exists yet and the engine is still composing the prompt from current_prompt."""
    history = getattr(evaluator, "conversation_history", None)
    if not history:
        return False
    head = history[0]
    content = head.get("content") or []
    if not content or "text" not in content[0]:
        return False
    if getattr(grounding, "_base_prompt", None) is None:
        grounding._base_prompt = content[0]["text"]          # the pristine prompt, stashed once
    content[0]["text"] = grounding._base_prompt + "\n\n" + grounding.block(refused=refused)
    return True


def _rescue_action(evaluator, grounding, template_action):
    """Build the engine Action for the grounding tool's rescue turn, cloning the model's own
    Action so the engine sees exactly the shape it expects."""
    name = grounding.rescue()
    meta = dict(getattr(template_action, "metadata", None) or {})
    meta.pop("granularity", None)
    is_move = name in action_algebra.MOVEMENT_ACTIONS
    if is_move:
        meta["granularity"] = "small"            # one cell keeps dead reckoning exact
    meta.update({"function_called": name,
                 "action_type": "move" if is_move else "turn",
                 "harness_note": "substituted: the model's answer was a frame-proven no-op"})
    return template_action.__class__(type="key_press", key=action_algebra.key_for(name),
                                     metadata=meta), name


async def decide(evaluator, frame_data, grounding=None):
    """One decision step -> engine Action, with the frozen model still choosing."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    if grounding is None:                                    # degrade to the anchor behaviour
        return await step(frame_data)

    image = getattr(frame_data, "image", None)
    if image is None:                                        # belt and braces: the engine's own
        image = frame_tool.capture(evaluator)                # screenshot moves in exact lockstep
    grounding.observe(image)
    injected = _inject(evaluator, grounding)

    act = await step(frame_data)
    name = _name_of(act)

    if injected:
        tries = 0
        while (name in grounding.proven_null() and tries < RETRY_LIMIT
               and grounding.vetoes < MAX_EXTRA_CALLS):
            grounding.vetoes += 1
            tries += 1
            _inject(evaluator, grounding, refused=name)
            act = await step(frame_data)
            name = _name_of(act)
        if name in grounding.proven_null():
            refused, (act, name) = name, _rescue_action(evaluator, grounding, act)
            grounding.substitutions += 1
            _log(evaluator, f"[grounding] step {grounding.step}: `{refused}` is a frame-proven "
                            f"no-op here; substituted `{name}` "
                            f"(substitutions={grounding.substitutions}, re-asks={grounding.vetoes}, "
                            f"no-ops so far={grounding.n_nulls}/{grounding.n_actions})")

    grounding.commit(name, _granularity_of(act))
    return act
