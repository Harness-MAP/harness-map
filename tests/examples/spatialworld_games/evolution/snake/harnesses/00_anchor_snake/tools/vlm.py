"""The perceive -> VLM -> parse step. Kept as a thin reuse of the evaluator's own pristine per-step
VLM fn (`evaluator.baseline_step`), which captures the frame (see tools/frame.py), builds the
29-turn image+text history, calls the VLM via `evaluator.client`, parses the JSON action and falls
back to a safe default — returning an engine `Action(type="key_press", ...)`. Reusing it verbatim is
what makes this rich anchor score identically to the prompt-only baseline. A proposer evolves THIS
file to change the decision logic (e.g. add a verification/retry pass, a second reasoning call, or
self-derived memory injection) while keeping the engine Action contract (see tools/action.py) intact.

NB `evaluator.baseline_step` is the UNPATCHED SpatialWorld per-step fn that eval_runner stashes
before it reassigns `evaluator.openai_ai_model = build_agent(evaluator)`. Delegating here rather
than to `openai_ai_model` (now this very agent) is what keeps the reuse recursion-free.
"""


async def decide(evaluator, frame_data):
    """One decision step -> engine Action. Reuses the evaluator's pristine VLM step + JSON parser."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    return await step(frame_data)
