"""Rich MAZE harness — the agent-factory injection point over the SAME SpatialWorld engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func` (config.py::_agent_func / _apply_harness). It REUSES the evaluator's already-built
`self.input_source` / `self.engine` / `self.client` / `baseline_step` and returns an engine `Action`
each step, so scoring (the maze `success` field) stays byte-compatible with the prompt-only
baseline. Nothing here reads engine state: the only observation is `frame_data.image`, the very
frame the frozen model is shown.

The per-step decision is a closed, measured loop rather than a pass-through. `build_agent` creates
one `tools.action.Chart` (the dead-reckoned map) and one `tools.action.Tape` (the planner and
governor) per episode and threads them through `tools.vlm.decide`, which decodes the frame exactly,
folds what it measures into the chart, states those facts back to the model in its system turn,
asks the model, and displaces only the proposals the measurement proves cannot help.

The planner's currency is DECISIONS, not cells: the episode is billed in model calls, so the route
is the cheapest sequence of legal decisions rather than the shortest path. Module map + lever table:
see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def compose_system_prompt() -> str:
    """Recompose the maze system prompt from the .md priors: AGENT.md (role + the perception facts
    this build actually renders) + skills/*.md (the navigation priority order) + rules.md (the legal
    action set and the granularity contract). Each concern is an independently-evolvable lever; the
    engine's build_full_prompt still appends the JSON RESPONSE FORMAT + tool list on top."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`.

    ONE chart and one tape per episode: build_agent is called once, immediately before
    run_evaluation, so every ledger they keep is episode-local and nothing leaks between levels.
    memory.md supplies the wording of the injected block and Tape.TEMPLATE stands in when the file
    is absent or its placeholders do not match `Tape.fields()`.

    NB `evaluator.game` does NOT exist yet here — run_evaluation builds it after this call — so
    nothing may touch engine state at construction time, and nothing does.
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import action, vlm

    evaluator.current_prompt = compose_system_prompt()
    try:
        budget = int(getattr(evaluator, "max_steps", 100) or 100)
    except (TypeError, ValueError):
        budget = 100
    template = _read("memory.md")
    body = template.split("-->")[-1].strip() if "-->" in template else template.strip()
    if "{ahead_line}" not in body or "{step}" not in body:
        body = ""                                      # the file is documentation, not a template
    tape = action.Tape(action.Chart(), max_steps=max(1, budget), template=body)

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data, tape)

    return agent
