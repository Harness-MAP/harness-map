"""Rich MAZE harness — the agent-factory injection point over the SAME SpatialWorld engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func`. It REUSES the evaluator's already-built `self.game` / `self.input_source` /
`self.engine` / `self.client` / `capture_game_screenshot` and returns an engine `Action` each step,
so scoring (the maze `success` field) stays byte-compatible with the prompt-only baseline.

The per-step decision is a CLOSED LOOP rather than a pass-through. `build_agent` opens ONE
`tools.plan.Pilot` for the episode and threads it through `tools.vlm.decide`, which reads the frame
as geometry, settles the previous decision against it, states the measurements back to the frozen
model through the system turn, asks the model, and then honours its answer wherever that answer is
free against the route the chart has paid for. The pilot reads nothing but `frame_data.image` — the
frame the agent itself is shown — and every pose it believes in follows from actions this harness
emitted.

Module map + lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def _budget(value, default: int = 100) -> int:
    """`value` as a positive int, else `default`. The step budget is stated to the agent and used to
    price nothing: the decision-optimal route is well inside it on every level, so it is display
    only and never a gate."""
    try:
        return int(value) if int(value) > 0 else default
    except (TypeError, ValueError):
        return default


def compose_system_prompt() -> str:
    """Recompose the maze system prompt from the .md priors: AGENT.md (role + the perception facts
    this build actually has) + skills/*.md (the navigation priority order) + rules.md (the legal
    action set, the granularity contract and the real climb precondition), in that order. The
    engine's build_full_prompt still appends the JSON RESPONSE FORMAT + tool list on top."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`.

    Reuses the evaluator's engine handles and owns only the per-step decision. `build_agent` is
    called once, immediately before the episode runs, so the pilot it opens is episode-local and
    nothing it learns leaks between levels. Nothing here may touch engine state: the evaluator
    builds `self.game` AFTER this call returns."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import plan, vlm

    evaluator.current_prompt = compose_system_prompt()
    pilot = plan.Pilot(budget=_budget(getattr(evaluator, "max_steps", None)),
                       template=_read("memory.md"))

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data, pilot)

    return agent
