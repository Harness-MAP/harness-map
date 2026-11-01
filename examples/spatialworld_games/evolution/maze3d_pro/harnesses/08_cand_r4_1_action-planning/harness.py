"""Rich MAZE harness — the agent-factory injection point over the SAME SpatialWorld engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func` (config.py::_agent_func / _apply_harness). It REUSES the evaluator's already-built
`self.game` / `self.input_source` / `self.engine` / `self.client` / `capture_game_screenshot` and
returns an engine `Action` each step, so scoring (the maze `success` field) stays byte-compatible
with the prompt-only baseline.

The per-step decision is a CLOSED, MEASURED loop rather than a pass-through. `build_agent` creates
one `tools.plan.ActionPlanner` per episode and threads it through `tools.vlm.decide`, which decodes
the frame exactly (the renderer's own fog/rect geometry, inverted), writes what it measures into a
dead-reckoned occupancy map, states those facts back to the frozen model in its system turn, asks
the model every step, and re-aims only the proposals the measurement proves cannot do anything. The
planner reads nothing but `frame_data.image` — the very frame the agent is shown.

Module map + lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def _positive_int(value, default: int) -> int:
    """`value` as a positive int, else `default`. The step budget is display-only, never a gate."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def compose_system_prompt() -> str:
    """Recompose the maze system prompt from the .md priors: AGENT.md (role + the perception facts
    this build actually renders) + skills/*.md (the navigation priority order) + rules.md (the legal
    action set, the granularity contract and the climb precondition). Each concern is an
    independently-evolvable lever; the engine's build_full_prompt still appends the JSON RESPONSE
    FORMAT + tool list on top of whatever this returns."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Reuses the evaluator's
    engine handles and owns only the per-step decision. Installs the composed prior as the
    evaluator's `current_prompt` (the first VLM turn reads it via build_full_prompt).

    NB `evaluator.game` does NOT exist yet here — run_evaluation builds it after this call — so
    nothing in the planner may touch engine state at construction time, and nothing does."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import plan, vlm

    evaluator.current_prompt = compose_system_prompt()

    # ONE planner per episode: build_agent is called once, immediately before run_evaluation, so
    # every ledger it keeps is episode-local and nothing leaks between levels. memory.md supplies
    # the wording of the injected block; plan.DEFAULT_TEMPLATE stands in when it is absent or its
    # placeholders do not match plan.ActionPlanner.fields().
    planner = plan.ActionPlanner(
        max_steps=_positive_int(getattr(evaluator, "max_steps", None), 100),
        template=_read("memory.md"),
    )

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data, planner)

    return agent
