"""Rich MAZE harness (change #5) — the agent-factory injection point over the SAME SpatialWorld
engine. `build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs
as `ai_model_func` (config.py::_agent_func / _apply_harness). It REUSES the evaluator's already-built
`self.game` / `self.input_source` / `self.engine` / `self.client` / `capture_game_screenshot` and
returns an engine `Action` each step, so scoring (the maze `success` field) is byte-compatible with
the prompt-only baseline.

The per-step decision is a CLOSED LOOP rather than a pass-through: `build_agent` creates one
`tools.plan.ActionPlanner` per episode and threads it through `tools.vlm.decide`, which asks the
frozen model every step, states the planner's self-derived working memory back to it, and displaces
only the proposals the frame itself has already settled. The planner reads nothing but
`frame_data.image` — the frame the agent is shown — and `tools.sight.Scene` now reads that frame as
GEOMETRY and not only as colour: the centre column is a RANGEFINDER (what stands dead ahead and how
many cells away) and the flanks are a MAP STRIP (which of the cells beside the corridor are open).
Both are exact inversions of the renderer's own polygons, scored at 1820/1820 against the engine's
layout, and they answer the two questions the clock turns on — how far the body can walk in one
decision, and which way is open — that neither the fingerprint nor the colour census could.

Module map + lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def _int_or(value, default: int) -> int:
    """`value` as a positive int, else `default` (the step budget is display-only, never a gate)."""
    try:
        return int(value) if int(value) > 0 else default
    except (TypeError, ValueError):
        return default


def compose_system_prompt() -> str:
    """Recompose the maze system prompt from the .md priors. Concatenating AGENT.md (role +
    perception) + skills/*.md (navigation strategy) + rules.md (legal actions + granularity)
    reproduces the shipped system_prompt.txt content IN ORDER, so this anchor scores like the
    prompt-only baseline; the split just makes each concern an independently-evolvable lever.
    The engine's build_full_prompt still appends the JSON RESPONSE FORMAT + tool list on top."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Reuses the evaluator's
    engine handles and owns only the per-step decision. Installs the composed prior as the
    evaluator's `current_prompt` (the first VLM turn reads it via build_full_prompt)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import plan, vlm

    evaluator.current_prompt = compose_system_prompt()

    # ONE planner per episode: build_agent is called once, immediately before run_evaluation, so the
    # ledgers it keeps are episode-local and nothing leaks between levels. memory.md supplies the
    # wording of the per-turn working-memory block; plan.DEFAULT_TEMPLATE stands in if it is absent.
    planner = plan.ActionPlanner(
        max_steps=_int_or(getattr(evaluator, "max_steps", None), 100),
        template=_read("memory.md"),
    )

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data, planner)

    return agent
