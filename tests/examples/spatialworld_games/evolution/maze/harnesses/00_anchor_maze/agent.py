"""Rich MAZE harness (change #4) — the agent-factory injection point over the SAME SpatialWorld
engine. `build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs
as `ai_model_func` (config.py::_agent_func / _apply_harness). It REUSES the evaluator's already-built
`self.game` / `self.input_source` / `self.engine` / `self.client` / `capture_game_screenshot` and
returns an engine `Action` each step, so scoring (the maze `success` field) is byte-compatible with
the prompt-only baseline. This anchor deliberately reproduces the baseline behaviour; it exists to
prove the rich-harness pattern and to give the agentic proposer a real multi-file surface to evolve.

Module map + lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


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
    from tools import vlm

    evaluator.current_prompt = compose_system_prompt()

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data)

    return agent
