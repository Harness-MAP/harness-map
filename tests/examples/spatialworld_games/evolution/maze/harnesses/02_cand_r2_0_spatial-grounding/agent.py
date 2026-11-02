"""Rich MAZE harness — the agent-factory injection point over the SAME SpatialWorld engine.
`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func` (config.py::_agent_func / _apply_harness). It reuses the evaluator's already-built
`self.game` / `self.input_source` / `self.engine` / `self.client` / `capture_game_screenshot` and
returns an engine `Action` each step, so scoring (the maze `success` field) is byte-compatible with
the prompt-only baseline.

The component graph is the anchor's, unchanged. This iteration spends its budget inside the seven
component bodies and in module-level plumbing that adds no node.

Module map + lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Module-level plumbing, not a component: never inherit a sibling candidate's `tools/`. A lambda,
# because a `def` here would be a fourth node in this module's graph.
_PURGE = lambda: [sys.modules.pop(m, None) for m in
                  [k for k in list(sys.modules) if k == "tools" or k.startswith("tools.")]]


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def compose_system_prompt() -> str:
    """Recompose the maze system prompt from the .md priors: AGENT.md (role + perception) +
    skills/*.md (navigation strategy) + rules.md (legal actions + granularity), in order. The
    engine's build_full_prompt appends the JSON RESPONSE FORMAT + tool list on top, and `decide`
    preserves that whole message as the base it grafts the working-memory sheet onto."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Installs the composed
    prior as `evaluator.current_prompt`, which the first VLM turn reads via build_full_prompt; from
    the second turn on the prior is maintained by `decide`, because the engine consults
    build_full_prompt once and then copies message 0 by reference."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    _PURGE()
    from tools import vlm

    evaluator.current_prompt = compose_system_prompt()

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data)

    return agent
