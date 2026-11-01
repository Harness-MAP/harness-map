"""Rich MAZE harness — the agent-factory injection point over the SAME SpatialWorld engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func`. It REUSES the evaluator's already-built engine handles and returns an engine
`Action` each step, so scoring is byte-compatible with the prompt-only baseline. This candidate's
change is entirely INSIDE the components it inherited: the component graph is still the anchor's
12 nodes and 4 edges, and `tools/vlm.py:decide` now measures the cell it stands in off the frame.

Module map and lever table: NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    """One prior file as text, or "" when the candidate does not ship it."""
    leaf = HERE / name
    if leaf.exists():
        return leaf.read_text(errors="replace")
    return ""


def compose_system_prompt() -> str:
    """Recompose the maze system prompt from the .md priors: AGENT.md + skills/*.md + rules.md,
    in that order. The engine's build_full_prompt still appends the tool list and the JSON
    RESPONSE FORMAT on top."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [_read("skills/" + leaf.name) for leaf in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(part.strip() for part in parts if part.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Installs the composed
    prior as `current_prompt`, which build_full_prompt freezes into `conversation_history[0]` —
    the one slot the engine carries forward verbatim, and so the only per-turn text channel."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                  # so `import tools...` resolves to THIS harness
    for name in [mod for mod in list(sys.modules)
                 if mod == "tools" or mod.startswith("tools.")]:
        sys.modules.pop(name, None)                    # defensive: never inherit a sibling's tools/
    from tools import vlm

    evaluator.current_prompt = compose_system_prompt()

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data)

    return agent
