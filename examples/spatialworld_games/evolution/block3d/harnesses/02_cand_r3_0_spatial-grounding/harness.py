"""Rich BLOCK3D harness — the agent-factory injection point over the SAME SpatialWorld engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func` (config.py::_agent_func / _apply_harness). It REUSES the evaluator's already-built
`self.game` / `self.input_source` / `self.engine` / `self.client` / `baseline_step` and returns an
engine `Action` each step, so scoring (the block3d `won` field) is byte-compatible with the prompt-only
baseline.

THIS CANDIDATE'S AXIS IS THE PLAN'S ARITHMETIC. The three target panels are not one 3-D puzzle: they
FACTOR BY HEIGHT. At a fixed height, FRONT fixes exactly which rows that layer must light and SIDE
exactly which columns, and TOP is the only constraint that couples layers at all. So each layer is an
independent minimum bipartite EDGE COVER, solved exactly by augmenting paths, and the union of those
covers plus one pass of TOP-debt repair is the SMALLEST structure the panels admit. tools/layers.py owns
that arithmetic; tools/serpent.py turns the result into a boustrophedon press tape whose every prefix is
a distinct candidate the engine tests for free; tools/vlm.py spends the presses, asking the frozen model
exactly once per decision whose answer could change the emitted key and staying silent otherwise.

Module map + lever table: see NOTES.md.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ASIDE = re.compile(r"<!--.*?-->", re.S)
_SKILLS = lambda room: [f.read_text() for f in sorted(room.glob("*.md"))]
_STACKED = lambda parts: "\n".join(p.strip() for p in parts if p.strip()) + "\n"
_MINE = lambda loaded: [m for m in loaded if m == "tools" or m.startswith("tools.")]


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def _template(name: str) -> str:
    """A prose file with its authoring comments stripped — what actually reaches the model."""
    return _ASIDE.sub("", _read(name)).strip()


def compose_system_prompt() -> str:
    """Recompose the block3d system prompt from the .md priors: AGENT.md (role + the perception facts
    this harness verified off the frame) + skills/*.md (the build strategy and the layer sweep) +
    rules.md (the legal action set, the control mapping and the one-action contract). The engine's
    build_full_prompt still appends the JSON RESPONSE FORMAT + tool list on top, and because
    `build_conversation_history` reuses the first system turn verbatim for every later call, this text is
    FIXED for the episode — the live state reaches the model through tools/vlm.py's per-turn brief."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += _SKILLS(skills)
    parts.append(_read("rules.md"))
    return _STACKED(parts)


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Owns only the per-step
    decision; installs the composed prior as `evaluator.current_prompt` and hands the live scratchpad
    TEMPLATE (memory.md) to the session that renders it each turn."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in _MINE(list(sys.modules)):
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import vlm

    evaluator.current_prompt = compose_system_prompt()
    vlm.session_for(evaluator).template = _template("memory.md")

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data)

    return agent
