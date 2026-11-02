"""Rich BLOCK3D harness — the agent-factory injection point over the SAME SpatialWorld engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as its
`openai_ai_model`. It REUSES the evaluator's already-built `self.input_source` / `self.engine` /
`self.client` / `capture_game_screenshot` and returns an engine `Action` each step, so scoring
(the block3d `won` field) stays byte-compatible with the prompt-only baseline.

This revision replaces the anchor's two-line pass-through with a RESIDUAL-PROJECTION-DEBT loop.
The three orthographic panels are fitted and read back out of the rendered frame, the cursor is
read off the highlight ink and cross-checked against a booked replay of this harness's own
returned actions, and the cell to work toward is re-derived every single decision from the panel
squares that the booked blocks do not yet explain. The frozen model is still asked once per
decision and still owns the commit-or-walk-on judgement wherever the panels leave it open; the
harness owns routing, refuses the steps the picture proves cannot move the cursor, and books
every toggle so the grid is never un-built by accident.

Nothing but the rendered frame and this harness's own returned actions is ever read.

Module map + lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def compose_system_prompt() -> str:
    """Recompose the block3d system prompt from the .md priors: AGENT.md (role + the perception
    facts) + skills/*.md (build strategy, then how to use the grounded-state block) + rules.md
    (legal actions + control mapping + observation tips), in that order. Each concern stays an
    independently-evolvable lever; the engine's build_full_prompt still appends the JSON RESPONSE
    FORMAT + the tool list on top, and tools/vlm.py appends the per-turn grounded state after it."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Reuses the evaluator's
    engine handles and owns only the per-step decision. Installs the composed prior as the
    evaluator's `current_prompt` (the first VLM turn reads it via build_full_prompt) and builds the
    episode-scoped arbiter that carries the panel reading and the ledger across the decisions."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import vlm

    evaluator.current_prompt = compose_system_prompt()

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data)

    return agent
