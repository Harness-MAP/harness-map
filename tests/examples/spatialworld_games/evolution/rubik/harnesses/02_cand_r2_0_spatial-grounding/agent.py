"""Rich RUBIK harness — the agent-factory injection point over the SpatialWorld rubik engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func` (config.py::_agent_func / _apply_harness). It REUSES the evaluator's already-built
`self.game` / `self.input_source` / `self.engine` / `self.client` / `capture_game_screenshot` and
returns an engine `Action` each step, so scoring (the rubik `cube_solved` field) stays byte
compatible with the prompt-only baseline: one decision, one engine action, one model call.

The prose priors are composed here and the decision itself lives in `tools/vlm.py`, which runs the
proposed action word past the near-ball frontier router in `tools/action.py` before anything is
played: the router keeps an exact cube state over the harness's own moves and honours any pick
that shortens the walk's distance to the nearest configuration near the episode's start that has
not been stood on yet.

Module map + lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def compose_system_prompt() -> str:
    """Recompose the rubik system prompt from the .md priors, in order: AGENT.md (the role and
    the six face letters, unchanged from the anchor), then every skills/*.md in sorted order
    (here the near-ball search contract: three unconditional rules and how to read the menu),
    then rules.md (the move taxonomy, also unchanged). The engine's build_full_prompt appends the AVAILABLE ACTIONS list and the JSON
    RESPONSE FORMAT on top, and `tools/vlm.py` splices the per-turn HARNESS STATE brief after
    that. Each concern stays an independently-evolvable prose lever."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Reuses the evaluator's
    engine handles and owns only the per-step decision. Installs the composed prior as the
    evaluator's `current_prompt` (the first VLM turn reads it via build_full_prompt); the ledger
    the brief is rendered from is created lazily by `tools/vlm.py` and cached on the evaluator,
    which is one object per episode because eval_runner runs one episode per process."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import vlm

    evaluator.current_prompt = compose_system_prompt()

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data)

    return agent
