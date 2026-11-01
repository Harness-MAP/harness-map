"""Rich MAZE harness — the agent-factory injection point over the SAME SpatialWorld engine.
`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func` (eval_runner::run_one). It REUSES the evaluator's already-built `self.game` /
`self.input_source` / `self.engine` / `self.client` / `capture_game_screenshot` and returns an
engine `Action` each step, so scoring (the maze `success` field) stays byte-compatible with the
prompt-only baseline.

This iteration adds SPATIAL GROUNDING: a per-episode `tools.grounding.Grounding` instance is
created here and threaded into every decision, so the harness can tell the frozen agent what its
previous action actually did and what the frame actually shows. See NOTES.md for the measurements
that motivated it (the parent spent 48-100% of every failed episode on actions that left the
frame byte-identical).

Module map + lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def compose_system_prompt() -> str:
    """Recompose the maze system prompt from the .md priors: AGENT.md (role + the perception
    facts measured off this build) + skills/*.md (navigation strategy) + rules.md (legal actions
    + granularity). The engine's build_full_prompt still appends the JSON RESPONSE FORMAT + tool
    list on top of whatever this returns, and that combined text is what gets stashed as the
    pristine base the per-step grounding block is appended to."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Reuses the
    evaluator's engine handles and owns only the per-step decision. Installs the composed prior
    as the evaluator's `current_prompt` (the first VLM turn reads it via build_full_prompt) and
    opens one fresh grounding state for this episode (one episode per process)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import vlm
    from tools.grounding import Grounding

    evaluator.current_prompt = compose_system_prompt()
    grounding = Grounding(template=_read("memory.md"))

    async def agent(frame_data):
        try:
            return await vlm.decide(evaluator, frame_data, grounding)
        except Exception:                              # noqa: BLE001
            # grounding must never cost an episode: fall back to the pristine per-step VLM fn,
            # and tell the tool its model of the body is no longer trustworthy
            grounding.desync()
            step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
            return await step(frame_data)

    return agent
