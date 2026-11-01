"""Rich MAZE harness — the agent-factory injection point over the SAME SpatialWorld engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`ai_model_func` (config.py::_agent_func / _apply_harness). It REUSES the evaluator's already-built
`self.game` / `self.input_source` / `self.engine` / `self.client` / `capture_game_screenshot`, and
returns an engine `Action` each step, so scoring (the maze `success` field) stays byte-compatible
with the prompt-only baseline.

What this harness adds over the anchor it descends from is ONE mechanism, in `tools/plan.py`: the
renderer's projection, inverted. The 800x600 first-person view draws the cell d cells ahead as a
rectangle of height 400/d about the vanishing point, and draws the cells immediately to either
side in x bands that nothing else can reach, so three single columns of the frame the model is
already shown (x = 400, 50, 750) yield the distance to the nearest obstacle as an INTEGER cell
count, that obstacle's identity, and the passability of both flanks — exactly, at every pose, with
nothing to threshold. `build_agent` opens ONE `SightedPlanner` per episode so those readings can
accumulate into a dead-reckoned per-floor lattice, states them back to the model through the only
per-turn text channel the engine has, and lets the lattice replace a decision only where the
pixels prove that decision cannot do anything at all.

Module map + the measured lever table: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def compose_system_prompt() -> str:
    """Recompose the maze system prompt from the .md priors: AGENT.md (role + the perception facts
    this build actually renders) + skills/*.md (navigation strategy) + rules.md (legal actions and
    the granularity contract), in that order. The engine's build_full_prompt still appends the JSON
    RESPONSE FORMAT + tool list on top, so the response contract is untouched. The split is what
    makes each concern an independently-evolvable lever; `system_prompt.txt` is a stale reference
    copy that nothing reads."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def memory_template() -> str:
    """The per-turn working-memory block, from memory.md when its placeholders match the fields
    the planner renders, otherwise the copy that lives inside tools/plan.py."""
    from tools import plan
    text = _read("memory.md")
    body = text.split("<!-- TEMPLATE -->", 1)[-1].strip() if "<!-- TEMPLATE -->" in text else ""
    if not body:
        return plan.DEFAULT_TEMPLATE
    try:
        body.format(**{k: "" for k in ("sight_line", "reach_line", "neighbours", "underfoot",
                                       "last_line", "mapped", "floor_line", "advice")})
    except Exception:
        return plan.DEFAULT_TEMPLATE
    return body


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`. Reuses the evaluator's
    engine handles and owns only the per-step decision. Installs the composed prior as the
    evaluator's `current_prompt` (the first VLM turn reads it via build_full_prompt) and opens one
    `SightedPlanner` for the whole episode."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                 # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import plan, vlm

    evaluator.current_prompt = compose_system_prompt()
    template = memory_template()
    planner = plan.SightedPlanner()

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data, planner=planner, template=template)

    agent.planner = planner                            # offline rigs read the counters off this
    return agent
