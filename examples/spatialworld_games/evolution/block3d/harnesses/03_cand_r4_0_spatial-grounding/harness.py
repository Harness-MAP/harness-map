"""Rich BLOCK3D harness — the agent-factory injection point over the SAME SpatialWorld engine.

`build_agent(evaluator)` returns the per-step decision fn the extended evaluator installs as
`evaluator.openai_ai_model`. It REUSES the evaluator's already-built `game` / `input_source` /
`engine` / `client` and its pristine per-step VLM fn (preserved as `evaluator.baseline_step`), and
returns an engine `Action` each step, so scoring (the block3d `won` field — exact set equality
between the placed set and the target) stays byte-compatible with the prompt-only baseline.

WHAT THIS HARNESS DOES, in one paragraph. It MEASURES the frame's own ruler — the panel lattice is
fitted from the lattice lines rather than asserted — reads the three orthographic target panels
through it, and back-projects them into the envelope of every cell the picture admits. Separately
it CALIBRATES the viewport's projection from the agent's own moves: the cursor highlight on the
first frame is an unoccluded cube footprint, so it is a template, and one applied move per axis
measures the affine translation, after which the cursor is READ off every later frame by
footprint containment instead of being dead-reckoned and hoped for. The identity fence turns a
clamped move — a frozen picture — into an exact coordinate rather than a bookkeeping error. On top
of that measured state the planner lays the proof first, lifts the volume slice by slice, core
before rim, and honours the frozen model's own action on every turn it is free to.

Module map + lever table + the change: see NOTES.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read(name: str) -> str:
    p = HERE / name
    return p.read_text() if p.exists() else ""


def compose_system_prompt() -> str:
    """Recompose the system prompt from the .md priors: AGENT.md (role + the verified perception
    facts) + skills/*.md (one concern per file, in sorted order) + rules.md (legal actions and the
    output contract). The engine's `build_full_prompt` still appends the JSON RESPONSE FORMAT and
    the tool list on top. `system_prompt.txt` / `system_prompt.md` are read by NOTHING: the prompt
    is rebuilt here, so the `.md` priors are the only prose lever."""
    parts = [_read("AGENT.md")]
    skills = HERE / "skills"
    if skills.is_dir():
        parts += [f.read_text() for f in sorted(skills.glob("*.md"))]
    parts.append(_read("rules.md"))
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def build_agent(evaluator):
    """Factory: return the async per-step fn `agent(frame_data) -> Action`.

    Owns only the per-step decision. Installs the composed prior as `evaluator.current_prompt`
    (the runner captures it so the first VLM turn reads it via `build_full_prompt`)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))                  # so `import tools...` resolves to THIS harness
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                      # defensive: never inherit a sibling's tools/
    from tools import vlm                              # a package: its modules import each other

    evaluator.current_prompt = compose_system_prompt()

    async def agent(frame_data):
        return await vlm.decide(evaluator, frame_data)

    return agent
