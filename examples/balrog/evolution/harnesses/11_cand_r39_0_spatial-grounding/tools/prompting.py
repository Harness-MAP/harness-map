"""prompting.py — assemble the per-step message list.

LEVER (code + the prose files it composes).

BEHAVIOURALLY NAIVE BY CONTENT, NOT BY MISSING MECHANISM. The prior-composition mechanism is
present and live — `compose_priors()` reads AGENT.md, rules.md and skills/*.md and appends whatever
they contain. Those files SHIP EMPTY, so the composed block is the empty string and the assembled
prompt is byte-identical to BALROG's NaiveAgent.

That distinction matters. Omitting the mechanism would make the .md files inert, and a proposer
that carefully rewrote AGENT.md would produce a candidate that scores exactly like its parent — a
NOOP the search cannot tell from a bad idea. Shipping the mechanism with empty content means the
anchor adds nothing, while an edit to any prose lever changes behaviour immediately.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NAIVE_INSTRUCTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)

PROSE_ORDER = ["AGENT.md", "rules.md", "memory.md"]
SKILL_DIR = "skills"


def compose_priors(root: Path = ROOT) -> str:
    """Concatenate the prose levers. Empty in the anchor -> returns ''."""
    parts = []
    for name in PROSE_ORDER:
        p = root / name
        if p.exists() and p.read_text().strip():
            parts.append(p.read_text().strip())
    sk = root / SKILL_DIR
    if sk.is_dir():
        for p in sorted(sk.glob("*.md")):
            if p.read_text().strip():
                parts.append(p.read_text().strip())
    return "\n\n---\n\n".join(parts)


GLYPH_NAME = {".": "floor", "#": "corridor/bars", ">": "stairs down", "<": "stairs up", "+": "door",
              "|": "wall", "-": "wall", "}": "lava/water", "`": "boulder", "0": "boulder",
              " ": "unseen"}


def render_chart(neigh, route) -> str:
    """neigh: [(dir, glyph, visits, blocked, deadly)]; route: (hop, kind, dist)."""
    if not neigh:
        return ""
    hop, kind, dist = route
    out = ["SPATIAL CHART (kept by code from your own observations on this level):"]
    if hop:
        out.append(f"ROUTE: nearest {kind} is {dist} step(s) away; first move toward it: {hop}")
    else:
        out.append("ROUTE: no charted path to stairs or unexplored ground")
    out.append("direction | sees | times stood there | blocked from here | hazard")
    for d, g, v, b, h in neigh:
        out.append(f"{d} | {GLYPH_NAME.get(g, 'creature/item ' + g)} | {v} | "
                   f"{'yes' if b else 'no'} | {'DEADLY' if h else '-'}")
    return "\n".join(out)


def build_messages(prompt_builder, chart_block: str = "", priors: str | None = None):
    """NaiveAgent's assembly + priors + the code-kept chart block."""
    messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        block = priors if priors is not None else compose_priors()
        tail = "\n\n".join(x for x in (block, chart_block, NAIVE_INSTRUCTION) if x)
        messages[-1].content += "\n\n" + tail
    return messages
