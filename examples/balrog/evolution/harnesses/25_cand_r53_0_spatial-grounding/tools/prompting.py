"""prompting.py — assemble the per-step message list, plus the ranked MOVE MENU block.

LEVER (code + the prose files it composes). The naive transcript is kept. After it comes a
best-first menu of compass moves that code has ranked (harness.Ledger.menu). The model gets an
ordered choice to make instead of a table it has to rank itself.
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
GLYPH_NAME = {".": "floor", "#": "corridor or bars", ">": "STAIRS DOWN (goal)", "<": "stairs up",
              "+": "door", "|": "wall or open door", "-": "wall or open door", "}": "lava/water",
              "`": "boulder", "0": "boulder", " ": "unseen", "@": "you"}


def compose_priors(root: Path = ROOT) -> str:
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


def glyph_name(g: str) -> str:
    if g.isalpha():
        return f"CREATURE {g} (moving there attacks it)"
    return GLYPH_NAME.get(g, "item " + g)


def render_menu(menu) -> str:
    """menu: ranked list of (direction, glyph, visits, just_left)."""
    if not menu:
        return ""
    out = ["MOVE MENU (best first, ranked by code from your own moves on this level):"]
    for i, (d, g, v, back) in enumerate(menu, 1):
        tag = ", just left" if back else ""
        out.append(f"{i}. {d} — {glyph_name(g)}, stood {v}{tag}")
    return "\n".join(out)


def build_messages(prompt_builder, menu=None, priors: str | None = None):
    messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        block = priors if priors is not None else compose_priors()
        table = render_menu(menu or [])
        tail = "\n\n".join(x for x in (block, table, NAIVE_INSTRUCTION) if x)
        messages[-1].content += "\n\n" + tail
    return messages
