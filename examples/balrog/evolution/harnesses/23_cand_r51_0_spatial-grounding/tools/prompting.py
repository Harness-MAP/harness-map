"""prompting.py — assemble the per-step message list, plus the grounded NEIGHBOUR TABLE.

LEVER (code + the prose files it composes). The naive transcript is kept; after it come the prose
priors and a table: one row per compass direction with the glyph, what the GAME TEXT names there
(the prose noun outranks the glyph), how often the agent stood there, whether a move that way
already failed from this square, and whether it is the square just left.
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
GLYPH_NAME = {".": "floor", "#": "corridor or bars", ">": "stairs down", "<": "stairs up",
              "+": "door", "|": "wall", "-": "wall or open door", "}": "lava/water",
              "`": "boulder", "0": "boulder", " ": "unseen"}
KIND_NAME = {"deadly": "DEADLY", "blocked": "BLOCKED", "creature": "creature/item - move = attack",
             "open": "passable", "": "-"}


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


def render_table(rows) -> str:
    if not rows:
        return ""
    out = ["NEIGHBOUR TABLE (kept by code from this level's frames):",
           "direction | map glyph | game text says | times stood there | move failed here | square you just left"]
    for d, g, v, f, kind, back in rows:
        out.append(f"{d} | {GLYPH_NAME.get(g, 'creature/item ' + g)} | {KIND_NAME[kind]} | {v} | "
                   f"{'yes' if f else 'no'} | {'yes' if back else 'no'}")
    return "\n".join(out)


def build_messages(prompt_builder, rows=None, priors: str | None = None):
    messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        block = priors if priors is not None else compose_priors()
        table = render_table(rows or [])
        tail = "\n\n".join(x for x in (block, table, NAIVE_INSTRUCTION) if x)
        messages[-1].content += "\n\n" + tail
    return messages
