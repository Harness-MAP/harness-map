"""prompting.py — assemble the per-step message list, plus the NEIGHBOUR LEDGER block.

LEVER (code + the prose files it composes). The naive transcript is kept; after it comes a
ledger table: one row per compass direction with what lies there (glyph from the map), how often the
agent has stood there, whether a move that way already failed from this square, and whether the
prose names a hazard there. The model reads facts, not a raw screen.
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
GLYPH_NAME = {".": "floor", "#": "corridor", ">": "stairs down", "<": "stairs up", "+": "door",
              "|": "wall", "-": "wall", "}": "lava/water", "`": "boulder", "0": "boulder",
              " ": "unseen"}


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


def render_ledger(rows) -> str:
    """rows: list of (direction, glyph, visits, failed, hazard) -> fixed-width table."""
    if not rows:
        return ""
    out = ["NEIGHBOUR LEDGER (from your own past moves on this level):",
           "direction | sees | times stood there | move failed here | hazard"]
    for d, g, v, f, h in rows:
        out.append(f"{d} | {GLYPH_NAME.get(g, 'creature/item ' + g)} | {v} | "
                   f"{'yes' if f else 'no'} | {'DEADLY' if h else '-'}")
    return "\n".join(out)


def build_messages(prompt_builder, ledger_rows=None, priors: str | None = None):
    messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        block = priors if priors is not None else compose_priors()
        table = render_ledger(ledger_rows or [])
        tail = "\n\n".join(x for x in (block, table, NAIVE_INSTRUCTION) if x)
        messages[-1].content += "\n\n" + tail
    return messages
