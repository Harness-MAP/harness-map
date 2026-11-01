"""prompting.py — assemble the per-step message list, plus the stage NOTES block.

LEVER (code + the prose files it composes). The naive transcript is kept; after it come the
priors and the short NOTES the pipeline stages emitted this step (the route's next hop, blocked
directions).
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


def render_notes(notes) -> str:
    """Lines the pipeline stages want the model to see this step (route, blocked directions)."""
    if not notes:
        return ""
    return "HARNESS NOTES (from your own map and moves on this level):\n" + "\n".join(notes)


def build_messages(prompt_builder, notes=None, priors: str | None = None):
    messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        block = priors if priors is not None else compose_priors()
        tail = "\n\n".join(x for x in (block, render_notes(notes or []), NAIVE_INSTRUCTION) if x)
        messages[-1].content += "\n\n" + tail
    return messages
