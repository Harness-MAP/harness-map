"""prompting.py — assemble the per-step message list (asked only when no stage answers).

LEVER (code + the prose files it composes). The naive transcript is kept; after it come the
composed priors and the "Checks on this frame" notes that pipeline stages injected.
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


def build_messages(prompt_builder, priors: str | None = None, notes=None):
    """NaiveAgent's assembly + the priors block + the pipeline stages' per-step notes."""
    messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        block = priors if priors is not None else compose_priors()
        if notes:
            block = (block + "\n\n" if block else "") + "Checks on this frame:\n" + "\n".join(
                "- " + n for n in notes)
        tail = "\n\n".join(x for x in (block, NAIVE_INSTRUCTION) if x)
        messages[-1].content += "\n\n" + tail
    return messages
