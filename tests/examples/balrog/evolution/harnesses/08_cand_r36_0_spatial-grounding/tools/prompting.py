"""prompting.py - compose the prose priors (AGENT.md, rules.md, skills/*.md) into the user turn."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NAIVE_INSTRUCTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)


def compose_priors(root=ROOT):
    parts = []
    for name in ["AGENT.md", "rules.md"]:
        p = root / name
        if p.exists() and p.read_text().strip():
            parts.append(p.read_text().strip())
    sk = root / "skills"
    if sk.is_dir():
        parts += [p.read_text().strip() for p in sorted(sk.glob("*.md")) if p.read_text().strip()]
    return "\n\n---\n\n".join(parts)


def build_messages(prompt_builder, priors, banned=()):
    messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        tail = NAIVE_INSTRUCTION
        if priors:
            tail = priors + "\n\n" + tail
        if banned:
            tail += "\nMoves that are closed or deadly from this square: " + ", ".join(sorted(banned)) + "."
        messages[-1].content += "\n\n" + tail
    return messages
