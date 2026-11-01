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

PROSE_ORDER = ["AGENT.md", "rules.md"]
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


def build_messages(prompt_builder, priors: str | None = None, grounding: str = "", messages=None):
    """NaiveAgent's assembly + the priors block + the harness-computed spatial grounding block."""
    if messages is None:
        messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        block = priors if priors is not None else compose_priors()
        parts = [p for p in (block, grounding, NAIVE_INSTRUCTION) if p]
        messages[-1].content += "\n\n" + "\n\n".join(parts)
    return messages
