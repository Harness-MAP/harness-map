"""reflex.py - model-free rungs that fire before the planner: modal prompts, stairs, adjacent foe.

Every answer is a verb from BALROG's MiniHack vocabulary. `esc` and `y` are NOT in it: the `y` key
is the verb `northwest`, and `more` closes both `--More--` and an item prompt ("Never mind.").
"""
from __future__ import annotations

import re

from perception import STEP, step, glyph

CREATURE = re.compile(r"[a-zA-Z&;:']")
TAME = {"@"}


def modal(fr, legal):
    """Answer an open prompt, or None when no prompt owns the keyboard."""
    if not fr["prompt"]:
        return None
    msg = fr["message"]
    low = msg.lower()
    if "[yn" in msg and "attack" in low and "northwest" in legal:
        return "northwest", "reflex:confirm-attack"
    if "more" in legal:
        return "more", "reflex:close-prompt"
    return None


def adjacent_foe(fr, moves, bad, blocked):
    pos = fr["pos"]
    for d in moves:
        if d in bad or (pos, d) in blocked:
            continue
        ch = glyph(fr, step(pos, d))
        if CREATURE.fullmatch(ch or " ") and ch not in TAME:
            return d
    return None


def lethal_moves(fr):
    bad = set()
    for noun, dirs in fr["adjacent"].items():
        if "lava" in noun or "water" in noun:
            bad |= dirs
    pos = fr["pos"]
    if pos:
        for d in STEP:
            if glyph(fr, step(pos, d)) == "}":
                bad.add(d)
    return bad
