"""action_io.py — project the model's reply onto the LEGAL action names.

LEVER (code). BALROG coerces any unlisted string to `north`; here the reply is matched whole-word
against the names parsed from the system prompt (earliest mention wins, longest name on a tie, so
`northeast` is not read as `north`). With no legal list, the stock regex cleanup is kept.
"""
from __future__ import annotations

import re


def legal_actions(system_prompt: str) -> list:
    names = []
    for line in (system_prompt or "").split("In a moment")[0].splitlines():
        m = re.match(r"^\s*([a-z][a-z ]*?):\s", line)
        if m:
            names.append(m.group(1).strip())
    return names


def mentions(text: str, legal: list) -> list:
    t = (text or "").lower()
    hits = []
    for name in legal:
        for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?![a-z])", t):
            hits.append((m.start(), -len(name), name))
    return [n for _, _, n in sorted(hits)]


def snap(text: str, legal: list) -> str:
    if legal:
        ms = mentions(text, legal)
        if ms:
            return ms[0]
    return re.sub(r"[^a-zA-Z\s:]", "", text or "").strip()


def extract_action(text: str, legal=None) -> str:
    return snap(text, legal or [])
