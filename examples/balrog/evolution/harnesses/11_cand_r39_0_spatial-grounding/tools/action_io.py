"""action_io.py — project the model's reply onto the LEGAL set, then veto against the chart.

LEVER (code). BALROG coerces any unlisted string to `north`. Here the reply is matched whole-word
against the action names parsed from the system prompt, and the earliest mention wins. A move the
chart marks deadly, or one already blocked from this square, is vetoed and the next mention is
taken. If nothing survives, the chart's BFS first hop (the fallback) is used.
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


def extract_action(text: str, legal=None, vetoed=(), fallback=None) -> str:
    if not legal:
        return re.sub(r"[^a-zA-Z\s:]", "", text or "")
    for name in mentions(text, legal):
        if name not in vetoed:
            return name
    if fallback:
        return fallback
    ms = mentions(text, legal)
    return ms[0] if ms else re.sub(r"[^a-zA-Z\s:]", "", text or "")
