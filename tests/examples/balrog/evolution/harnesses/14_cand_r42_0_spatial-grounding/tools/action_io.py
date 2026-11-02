"""action_io.py — turn the model's reply into the action string the env receives.

LEVER (code). Reproduces BALROG NaiveAgent's `_extract_final_answer` exactly: keep only letters,
whitespace and colons. It is deliberately NOT robust — the stock agent is not, and making it robust
here would be an unearned improvement credited to the search.
"""
from __future__ import annotations

import re


def extract_action(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s:]", "", text or "")


def legal_actions(system_prompt: str) -> list:
    """The legal vocabulary: one `name: description` line per action in BALROG's system prompt."""
    acts = []
    for line in (system_prompt or "").splitlines():
        m = re.match(r"^\s*([a-z][a-z ]{0,20}?):\s", line)
        if m and m.group(1) not in acts:
            acts.append(m.group(1))
    return acts


def snap(text: str, legal: list) -> str:
    """Stock filter, then the LAST whole-word legal mention (longest name first on a tie)."""
    clean = extract_action(text).strip().lower()
    if not legal or clean in legal:
        return clean
    best, at = None, -1
    for a in sorted(legal, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(a)}\b", clean):
            if m.start() > at:
                best, at = a, m.start()
    if best:
        return best
    # 16-point compounds (`southsoutheast`) have no word boundary: take the 8-point suffix/prefix.
    for a in sorted(legal, key=len, reverse=True):
        if clean.endswith(a) or clean.startswith(a):
            return a
    return clean
