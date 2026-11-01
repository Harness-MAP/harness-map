"""action_io.py — legal-set extraction, modal-prompt handling, and reply ARBITRATION.

The model no longer writes straight to the env. Its reply is parsed against the legal vocabulary
(whole words, longest first), then checked against the grounding's vetoes. A vetoed or unparseable
reply is replaced by the grounding's recommended move — never by BALROG's silent `north` coercion.
"""
from __future__ import annotations

import re


def legal_actions(system_prompt: str) -> list:
    """Parse `name: description` lines from BALROG's system prompt."""
    acts = []
    for line in (system_prompt or "").splitlines():
        m = re.match(r"^\s*([a-z][a-z ]{0,20}?):\s", line)
        if m and m.group(1) not in acts:
            acts.append(m.group(1))
    return acts


def extract_action(text: str, legal: list | None = None) -> str:
    """Stock filter, then snap to a legal action by whole-word match (last mention wins)."""
    clean = re.sub(r"[^a-zA-Z\s:]", "", text or "").strip().lower()
    if not legal:
        return clean
    if clean in legal:
        return clean
    best, best_pos = None, -1
    for a in sorted(legal, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(a)}\b", clean):
            if m.start() > best_pos:
                best, best_pos = a, m.start()
    return best or clean


def modal_response(message: str, legal: list):
    """An open game prompt must be answered before anything else can happen."""
    m = (message or "").lower()
    if not m:
        return None
    if "--more--" in m and "more" in legal:
        return "more"
    if "[yn" in m or "what do you want to" in m or "in what direction" in m:
        for a in ("esc", "more"):
            if a in legal:
                return a
    if "there is a staircase down here" in m and "down" in legal:
        return "down"
    return None


def arbitrate(reply: str, grounding, legal: list):
    """Return (action, source)."""
    allowed = {o.direction for o in grounding.allowed}
    vetoed = {o.direction for o in grounding.options if o.veto}
    a = extract_action(reply, legal)
    if a in vetoed:
        return (grounding.recommended or a), "override-veto"
    if a in allowed or (legal and a in legal):
        return a, "model"
    if grounding.recommended:
        return grounding.recommended, "fallback-route"
    return a, "model-raw"
