"""action_io.py - legal-set extraction and reply projection. NEVER returns an unlisted string."""
from __future__ import annotations

import re

from perception import ORDER, step


def legal_actions(system_prompt):
    names = []
    for line in (system_prompt or "").split("In a moment")[0].splitlines():
        m = re.match(r"^\s*([a-z][a-z ]*?):\s", line)
        if m:
            names.append(m.group(1).strip())
    return names or list(ORDER)


def decide_action(reply, legal, banned=()):
    """Whole-word projection of the reply onto the legal vocabulary (first mention wins)."""
    text = (reply or "").lower()
    hits = []
    for name in legal:
        for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?![a-z])", text):
            hits.append((m.start(), -len(name), name))
    for _, _, name in sorted(hits):
        if name not in banned:
            return name
    return None


def least_visited(pos, legal, banned, visits):
    safe = [d for d in ORDER if d in legal and d not in banned]
    if not safe or pos is None:
        return None
    return min(safe, key=lambda d: (visits.get(step(pos, d), 0), ORDER.index(d)))


def arbitrate(reply, legal, banned, pos, visits, foe=None):
    """Return (action, source)."""
    pick = decide_action(reply, legal, banned)
    if pick is not None:
        return pick, "model"
    low = (reply or "").lower()
    if foe and re.search(r"\b(attack|fight|yes|kill|hit)\b", low):
        return foe, "arbiter:attack->bearing"
    lv = least_visited(pos, legal, banned, visits)
    if lv:
        return lv, "arbiter:least-visited"
    return (decide_action(reply, legal) or (legal[0] if legal else "north")), "arbiter:legal-first"
