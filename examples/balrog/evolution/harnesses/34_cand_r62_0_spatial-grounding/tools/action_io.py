"""action_io.py — project the model's reply onto the LEGAL set, then arbitrate against the ledger.

LEVER (code). BALROG coerces any unlisted string to `north`; here the reply is matched whole-word
against the action names parsed from the system prompt (earliest mention wins, `attack <dir>` reads
as `<dir>` because moving into a monster attacks it). Then `arbitrate`:
  1. a move onto a square the prose names deadly/blocking, or one that already failed from this
     square, is VETOED -> next mention, else the ledger's best open direction;
  2. a move straight BACK to the square just left (the A->B->A orbit) with no creature adjacent is
     OVERRIDDEN by a strictly less-visited open direction when one exists.
Every override is named so the harness can write it to the reasoning column.
"""
from __future__ import annotations

import re

WALLISH = "|- "


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


def rank_open(rows, legal):
    """Open, non-vetoed directions, best first: drawn before undrawn, prose-open before wallish, fewest visits, not back."""
    cand = []
    for i, (d, g, v, f, kind, back) in enumerate(rows):
        if d not in legal or f or kind in ("deadly", "blocked"):
            continue
        cand.append((g == " " or kind == "unknown", (g in WALLISH) and kind != "open", 0 if g == ">" else 1, v, back, i, d))
    return [c[-1] for c in sorted(cand)]


def arbitrate(text: str, legal: list, rows) -> tuple:
    """-> (action, rule) where rule names who decided."""
    if not legal:
        return re.sub(r"[^a-zA-Z\s:]", "", text or ""), "naive"
    by_dir = {r[0]: r for r in rows}
    vetoed = {d for d, _, _, f, kind, _ in rows if f or kind in ("deadly", "blocked")}
    ranked = rank_open(rows, legal)
    pick, rule = None, "model"
    for name in mentions(text, legal):
        if name in vetoed:
            rule = "veto"
            continue
        pick = name
        break
    if pick is None:
        if ranked:
            return ranked[0], "fallback:" + rule
        ms = mentions(text, legal)
        return (ms[0] if ms else re.sub(r"[^a-zA-Z\s:]", "", text or "")), "unmatched"
    row = by_dir.get(pick)
    fighting = any(r[4] == "creature" for r in rows)
    if row and row[5] and not fighting and ranked:
        best = ranked[0]
        if best != pick and by_dir[best][2] < row[2]:
            return best, "orbit-break"
    return pick, rule


def extract_action(text: str, legal=None, rows=()) -> str:
    return arbitrate(text, legal or [], rows)[0]
