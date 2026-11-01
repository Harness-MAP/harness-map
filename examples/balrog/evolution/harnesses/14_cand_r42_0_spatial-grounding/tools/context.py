"""context.py — the ONE per-step object every pipeline stage reads and writes.

A StepContext is built from the raw observation (observation-only: map block, cursor, message,
adjacent-noun prose). Stages never talk to each other directly; they communicate through the context
(`ctx.notes` for prompt lines, `ctx.trail` for why the action changed hands).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DELTA = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, -1), "northwest": (-1, -1), "southeast": (1, 1), "southwest": (-1, 1),
}
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east",
            "northeast": "southwest", "southwest": "northeast",
            "northwest": "southeast", "southeast": "northwest"}


@dataclass
class StepContext:
    obs: object = None
    raw: str = ""
    legal: list = field(default_factory=list)
    message: str = ""
    pos: tuple | None = None
    cells: dict = field(default_factory=dict)       # (x, y) -> glyph
    nouns: dict = field(default_factory=dict)       # direction -> {noun}
    notes: list = field(default_factory=list)       # lines a stage wants the model to see
    trail: list = field(default_factory=list)       # provenance of the final action
    action: str | None = None

    def target(self, d):
        if self.pos is None or d not in DELTA:
            return None
        return (self.pos[0] + DELTA[d][0], self.pos[1] + DELTA[d][1])

    def moves(self):
        return [d for d in DELTA if d in self.legal]


def _texts(obs):
    t = obs.get("text", obs) if isinstance(obs, dict) else obs
    if isinstance(t, dict):
        return str(t.get("short_term_context") or ""), str(t.get("long_term_context") or "")
    return str(t or ""), ""


def _block(text, head):
    m = re.search(rf"(?m)^{re.escape(head)}\s*$", text)
    if not m:
        return None
    rest = text[m.end():]
    cut = [rest.find("\n" + h) for h in ("message:", "language observation:", "cursor:", "map:") if h != head]
    cut = [c for c in cut if c >= 0]
    return rest[:min(cut)] if cut else rest


def build(obs, legal) -> StepContext:
    short, long_ = _texts(obs)
    raw = short + "\n" + long_
    c = StepContext(obs=obs, legal=list(legal), raw=raw)
    msg = _block(raw, "message:")
    c.message = " ".join(msg.split()) if msg else ""
    for line in (_block(raw, "language observation:") or "").splitlines():
        m = re.match(r"\s*(.+?) adjacent (.+)$", line.strip())
        if m:
            for d in m.group(2).replace(" and ", ",").split(","):
                if d.strip() in DELTA:
                    c.nouns.setdefault(d.strip(), set()).add(m.group(1).strip().lower())
    cur = re.search(r"\(x=(\d+),\s*y=(\d+)\)", raw)
    if cur:
        c.pos = (int(cur.group(1)), int(cur.group(2)))
    text = long_ if "map:" in long_ else raw
    if "map:" in text:
        rows = text.split("map:", 1)[1].split("\n")[1:]
        rows = [r for r in rows if "Dlvl:" not in r and not ("St:" in r and "Dx:" in r)]
        for y, row in enumerate(rows, start=1):
            for x, ch in enumerate(row):
                if ch != " ":
                    c.cells[(x, y)] = ch
        if c.pos and c.cells.get(c.pos) != "@":
            c.pos = None
    return c
