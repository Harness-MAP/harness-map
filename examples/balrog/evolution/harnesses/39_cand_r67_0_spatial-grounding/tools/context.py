"""context.py — the ONE per-step object every pipeline stage reads and writes (observation only).

Parses the frame text: cursor (x, y) -> map row y-1, the `map:` block (21 rows), `Dlvl`, the message
line, and the "<noun> adjacent <dirs>" prose. Stages communicate only through this object:
`notes` are lines shown to the model, `trail` is the provenance written to `response.reasoning`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DELTA = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
         "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
DIRS_RE = r"northeast|northwest|southeast|southwest|north|south|east|west"


@dataclass
class StepContext:
    text: str = ""
    legal: list = field(default_factory=list)
    level: str = "?"
    message: str = ""
    pos: tuple | None = None                       # (x, map_row)
    grid: list = field(default_factory=list)       # map rows (strings)
    nouns: dict = field(default_factory=dict)      # direction -> [noun, ...]
    notes: list = field(default_factory=list)
    trail: list = field(default_factory=list)

    def glyph(self, x, y):
        if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[y]):
            return self.grid[y][x]
        return " "

    def target(self, d):
        if self.pos is None or d not in DELTA:
            return None
        return (self.pos[0] + DELTA[d][0], self.pos[1] + DELTA[d][1])

    def moves(self):
        return [d for d in DELTA if d in self.legal]


def _section(text, head):
    m = re.search(rf"(?m)^{re.escape(head)}\s*$", text)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"(?m)^(message|language observation|cursor|map):\s*$", rest)
    return rest[:nxt.start()] if nxt else rest


def build(obs, legal) -> StepContext:
    try:
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
    except (KeyError, TypeError):
        text = str(obs)
    text = text or ""
    c = StepContext(text=text, legal=list(legal))
    c.message = " ".join(_section(text, "message:").split())
    dl = re.search(r"Dlvl:(\d+)", text)
    c.level = dl.group(1) if dl else "?"
    cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
    if cm:
        c.pos = (int(cm.group(1)), int(cm.group(2)) - 1)
    if "map:" in text:
        c.grid = text.split("map:", 1)[1].split("\n")[1:22]
    for line in _section(text, "language observation:").splitlines():
        if " adjacent " not in line:
            continue
        noun, rest = line.split(" adjacent ", 1)
        for d in re.findall(DIRS_RE, rest):
            c.nouns.setdefault(d, []).append(noun.strip().lower())
    return c
