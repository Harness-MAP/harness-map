"""context.py — the ONE per-step object every pipeline stage reads and writes.

Built from the observation only: cursor (row = y-1 into the `map:` block), `Dlvl`, the message line,
the map rows and the "<noun> adjacent <dirs>" prose. Stages talk only through this object:
`notes` = lines a stage wants the model to read, `trail` = who changed the action and why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
DIRS_RE = r"northeast|northwest|southeast|southwest|north|south|east|west"
COMBAT = re.compile(r"\byou (hit|miss|kill|destroy|smite)|\bbites\b|\bhits\b|\bmisses\b|\battacks?\b"
                    r"|\bstings\b|\bclaws\b|\bbutts\b|\bkicks\b", re.I)


def adjacent_nouns(text: str) -> dict:
    """{direction: [noun, ...]} from lines like 'lava adjacent northeast, east, and southeast'."""
    out = {}
    for line in (text or "").splitlines():
        if " adjacent " not in line:
            continue
        noun, rest = line.split(" adjacent ", 1)
        for d in re.findall(DIRS_RE, rest):
            out.setdefault(d, []).append(noun.strip().lower())
    return out


@dataclass
class StepContext:
    text: str = ""
    legal: list = field(default_factory=list)
    level: str = "?"
    pos: tuple | None = None
    rows: list = field(default_factory=list)
    nouns: dict = field(default_factory=dict)
    combat: bool = False
    notes: list = field(default_factory=list)
    trail: list = field(default_factory=list)

    def moves(self):
        return [d for d in STEP if d in self.legal]

    def target(self, d):
        if self.pos is None or d not in STEP:
            return None
        return (self.pos[0] + STEP[d][0], self.pos[1] + STEP[d][1])

    def glyph(self, cell):
        if cell is None:
            return " "
        x, y = cell
        return self.rows[y][x] if 0 <= y < len(self.rows) and 0 <= x < len(self.rows[y]) else " "

    def creature_dirs(self):
        """Neighbours drawn as a monster letter (MiniHack monsters are letters; items are symbols)."""
        return [d for d in self.moves() if self.glyph(self.target(d)).isalpha()]


def build(obs, legal) -> StepContext:
    try:
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
    except (KeyError, TypeError):
        text = str(obs)
    text = text or ""
    c = StepContext(text=text, legal=list(legal))
    cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
    c.pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
    dl = re.search(r"Dlvl:(\d+)", text)
    c.level = dl.group(1) if dl else "?"
    c.rows = text.split("map:", 1)[1].split("\n")[1:22] if "map:" in text else []
    c.nouns = adjacent_nouns(text)
    msg = text.split("message:", 1)[1].split("\n\n", 1)[0] if "message:" in text else ""
    c.combat = bool(COMBAT.search(msg))
    return c
