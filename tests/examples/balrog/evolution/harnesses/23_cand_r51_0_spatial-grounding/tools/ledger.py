"""ledger.py — self-derived per-level spatial record, grounded in the frame's own words.

LEVER (code). Parses the current observation only (cursor, `map:` block, `Dlvl`, the
"<noun> adjacent <dirs>" prose lines, the message) and keeps, per level:
  · visits[(x, y)]      — how often each square was stood on
  · failed{(sq, dir)}   — a move that left the cursor in place WITHOUT a combat message
                          (a swing that leaves the rat alive is NOT a failed move)
Per frame it grounds each of the eight neighbours with BOTH its glyph and the prose noun the game
names there. The prose noun outranks the glyph (`#` is corridor OR bars; only the prose says which).
"""
from __future__ import annotations

import re

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
DIRS_RE = r"northeast|northwest|southeast|southwest|north|south|east|west"
DEADLY = ("lava", "water")
BLOCKING = ("wall", "stone", "bars", "tree")          # a noun that refuses the step
TERRAIN = DEADLY + BLOCKING + ("floor", "room", "corridor", "doorway", "door", "stairs",
                               "ladder", "fountain", "altar", "grave", "sink", "ice",
                               "boulder", "dark", "air", "cloud", "throne")
COMBAT = re.compile(r"\byou (hit|miss|kill|destroy)|\bbites\b|\bhits\b|\bmisses\b|\battacks?\b", re.I)


def adjacent_nouns(text: str) -> dict:
    """{direction: [noun, ...]} from lines like 'lava adjacent northeast, east, and southeast'."""
    out = {}
    for line in (text or "").splitlines():
        if " adjacent " not in line or " far " in f" {line} ":
            continue
        noun, rest = line.split(" adjacent ", 1)
        noun = noun.strip().lower()
        for d in re.findall(DIRS_RE, rest):
            out.setdefault(d, []).append(noun)
    return out


def classify(nouns) -> str:
    """'deadly' | 'blocked' | 'creature' | 'open' | '' from the prose nouns at one square."""
    joined = " ".join(nouns)
    if any(w in joined for w in DEADLY):
        return "deadly"
    if any(re.search(rf"\b{w}\b", joined) for w in BLOCKING):
        return "blocked"
    for n in nouns:
        if not any(w in n for w in TERRAIN):
            return "creature"      # a non-terrain noun: a monster or an item; moving there is fine
    return "open" if nouns else ""


class Ledger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level, self.visits, self.failed = None, {}, set()
        self.last, self.prev_pos, self.pos = None, None, None

    def observe(self, text: str):
        text = text or ""
        cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
        pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
        dl = re.search(r"Dlvl:(\d+)", text)
        lvl = dl.group(1) if dl else "?"
        if lvl != self.level:
            self.level, self.visits, self.failed = lvl, {}, set()
            self.last, self.prev_pos, self.pos = None, None, None
        if pos is not None:
            if self.last and self.last[0] == pos:
                if not COMBAT.search(text):
                    self.failed.add(self.last)
            elif self.pos is not None and self.pos != pos:
                self.prev_pos = self.pos
            self.visits[pos] = self.visits.get(pos, 0) + 1
            self.pos = pos
        mp = text.split("map:", 1)[1].split("\n")[1:22] if "map:" in text else []
        return pos, mp, adjacent_nouns(text)

    def rows(self, pos, mp, adj):
        """[(dir, glyph, visits, failed, kind, is_back)] for the eight neighbours."""
        if pos is None:
            return []
        out = []
        for d, (dx, dy) in STEP.items():
            x, y = pos[0] + dx, pos[1] + dy
            g = mp[y][x] if 0 <= y < len(mp) and 0 <= x < len(mp[y]) else " "
            out.append((d, g, self.visits.get((x, y), 0), (pos, d) in self.failed,
                        classify(adj.get(d, [])), (x, y) == self.prev_pos))
        return out

    def commit(self, pos, action):
        self.last = (pos, action) if pos is not None and action in STEP else None
