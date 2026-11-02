"""chart.py — a self-derived per-level MAP MEMORY and a BFS route to the next useful square.

LEVER (code). Every observation carries a `map:` block (row index = cursor.y - 1) and the cursor
`(x=.., y=..)`. The Chart accumulates, per dungeon level:
  · glyphs  — the last glyph seen at each square (so squares that scroll out of the lit area persist)
  · visits  — how often the agent stood on each square
  · blocked — (square, direction) moves that left the cursor where it was (wall, bars, stuck boulder)
  · hazard  — directions the prose names lava/water adjacent this turn
Each step `plan()` runs a BFS over charted passable squares (honouring blocked edges) to
  1. a charted `>` if there is one, else
  2. the nearest FRONTIER: a passable, never-stood-on square, or a passable square bordering unseen.
It returns the first hop direction, the target kind and its distance. Nothing here is oracle: every
fact is read off the observation text the agent itself received.
"""
from __future__ import annotations

import re
from collections import deque

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
DIRS = "northeast|northwest|southeast|southwest|north|south|east|west"
WALL = set("|- ")
DEADLY = set("}")
MAP_ROWS = 21


def parse(text: str):
    text = text or ""
    cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
    pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
    dl = re.search(r"Dlvl:(\d+)", text)
    lvl = dl.group(1) if dl else "?"
    mp = text.split("map:", 1)[1].split("\n")[1:1 + MAP_ROWS] if "map:" in text else []
    hazard = set()
    for line in text.splitlines():
        if ("lava" in line or "water" in line) and "adjacent" in line:
            hazard |= set(re.findall(DIRS, line.split("adjacent", 1)[1]))
    return pos, lvl, mp, hazard


def passable(g: str) -> bool:
    # floor, corridor, stairs, doors, items, boulders(pushable), the agent itself; NOT walls,
    # unseen, lava/water, or a creature letter (routing through a monster starts a fight).
    return g not in WALL and g not in DEADLY and not g.isalpha() or g == "@"


class Chart:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level, self.glyphs, self.visits, self.blocked = None, {}, {}, set()
        self.last, self.pos, self.hazard = None, None, set()

    def observe(self, text: str):
        pos, lvl, mp, hazard = parse(text)
        if lvl != self.level:
            self.level, self.glyphs, self.visits, self.blocked, self.last = lvl, {}, {}, set(), None
        for y, row in enumerate(mp):
            for x, g in enumerate(row):
                if g != " ":
                    self.glyphs[(x, y)] = g
        if pos is not None:
            if self.last and self.last[0] == pos:
                self.blocked.add(self.last)
            self.visits[pos] = self.visits.get(pos, 0) + 1
        self.pos, self.hazard = pos, hazard
        return pos

    def glyph(self, sq):
        return self.glyphs.get(sq, " ")

    def neighbours(self):
        """[(direction, glyph, visits, blocked_from_here, deadly)] around the cursor."""
        if self.pos is None:
            return []
        out = []
        for d, (dx, dy) in STEP.items():
            sq = (self.pos[0] + dx, self.pos[1] + dy)
            g = self.glyph(sq)
            out.append((d, g, self.visits.get(sq, 0), (self.pos, d) in self.blocked,
                        d in self.hazard or g in DEADLY))
        return out

    def _frontier(self, sq):
        if sq == self.pos:
            return False
        if self.visits.get(sq, 0) == 0:
            return True
        return any(self.glyph((sq[0] + dx, sq[1] + dy)) == " " for dx, dy in STEP.values())

    def plan(self, legal=None):
        """BFS from the cursor -> (first_hop, kind, dist) or (None, None, None)."""
        if self.pos is None:
            return None, None, None
        dirs = [d for d in STEP if not legal or d in legal]
        start = self.pos
        prev = {start: None}
        q = deque([(start, 0)])
        best_frontier = None
        while q:
            sq, dist = q.popleft()
            if sq != start:
                if self.glyph(sq) == ">":
                    return self._hop(prev, sq), "stairs down", dist
                if best_frontier is None and self._frontier(sq):
                    best_frontier = (sq, dist)
            for d in dirs:
                dx, dy = STEP[d]
                nxt = (sq[0] + dx, sq[1] + dy)
                if nxt in prev or (sq, d) in self.blocked:
                    continue
                if sq == start and d in self.hazard:
                    continue
                g = self.glyph(nxt)
                if not passable(g) and g != ">":
                    continue
                prev[nxt] = (sq, d)
                q.append((nxt, dist + 1))
        if best_frontier:
            return self._hop(prev, best_frontier[0]), "unexplored", best_frontier[1]
        return None, None, None

    @staticmethod
    def _hop(prev, sq):
        d = None
        while prev[sq] is not None:
            sq, d = prev[sq]
        return d
