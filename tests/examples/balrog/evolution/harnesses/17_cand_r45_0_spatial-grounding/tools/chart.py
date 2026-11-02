"""chart.py — self-derived per-level MAP MEMORY + BFS route (all facts read off the observation).

LEVER (code). Per dungeon level the Chart keeps
  · glyphs  — last glyph seen per square (`map:` block, row = cursor.y - 1)
  · visits  — times the agent stood on each square
  · walls   — squares the prose NAMES as wall/bars adjacent (square-scoped, never glyph-scoped:
              `#` is corridor in a maze but bars in Boxoban, so only the named square is convicted)
  · blocked — (square, direction) edges refused by TERRAIN. A no-move is booked only when the
              message is a terrain refusal (or silent) and the target is not a creature: a melee
              swing also leaves the cursor in place and must never convict the edge (r029 bug).
`plan()` BFS: a charted `>` first, else the nearest FRONTIER = passable, never stood on AND
bordering an uncharted square (AND, not OR: the OR form makes every dark-maze square a frontier
and the route a 2-cycle, r039 bug).
"""
from __future__ import annotations

import re
from collections import deque

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
DIRS = r"northeast|northwest|southeast|southwest|north|south|east|west"
MAP_ROWS = 21
REFUSAL = re.compile(r"It's a wall|It's solid stone|cannot pass through the bars|in vain|"
                     r"door is locked|door is closed|You can't move diagonally|squeeze|"
                     r"Perhaps that's why you cannot move past it", re.I)
COMBAT = re.compile(r"You (hit|miss|kill|destroy)|bites|misses|hits!|Really attack|"
                    r"You displace|You swap places|You stop", re.I)
WALL_NOUN = re.compile(r"^(?:vertical wall|horizontal wall|wall|bars|dark part of a room|"
                       r"(?:top|bottom) (?:left|right) corner|corner|stone|solid stone)\s+adjacent\s+(.*)$")


def parse(text: str):
    text = text or ""
    cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
    pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
    dl = re.search(r"Dlvl:(\d+)", text)
    lvl = dl.group(1) if dl else "?"
    mp = text.split("map:", 1)[1].split("\n")[1:1 + MAP_ROWS] if "map:" in text else []
    msg = ""
    if "message:" in text:
        msg = text.split("message:", 1)[1].split("language observation:", 1)[0].strip()
    hazard, walled = set(), set()
    for line in text.splitlines():
        line = line.strip()
        if "adjacent" not in line:
            continue
        if "lava" in line or "water" in line:
            hazard |= set(re.findall(DIRS, line.split("adjacent", 1)[1]))
        m = WALL_NOUN.match(line)
        if m and "dark part" not in line:
            walled |= set(re.findall(DIRS, m.group(1)))
    return pos, lvl, mp, msg, hazard, walled


def is_creature(g: str) -> bool:
    return g.isalpha() or g == "@"


def passable(g: str) -> bool:
    return g not in "|- }" and not is_creature(g)


class Chart:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level, self.glyphs, self.visits, self.walls, self.blocked = None, {}, {}, set(), set()
        self.last, self.pos, self.hazard, self.walled, self.msg = None, None, set(), set(), ""

    def observe(self, text: str):
        pos, lvl, mp, msg, hazard, walled = parse(text)
        if lvl != self.level:
            self.level, self.glyphs, self.visits, self.walls, self.blocked = lvl, {}, {}, set(), set()
            self.last = None
        for y, row in enumerate(mp):
            for x, g in enumerate(row):
                if g != " ":
                    self.glyphs[(x, y)] = g
        if pos is not None:
            self.glyphs[pos] = "."  # the square under the hero is walkable (drawn as @)
            for d in walled:
                dx, dy = STEP[d]
                self.walls.add((pos[0] + dx, pos[1] + dy))
            if self.last and self.last[0] == pos:
                sq0, d0 = self.last
                tgt = (sq0[0] + STEP[d0][0], sq0[1] + STEP[d0][1])
                if not COMBAT.search(msg) and (REFUSAL.search(msg) or not is_creature(self.glyph(tgt))):
                    self.blocked.add(self.last)
            self.visits[pos] = self.visits.get(pos, 0) + 1
        self.pos, self.hazard, self.walled, self.msg, self.last = pos, hazard, walled, msg, None
        return pos

    def glyph(self, sq):
        if sq == self.pos:
            return "@"
        return self.glyphs.get(sq, " ")

    def target(self, d):
        dx, dy = STEP[d]
        return (self.pos[0] + dx, self.pos[1] + dy)

    def neighbours(self):
        """[(direction, glyph, visits, blocked_from_here, deadly, walled)] around the cursor."""
        if self.pos is None:
            return []
        out = []
        for d in STEP:
            sq = self.target(d)
            g = self.glyph(sq)
            out.append((d, g, self.visits.get(sq, 0), (self.pos, d) in self.blocked,
                        d in self.hazard or g == "}", d in self.walled or sq in self.walls))
        return out

    def creatures(self, legal=None):
        if self.pos is None:
            return []
        return [d for d in STEP if (not legal or d in legal) and is_creature(self.glyph(self.target(d)))]

    def _open(self, sq):
        return passable(self.glyph(sq)) and self.glyph(sq) != " " and sq not in self.walls

    def _frontier(self, sq):
        if sq == self.pos or self.visits.get(sq, 0):
            return False
        return any(self.glyphs.get((sq[0] + dx, sq[1] + dy), " ") == " " for dx, dy in STEP.values())

    def plan(self, legal=None):
        """BFS from the cursor -> (first_hop, kind, dist) or (None, None, None). Total order:
        distance, then STEP order, so re-electing every step is a fixed point."""
        if self.pos is None:
            return None, None, None
        dirs = [d for d in STEP if not legal or d in legal]
        start = self.pos
        prev = {start: None}
        q = deque([(start, 0)])
        frontier = None
        while q:
            sq, dist = q.popleft()
            if sq != start:
                if self.glyph(sq) == ">":
                    return self._hop(prev, sq), "stairs down", dist
                if frontier is None and self._frontier(sq):
                    frontier = (sq, dist)
            for d in dirs:
                dx, dy = STEP[d]
                nxt = (sq[0] + dx, sq[1] + dy)
                if nxt in prev or (sq, d) in self.blocked:
                    continue
                if sq == start and (d in self.hazard or d in self.walled):
                    continue
                if not (self._open(nxt) or self.glyph(nxt) == ">"):
                    continue
                prev[nxt] = (sq, d)
                q.append((nxt, dist + 1))
        if frontier:
            return self._hop(prev, frontier[0]), "unexplored", frontier[1]
        return None, None, None

    @staticmethod
    def _hop(prev, sq):
        d = None
        while prev[sq] is not None:
            sq, d = prev[sq]
        return d
