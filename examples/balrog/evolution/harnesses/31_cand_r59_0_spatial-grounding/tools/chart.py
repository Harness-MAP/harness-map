"""chart.py - per-level map memory and the reachability-seeded potential field over it."""
from __future__ import annotations

from collections import deque

from field import STEP

WALL = set("|- ")
LETHAL = set("}")
DOOR_NOUNS = ("doorway", "open door", "broken door")


class Chart:
    """Drawn glyphs, door squares, bump-closed edges, visits. A bump convicts the EDGE; a door is a SQUARE."""

    def __init__(self):
        self.clear("?")

    def clear(self, level):
        self.level = level
        self.terrain, self.visits = {}, {}
        self.blocked, self.doors = set(), set()
        self.locked = {}                       # (square, dir) -> kicks spent on that locked door
        self.bars = False
        self.pending = None

    def fold(self, fr):
        if fr["level"] != self.level:
            self.clear(fr["level"])
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch not in " @":
                    self.terrain[(x, y)] = ch
                    if ch == "+":
                        self.doors.add((x, y))
        pos, msg = fr["pos"], fr["message"].lower()
        if "bars" in msg:
            self.bars = True
        if pos is None or pos[1] < 0:
            return
        if self.terrain.get(pos, "@") == "@":
            self.terrain[pos] = "."
        for noun, dirs in fr["adjacent"].items():
            if any(k in noun for k in DOOR_NOUNS):
                self.doors |= {(pos[0] + STEP[d][0], pos[1] + STEP[d][1]) for d in dirs}
        if self.pending and not fr["modal"] and self.pending[0] == pos:
            d = self.pending[1]
            if "door opens" in msg:
                self.doors.add((pos[0] + STEP[d][0], pos[1] + STEP[d][1]))
            elif "door is locked" in msg and len(d) <= 5:
                self.locked.setdefault(self.pending, 0)
            else:
                self.blocked.add(self.pending)
        self.pending = None
        self.visits[pos] = self.visits.get(pos, 0) + 1

    def lock_to_kick(self, pos, cap):
        """A locked door beside `pos` that is still drawn shut and has kicks left -> dir, else None.
        A door past the cap is convicted as a closed EDGE, so the field routes around it."""
        for (a, d), n in sorted(self.locked.items()):
            if a != pos:
                continue
            b = (a[0] + STEP[d][0], a[1] + STEP[d][1])
            if self.terrain.get(b) != "+":
                continue
            if n >= cap:
                self.blocked.add((a, d))
                continue
            self.locked[(a, d)] = n + 1
            return d
        return None

    def passable(self, c):
        if c in self.doors:
            return True
        ch = self.terrain.get(c)
        if ch is None or ch in WALL or ch in LETHAL:
            return False
        return not (self.bars and ch == "#")

    def edge(self, a, d):
        """Is the move a -> a+d open on the chart (no bump record, no diagonal through a door)?"""
        b = (a[0] + STEP[d][0], a[1] + STEP[d][1])
        if (a, d) in self.blocked or not self.passable(b):
            return None
        if len(d) > 5 and (a in self.doors or b in self.doors):
            return None
        return b

    def reach(self, pos):
        seen, q = {pos}, deque([pos])
        while q:
            cur = q.popleft()
            for d in STEP:
                nxt = self.edge(cur, d)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return seen

    def frontier(self, c):
        return not self.visits.get(c) and any((c[0] + dx, c[1] + dy) not in self.terrain
                                              for dx, dy in STEP.values())

    def field(self, pos):
        """Potential over reachable cells: 0 at a REACHABLE `>`, else 0 at reachable frontier squares.
        Returns (potential, kind); kind is 'stairs', 'frontier' or None (field flat)."""
        region = self.reach(pos)
        seeds = [c for c in region if self.terrain.get(c) == ">"]
        kind = "stairs"
        if not seeds:
            seeds, kind = [c for c in region if c != pos and self.frontier(c)], "frontier"
        if not seeds:
            return {}, None
        pot = {c: 0.0 for c in seeds}
        q = deque(seeds)
        while q:
            cur = q.popleft()
            for d, (dx, dy) in STEP.items():
                prev = (cur[0] - dx, cur[1] - dy)
                if prev not in region or self.edge(prev, d) != cur:
                    continue
                cost = pot[cur] + 1.0 + 0.5 * min(self.visits.get(prev, 0), 6)
                if cost < pot.get(prev, 1e9):
                    pot[prev] = cost
                    q.append(prev)
        return pot, kind
