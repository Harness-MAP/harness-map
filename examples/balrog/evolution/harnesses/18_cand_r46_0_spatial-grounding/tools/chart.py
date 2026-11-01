"""tools/chart.py - per-level map memory (drawn glyphs, doors, bump-closed edges, visits) + BFS routing."""
from __future__ import annotations

from collections import deque



class Chart:
    def __init__(self, frame):
        self.F = frame
        self.clear()

    def clear(self):
        self.drawn = {}
        self.stood = set()
        self.visits = {}
        self.blocked = set()
        self.doors = set()

    def fold(self, fr):
        """Fold the frame's rows into memory; returns how many cells were newly drawn."""
        new_cells = 0
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@":
                    new_cells += (x, y) not in self.drawn
                    self.drawn[(x, y)] = ch
        return new_cells

    def glyph(self, cell):
        return self.drawn.get(cell, " ")

    def passable(self, cell):
        if cell in self.doors:
            return True
        ch = self.drawn.get(cell)
        return ch is not None and ch not in self.F.WALL and ch not in self.F.LETHAL

    def frontier(self, cell):
        if self.visits.get(cell, 0):
            return False
        x, y = cell
        return any((x + dx, y + dy) not in self.drawn for dx, dy in self.F.STEP.values())

    def bfs(self, start, moves, bad=()):
        """Breadth-first tree over charted passable squares -> {cell: (parent, dir) | None}, in visit order."""
        STEP = self.F.STEP
        seen = {start: None}
        order = []
        q = deque([start])
        while q:
            cur = q.popleft()
            order.append(cur)
            for d in moves:
                if (cur, d) in self.blocked or (cur == start and d in bad):
                    continue
                nxt = (cur[0] + STEP[d][0], cur[1] + STEP[d][1])
                if nxt in seen or not self.passable(nxt):
                    continue
                if len(d) > 5 and (nxt in self.doors or cur in self.doors):
                    continue
                seen[nxt] = (cur, d)
                q.append(nxt)
        return seen, order

    @staticmethod
    def first_hop(seen, goal):
        hop, n = None, 0
        while seen[goal] is not None:
            goal, hop = seen[goal]
            n += 1
        return hop, n

    def route(self, start, moves, bad):
        """First hop toward a drawn reachable `>` (kind 'stairs'), else the nearest unstood frontier
        (kind 'frontier'), else (None, None)."""
        seen, order = self.bfs(start, moves, bad)
        stairs = [c for c in order if c != start and self.drawn.get(c) == ">"]
        if stairs:
            return self.first_hop(seen, stairs[0])[0], "stairs"
        for c in order:
            if c != start and self.frontier(c):
                return self.first_hop(seen, c)[0], "frontier"
        return None, None

    def toward(self, start, moves, bad, goals):
        """First hop to the nearest of `goals` (ties in level (y, x) order), or None."""
        seen, order = self.bfs(start, moves, bad)
        for c in order:
            if c != start and c in goals:
                return self.first_hop(seen, c)
        return None, 0
