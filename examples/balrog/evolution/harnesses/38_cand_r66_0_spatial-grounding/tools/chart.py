"""tools/chart.py - per-level map memory: drawn glyphs, prose-credited doors and bars, bump-closed
(square, bearing) edges, visits; BFS routing to a drawn `>` / nearest frontier / arbitrary goals."""
from __future__ import annotations

from collections import deque

DOOR_NOUNS = ("doorway", "open door", "broken door")
BAR_NOUNS = ("iron bars", "bars")


class Chart:
    def __init__(self, frame):
        self.F = frame
        self.clear()

    def clear(self):
        self.drawn = {}
        self.visits = {}
        self.stood = set()
        self.blocked = set()
        self.doors = set()
        self.bars = set()

    def fold(self, fr):
        """Fold the frame's rows and the adjacent-prose nouns into memory."""
        STEP = self.F.STEP
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@":
                    self.drawn[(x, y)] = ch
        pos = fr["pos"]
        if pos is None:
            return
        if self.drawn.get(pos, "@") == "@":
            self.drawn[pos] = "."
        for noun, dirs in fr["adjacent"].items():
            cells = {(pos[0] + STEP[d][0], pos[1] + STEP[d][1]) for d in dirs}
            if any(k in noun for k in DOOR_NOUNS):
                self.doors |= cells
            elif any(noun == k or noun.endswith(" " + k) for k in BAR_NOUNS):
                self.bars |= cells
        self.visits[pos] = self.visits.get(pos, 0) + 1
        self.stood.add(pos)

    def book(self, prev, move, pos, message):
        """Credit the outcome of the last move: a door opening marks a door, any other no-move closes an edge."""
        if prev != pos or move not in self.F.STEP:
            return
        target = (prev[0] + self.F.STEP[move][0], prev[1] + self.F.STEP[move][1])
        if "door opens" in message.lower():
            self.doors.add(target)
        else:
            self.blocked.add((prev, move))

    def glyph(self, cell):
        return self.drawn.get(cell, " ")

    def passable(self, cell):
        if cell in self.bars:
            return False
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
        """Breadth-first tree over charted passable squares; no diagonal edge into or out of a door."""
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
        """First hop toward a reachable drawn `>` ('stairs'), else the nearest unstood frontier ('frontier')."""
        seen, order = self.bfs(start, moves, bad)
        for c in order:
            if c != start and self.drawn.get(c) == ">":
                return self.first_hop(seen, c)[0], "stairs"
        for c in order:
            if c != start and self.frontier(c):
                return self.first_hop(seen, c)[0], "frontier"
        return None, None

    def toward(self, start, moves, bad, goals):
        """First hop to the nearest of `goals`, or (None, 0)."""
        seen, order = self.bfs(start, moves, bad)
        for c in order:
            if c != start and c in goals:
                return self.first_hop(seen, c)
        return None, 0
