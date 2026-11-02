"""tools/chart.py - per-level map memory (drawn glyphs, doors, bump-closed edges, visits) + BFS routing.

The wall table is REGIME-aware: when the legal move set is exactly the four cardinals (Boxoban),
`#` is iron bars and a boulder is not floor. Doors (`+`, prose-named doorways, `The door opens.`,
and `-`/`|` flanked by walkable ground on both opposite sides) are passable but orthogonal-only."""
from __future__ import annotations

from collections import deque


class Chart:
    def __init__(self, frame):
        self.F = frame
        self.bars = False
        self.clear()

    def clear(self):
        self.drawn = {}
        self.visits = {}
        self.blocked = set()
        self.doors = set()

    def set_regime(self, moves):
        self.bars = set(moves) == set(self.F.CARDINALS)

    def walls(self):
        return self.F.WALL | (set("#`") if self.bars else set())

    def fold(self, fr):
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@":
                    self.drawn[(x, y)] = ch
                    if ch == "+":
                        self.doors.add((x, y))
        pos = fr["pos"]
        if pos is not None and self.drawn.get(pos) in (None, "@"):
            self.drawn[pos] = "."
        self.infer_doors()

    def infer_doors(self):
        """An open door is drawn with a wall glyph; spot it by walkable ground on both opposite sides."""
        if self.bars:
            return
        walk = lambda c: self.drawn.get(c) is not None and self.drawn[c] not in self.F.WALL \
            and self.drawn[c] not in self.F.LETHAL
        for (x, y), ch in self.drawn.items():
            if (x, y) in self.doors:
                continue
            g = self.drawn.get
            # a door in a vertical `|` wall opens to `-`, in a horizontal `-` wall to `|`
            if ch == "-" and walk((x - 1, y)) and walk((x + 1, y)) and "|" in (g((x, y - 1)), g((x, y + 1))):
                self.doors.add((x, y))
            elif ch == "|" and walk((x, y - 1)) and walk((x, y + 1)) and "-" in (g((x - 1, y)), g((x + 1, y))):
                self.doors.add((x, y))

    def mark_door(self, pos, d):
        dx, dy = self.F.STEP[d]
        self.doors.add((pos[0] + dx, pos[1] + dy))

    def passable(self, cell):
        if cell in self.doors:
            return True
        ch = self.drawn.get(cell)
        return ch is not None and ch not in self.walls() and ch not in self.F.LETHAL

    @property
    def terrain(self):
        """The drawn-glyph map, as the Quest regime (tools/quest.py) reads it."""
        return self.drawn

    def edge(self, a, d):
        """The square one step from `a` along d, or None if that edge is closed / impassable."""
        dx, dy = self.F.STEP[d]
        b = (a[0] + dx, a[1] + dy)
        if (a, d) in self.blocked or not self.passable(b):
            return None
        if len(d) > 5 and (a in self.doors or b in self.doors):
            return None
        return b

    def frontier(self, cell):
        if self.visits.get(cell, 0):
            return False
        x, y = cell
        return any((x + dx, y + dy) not in self.drawn for dx, dy in self.F.STEP.values())

    def bfs(self, start, moves, bad=()):
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
        hop = None
        while seen[goal] is not None:
            goal, hop = seen[goal]
        return hop

    def route(self, start, moves, bad):
        """First hop toward a reachable `>` ('stairs'), else the nearest unstood frontier ('frontier')."""
        seen, order = self.bfs(start, moves, bad)
        for c in order:
            if c != start and self.drawn.get(c) == ">":
                return self.first_hop(seen, c), "stairs"
        for c in order:
            if c != start and self.frontier(c):
                return self.first_hop(seen, c), "frontier"
        return None, None

    def probe(self, start, moves, bad):
        """Dark-room fallback: first hop to the nearest stood square with an UNDRAWN orthogonal
        neighbour, then the step into it (a bump closes it through the edge ledger)."""
        STEP = self.F.STEP
        seen, order = self.bfs(start, moves, bad)
        for c in order:
            for d in self.F.CARDINALS:
                if d not in moves or (c, d) in self.blocked or (c == start and d in bad):
                    continue
                n = (c[0] + STEP[d][0], c[1] + STEP[d][1])
                if n not in self.drawn:
                    return d if c == start else self.first_hop(seen, c)
        return None
