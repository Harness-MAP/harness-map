"""chart.py - a per-level memory of the drawn map and the BFS walker that plans on it."""
from __future__ import annotations

import re
from collections import deque

from grid import BEARINGS

ORDER = ["north", "east", "south", "west", "northeast", "southeast", "southwest", "northwest"]
BLOCKING = set("-|}`")
DOORS = set("+")
FIGHT = re.compile(r"\byou (?:miss|hit|kill|destroy)|\bbites\b|\bhits!|\bmisses\b")


class Chart:
    """Terrain folded across frames; bumps convict a square or an edge, never a glyph."""

    def __init__(self):
        self.level = None
        self.clear()

    def clear(self):
        self.terrain = {}
        self.stone = set()
        self.opened = set()
        self.blocked = set()
        self.visits = {}
        self.pending = None

    def fold(self, text, frame):
        """Fold one frame (and the verdict on the previous move) into the chart."""
        dl = re.search(r"Dlvl:\s*(\d+)", text or "")
        level = dl.group(1) if dl else "?"
        if level != self.level:
            self.level = level
            self.clear()
        rows = frame.get("rows") or []
        for r, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@" and not ch.isalpha():
                    self.terrain[(x, r + 1)] = ch
        pos = frame["pos"]
        if pos is None:
            self.pending = None
            return
        self.terrain.setdefault(pos, ".")
        if self.terrain.get(pos) in "-|":
            self.opened.add(pos)
        if self.pending and self.pending[0] == pos:
            src, d = self.pending
            target = (src[0] + BEARINGS[d][0], src[1] + BEARINGS[d][1])
            msg = (text or "").lower()
            if "door opens" in msg:
                self.opened.add(target)
            elif "solid stone" in msg or target not in self.terrain:
                self.stone.add(target)
            elif not FIGHT.search(msg):
                self.blocked.add((src, d))
        for dx, dy in BEARINGS.values():
            n = (pos[0] + dx, pos[1] + dy)
            if n not in self.terrain:
                self.stone.add(n)
        self.visits[pos] = self.visits.get(pos, 0) + 1
        self.pending = None

    def doorish(self, c):
        return c in self.opened or self.terrain.get(c) in DOORS

    def passable(self, c, bars=False):
        if c in self.stone:
            return False
        if c in self.opened:
            return True
        ch = self.terrain.get(c)
        if ch is None or ch in BLOCKING:
            return False
        return not (ch == "#" and bars)

    def step_ok(self, a, d, bars=False):
        dx, dy = BEARINGS[d]
        b = (a[0] + dx, a[1] + dy)
        if (a, d) in self.blocked or not self.passable(b, bars):
            return None
        if dx and dy and (self.doorish(a) or self.doorish(b)):
            return None
        return b

    def frontier(self, c):
        """A charted open square with an undrawn, unconvicted neighbour."""
        for dx, dy in BEARINGS.values():
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < 79 and 1 <= n[1] <= 21 and n not in self.terrain and n not in self.stone:
                return True
        return False

    def plan(self, pos, legal, bars=False, lethal=()):
        """First hop toward a reachable `>`, else toward the best frontier square; None if neither."""
        dirs = [d for d in ORDER if d in legal and d not in lethal]
        first, dist = {pos: None}, {pos: 0}
        q = deque([pos])
        while q:
            cur = q.popleft()
            for d in (dirs if cur == pos else ORDER):
                nxt = self.step_ok(cur, d, bars)
                if nxt is None or nxt in dist:
                    continue
                dist[nxt] = dist[cur] + 1
                first[nxt] = d if cur == pos else first[cur]
                q.append(nxt)
        downs = [c for c, ch in self.terrain.items() if ch == ">"]
        hit = [c for c in downs if c in dist and c != pos]
        if hit:
            best = min(hit, key=lambda c: (dist[c], c[1], c[0]))
            return first[best], "chart:route"
        goals = [c for c in dist if c != pos and self.frontier(c)]
        if not goals:
            return None, ""

        def key(c):
            pull = min((max(abs(c[0] - g[0]), abs(c[1] - g[1])) for g in downs), default=0)
            return (pull, dist[c], self.visits.get(c, 0), c[1], c[0]) if downs else \
                (dist[c], self.visits.get(c, 0), c[1], c[0])
        best = min(goals, key=key)
        return first[best], ("chart:seek" if downs else "chart:explore")
