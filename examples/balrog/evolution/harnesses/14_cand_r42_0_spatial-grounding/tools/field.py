"""field.py — persistent per-level chart + the potential field a pipeline stage descends.

The chart folds every frame's `map:` glyphs (same (x, y) frame as ctx.pos / ctx.cells), a visit
count per square and a (square, direction) refusal ledger. `flood` runs a reverse multi-source
Dijkstra from the drawn `>` (else from every unvisited open square bordering undrawn space); each
step's cost grows with how often the square was trodden, so the field has no fixed-point 2-cycle.
"""
from __future__ import annotations

import heapq

from context import DELTA

SOLID = set("|- }0`")
MELEE = ("you miss", "you hit", "you kill", "bites", "hits!", "misses", "you destroy", "attack")


class Chart:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level = None
        self.terrain = {}
        self.visits = {}
        self.blocked = set()

    def open(self, c):
        ch = self.terrain.get(c)
        return ch is not None and ch not in SOLID

    def observe(self, ctx, level, pending):
        if level != self.level:
            self.reset()
            self.level = level
        for c, ch in ctx.cells.items():
            if ch != "@":
                self.terrain[c] = ch
        if ctx.pos is None:
            return
        if self.terrain.get(ctx.pos) is None:
            self.terrain[ctx.pos] = "."
        if pending and pending[0] == ctx.pos:
            m = ctx.message.lower()
            tgt = ctx.target(pending[1])
            foe = tgt is not None and ctx.cells.get(tgt, " ").isalpha()
            if not foe and not any(k in m for k in MELEE):
                self.blocked.add(pending)
        self.visits[ctx.pos] = self.visits.get(ctx.pos, 0) + 1

    def _frontier(self):
        return [c for c in self.terrain if self.open(c) and not self.visits.get(c)
                and any((c[0] + dx, c[1] + dy) not in self.terrain for dx, dy in DELTA.values())]

    def flood(self, seeds):
        pot = {c: 0.0 for c in seeds}
        heap = [(0.0, c) for c in seeds]
        while heap:
            v, cur = heapq.heappop(heap)
            if v > pot.get(cur, 1e9):
                continue
            for d, (dx, dy) in DELTA.items():
                prev = (cur[0] - dx, cur[1] - dy)
                if not self.open(prev) or (prev, d) in self.blocked:
                    continue
                cost = v + 1.0 + 0.5 * min(self.visits.get(prev, 0), 6)
                if cost < pot.get(prev, 1e9):
                    pot[prev] = cost
                    heapq.heappush(heap, (cost, prev))
        return pot

    def fields(self):
        """Goal field first (drawn `>`), then the exploration field as the fallback."""
        stairs = [c for c, ch in self.terrain.items() if ch == ">"]
        out = []
        if stairs:
            out.append(("stairs", self.flood(stairs)))
        out.append(("frontier", self.flood(self._frontier())))
        return out
