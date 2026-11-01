"""field.py — the per-level CHART and the potential field that makes the move decision.

LEVER (code). Folds each frame's `map:` block (row = cursor.y - 1) into a per-level terrain chart and
runs a reverse multi-source Dijkstra over it, RE-ELECTED every step:
  · sources = every drawn `>`; with none, every frontier cell — a drawn open cell never stood on that
    borders an undrawn cell, plus an undrawn neighbour the prose names passable (a dark room floor);
  · an edge the ledger booked as failed is closed, `}` / `|` / `-` / blank are never entered;
  · the cost of entering a cell grows with how often it was stood on, so the field has no 2-cycle
    fixed point (the orbit the one-hop arbiter kept falling into).
The harness steps down the gradient; the model is only asked where the field is flat.
"""
from __future__ import annotations

import heapq

from ledger import STEP

SOLID = set("|- }")


class Chart:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level, self.terrain, self.virtual = None, {}, set()

    def fold(self, level, mp, pos, rows):
        if level != self.level:
            self.level, self.terrain, self.virtual = level, {}, set()
        for y, row in enumerate(mp):
            for x, ch in enumerate(row):
                if ch not in " @":
                    self.terrain[(x, y)] = ch
        if pos is not None:
            if self.terrain.get(pos, "@") == "@":
                self.terrain[pos] = "."
            for d, g, v, f, kind, back in rows:
                c = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
                if kind in ("deadly",):
                    self.terrain[c] = "}"
                elif kind == "open" and c not in self.terrain:
                    self.virtual.add(c)            # prose-named passable square the map left blank
        self.virtual -= set(self.terrain)

    def passable(self, c):
        ch = self.terrain.get(c)
        if ch is None:
            return c in self.virtual
        return ch not in SOLID

    def flood(self, visits, failed):
        """{cell: potential}; 0 at the sources, +1 per hop, + a visit surcharge on the entered cell."""
        seeds = [c for c, ch in self.terrain.items() if ch == ">"]
        if not seeds:
            seeds = [c for c in list(self.terrain) + list(self.virtual)
                     if self.passable(c) and not visits.get(c)
                     and any((c[0] + dx, c[1] + dy) not in self.terrain for dx, dy in STEP.values())]
        pot = {c: 0.0 for c in seeds}
        heap = [(0.0, c) for c in seeds]
        heapq.heapify(heap)
        while heap:
            p, cur = heapq.heappop(heap)
            if p > pot.get(cur, 1e9):
                continue
            for d, (dx, dy) in STEP.items():
                prev = (cur[0] - dx, cur[1] - dy)
                if not self.passable(prev) or (prev, d) in failed:
                    continue
                cost = p + 1.0 + 0.5 * min(visits.get(cur, 0), 6)
                if cost < pot.get(prev, 1e9):
                    pot[prev] = cost
                    heapq.heappush(heap, (cost, prev))
        return pot
