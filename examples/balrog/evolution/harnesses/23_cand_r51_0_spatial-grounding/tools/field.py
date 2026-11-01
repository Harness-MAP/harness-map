"""field.py — per-level chart + potential field, with the game's prose nouns as the edge oracle.

LEVER (code). Folds every frame's `map:` block into a per-`Dlvl` chart and runs a reverse Dijkstra
(heap) from a drawn `>` at cost 0 AND from every unstood frontier cell at cost FRONTIER_OFFSET, so an
unreachable `>` never flattens the field. Passability is decided per SQUARE, never per glyph:
  · blank is unknown = not passable (never a frontier target by itself)
  · `|`/`-`/`}` are solid UNLESS the square is a known door (ever drawn `+`, or `The door opens.`)
  · a square the prose names wall/stone/bars/lava/water (tools/ledger.classify) is booked solid
  · a no-move convicts the (square, dir) EDGE, except `The door opens.` which books a door square
  · no diagonal step into or out of a door square
"""
from __future__ import annotations

import heapq
import re

from ledger import STEP, classify

SOLID = set("|-}")
FRONTIER_OFFSET = 25.0
DOOR_OPENS = re.compile(r"door opens", re.I)
LOCKED = re.compile(r"door is locked", re.I)
COMBAT = re.compile(r"\byou (hit|miss|kill|destroy)|\bbites\b|\bhits!?\b|\bmisses\b", re.I)


class Chart:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level = None
        self.terrain, self.visits, self.blocked = {}, {}, set()
        self.doors, self.solid, self.locked, self.peaceful = set(), set(), {}, set()
        self.pending = None
        self.field = {}

    # ---- folding ----------------------------------------------------------------------------
    def fold(self, level, pos, rows, adj, msg, modal):
        if level != self.level:
            lvl = level
            self.reset()
            self.level = lvl
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch in " @":
                    continue
                if ch.isalpha() or ch in "&'":
                    continue                       # a creature letter is not terrain
                self.terrain[(x, y)] = ch
                if ch == "+":
                    self.doors.add((x, y))
        if modal or pos is None:
            return
        if self.terrain.get(pos) is None:
            self.terrain[pos] = "."
        for d, nouns in adj.items():
            c = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
            kind = classify(nouns)
            if kind in ("deadly", "blocked"):
                self.solid.add(c)
            elif any(w in " ".join(nouns) for w in ("open door", "closed door")):
                self.doors.add(c)          # prose `doorway` is doorless: diagonals stay legal
        pend, self.pending = self.pending, None
        if pend and pend[0] == pos:
            d = pend[1]
            tgt = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
            if DOOR_OPENS.search(msg):
                self.doors.add(tgt)
                self.solid.discard(tgt)
            elif LOCKED.search(msg):
                self.doors.add(tgt)
                self.locked[tgt] = self.locked.get(tgt, 0)
            elif not COMBAT.search(msg):
                self.blocked.add(pend)
        self.visits[pos] = self.visits.get(pos, 0) + 1

    # ---- geometry ---------------------------------------------------------------------------
    def passable(self, c):
        if c in self.solid:
            return False
        ch = self.terrain.get(c)
        if ch is None:
            return False
        if c in self.locked:
            return False
        return c in self.doors or ch not in SOLID

    def edge(self, a, d):
        dx, dy = STEP[d]
        b = (a[0] + dx, a[1] + dy)
        if (a, d) in self.blocked or not self.passable(b):
            return False
        if dx and dy and (a in self.doors or b in self.doors):
            return False
        return True

    def frontier(self):
        out = []
        for c in self.terrain:
            if not self.passable(c) or self.visits.get(c):
                continue
            if any((c[0] + dx, c[1] + dy) not in self.terrain for dx, dy in STEP.values()):
                out.append(c)
        return out

    def flood(self):
        """Reverse Dijkstra: `>` at 0, unstood frontier at FRONTIER_OFFSET; cost grows with visits."""
        pot, heap = {}, []
        for c, ch in self.terrain.items():
            if ch == ">" and self.passable(c):
                pot[c] = 0.0
        for c in self.frontier():
            pot.setdefault(c, FRONTIER_OFFSET)
        heap = [(v, c) for c, v in pot.items()]
        heapq.heapify(heap)
        while heap:
            v, cur = heapq.heappop(heap)
            if v > pot.get(cur, 1e18):
                continue
            for d, (dx, dy) in STEP.items():
                prev = (cur[0] - dx, cur[1] - dy)
                if not self.passable(prev) or not self.edge(prev, d):
                    continue
                cost = v + 1.0 + 0.5 * min(self.visits.get(prev, 0), 6)
                if cost < pot.get(prev, 1e18):
                    pot[prev] = cost
                    heapq.heappush(heap, (cost, prev))
        self.field = pot
        return pot
