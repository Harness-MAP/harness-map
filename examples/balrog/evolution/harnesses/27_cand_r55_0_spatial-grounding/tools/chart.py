"""chart.py — per-level MAP MEMORY + BFS route (the multi-hop spatial model the ledger lacked).

Keeps, per `Dlvl`: every glyph ever drawn, visits per square, and bump-closed (square, dir) edges
(a no-move WITHOUT a combat message — a swing at a surviving rat is not a refusal). `route()` runs BFS
over charted passable squares and returns the first hop toward a drawn `>`, else toward the NEAREST
unvisited square that borders undrawn space (a frontier). The order is a total order (distance, then
fixed compass order), re-elected every step, so A->B->A orbits cannot persist: once B is visited it is
no longer a frontier and a strictly different target wins.
"""
from __future__ import annotations

from collections import deque

from context import STEP

WALL = set("|- ")
LETHAL = set("}")
BLOCK_NOUNS = ("wall", "stone", "bars", "tree")
DOOR_NOUNS = ("door",)


class Chart:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level, self.drawn, self.visits, self.blocked = None, {}, {}, set()
        self.doors, self.walls, self.lethal, self.last = set(), set(), set(), None

    def fold(self, ctx):
        if ctx.level != self.level:
            lvl = ctx.level
            self.reset()
            self.level = lvl
        if ctx.pos is None:
            self.last = None
            return
        if self.last is not None and self.last[0] == ctx.pos and not ctx.combat:
            self.blocked.add(self.last)
        self.last = None
        self.visits[ctx.pos] = self.visits.get(ctx.pos, 0) + 1
        for y, row in enumerate(ctx.rows):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@":
                    self.drawn[(x, y)] = ch
        # the prose noun outranks the glyph (an open door is drawn `-`/`|`, `#` may be bars)
        for d, nouns in ctx.nouns.items():
            cell = ctx.target(d)
            j = " ".join(nouns)
            if "lava" in j or "water" in j:
                self.lethal.add(cell)
            elif any(w in j for w in DOOR_NOUNS) and "broken wall" not in j:
                self.doors.add(cell)
            elif any(w in j for w in BLOCK_NOUNS):
                self.walls.add(cell)

    def commit(self, pos, action):
        self.last = (pos, action) if pos is not None and action in STEP else None

    def passable(self, cell):
        if cell in self.lethal or cell in self.walls:
            return False
        if cell in self.doors:
            return True
        ch = self.drawn.get(cell)
        return ch is not None and ch not in WALL and ch not in LETHAL and ch not in "`0"

    def deadly(self, cell):
        return cell in self.lethal or self.drawn.get(cell) in LETHAL

    def frontier(self, cell):
        if self.visits.get(cell):
            return False
        x, y = cell
        return any((x + dx, y + dy) not in self.drawn for dx, dy in STEP.values())

    def _doorish(self, cell):
        return cell in self.doors or self.drawn.get(cell) == "+"

    def route(self, start, moves, avoid=()):
        """-> (first_hop, kind, distance) toward `>` or the nearest frontier, else (None, None, 0)."""
        if start is None:
            return None, None, 0
        seen = {start: None}
        q = deque([(start, 0)])
        stairs = front = None
        while q:
            cur, dist = q.popleft()
            if cur != start:
                if self.drawn.get(cur) == ">":
                    stairs = (cur, dist)
                    break
                if front is None and self.frontier(cur):
                    front = (cur, dist)
            for d in moves:
                if (cur, d) in self.blocked or (cur == start and d in avoid):
                    continue
                nxt = (cur[0] + STEP[d][0], cur[1] + STEP[d][1])
                if nxt in seen or not self.passable(nxt):
                    continue
                if len(d) > 5 and (self._doorish(nxt) or self._doorish(cur)):
                    continue            # NetHack forbids diagonal moves through doorways
                seen[nxt] = (cur, d)
                q.append((nxt, dist + 1))
        goal, kind = (stairs, "stairs") if stairs else ((front, "frontier") if front else (None, None))
        if goal is None:
            return None, None, 0
        cell, hop = goal[0], None
        while seen[cell] is not None:
            cell, hop = seen[cell]
        return hop, kind, goal[1]
