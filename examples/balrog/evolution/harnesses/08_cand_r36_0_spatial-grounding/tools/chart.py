"""chart.py - the episode's self-derived spatial memory and planner (NEVER oracle).

  terrain  - (x, y) -> last glyph drawn there
  visits   - (x, y) -> decisions taken standing there
  blocked  - (square, bearing) the env refused (a bump convicts the SQUARE, never the glyph)
  doors    - cells that are / were a real door: keep door identity when `+` is redrawn `-`/`|`;
             passable, but NetHack forbids diagonal moves into or out of them
  doorways - doorless / broken doorways: passable, diagonals allowed
  searched - (x, y) -> `search` turns spent there (budget is keyed on the probed cell)
"""
from __future__ import annotations

from collections import deque

from perception import STEP, DIAG, step

LETHAL = set("}")
WALL = set("|- ")
FLOOR = set(".#<>{")
SEARCH_BUDGET = 8


class Chart:
    def __init__(self):
        self.reset()

    def reset(self):
        self.terrain = {}
        self.visits = {}
        self.blocked = set()
        self.doors = set()
        self.doorways = set()
        self.searched = {}
        self.last = None
        self.level = None

    def observe(self, fr):
        if fr["dlvl"] != self.level:
            self.reset()
            self.level = fr["dlvl"]
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@":
                    self.terrain[(x, y)] = ch
                    if ch == "+":
                        self.doors.add((x, y))
        pos = fr["pos"]
        if pos is None:
            return
        for d in fr["door_dirs"]:
            self.doors.add(step(pos, d))
        for d in fr["doorway_dirs"]:
            self.doorways.add(step(pos, d))
        if pos not in self.terrain or self.terrain[pos] == "@":
            self.terrain[pos] = "."
        msg = fr["message"].lower()
        if self.last and not fr["prompt"]:
            prev, move, kind = self.last
            if prev == pos and move in STEP and kind != "swing":
                self.blocked.add((prev, move))
                if "door opens" in msg:
                    self.doors.add(step(prev, move))
        if "door opens" in msg or "crashes open" in msg:
            # the square's own verdict clears a record of a closed door bump
            self.blocked = {(c, d) for (c, d) in self.blocked if c != pos}
        self.visits[pos] = self.visits.get(pos, 0) + 1

    def is_door(self, cell):
        if cell in self.doors:
            return True
        ch = self.terrain.get(cell)
        if ch not in ("-", "|"):
            return False
        # an OPEN door is drawn perpendicular to its wall: `-` in a vertical wall is flanked
        # east/west by walkable cells and north/south by `|`; `|` in a horizontal wall is flanked
        # north/south by walkable cells and east/west by `-`. (Two stacked doorways `.|.` are not.)
        if ch == "-":
            walk, side, wall = ("west", "east"), ("north", "south"), "|"
        else:
            walk, side, wall = ("north", "south"), ("west", "east"), "-"
        return (all(self.terrain.get(step(cell, d)) in FLOOR for d in walk)
                and any(self.terrain.get(step(cell, d)) == wall for d in side))

    def passable(self, cell):
        ch = self.terrain.get(cell)
        if ch is None or ch in LETHAL:
            return False
        if cell in self.doorways or self.is_door(cell):
            return True
        return ch not in WALL

    def edge_ok(self, cur, d, nxt):
        if d in DIAG and (self.is_door(cur) or self.is_door(nxt)):
            return False
        return True

    def frontier(self, cell):
        if self.visits.get(cell, 0):
            return False
        return any(step(cell, d) not in self.terrain for d in STEP)

    def _bfs(self, start, moves, bad):
        seen = {start: None}
        order = []
        q = deque([start])
        while q:
            cur = q.popleft()
            order.append(cur)
            for d in moves:
                if (cur, d) in self.blocked or (cur == start and d in bad):
                    continue
                nxt = step(cur, d)
                if nxt in seen or not self.passable(nxt) or not self.edge_ok(cur, d, nxt):
                    continue
                seen[nxt] = (cur, d)
                q.append(nxt)
        return seen, order

    @staticmethod
    def _first_hop(seen, cell):
        hop = None
        while seen[cell] is not None:
            cell, hop = seen[cell]
        return hop

    def route(self, start, moves, bad=()):
        """First hop toward a drawn `>`, else the nearest frontier (re-elected every step)."""
        seen, order = self._bfs(start, moves, bad)
        stairs = [c for c in order if c != start and self.terrain.get(c) == ">"]
        if stairs:
            return self._first_hop(seen, stairs[0])
        for c in order:
            if c != start and self.frontier(c):
                return self._first_hop(seen, c)
        return None

    def stairs_charted(self):
        return any(ch == ">" for ch in self.terrain.values())

    def dead_end(self, cell, moves):
        n = sum(1 for d in moves if self.passable(step(cell, d)))
        undrawn = sum(1 for d in STEP if step(cell, d) not in self.terrain)
        return n <= 1 or (self.terrain.get(cell) == "#" and undrawn >= 3)

    def search_plan(self, start, moves, bad=()):
        """When the chart is exhausted: walk to the nearest un-budgeted dead end and `search`.

        Returns ("search", None) when standing on the probe cell, ("walk", hop), or None.
        """
        seen, order = self._bfs(start, moves, bad)
        for c in order:
            if self.searched.get(c, 0) >= SEARCH_BUDGET or not self.dead_end(c, moves):
                continue
            if c == start:
                self.searched[c] = self.searched.get(c, 0) + 1
                return "search", None
            hop = self._first_hop(seen, c)
            if hop:
                return "walk", hop
        return None
