"""sokoban.py - the Boxoban stage owner: an exact push-level A* over the drawn frame.

Armed from the observation only: the legal moves are exactly the 4 cardinals (Boxoban offers no
diagonals) and the frame draws a boulder `` ` `` and a fountain `{`. The level is premapped and lit;
`#` is iron bars, so `#`, `-`, `|` and blank are all walls. Boulders are read from the CURRENT frame
(they move), fountains are ACCUMULATED across frames (a goal under the hero or a boulder is drawn
once uncovered). The hero's square is floor, never a boulder.

The plan is solved once and replayed one move per step; any mismatch between the expected
(hero, boulders) and the observed one, or a newly drawn goal, triggers a replan from this frame.
"""
from __future__ import annotations

import heapq
import time
from collections import deque

CARD = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
WALLS = set("#-| ")
BOULDER = set("`0")
GOAL = "{"


def _add(c, d):
    return (c[0] + d[0], c[1] + d[1])


def read(rows, pos):
    """-> floor set, boulders frozenset, goals set drawn in this frame."""
    floor, boxes, goals = set(), set(), set()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in WALLS:
                continue
            floor.add((x, y))
            if ch in BOULDER:
                boxes.add((x, y))
            elif ch == GOAL:
                goals.add((x, y))
    if pos is not None:
        floor.add(pos)
        boxes.discard(pos)
    return floor, frozenset(boxes), goals


def _pull_dist(floor, goals):
    """Min pushes from each square to its nearest goal (walls only; boulders ignored)."""
    dist = {g: 0 for g in goals}
    q = deque(goals)
    while q:
        b = q.popleft()
        for d in CARD.values():
            prev = (b[0] - d[0], b[1] - d[1])       # box came from here, pushed along d
            player = (prev[0] - d[0], prev[1] - d[1])
            if prev in floor and player in floor and prev not in dist:
                dist[prev] = dist[b] + 1
                q.append(prev)
    return dist


def _reach(floor, boxes, start):
    seen = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        for a, d in CARD.items():
            n = _add(c, d)
            if n in floor and n not in boxes and n not in seen:
                seen[n] = (c, a)
                q.append(n)
    return seen


def _path(seen, target):
    out = []
    while seen[target] is not None:
        target, a = seen[target]
        out.append(a)
    return out[::-1]


def solve(floor, boxes, goals, pos, budget=4.0, weight=1.0, cap=400000):
    """Push-level A* (cost = moves). Returns a list of cardinal moves, or None."""
    goals = frozenset(goals)
    if not boxes or not goals or len(goals) > len(boxes):
        return None
    live = _pull_dist(floor, goals)
    if any(g not in floor for g in goals):
        return None

    def h(bx):
        tot = 0
        for b in bx:
            if b not in live:
                return None
            tot += live[b]
        return tot

    h0 = h(boxes)
    if h0 is None:
        # a boulder already on a dead square that is not a goal-covering need: tolerate extras
        extra = [b for b in boxes if b not in live]
        if len(boxes) - len(extra) < len(goals):
            return None
        live.update({b: 0 for b in extra})
        h0 = h(boxes)
    t0 = time.time()
    start_reach = _reach(floor, boxes, pos)
    key0 = (min(start_reach), boxes)
    best_g = {key0: 0}
    parent = {key0: (None, pos, [])}
    tick = 0
    heap = [(weight * h0, 0, tick, key0, pos)]
    while heap:
        f, g, _, key, p = heapq.heappop(heap)
        if g > best_g.get(key, 1 << 30):
            continue
        bx = key[1]
        if goals <= bx:
            plan = []
            while key is not None:
                par, _, moves = parent[key]
                plan = moves + plan
                key = par
            return plan
        if len(best_g) > cap or time.time() - t0 > budget:
            return None
        seen = _reach(floor, bx, p)
        for b in bx:
            for a, d in CARD.items():
                stand = (b[0] - d[0], b[1] - d[1])
                if stand not in seen:
                    continue
                dest = _add(b, d)
                if dest not in floor or dest in bx or dest not in live:
                    continue
                nb = frozenset((bx - {b}) | {dest})
                hh = h(nb)
                if hh is None:
                    continue
                walk = _path(seen, stand)
                ng = g + len(walk) + 1
                nreach = _reach(floor, nb, b)
                nkey = (min(nreach), nb)
                if ng < best_g.get(nkey, 1 << 30):
                    best_g[nkey] = ng
                    parent[nkey] = (key, b, walk + [a])
                    tick += 1
                    heapq.heappush(heap, (ng + weight * hh, ng, tick, nkey, b))
    return None


def simulate(pos, boxes, move):
    d = CARD[move]
    n = _add(pos, d)
    if n in boxes:
        return n, frozenset((boxes - {n}) | {_add(n, d)})
    return n, boxes


class Sokoban:
    """Per-level plan cache; `forget` on reset/level change."""

    def __init__(self):
        self.goals = set()
        self.start = None
        self.plan = []
        self.expect = None
        self.planned_goals = None
        self.failed = set()

    @staticmethod
    def armed(moves, rows):
        if set(moves) != set(CARD):
            return False
        flat = "".join(rows)
        return any(ch in flat for ch in BOULDER) and GOAL in flat

    def target_goals(self, floor, boxes, drawn):
        goals = set(self.goals)
        if len(goals) < len(boxes) and self.start is not None and self.start not in drawn:
            goals.add(self.start)                   # a fountain may hide under the start square
        return goals

    def step(self, rows, pos, drawn):
        """-> (move, why) or None to abstain."""
        if self.start is None:
            self.start = pos
        floor, boxes, seen_goals = read(rows, pos)
        self.goals |= seen_goals
        goals = self.target_goals(floor, boxes, drawn)
        state = (pos, boxes)
        if self.plan and (self.expect != state or self.planned_goals != goals):
            self.plan = []
        if not self.plan:
            tag = (pos, boxes, frozenset(goals))
            if tag in self.failed:
                return None
            plan = solve(floor, boxes, goals, pos, budget=4.0, weight=1.0)
            if plan is None:
                plan = solve(floor, boxes, goals, pos, budget=6.0, weight=3.0)
            if not plan:
                self.failed.add(tag)
                return None
            self.plan = plan
            self.planned_goals = goals
        move = self.plan.pop(0)
        self.expect = simulate(pos, boxes, move)
        return move, "sokoban:plan:%d" % len(self.plan)
