"""route.py — BFS over the decoded `map:` block: the first hop toward `>` or the nearest frontier.

LEVER (code). Observation-only. The map rows are the lines after `map:` (row index = cursor.y - 1,
same convention as tools/ledger.py). A square is walkable when its glyph is terrain the hero can stand
on (`. # < > + { ( _` and item glyphs); walls `|`/`-`, unknown ` `, lava/water `}`, boulders and
creature letters are not. Edges the ledger saw fail (square, dir) are removed, squares the prose names
blocked/deadly are removed, and a `+` door cannot be entered or left diagonally.

  goal     `>` charted and reachable        -> first hop of the shortest path
  frontier nearest walkable, never-stood-on square with an unknown (` `) 8-neighbour
           -> first hop; ties broken by a TOTAL order (dist, visits, y, x, hop index) so that
              re-electing every step is a fixed point (no A->B->A cycles).
Returns (direction, kind, target, length) or None.
"""
from __future__ import annotations

from collections import deque

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
ORDER = list(STEP)
BLOCK = set("|- }`0@^") | {" "}
ITEMS = set("()[]%?!/=\"*$_{<>.#+")


def walkable(g: str) -> bool:
    if g in BLOCK:
        return False
    if g.isalpha():
        return False            # a creature: never route through it
    return g in ITEMS or not g.isalnum()


def glyph(mp, x, y):
    if 0 <= y < len(mp) and 0 <= x < len(mp[y]):
        return mp[y][x]
    return " "


def plan(mp, pos, visits, failed, banned=frozenset(), legal=None):
    """BFS from pos. failed: {((x,y), dir)}; banned: {(x,y)} named blocked/deadly."""
    if pos is None or not mp:
        return None
    dirs = [d for d in ORDER if legal is None or d in legal]
    if not dirs:
        return None
    first = {pos: None}
    dist = {pos: 0}
    q = deque([pos])
    goals, fronts = [], []
    while q:
        cur = q.popleft()
        cg = glyph(mp, *cur) if cur != pos else "."
        for d in dirs:
            if (cur, d) in failed:
                continue
            dx, dy = STEP[d]
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in dist or nxt in banned:
                continue
            g = glyph(mp, *nxt)
            if not walkable(g):
                continue
            if dx and dy and ("+" in (g, cg)):
                continue        # doors refuse diagonal entry and exit
            dist[nxt] = dist[cur] + 1
            first[nxt] = first[cur] or d
            q.append(nxt)
            if g == ">":
                goals.append(nxt)
            elif visits.get(nxt, 0) == 0 and any(
                    glyph(mp, nxt[0] + ex, nxt[1] + ey) == " " for ex, ey in STEP.values()):
                fronts.append(nxt)
    if goals:
        t = min(goals, key=lambda c: (dist[c], c[1], c[0]))
        return first[t], "goal", t, dist[t]
    if fronts:
        t = min(fronts, key=lambda c: (dist[c], visits.get(c, 0), c[1], c[0],
                                       ORDER.index(first[c])))
        return first[t], "frontier", t, dist[t]
    return None
