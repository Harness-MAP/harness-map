"""route.py — a chart + BFS router over the drawn map (observation + self-derived ledger only).

LEVER (code). Goal election, every step (a total order, so re-election is a fixed point):
  1. a drawn `>` reachable over passable glyphs  -> 'stairs'
  2. else the best FRONTIER: a reachable passable cell with blank neighbours that no stood-on square
     has probed; key (dist - 2*gain, dist, y, x)                   -> 'explore'
  3. else None (the model decides).
Returns the first hop direction. Walls, blanks, boulders, `}`, prose-blocked/deadly cells and
booked failed (square, dir) edges are never walked; diagonals into/out of a `+` door are refused.
"""
from __future__ import annotations

from collections import deque

from context import DELTA

PASSABLE = set(".#<>+{_$)[=!?/\"*(%,:;^'&")
IMPASSABLE = set("|- }`0")


def walkable(ctx, ledger, cell) -> bool:
    x, y = cell
    if cell in ledger.blocked or cell in ledger.deadly:
        return False
    g = ctx.glyph(x, y)
    if g in IMPASSABLE or (g == "#" and ledger.hash_is_bars):
        return False
    return g in PASSABLE or g.isalpha() or g == "@"


def is_creature(ctx, cell) -> bool:
    g = ctx.glyph(*cell)
    return g.isalpha() and cell != ctx.pos


def _step_ok(ctx, ledger, a, d, b) -> bool:
    if (a, d) in ledger.failed or not walkable(ctx, ledger, b):
        return False
    if len(d) > 5 and (ctx.glyph(*a) == "+" or ctx.glyph(*b) == "+"):
        return False                                   # no diagonal through a door
    return True


def plan(ctx, ledger):
    """-> (first_hop_direction, kind, goal_cell) or (None, None, None)."""
    if ctx.pos is None or not ctx.grid:
        return None, None, None
    moves = ctx.moves()
    start = ctx.pos
    first = {start: None}
    dist = {start: 0}
    q = deque([start])
    order = []
    while q:
        a = q.popleft()
        order.append(a)
        if a != start and is_creature(ctx, a):
            continue                                   # attack it, do not path through it
        for d in moves:
            dx, dy = DELTA[d]
            b = (a[0] + dx, a[1] + dy)
            if b in dist or not _step_ok(ctx, ledger, a, d, b):
                continue
            dist[b] = dist[a] + 1
            first[b] = d if a == start else first[a]
            q.append(b)
    stairs = [c for c in order if c != start and ctx.glyph(*c) == ">"]
    if stairs:
        g = min(stairs, key=lambda c: (dist[c], c[1], c[0]))
        return first[g], "stairs", g
    best = None
    for c in order:
        if c == start or is_creature(ctx, c):
            continue
        gain = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (c[0] + dx, c[1] + dy)
                if n != c and ctx.glyph(*n) == " " and not ledger.probed(n) \
                        and n not in ledger.blocked and 0 <= n[1] < 21 and n[0] >= 0:
                    gain += 1
        if gain:
            key = (dist[c] - 2 * gain, dist[c], c[1], c[0])
            if best is None or key < best[0]:
                best = (key, c)
    if best:
        return first[best[1]], "explore", best[1]
    return None, None, None
