"""field.py — a multi-hop potential field over the self-charted level (region of r027).

LEVER (code). The r053 menu ranked moves one hop deep: a `>` two squares away or a frontier ten
squares away was invisible to it. Here a reverse multi-source BFS runs over the chart the Ledger
keeps: sources are every drawn `>` (else every unvisited open cell touching an uncharted square),
edge cost grows with how often the square was stood on, and (square, direction) edges that already
failed are cut. Unlike r027, a `-`/`|` that was once charted as a door `+` stays passable (an open
door is drawn `-`/`|`), so opening a door no longer deletes the route through it.
"""
from __future__ import annotations

from collections import deque

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
SOLID = set(" }`0")          # unseen/stone, lava/water, boulders
WALL = set("|-")


def passable(cell, terrain, doors):
    ch = terrain.get(cell)
    if ch is None or ch in SOLID:
        return False
    if ch in WALL:
        return cell in doors
    return True


def flood(terrain, visits, failed, doors):
    """Potential per charted cell: 0 at a source, +1 (+0.5 per past visit, capped) per hop."""
    seeds = [c for c, ch in terrain.items() if ch == ">"]
    if not seeds:
        seeds = [c for c in terrain if passable(c, terrain, doors) and not visits.get(c)
                 and any((c[0] + dx, c[1] + dy) not in terrain for dx, dy in STEP.values())]
    pot = {c: 0.0 for c in seeds}
    q = deque(seeds)
    while q:
        cur = q.popleft()
        for d, (dx, dy) in STEP.items():
            prev = (cur[0] - dx, cur[1] - dy)
            if not passable(prev, terrain, doors) or (prev, d) in failed:
                continue
            cost = pot[cur] + 1.0 + 0.5 * min(visits.get(prev, 0), 6)
            if cost < pot.get(prev, 1e9):
                pot[prev] = cost
                q.append(prev)
    return pot
