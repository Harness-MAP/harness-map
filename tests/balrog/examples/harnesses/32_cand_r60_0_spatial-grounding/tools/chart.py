"""chart.py - a per-level remembered map and the BFS route that owns the move when one exists."""
from __future__ import annotations

import re
from collections import deque

from grid import BEARINGS

PASSABLE = set(".#<>+%()[]!?=/\"$*{^_:")
ORDER = ["north", "east", "south", "west", "northeast", "southeast", "southwest", "northwest"]


def new_chart():
    return {"level": None, "terrain": {}, "stood": set(), "blocked": set()}


def message_of(text):
    m = re.search(r"(?m)^message:[ \t]*(.*)$", text or "")
    return (m.group(1) if m else "").strip()


def merge(chart, text, frame):
    """Fold this frame's `map:` rows into the chart (blank and `@` never overwrite a known glyph)."""
    dl = re.search(r"Dlvl:(\d+)", text or "")
    level = dl.group(1) if dl else chart["level"]
    if level != chart["level"]:
        chart.update(level=level, terrain={}, stood=set(), blocked=set())
    if not text or "map:" not in text:
        return
    rows = text.split("map:", 1)[1].split("\n")[1:]
    rows = [r for r in rows if "Dlvl:" not in r and not ("St:" in r and "Dx:" in r)]
    for yy, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in " @":
                chart["terrain"][(x, yy + 1)] = ch
    if frame["pos"] is not None:
        chart["stood"].add(frame["pos"])


def passable(chart, c, bars):
    if c in chart["stood"]:
        return True  # a square we stood on is floor, whatever glyph now draws it (open doors are `-`/`|`)
    ch = chart["terrain"].get(c)
    if ch is None or ch == "}":
        return False
    if ch == "#" and bars:
        return False
    return ch in PASSABLE


def route(chart, pos, bars, banned=()):
    """First hop of the shortest path to a charted `>`, else to the nearest frontier square.

    A frontier is a passable, never-stood square with an uncharted neighbour.
    Returns (hop, kind) or (None, None).
    """
    if pos is None:
        return None, None
    goals = {c for c, ch in chart["terrain"].items() if ch == ">"}
    kind = "stairs"

    def frontier(c):
        if c in chart["stood"] or not passable(chart, c, bars):
            return False
        return any((c[0] + dx, c[1] + dy) not in chart["terrain"] for dx, dy in BEARINGS.values())

    for attempt in (0, 1):
        first = {}
        seen = {pos}
        q = deque([pos])
        while q:
            cur = q.popleft()
            if cur != pos and ((attempt == 0 and cur in goals) or (attempt == 1 and frontier(cur))):
                return first[cur], kind
            for d in ORDER:
                dx, dy = BEARINGS[d]
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in seen or (cur, d) in chart["blocked"] or not passable(chart, nxt, bars):
                    continue
                if cur == pos and d in banned:
                    continue
                seen.add(nxt)
                first[nxt] = first.get(cur, d)
                q.append(nxt)
        kind = "frontier"
    return None, None
