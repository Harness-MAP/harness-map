"""grid.py - decode one text frame into the hero's square and its eight neighbours."""
from __future__ import annotations

import re

BEARINGS = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, -1), "northwest": (-1, -1), "southeast": (1, 1), "southwest": (-1, 1),
}
WALLS = set("-|")
LIQUID = set("}")
FLOORISH = set(".#<>+%()[]!?=/\"$*`{^_:")


def ring_of_eight(rows, x, y, bars):
    """Classify the eight squares around (x, y)."""
    out = {}
    for name, (dx, dy) in BEARINGS.items():
        cx, cy = x + dx, y + dy
        ch = " "
        if 0 <= cy - 1 < len(rows) and 0 <= cx < len(rows[cy - 1]):
            ch = rows[cy - 1][cx]
        if ch in WALLS or (ch == "#" and bars):
            kind = "wall"
        elif ch in LIQUID:
            kind = "lava"
        elif ch.isalpha() and ch != "@":
            kind = "creature"
        elif ch == ">":
            kind = "stairs down"
        elif ch == "`":
            kind = "boulder"
        elif ch in FLOORISH:
            kind = "open"
        else:
            kind = "dark"
        out[name] = (kind, ch, (cx, cy))
    return out


def parse_glyph_map(text, bars=False):
    """Hero square, ring, prose hazards and the down-stairs square."""
    frame = {"pos": None, "ring": {}, "lava": set(), "foes": {}, "down": None, "bars": False}
    if not text:
        return frame
    frame["bars"] = bars or bool(re.search(r"\bbars\b", text))
    m = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
    for noun, dirs in re.findall(r"(?m)^\s*([a-z ]+?) adjacent ([a-z, ]+?)\s*$", text):
        found = set(re.findall(r"\b(?:north|south)(?:east|west)?\b|\beast\b|\bwest\b", dirs))
        if noun.strip() in ("lava", "water"):
            frame["lava"] |= found
        elif noun.strip() not in ("wall", "dark area", "floor", "doorway", "corridor", "bars",
                                  "boulder", "stairs up", "stairs down", "fountain", "door"):
            for d in found:
                frame["foes"][d] = noun.strip()
    if "map:" not in text or not m:
        return frame
    rows = text.split("map:", 1)[1].split("\n")[1:]
    rows = [r for r in rows if "Dlvl:" not in r and not ("St:" in r and "Dx:" in r)]
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= y - 1 < len(rows) and 0 <= x < len(rows[y - 1]) and rows[y - 1][x] == "@"):
        return frame
    frame["pos"] = (x, y)
    frame["ring"] = ring_of_eight(rows, x, y, frame["bars"])
    for yy, row in enumerate(rows):
        i = row.find(">")
        if i >= 0:
            frame["down"] = (i, yy + 1)
    return frame


SOLID = set("-| }")


def fold_chart(text, chart):
    """Merge this frame's `map:` rows into the level chart, keyed like the cursor: (x, y) = rows[y-1][x]."""
    dl = re.search(r"Dlvl:(\d+)", text or "")
    level = dl.group(1) if dl else chart.get("level", "?")
    fresh = level != chart.get("level")
    if fresh:
        chart.clear()
        chart.update(level=level, terrain={}, probes={})
    if not text or "map:" not in text:
        return fresh
    rows = text.split("map:", 1)[1].split("\n")[1:]
    rows = [r for r in rows if "Dlvl:" not in r and not ("St:" in r and "Dx:" in r)]
    for yy, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "@":
                chart["terrain"].setdefault((x, yy + 1), ".")
            elif ch != " ":
                chart["terrain"][(x, yy + 1)] = ch
    return fresh


def flood_field(chart, visits, refused, bars=False):
    """Reverse Dijkstra-by-relaxation from a drawn `>` (else every unstood open square touching blank rock).

    Refusals apply to every hop; each square's cost grows with how often it was stood on, so a
    field that re-descends each step has no fixed-point 2-cycle."""
    terrain = chart.get("terrain", {})

    def passable(c):
        ch = terrain.get(c)
        return ch is not None and ch not in SOLID and not (ch == "#" and bars)

    seeds = [c for c, ch in terrain.items() if ch == ">"]
    if not seeds:
        seeds = [c for c in terrain if passable(c) and not visits.get(c)
                 and any((c[0] + dx, c[1] + dy) not in terrain for dx, dy in BEARINGS.values())]
    pot = {c: 0.0 for c in seeds}
    queue = list(seeds)
    while queue:
        cur = queue.pop(0)
        for d, (dx, dy) in BEARINGS.items():
            prev = (cur[0] - dx, cur[1] - dy)
            if not passable(prev) or (prev, d) in refused:
                continue
            cost = pot[cur] + 1.0 + 0.5 * min(visits.get(prev, 0), 6)
            if cost < pot.get(prev, 1e9):
                pot[prev] = cost
                queue.append(prev)
    return pot
