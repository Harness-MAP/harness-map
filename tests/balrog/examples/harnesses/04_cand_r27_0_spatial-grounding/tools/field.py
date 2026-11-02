"""field.py - frame decoder and the potential field it feeds."""
from __future__ import annotations

import re
from collections import deque

STEP = {
    "north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1),
}
SOLID = set("|- }")
SECTIONS = ["message", "language observation", "cursor", "map"]
WORDS = "northeast|northwest|southeast|southwest|north|south|east|west"


def flood_field(chart):
    """Reverse multi-source Dijkstra: 0 at a drawn `>` (else at every open-edge cell), cost grows with visits."""
    terrain, visits, blocked = chart["terrain"], chart["visits"], chart["blocked"]

    def open_cell(c):
        ch = terrain.get(c)
        return ch is not None and ch not in SOLID

    seeds = [c for c, ch in terrain.items() if ch == ">"]
    if not seeds:
        seeds = [c for c in terrain if open_cell(c) and not visits.get(c)
                 and any((c[0] + dx, c[1] + dy) not in terrain for dx, dy in STEP.values())]
    pot = {c: 0.0 for c in seeds}
    q = deque(seeds)
    while q:
        cur = q.popleft()
        for d, (dx, dy) in STEP.items():
            prev = (cur[0] - dx, cur[1] - dy)
            if not open_cell(prev) or (prev, d) in blocked:
                continue
            cost = pot[cur] + 1.0 + 0.5 * min(visits.get(prev, 0), 6)
            if cost < pot.get(prev, 1e9):
                pot[prev] = cost
                q.append(prev)
    return pot


def parse_tty_frame(text, chart):
    """Decode message/prose/cursor/map (row = cursor.y - 1), fold into chart, refresh chart['field']."""
    text = text or ""
    parts = {}
    for i, name in enumerate(SECTIONS):
        m = re.search(rf"(?m)^{name}:[ \t]*\n?", text)
        if not m:
            parts[name] = ""
            continue
        rest = text[m.end():]
        cut = len(rest)
        for other in SECTIONS[i + 1:]:
            k = re.search(rf"(?m)^{other}:", rest)
            if k:
                cut = min(cut, k.start())
        parts[name] = rest if name == "map" else rest[:cut]
    msg = parts["message"].strip()
    cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", parts["cursor"])
    pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
    dl = re.search(r"Dlvl:(\d+)", text)
    level = dl.group(1) if dl else "?"
    modal = ("[yn" in msg) or ("What do you want" in msg) or ("In what direction" in msg) or msg.endswith("--More--")
    if level != chart.get("level"):
        chart.update(level=level, terrain={}, visits={}, blocked=set(), pending=None)
    rows = parts["map"].split("\n")[:21]
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in " @":
                chart["terrain"][(x, y)] = ch
    lethal = set()
    for line in parts["language observation"].splitlines():
        if "lava" in line or "water" in line:
            if "adjacent" in line:
                lethal |= set(re.findall(WORDS, line.split("adjacent", 1)[1]))
    if pos is not None:
        if chart["terrain"].get(pos, "@") == "@":
            chart["terrain"][pos] = "."
        pend = chart.get("pending")
        if pend and not modal and pend[0] == pos:
            chart["blocked"].add(pend)
        chart["visits"][pos] = chart["visits"].get(pos, 0) + 1
        for d, (dx, dy) in STEP.items():
            if chart["terrain"].get((pos[0] + dx, pos[1] + dy)) == "}":
                lethal.add(d)
    chart["pending"] = None
    chart["field"] = flood_field(chart)
    return {"message": msg, "pos": pos, "rows": rows, "modal": modal, "lethal": lethal}
