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
        elif ch == " ":
            kind = "stone"
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
    names = {}
    for noun, dirs in re.findall(r"(?m)^\s*([a-z ]+?) adjacent ([a-z, ]+?)\s*$", text):
        found = set(re.findall(r"\b(?:north|south)(?:east|west)?\b|\beast\b|\bwest\b", dirs))
        if noun.strip() in ("lava", "water"):
            frame["lava"] |= found
        else:
            for d in found:
                names[d] = noun.strip()
    if "map:" not in text or not m:
        return frame
    rows = text.split("map:", 1)[1].split("\n")[1:]
    rows = [r for r in rows if "Dlvl:" not in r and not ("St:" in r and "Dx:" in r)]
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= y - 1 < len(rows) and 0 <= x < len(rows[y - 1]) and rows[y - 1][x] == "@"):
        return frame
    frame["pos"] = (x, y)
    frame["ring"] = ring_of_eight(rows, x, y, frame["bars"])
    for d, (kind, ch, _) in frame["ring"].items():
        if kind == "creature":  # a foe is a LETTER on this frame; prose only names it
            frame["foes"][d] = names.get(d, "creature `%s`" % ch)
    for yy, row in enumerate(rows):
        i = row.find(">")
        if i >= 0:
            frame["down"] = (i, yy + 1)
    return frame
