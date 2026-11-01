"""perception.py - decode ONE BALROG MiniHack text observation into a frame dict.

Observation-only: message / language observation / cursor / map. Map row index == cursor.y - 1.
Also extracts the prose door facts (`open door adjacent east`, `doorway adjacent north`) that the
glyph layer loses once a `+` is opened and redrawn as `-`/`|`.
"""
from __future__ import annotations

import re

STEP = {
    "north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1),
}
ORDER = list(STEP)
DIAG = {"northeast", "southeast", "southwest", "northwest"}
SECTIONS = ["message", "language observation", "cursor", "map"]
DIR_RE = r"(northeast|northwest|southeast|southwest|north|south|east|west)"


def step(cell, d):
    return (cell[0] + STEP[d][0], cell[1] + STEP[d][1])


def clause_index(lang):
    """Prose clauses `<noun> adjacent <dirs>` -> {noun: set(dirs)}."""
    out = {}
    for line in (lang or "").splitlines():
        m = re.match(r"^\s*(.+?)\s+adjacent\s+(.+?)\s*$", line.strip())
        if not m:
            continue
        dirs = re.findall(DIR_RE, m.group(2))
        out.setdefault(m.group(1).strip().lower(), set()).update(dirs)
    return out


def door_bearings(adjacent):
    """Split prose door nouns: `doors` seal the diagonal, `doorways` (doorless/broken) do not."""
    doors, doorways = set(), set()
    for noun, dirs in adjacent.items():
        if "doorway" in noun or "broken door" in noun:
            doorways |= dirs
        elif re.search(r"\b(open|closed)?\s*door\b", noun):
            doors |= dirs
    return doors, doorways


def parse_glyph_map(text):
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
    rows = parts["map"].split("\n")[:21]
    cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", parts["cursor"])
    pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
    dl = re.search(r"Dlvl:(\d+)", text)
    msg = parts["message"].strip()
    adjacent = clause_index(parts["language observation"])
    doors, doorways = door_bearings(adjacent)
    return {
        "message": msg,
        "adjacent": adjacent,
        "door_dirs": doors,
        "doorway_dirs": doorways,
        "rows": rows,
        "pos": pos,
        "dlvl": dl.group(1) if dl else "?",
        # cursor on the message line (row 0) means a modal prompt owns the keyboard
        "prompt": ("[yn" in msg) or ("What do you want" in msg) or ("In what direction" in msg)
        or msg.endswith("--More--") or (pos is not None and pos[1] < 0),
    }


def glyph(fr, cell):
    x, y = cell
    rows = fr["rows"]
    if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
        return rows[y][x]
    return " "
