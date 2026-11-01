"""field.py - frame decoder: message / prose clauses / cursor / map rows (row = cursor.y - 1)."""
from __future__ import annotations

import re

STEP = {
    "north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1),
}
SECTIONS = ["message", "language observation", "cursor", "map"]
WORDS = "northeast|northwest|southeast|southwest|north|south|east|west"


def clauses(lang):
    """Prose lines `<noun> adjacent <dirs>` -> {noun: set(dirs)}."""
    out = {}
    for line in (lang or "").splitlines():
        m = re.match(r"^\s*(.+?)\s+adjacent\s+(.+?)\s*$", line.strip())
        if m:
            out.setdefault(m.group(1).strip().lower(), set()).update(re.findall(WORDS, m.group(2)))
    return out


def parse_tty_frame(text):
    """Split the observation into its sections; no chart state is touched here."""
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
    modal = (("[yn" in msg) or ("What do you want" in msg) or ("In what direction" in msg)
             or msg.endswith("--More--") or (pos is not None and pos[1] < 0))
    return {"message": msg, "pos": pos, "rows": parts["map"].split("\n")[:21], "modal": modal,
            "adjacent": clauses(parts["language observation"]), "level": dl.group(1) if dl else "?"}


def glyph(fr, cell):
    x, y = cell
    rows = fr["rows"]
    return rows[y][x] if 0 <= y < len(rows) and 0 <= x < len(rows[y]) else " "
