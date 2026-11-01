"""tools/frame.py - the observation parser: message / prose / cursor / map rows, and reply projection."""
from __future__ import annotations

import re

STEP = {
    "north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1),
}
ORDER = list(STEP)
CARDINALS = ORDER[:4]
LETHAL = set("}")
WALL = set("|- ")
SECTIONS = ["message", "language observation", "cursor", "map"]


def clause_index(lang):
    """Prose clauses `<noun> adjacent <dirs>` -> {noun: set(dirs)}."""
    out = {}
    for line in (lang or "").splitlines():
        m = re.match(r"^\s*(.+?)\s+adjacent\s+(.+?)\s*$", line.strip())
        if not m:
            continue
        dirs = re.findall(r"(northeast|northwest|southeast|southwest|north|south|east|west)", m.group(2))
        out.setdefault(m.group(1).strip().lower(), set()).update(dirs)
    return out


def parse_glyph_map(text):
    """Split message / prose / cursor / rows; row index = cursor.y - 1."""
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
    return {
        "message": msg,
        "adjacent": clause_index(parts["language observation"]),
        "rows": rows,
        "pos": pos,
        "dlvl": dl.group(1) if dl else "?",
        "prompt": ("[yn" in msg) or ("What do you want" in msg) or ("In what direction" in msg)
        or msg.endswith("--More--"),
    }


def glyph(fr, cell):
    x, y = cell
    rows = fr["rows"]
    if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
        return rows[y][x]
    return " "


def decide_action(reply, legal, allowed=None):
    """Whole-word projection of a reply onto the legal vocabulary (optionally a subset of it)."""
    text = (reply or "").lower()
    pool = [a for a in legal if allowed is None or a in allowed]
    hits = []
    for name in pool:
        for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?![a-z])", text):
            hits.append((m.start(), -len(name), name))
    return sorted(hits)[0][2] if hits else None
