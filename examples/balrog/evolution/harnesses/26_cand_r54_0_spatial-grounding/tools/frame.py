"""tools/frame.py - observation parser, glyph tables and reply projection onto the legal vocabulary."""
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
CREATURE = re.compile(r"[a-zA-Z&;:']")
TAME = {"@"}
SECTIONS = ["message", "language observation", "cursor", "map"]
DOOR_NOUNS = ("doorway", "open door", "broken door")


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
    """Decode message / prose / cursor / map rows; map row = cursor.y - 1."""
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
    # a prompt is OPEN only while the cursor sits on the message row (cursor.y == 0)
    open_row = pos is not None and pos[1] == -1
    marker = ("[yn" in msg) or ("What do you want" in msg) or ("In what direction" in msg) \
        or msg.endswith("--More--") or ("?" in msg and "[" in msg)
    return {
        "message": msg,
        "adjacent": clause_index(parts["language observation"]),
        "rows": rows,
        "pos": None if open_row else pos,
        "dlvl": dl.group(1) if dl else "?",
        "prompt": open_row or (pos is None and marker),
        "yn": "[yn" in msg,
    }


def glyph(fr, cell):
    x, y = cell
    rows = fr["rows"]
    if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
        return rows[y][x]
    return " "


def decide_action(reply, legal, banned=()):
    """Whole-word projection of the model reply onto the legal vocabulary, skipping banned moves."""
    text = (reply or "").lower()
    hits = []
    for name in legal:
        for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?![a-z])", text):
            hits.append((m.start(), -len(name), name))
    for _, _, name in sorted(hits):
        if name not in banned:
            return name
    return None
