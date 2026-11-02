"""perception.py — decode ONE BALROG MiniHack text observation into structured spatial facts.

Observation-only. Reads the rendered text the wrapper hands the agent (obs["text"]): the `message:`
line, the `language observation:` prose, the `cursor:` (x, y) and the `map:` block. Nothing else.

Facts produced (`Percept`):
  message  — the top-line game message ('' when none)
  pos      — (x, y) of the agent from the cursor block, or None
  grid     — dict (x, y) -> glyph decoded from the map block, row-aligned so the '@' lands on the
             cursor cell (empirically map row index == cursor.y - 1)
  adjacent — dict direction -> set of prose nouns reported "adjacent <dirs>"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DIRS = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, -1), "northwest": (-1, -1), "southeast": (1, 1), "southwest": (-1, 1),
}


@dataclass
class Percept:
    message: str = ""
    pos: tuple | None = None
    grid: dict = field(default_factory=dict)
    adjacent: dict = field(default_factory=dict)
    raw: str = ""


def obs_text(obs) -> str:
    """Flatten whatever text the wrapper provides into one string."""
    if isinstance(obs, str):
        return obs
    t = obs.get("text", obs) if isinstance(obs, dict) else obs
    if isinstance(t, dict):
        parts = [t.get("short_term_context") or "", t.get("long_term_context") or ""]
        # keep order stable; drop exact duplicates
        seen, out = set(), []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return "\n".join(out)
    return str(t or "")


def _section(text: str, name: str, stops=("message:", "language observation:", "cursor:", "map:")):
    m = re.search(rf"(?m)^{re.escape(name)}\s*$", text)
    if not m:
        return None
    rest = text[m.end():]
    ends = [rest.find("\n" + s) for s in stops if s != name]
    ends = [e for e in ends if e >= 0]
    return rest[: min(ends)] if ends else rest


def _split_dirs(s: str):
    s = s.replace(" and ", ",")
    return [d.strip() for d in s.split(",") if d.strip() in DIRS]


def parse(obs) -> Percept:
    text = obs_text(obs)
    p = Percept(raw=text)
    msg = _section(text, "message:")
    if msg is not None:
        p.message = " ".join(msg.split())
    lang = _section(text, "language observation:") or ""
    for line in lang.splitlines():
        m = re.match(r"\s*(.+?) adjacent (.+)$", line.strip())
        if m:
            noun = m.group(1).strip().lower()
            for d in _split_dirs(m.group(2)):
                p.adjacent.setdefault(d, set()).add(noun)
    cur = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
    if cur:
        p.pos = (int(cur.group(1)), int(cur.group(2)))
    mp = re.search(r"(?m)^map:\s*$", text)
    if mp:
        lines = text[mp.end() + 1:].split("\n")
        # the map block ends at the status lines (start with an alphabetic name / 'Dlvl')
        rows = []
        for ln in lines:
            if re.match(r"\s*Dlvl:", ln):
                break
            rows.append(ln)
        # drop the trailing first status line (e.g. 'Agent the Hatamoto  St:..')
        if rows and re.search(r"St:\d+", rows[-1]):
            rows = rows[:-1]
        at = [(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == "@"]
        dy = 0
        if p.pos and at:
            # align so the '@' nearest the cursor sits on the cursor cell
            ax, ay = min(at, key=lambda c: abs(c[0] - p.pos[0]) + abs(c[1] - p.pos[1]))
            dy = p.pos[1] - ay
        elif p.pos:
            dy = 1
        for y, r in enumerate(rows):
            for x, c in enumerate(r):
                if c != " ":
                    p.grid[(x, y + dy)] = c
    return p


def step(pos, d):
    dx, dy = DIRS[d]
    return (pos[0] + dx, pos[1] + dy)
