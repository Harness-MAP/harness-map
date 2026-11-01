"""terrain.py - square-level terrain semantics: a glyph is only a hint, a prose noun convicts the SQUARE.

MiniHack draws an OPEN door as `-`/`|` (same as a wall) and Boxoban draws iron BARS as `#` (same as a
corridor). The glyph table alone therefore gets both wrong. This module turns the adjacent-noun prose
(`open door adjacent northwest`, `bars adjacent north`) into per-cell facts the chart consults first.
"""
from __future__ import annotations

STEP = {
    "north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1),
}
DIAG = {"northeast", "southeast", "southwest", "northwest"}
LETHAL = set("}")
WALL = set("|- ")

# fact kinds, most specific first
DOOR = "door"          # real door (open or closed): passable, NO diagonal in or out
DOORWAY = "doorway"    # doorless / broken: passable, diagonals fine
SOLID = "solid"        # wall, stone, bars, tree: never passable
LIQUID = "liquid"      # lava / water: never passable


def noun_kind(noun):
    n = noun.lower()
    if "lava" in n or "water" in n or "moat" in n:
        return LIQUID
    if "doorway" in n or "broken door" in n:
        return DOORWAY
    if "door" in n:
        return DOOR
    if "bars" in n or "wall" in n or "stone" in n or "tree" in n:
        return SOLID
    return None


def prose_cell_facts(adjacent, pos):
    """{cell: kind} for every adjacent prose noun whose kind matters to passability."""
    out = {}
    if pos is None:
        return out
    for noun, dirs in (adjacent or {}).items():
        kind = noun_kind(noun)
        if kind is None:
            continue
        for d in dirs:
            if d in STEP:
                out[(pos[0] + STEP[d][0], pos[1] + STEP[d][1])] = kind
    return out


def level_has_bars(adjacent):
    return any("bars" in noun for noun in (adjacent or {}))


def passable(glyph, fact, bars_level=False):
    """Square verdict: a prose fact outranks the glyph; `#` is bars on a level that names bars."""
    if fact in (DOOR, DOORWAY):
        return True
    if fact in (SOLID, LIQUID):
        return False
    if glyph is None or glyph in WALL or glyph in LETHAL:
        return False
    if glyph == "#" and bars_level:
        return False
    return True


def edge_ok(src, dst, move, facts, glyph_at):
    """NetHack: no diagonal step into or out of a real door square (`+` ever drawn counts as a door)."""
    if move not in DIAG:
        return True
    for cell in (src, dst):
        if facts.get(cell) == DOOR or glyph_at(cell) == "+":
            return False
    return True
