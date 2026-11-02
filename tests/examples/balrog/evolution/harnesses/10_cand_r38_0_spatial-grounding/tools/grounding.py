"""grounding.py — turn percept + spatial memory into a GROUNDED MOVE MENU and a planned route.

This is where the decision space is shaped. For every legal compass move it computes the target
square's glyph, the prose nouns the env reports there, how often we have stood there, and whether
the env already refused that exact (cell, dir). Moves that are provably lethal (lava/water named
adjacent, or drawn `}`) or already refused are VETOED — they never reach the env.

It also plans: BFS over the remembered map to a drawn `>` if one is reachable, else to the nearest
least-visited frontier square (a passable square touching undrawn space). The first hop of that
route is the RECOMMENDED move, which is both shown to the model and used as the fallback.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from perception import DIRS, step

LETHAL_NOUNS = ("lava", "water", "pool", "moat")
WALL_NOUNS = ("wall", "stone", "bars", "tree", "room corner")
OPENING_NOUNS = ("doorway", "open door", "broken door")
WALL_GLYPHS = set("-|")
LETHAL_GLYPHS = set("}")
BLOCK_GLYPHS = WALL_GLYPHS | LETHAL_GLYPHS | {" "}
DIAG = {"northeast", "northwest", "southeast", "southwest"}


@dataclass
class MoveOption:
    direction: str
    glyph: str
    nouns: list
    visits: int
    veto: str = ""          # non-empty -> reason this move is ruled out


@dataclass
class Grounding:
    options: list = field(default_factory=list)
    recommended: str | None = None
    route_goal: str = ""
    route_len: int = 0

    @property
    def allowed(self):
        return [o for o in self.options if not o.veto]


def _passable(g):
    return g is not None and g not in BLOCK_GLYPHS


def _neighbours(cell, seen, refused, legal_dirs):
    for d in legal_dirs:
        n = step(cell, d)
        if (cell, d) in refused:
            continue
        if not _passable(seen.get(n)):
            continue
        # doorways forbid diagonal entry/exit in NetHack
        if d in DIAG and (seen.get(n) in ("+",) or seen.get(cell) in ("+",)):
            continue
        yield d, n


def plan_route(pos, seen, refused, visits, legal_dirs):
    """BFS; returns (first_dir, goal_kind, length) or (None, '', 0)."""
    if pos is None:
        return None, "", 0
    prev = {pos: None}
    q = deque([pos])
    order = []
    while q:
        c = q.popleft()
        order.append(c)
        for d, n in _neighbours(c, seen, refused, legal_dirs):
            if n not in prev:
                prev[n] = (c, d)
                q.append(n)

    def first_hop(target):
        c, length = target, 0
        d0 = None
        while prev[c] is not None:
            pc, d = prev[c]
            d0, c = d, pc
            length += 1
        return d0, length

    stairs = [c for c in order if seen.get(c) == ">" and c != pos]
    if stairs:
        d, n = first_hop(stairs[0])
        return d, "stairs down `>`", n

    def frontier(c):
        # a square we have STOOD on has already drawn everything it can reveal: retire it,
        # otherwise two stood squares bordering dark rock elect each other forever (A<->B orbit)
        if visits[c] > 0:
            return False
        return any(seen.get(step(c, d)) is None for d in DIRS)

    best, best_key = None, None
    for i, c in enumerate(order):
        if c == pos or not frontier(c):
            continue
        d, n = first_hop(c)
        key = (n, i)                         # nearest unstood frontier (a total order)
        if best_key is None or key < best_key:
            best, best_key = c, key
    if best is not None:
        d, n = first_hop(best)
        return d, "nearest unexplored edge", n
    return None, "", 0


def ground(percept, memory, legal):
    legal_dirs = [d for d in DIRS if d in legal]
    g = Grounding()
    pos = percept.pos
    if pos is None:
        return g
    for d in legal_dirs:
        tgt = step(pos, d)
        glyph = percept.grid.get(tgt, memory.seen.get(tgt, " "))
        nouns = sorted(percept.adjacent.get(d, ()))
        o = MoveOption(d, glyph, nouns, memory.visits[tgt])
        opening = any(any(k in n for k in OPENING_NOUNS) for n in nouns)
        if any(any(k in n for k in LETHAL_NOUNS) for n in nouns) or glyph in LETHAL_GLYPHS:
            o.veto = "lethal (" + (", ".join(nouns) or glyph) + ")"
        elif (pos, d) in memory.refused:
            o.veto = "already refused from this square"
        elif not opening and any(any(k in n for k in WALL_NOUNS) for n in nouns) and \
                not any(k in n for n in nouns for k in ("doorway", "door")):
            o.veto = "blocked (" + ", ".join(nouns) + ")"
        elif glyph in WALL_GLYPHS and not opening and not nouns:
            o.veto = "wall glyph"
        g.options.append(o)
    rec, kind, n = plan_route(pos, {**memory.seen, **{c: v for c, v in percept.grid.items() if v != "@"}, pos: "."},
                              memory.refused, memory.visits, legal_dirs)
    allowed = {o.direction for o in g.allowed}
    if rec in allowed:
        g.recommended, g.route_goal, g.route_len = rec, kind, n
    elif g.allowed:
        # no route: least-visited allowed move (ties broken by fixed compass order)
        g.recommended = min(g.allowed, key=lambda o: (o.visits, legal_dirs.index(o.direction))).direction
        g.route_goal, g.route_len = "least-visited neighbour", 1
    return g


def render(g, percept, memory) -> str:
    lines = ["SPATIAL GROUNDING (computed by the harness from this observation and your own history):"]
    if percept.pos:
        lines.append(f"you are at x={percept.pos[0]}, y={percept.pos[1]}; stood here {memory.visits[percept.pos]} time(s)")
    if percept.message:
        lines.append(f"last game message: {percept.message}")
    lines.append("moves:")
    for o in g.options:
        what = ", ".join(o.nouns) if o.nouns else f"glyph '{o.glyph}'"
        tag = f"RULED OUT — {o.veto}" if o.veto else f"allowed, visited {o.visits}x"
        lines.append(f"  {o.direction:<10} -> {what}; {tag}")
    if g.recommended:
        lines.append(f"recommended: {g.recommended}  (route to {g.route_goal}, {g.route_len} step(s))")
    return "\n".join(lines)
