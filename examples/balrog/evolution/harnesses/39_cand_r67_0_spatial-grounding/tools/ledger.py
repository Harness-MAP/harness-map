"""ledger.py — self-derived per-level memory: visits, combat-aware failed moves, prose-blocked cells.

LEVER (code). Keyed on `Dlvl`; cleared on a level change or a new task.
  visits[(x, y)]   squares stood on (a square next to a stood-on square has been SEEN: a blank there
                   is stone, not unexplored — this is what separates a real frontier from a wall face)
  failed{(sq,dir)} a move that left the cursor in place with NO combat message (a swing is not a bump)
  blocked{(x, y)}  cells the prose named wall/stone/bars/tree, or lava/water (deadly) — the noun
                   outranks the glyph (`#` is a corridor OR iron bars; only the prose says which)
  hash_is_bars     once any prose line names bars on this level, every `#` is treated as bars
"""
from __future__ import annotations

import re

from context import DELTA

DEADLY = ("lava", "water")
BLOCKING = ("wall", "stone", "bars", "tree")
COMBAT = re.compile(r"\byou (hit|miss|kill|destroy)|\bbites\b|\bhits\b|\bmisses\b|\bkilled\b", re.I)


def noun_kind(nouns) -> str:
    joined = " ".join(nouns or ())
    if any(w in joined for w in DEADLY):
        return "deadly"
    if any(re.search(rf"\b{w}\b", joined) for w in BLOCKING):
        return "blocked"
    return ""


class Ledger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level = None
        self.visits, self.failed, self.blocked, self.deadly = {}, set(), set(), set()
        self.last = None
        self.hash_is_bars = False

    def observe(self, ctx):
        if ctx.level != self.level:
            lvl = ctx.level
            self.reset()
            self.level = lvl
        if re.search(r"(?m)^\s*(iron )?bars (adjacent|very near|near|far)\b", ctx.text):
            self.hash_is_bars = True                   # the prose names `#` as bars on this level
        pos = ctx.pos
        if pos is None:
            return
        if self.last and self.last[0] == pos and not COMBAT.search(ctx.message):
            self.failed.add(self.last)
        self.visits[pos] = self.visits.get(pos, 0) + 1
        for d, nouns in ctx.nouns.items():
            t = ctx.target(d)
            k = noun_kind(nouns)
            if k == "deadly":
                self.deadly.add(t)
            elif k == "blocked":
                self.blocked.add(t)

    def probed(self, cell) -> bool:
        x, y = cell
        return any((x + dx, y + dy) in self.visits for dx in (-1, 0, 1) for dy in (-1, 0, 1))

    def commit(self, pos, action):
        self.last = (pos, action) if pos is not None and action in DELTA else None
