"""spatial_memory.py — the episode's self-derived spatial state (NEVER oracle).

  visits   — (x, y) -> how many decisions were taken standing there
  refused  — set of (cell, direction) the env refused (a bump convicts the SQUARE+DIR, never the glyph)
  seen     — (x, y) -> last glyph ever drawn there (the map persists what was seen once)
Cleared in reset(); also cleared when the level (Dlvl) or the system prompt changes.
"""
from __future__ import annotations

from collections import Counter

from perception import DIRS, step

REFUSAL_MARKERS = (
    "solid stone", "cannot pass", "in vain", "you can't move diagonally", "there is a wall",
    "it's a wall", "is locked", "door is closed", "you are carrying too much", "no room",
    "perhaps that's why you cannot move past it",
)
# a peaceful blocker: the move is refused, but only for a while (it may wander off)
TEMP_REFUSAL_MARKERS = ("really attack", "unknown command", "you displace", "you stop. ")
TEMP_TTL = 6
COMBAT_MARKERS = ("you hit", "you miss", "you kill", "bites", "hits!", "misses", "you destroy")


class SpatialMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.visits = Counter()
        self.refused = set()
        self.temp = {}             # (cell, dir) -> step at which the temporary conviction expires
        self.searched = Counter()  # probed cell -> `search` turns spent there
        self.t = 0
        self.seen = {}
        self.last_pos = None
        self.last_move = None
        self.trail = []

    def observe(self, percept):
        """Fold one percept in; convict (cell, dir) if the previous move did not move us."""
        for c, g in percept.grid.items():
            if g != "@":
                self.seen[c] = g
        pos = percept.pos
        if pos is None:
            return
        msg = percept.message.lower()
        if self.last_move and self.last_pos == pos:
            fought = any(m in msg for m in COMBAT_MARKERS)
            if any(m in msg for m in TEMP_REFUSAL_MARKERS):
                self.temp[(pos, self.last_move)] = self.t + TEMP_TTL
            elif not fought and (any(m in msg for m in REFUSAL_MARKERS) or not msg):
                self.refused.add((pos, self.last_move))
        if "the door opens" in msg or "door crashes open" in msg or ("wham" in msg and "open" in msg):
            self.refused = {(c, d) for (c, d) in self.refused if c != pos}
        self.t += 1
        self.temp = {k: v for k, v in self.temp.items() if v > self.t}
        self.visits[pos] += 1
        self.trail.append(pos)
        self.last_pos = pos

    def note_action(self, action):
        self.last_move = action if action in DIRS else None
        if action == "search" and self.last_pos is not None:
            self.searched[self.last_pos] += 1

    @property
    def blocked(self):
        """Permanent + live temporary convictions — what the planner must not route through."""
        return self.refused | set(self.temp)

    def neighbour_visits(self, pos, d):
        return self.visits[step(pos, d)]
