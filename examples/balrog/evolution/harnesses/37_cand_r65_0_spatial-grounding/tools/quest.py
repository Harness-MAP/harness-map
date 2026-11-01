"""tools/quest.py - (ported from r058/r052 onto tools/chart.Chart via its `terrain`/`edge` view) the lava-level regime the potential field cannot represent (Quest pair only).

Armed on the printed verb list ({pickup, kick, zap} <= legal), which is true of the Quest pair and of
no other task, so every other task never reaches this module. It owns:
  * ACQUISITION - the hero spawns ON a 2-object pile, and `pickup` there is SILENT. A silent pickup
    marks the square a pile; the hero steps to a post beside it whose ray THROUGH the pile ends on a
    wall before any `}` (a kick toward lava burns the horn), kicks it ONCE (a second kick re-piles),
    then picks up `(` before `/` (letter arithmetic: horn-first leaves the wand on a typeable key).
  * CROSSING - holding a horn/wand on a letter some offered verb types, stand on a bank and fire it
    ACROSS the lava; `The lava cools and solidifies.` turns the river into floor on the chart.
  * PROMPTS - `In what direction?` gets the stored aim, `[ij or ?*]` the verb bound to the letter,
    `Improvise?` gets `northwest` (= y).
  * HOLD - with a hostile named in the prose and the hero on a <=2-lane square, spend the turn
    instead of advancing into the open room (a hold, never a retreat).
All inputs are the observation text; nothing privileged."""
from __future__ import annotations

import re
from collections import deque

STEP = {
    "north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1),
}

LETTER_VERB = {
    "a": "apply", "b": "southwest", "c": "close", "e": "eat", "h": "west",
    "j": "south", "k": "north", "l": "east", "n": "southeast", "o": "open",
    "q": "quaff", "s": "search", "u": "northeast", "y": "northwest", "z": "zap",
}
YES = "northwest"
TOOLS = "(/"                      # a horn is `(`, a wand is `/`
FILLER = "!?=\")[*%"               # any other object: picked up only to burn an untypeable letter
BLOCK = set("|- ")
ZAP_CAP = 2
HOLD_CAP = 15
CARDINALS = ("north", "east", "south", "west")
_BRACKET = re.compile(r"\[([a-zA-Z]+)(?:\s+or\s+[^\]]*)?\]")
_ASSIGN = re.compile(r"\b([a-zA-Z]) - ([^.]+)\.")
FOE = re.compile(r"\b(rat|rats|newt|bug|bugs|jackal|fox|kobold|goblin|lizard|bat|snake|ant|beetle|"
                 r"wolf|orc|gnome|dwarf|imp|homunculus|mold|zombie|centipede|worm|blob|hobbit|"
                 r"gecko|lichen|mouse|sewer)\b")


class Quest:
    def __init__(self):
        self.clear()

    def clear(self):
        self.inv = {}
        self.tried = set()
        self.pile = None
        self.kicked = False
        self.opened = False
        self.last_pickup = None
        self.zaps = 0
        self.crossed = False
        self.want = None
        self.aim = None
        self.holds = 0

    @staticmethod
    def armed(legal):
        return {"pickup", "kick", "zap"} <= set(legal)

    # -- reading ---------------------------------------------------------------------------------
    def observe(self, fr):
        msg = fr["message"]
        low = msg.lower()
        if "solidif" in low or "freezes" in low:
            self.crossed = True
        got = _ASSIGN.findall(msg)
        for letter, desc in got:
            self.inv[letter] = desc.lower()
        if self.last_pickup is not None and not fr["modal"]:
            if self.last_pickup == fr["pos"] and not got and "nothing here" not in low \
                    and not self.kicked and self.pile is None:
                self.pile = self.last_pickup          # silent: more than one object lies here
                self.tried.discard(self.pile)
            self.last_pickup = None

    def held(self, word):
        return any(word in d for d in self.inv.values())

    def next_letter(self):
        """The letter NetHack will deal next (lowest free above the kit), once any is known."""
        if not self.inv:
            return None
        return chr(ord(max(self.inv)) + 1)

    def letter_dead(self, legal):
        """True when the next tool picked up would land on a key no offered verb can type."""
        nxt = self.next_letter()
        return nxt is not None and LETTER_VERB.get(nxt) not in legal

    def usable(self, legal):
        best = None
        for letter, desc in sorted(self.inv.items()):
            verb = LETTER_VERB.get(letter)
            if not verb or verb not in legal:
                continue
            if "wand" in desc and "zap" in legal:
                return letter, "zap"
            if "horn" in desc and "apply" in legal and best is None:
                best = (letter, "apply")
        return best

    def hostile(self, fr):
        for line in (fr.get("lang") or "").lower().splitlines():
            if "tame" in line or "peaceful" in line or "corpse" in line:
                continue
            if FOE.search(line):
                return True
        return False

    # -- prompts ---------------------------------------------------------------------------------
    def answer(self, msg, legal):
        low = msg.lower()
        if "in what direction" in low and self.aim:
            d, self.aim = self.aim, None
            return (d, "quest:aim:" + d) if d in legal else None
        if "improvise" in low and YES in legal:
            return YES, "quest:improvise"
        if "what do you want to" in low and self.want:
            hit = _BRACKET.search(msg)
            letters = list(hit.group(1)) if hit else []
            pick, self.want = self.want, None
            verb = LETTER_VERB.get(pick)
            if pick in letters and verb in legal:
                return verb, "quest:letter:" + pick
            self.aim = None
            return ("more", "quest:letter:cancel") if "more" in legal else None
        return None

    # -- geometry --------------------------------------------------------------------------------
    @staticmethod
    def ray_safe(chart, start, d):
        """Does a ray leaving `start` along d stop on a wall (or undrawn) before any lava?"""
        dx, dy = STEP[d]
        x, y = start[0] + dx, start[1] + dy
        for _ in range(16):
            ch = chart.terrain.get((x, y), " ")
            if ch == "}":
                return False
            if ch in BLOCK:
                return True
            x, y = x + dx, y + dy
        return True

    @staticmethod
    def crossing(chart, pos, d):
        dx, dy = STEP[d]
        x, y = pos[0] + dx, pos[1] + dy
        run = 0
        while chart.terrain.get((x, y)) == "}" and run < 12:
            run += 1
            x, y = x + dx, y + dy
        if not run or chart.terrain.get((x, y), " ") in "-|":
            return None
        return run

    def fire_ok(self, kind):
        if kind == "stairs" or self.zaps >= ZAP_CAP:
            return False
        return not self.crossed or kind is None

    def kick_dir(self, chart, pos):
        """Bearing pos -> pile if pos is a safe kick post for the pile, else None."""
        if self.pile is None:
            return None
        for d, (dx, dy) in STEP.items():
            if (pos[0] + dx, pos[1] + dy) == self.pile and self.ray_safe(chart, self.pile, d):
                return d
        return None

    # -- acting ----------------------------------------------------------------------------------
    def cross(self, chart, pos, legal, kind):
        """Fire a held horn/wand across adjacent lava -> (word, rung) or None."""
        item = self.usable(legal)
        if not item or not self.fire_ok(kind):
            return None
        aims = sorted((r, d) for d in CARDINALS if d in legal
                      for r in [self.crossing(chart, pos, d)] if r)
        if not aims:
            return None
        letter, verb = item
        self.want, self.aim = letter, aims[0][1]
        self.zaps += 1
        return verb, "quest:%s:%s" % (verb, self.aim)

    def hold(self, fr, chart, pos, legal, kind):
        if kind == "stairs" or not self.hostile(fr):
            self.holds = 0
            return None
        lanes = sum(1 for d in STEP if chart.edge(pos, d) is not None)
        if lanes > 2 or self.holds >= HOLD_CAP:
            return None
        self.holds += 1
        word = "wait" if "wait" in legal else "search"
        return (word, "quest:hold") if word in legal else None

    def kit(self, chart, pos, legal):
        """A verb on THIS square: open the pile, kick it, pick a tool (or a letter filler) up."""
        if not self.opened:
            self.opened = True
            self.last_pickup = pos
            return "pickup", "quest:pickup:spawn"
        if self.pile is not None and not self.kicked:
            d = self.kick_dir(chart, pos)
            if d:
                self.kicked, self.aim = True, d
                self.tried.discard(self.pile)
                return "kick", "quest:kick:" + d
        sealed = pos == self.pile and not self.kicked
        if pos not in self.tried and chart.terrain.get(pos, ".") in TOOLS and not sealed:
            if chart.terrain.get(pos, ".") == "/" and not self.held("horn") and self.items(chart, "("):
                return None                            # horn first: it decides the wand's letter
            if not self.usable(legal) and self.letter_dead(legal) and self.items(chart, FILLER):
                return None                            # burn the dead letter on a filler first
            self.tried.add(pos)
            self.last_pickup = pos
            return "pickup", "quest:pickup:item"
        if pos not in self.tried and chart.terrain.get(pos, ".") in FILLER and self.filling(chart, legal):
            self.tried.add(pos)
            self.last_pickup = pos
            return "pickup", "quest:pickup:item"
        return None

    def filling(self, chart, legal):
        """Tools still on the floor, none held usable, and the next letter is dead."""
        return bool(self.items(chart)) and not self.usable(legal) and self.letter_dead(legal)

    def items(self, chart, glyphs=TOOLS):
        return {c for c, g in chart.terrain.items() if g in glyphs and c not in self.tried}

    def targets(self, chart, legal, kind):
        """Squares worth walking to: a kick post, then tools (horn first), then lava banks."""
        if self.pile is not None and not self.kicked:
            posts = set()
            for d, (dx, dy) in STEP.items():
                post = (self.pile[0] - dx, self.pile[1] - dy)
                if chart.passable(post) and self.ray_safe(chart, self.pile, d):
                    posts.add(post)
            if posts:
                return posts
        if self.inv and self.filling(chart, legal) and self.items(chart, FILLER):
            return self.items(chart, FILLER)
        if not self.held("horn") and self.items(chart, "("):
            return self.items(chart, "(")
        if self.items(chart):
            return self.items(chart)
        if not self.usable(legal) or not self.fire_ok(kind):
            return set()
        banks = set()
        for c, g in chart.terrain.items():
            if g != "}":
                continue
            for d in CARDINALS:
                dx, dy = STEP[d]
                bank = (c[0] - dx, c[1] - dy)
                if chart.passable(bank) and self.crossing(chart, bank, d):
                    banks.add(bank)
        return banks

    @staticmethod
    def toward(chart, pos, goals, occupied=()):
        """First hop of a shortest charted walk from pos to any goal -> (dir, length) or (None, 0).
        Squares a creature is drawn on this frame are not walked through (a peaceful one re-asks
        `Really attack?` forever)."""
        if not goals or pos in goals:
            return None, 0
        seen, q = {pos: None}, deque([pos])
        while q:
            cur = q.popleft()
            for d in STEP:
                nxt = chart.edge(cur, d)
                if nxt is None or nxt in seen or nxt in occupied:
                    continue
                seen[nxt] = (cur, d)
                if nxt in goals:
                    n, hop = 1, d
                    while seen[nxt][0] != pos:
                        nxt = seen[nxt][0]
                        hop = seen[nxt][1]
                        n += 1
                    return hop, n
                q.append(nxt)
        return None, 0
