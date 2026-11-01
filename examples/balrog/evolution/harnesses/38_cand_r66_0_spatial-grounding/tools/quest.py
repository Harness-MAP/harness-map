"""tools/quest.py - the lava-crossing stage: acquire a horn/wand, walk to a bank, fire it ACROSS the lava.

Owns three things the chart explorer has no notion of: floor ITEMS (worth a `pickup` turn), the
INVENTORY letters read off `x - a <thing>.` messages, and the multi-prompt verb sequence
`apply|zap` -> item letter -> (`Improvise?` yes) -> direction. Letters are typed via the vocabulary
word the env binds to that key (e.g. `f` is not typeable; `a` is `apply`, `h` is `west`)."""
from __future__ import annotations

import re

LETTER_VERB = {
    "a": "apply", "b": "southwest", "c": "close", "e": "eat", "h": "west",
    "j": "south", "k": "north", "l": "east", "n": "southeast", "o": "open",
    "q": "quaff", "s": "search", "u": "northeast", "y": "northwest", "z": "zap",
}
YES = "northwest"                     # the `y` key
ITEM = set("()/!?=\"[*$")             # never `%`: a corpse is never the tool and eats a letter
ZAP_CAP = 4
_BRACKET = re.compile(r"\[([a-zA-Z]+)(?:\s+or\s+[^\]]*)?\]")
_ASSIGN = re.compile(r"\b([a-zA-Z]) - ([^.]+)\.")


class Quest:
    def __init__(self, frame):
        self.F = frame
        self.clear()

    def clear(self):
        self.inv = {}
        self.tried = set()
        self.opened = False
        self.zaps = 0
        self.crossed = False
        self.want = None
        self.aim = None
        self.fired_at = None

    # -- reading -----------------------------------------------------------------------------
    def observe(self, fr, chart):
        msg = fr["message"]
        if "solidif" in msg or "freez" in msg.lower():
            self.crossed = True
        if self.fired_at is not None and chart.glyph(self.fired_at) not in self.F.LETHAL:
            self.crossed = True
            self.fired_at = None
        for letter, desc in _ASSIGN.findall(msg):
            self.inv[letter] = desc.lower()

    @staticmethod
    def armed(chart, legal):
        return "pickup" in legal and any(g in chart.F.LETHAL for g in chart.drawn.values())

    def usable(self, legal):
        for letter, desc in sorted(self.inv.items()):
            verb = LETTER_VERB.get(letter)
            if not verb or verb not in legal:
                continue
            if "wand" in desc and "zap" in legal:
                return letter, "zap"
            if "horn" in desc and "apply" in legal:
                return letter, "apply"
        return None

    # -- the verb's prompts ------------------------------------------------------------------
    def answer(self, msg, legal):
        low = msg.lower()
        if "in what direction" in low and self.aim:
            d, self.aim = self.aim, None
            return d, "quest:aim:" + d
        if "improvise" in low and self.aim and YES in legal:
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

    # -- acting ------------------------------------------------------------------------------
    def crossing(self, chart, pos, d):
        dx, dy = self.F.STEP[d]
        x, y = pos[0] + dx, pos[1] + dy
        run = 0
        while chart.glyph((x, y)) in self.F.LETHAL and run < 12:
            run += 1
            x, y = x + dx, y + dy
        if not run or chart.glyph((x, y)) in "-|" or (x, y) in chart.stood:
            return None
        return run, d

    def fire_ok(self, kind):
        if kind == "stairs" or self.zaps >= ZAP_CAP:
            return False
        return not self.crossed or kind is None

    def step(self, chart, pos, legal, kind):
        """A verb on THIS square -> (word, rung) or None."""
        item = self.usable(legal)
        if item and self.fire_ok(kind):
            letter, verb = item
            aims = [a for a in (self.crossing(chart, pos, d) for d in self.F.CARDINALS if d in legal) if a]
            if aims:
                _run, d = min(aims)
                self.fired_at = (pos[0] + self.F.STEP[d][0], pos[1] + self.F.STEP[d][1])
                self.want, self.aim = letter, d
                self.zaps += 1
                return verb, "quest:%s:%s" % (verb, d)
        if not self.opened:
            self.opened = True
            self.tried.add(pos)
            return "pickup", "quest:pickup:spawn"
        if not item and pos not in self.tried and chart.glyph(pos) in ITEM:
            self.tried.add(pos)
            return "pickup", "quest:pickup:item"
        return None

    def targets(self, chart, legal, kind):
        """Squares worth walking to: uncollected items until a tool is held, then lava banks."""
        if not self.usable(legal):
            return {c for c, g in chart.drawn.items() if g in ITEM and c not in self.tried}
        if not self.fire_ok(kind):
            return set()
        banks = set()
        for c, g in chart.drawn.items():
            if g in self.F.LETHAL:
                for d in self.F.CARDINALS:
                    dx, dy = self.F.STEP[d]
                    bank = (c[0] - dx, c[1] - dy)
                    if chart.passable(bank) and self.crossing(chart, bank, d):
                        banks.add(bank)
        return banks
