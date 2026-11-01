"""harness.py — ENTRYPOINT. Naive transcript + a code-RANKED move menu + legal projection.

Region of cand_r029 (a code-kept neighbour ledger that grounds a single model call), with the
decision aid restructured. r029 rendered an unranked fact table, and its diagnosis showed the model
never reads it: orbits stayed at 38-48% of steps. It also booked a melee swing as a failed move,
so the agent never swung twice at a rat (0/35). Here:
  · the Ledger RANKS the compass moves in code (stairs > adjacent creature > never-stood >
    least-visited, and the square just left goes last), and the model sees a best-first MENU;
  · a no-move counts as `failed` only when it was not a fight (target not a creature, no combat text);
  · walls and bars that the prose names adjacent are vetoed BEFORE the first bump, as lava/water are;
  · orbit override: a reply that walks back to the square just left, while the menu still offers a
    never-stood square, is replaced by menu option 1.
One model call per step, as in naive.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages                  # noqa: E402
from action_io import extract_action, legal_actions   # noqa: E402

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
DIRS = "northeast|northwest|southeast|southwest|north|south|east|west"
DEADLY_NOUNS = ("lava", "water")
BLOCK_NOUNS = ("wall", "bars")
COMBAT = ("you hit", "you miss", "bites", "you kill", "you destroy", "hits!", "misses")
WALLISH = "|- "


class Ledger:
    """Self-derived per-level record: visits per square, failed (square, direction) moves."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.level, self.visits, self.failed = None, {}, set()
        self.last, self.last_glyph, self.prev_pos = None, " ", None

    def observe(self, text):
        text = text or ""
        cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
        pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
        dl = re.search(r"Dlvl:(\d+)", text)
        lvl = dl.group(1) if dl else "?"
        if lvl != self.level:
            self.reset()
            self.level = lvl
        low = text.lower()
        if pos is not None:
            if self.last and self.last[0] == pos:
                fight = self.last_glyph.isalpha() or any(c in low for c in COMBAT)
                if not fight:
                    self.failed.add(self.last)
            elif self.last and self.last[0] != pos:
                self.prev_pos = self.last[0]
            self.visits[pos] = self.visits.get(pos, 0) + 1
        mp = text.split("map:", 1)[1].split("\n")[1:22] if "map:" in text else []
        deadly, blocked = set(), set()
        for line in low.splitlines():
            if "adjacent" not in line:
                continue
            noun, rest = line.split("adjacent", 1)
            dirs = set(re.findall(DIRS, rest))
            if any(n in noun for n in DEADLY_NOUNS):
                deadly |= dirs
            elif any(n in noun for n in BLOCK_NOUNS) and "door" not in noun:
                blocked |= dirs
        return pos, mp, deadly, blocked

    def menu(self, pos, mp, deadly, blocked, legal):
        """Ranked (direction, glyph, visits, just_left) for every open, legal compass move."""
        if pos is None:
            return []
        ranked = []
        for i, (d, (dx, dy)) in enumerate(STEP.items()):
            if d not in legal or d in deadly or d in blocked or (pos, d) in self.failed:
                continue
            x, y = pos[0] + dx, pos[1] + dy
            g = mp[y][x] if 0 <= y < len(mp) and 0 <= x < len(mp[y]) else " "
            if g == "}":
                continue
            v = self.visits.get((x, y), 0)
            back = (x, y) == self.prev_pos
            tier = 0 if g == ">" else 1 if g.isalpha() else 2
            ranked.append(((tier, g in WALLISH, back, v, i), (d, g, v, back)))
        return [r for _, r in sorted(ranked)]

    def commit(self, pos, action, glyph):
        ok = pos is not None and action in STEP
        self.last = (pos, action) if ok else None
        self.last_glyph = glyph if ok else " "


class MenuHarness:
    """Naive agent whose single call chooses from, and is filtered through, a code-ranked menu."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.ledger = Ledger()

    def reset(self):
        self.prompt_builder.reset()
        self.ledger.reset()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        pos, mp, deadly, blocked = self.ledger.observe(text)
        legal = legal_actions(getattr(self.prompt_builder, "system_prompt", "") or "")
        menu = self.ledger.menu(pos, mp, deadly, blocked, legal)
        vetoed = ({d for d in STEP if d in deadly or d in blocked or (pos, d) in self.ledger.failed}
                  if pos is not None else set())
        top = menu[0] if menu else None
        messages = build_messages(self.prompt_builder, menu)
        response = self.client.generate(messages)
        out = copy.deepcopy(response)
        action = extract_action(out.completion, legal, vetoed, top[0] if top else None)
        row = next((r for r in menu if r[0] == action), None)
        if row and row[3] and top and top[0] != action and top[2] == 0 and not row[1].isalpha():
            action, row = top[0], top            # orbit override: never walk back past a fresh square
        self.ledger.commit(pos, action, row[1] if row else " ")
        return out._replace(completion=action)


def build_agent(client_factory, prompt_builder, config=None):
    return MenuHarness(client_factory, prompt_builder, config)
