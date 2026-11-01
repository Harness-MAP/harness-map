"""orchestrator.py — ENTRYPOINT. Naive transcript + a code-RANKED move menu + legal projection.

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
r063 (steer toward r027's gradient walker): the Ledger also keeps a terrain chart and a multi-hop
potential field (tools/field.py: BFS from a drawn `>`, else from the frontier). The menu is ranked
by that field, and whenever the field (or a stairs/creature reflex) decides, the step is emitted
by code with NO model call; the model is consulted, with the menu, only where the field is flat.
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
from field import flood, passable                     # noqa: E402

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
DIRS = "northeast|northwest|southeast|southwest|north|south|east|west"
DEADLY_NOUNS = ("lava", "water")
BLOCK_NOUNS = ("wall", "bars", "dark area")
COMBAT = ("you hit", "you miss", "bites", "you kill", "you destroy", "hits!", "misses")
WALLISH = "|- "


class Ledger:
    """Self-derived per-level record: visits per square, failed (square, direction) moves."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.level, self.visits, self.failed = None, {}, set()
        self.last, self.last_glyph, self.prev_pos = None, " ", None
        self.terrain, self.doors, self.field = {}, set(), {}

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
        self.chart(pos, mp)
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

    def chart(self, pos, mp):
        """Fold the drawn map into the per-level terrain chart and refresh the potential field."""
        for y, row in enumerate(mp):
            for x, ch in enumerate(row):
                if ch in " @":
                    continue
                c = (x, y)
                if ch == "+":
                    self.doors.add(c)
                if ch.isalpha():                      # a creature stands on something walkable
                    self.terrain.setdefault(c, ".")
                else:
                    self.terrain[c] = ch
        if pos is not None and self.terrain.get(pos, "@") in "@|-":
            if self.terrain.get(pos) in ("|", "-"):
                self.doors.add(pos)                   # standing on it: it is an open doorway
            else:
                self.terrain[pos] = "."
        self.field = flood(self.terrain, self.visits, self.failed, self.doors)

    def potential(self, pos, d):
        x, y = pos[0] + STEP[d][0], pos[1] + STEP[d][1]
        if not passable((x, y), self.terrain, self.doors) or (x, y) not in self.field:
            return None
        return self.field[(x, y)] + 0.25 * self.visits.get((x, y), 0)

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
            f = self.potential(pos, d)
            ranked.append(((tier, f is None, f or 0.0, g in WALLISH, back, v, i), (d, g, v, back, f)))
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

    def reply(self, action, why):
        """A model-free step: the field (or a reflex) decided, so no model call is made."""
        try:
            from balrog.client import LLMResponse
            return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                               input_tokens=0, output_tokens=0, reasoning=why)
        except Exception:
            from collections import namedtuple
            R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
            return R("harness", action, "stop", 0, 0, why)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        pos, mp, deadly, blocked = self.ledger.observe(text)
        legal = legal_actions(getattr(self.prompt_builder, "system_prompt", "") or "")
        mm = re.search(r"(?ms)^message:[ \t]*\n?(.*?)(?=^language observation:|^cursor:|^map:|\Z)", text)
        msg = mm.group(1).strip() if mm else ""
        if "[yn" in msg and "attack" in msg.lower() and "y" in legal:
            self.ledger.commit(pos, "y", " ")
            return self.reply("y", "reflex:confirm-attack")
        if msg.endswith("--More--") and "more" in legal:
            self.ledger.commit(pos, "more", " ")
            return self.reply("more", "reflex:more")
        if pos is not None and self.ledger.terrain.get(pos) == ">" and "down" in legal:
            self.ledger.commit(pos, "down", " ")
            return self.reply("down", "reflex:down")
        menu = self.ledger.menu(pos, mp, deadly, blocked, legal)
        top = menu[0] if menu else None
        if top and (top[1] == ">" or top[1].isalpha() or top[4] is not None):
            why = "reflex:stairs" if top[1] == ">" else "reflex:swing" if top[1].isalpha() else "field:descend"
            self.ledger.commit(pos, top[0], top[1])
            return self.reply(top[0], why)
        vetoed = ({d for d in STEP if d in deadly or d in blocked or (pos, d) in self.ledger.failed}
                  if pos is not None else set())
        messages = build_messages(self.prompt_builder, [r[:4] for r in menu])
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
