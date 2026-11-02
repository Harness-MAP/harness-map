"""agent.py - gradient walker wired as an ordered STAGE table over tools/ modules.

Each step builds one context `c`; `STAGES` run in order and the first to answer is emitted, its
rung written to `reasoning`. Modules:
  tools/field.py - frame decoder (sections, prose clauses, glyph lookup)
  tools/chart.py - per-level map memory + a potential field seeded only on REACHABLE goals
  tools/quest.py - the lava-level regime (Quest pair only, armed on the verb list): kick-split the
                   spawn pile, pick up horn/wand, fire across the lava, hold the corridor mouth
The model is consulted only when the field is flat, and never on an armed Quest level."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_NS = "_hm_" + hashlib.md5(str(ROOT).encode()).hexdigest()[:10]


def _load(name):
    """Load tools/<name>.py under a per-candidate module name (no sys.path cache collisions)."""
    key = "%s_%s" % (_NS, name)
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    if name in ("field", "chart"):
        sys.modules[name] = mod  # chart.py / quest.py do `from field import STEP`
    spec.loader.exec_module(mod)
    return mod


F = _load("field")
Chart = _load("chart").Chart
Quest = _load("quest").Quest
STEP = F.STEP

TAIL = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)
FOE = re.compile(r"[a-zA-Z&']")
STAGES = ("prompt", "orient", "stairs", "swing", "cross", "hold", "kit", "fetch", "descend", "consult")


def choose_action(reply, legal, banned=()):
    """Whole-word projection of a reply onto the legal set, earliest mention first."""
    text = (reply or "").lower()
    hits = []
    for name in legal:
        for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?![a-z])", text):
            hits.append((m.start(), -len(name), name))
    for _, _, name in sorted(hits):
        if name not in banned:
            return name
    return None


class GradientAgent:
    """Hub: owns the Chart; `act` runs STAGES in order, each abstains (None) or answers."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.priors = self.load_priors()
        self.chart = Chart()
        self.quest = Quest()
        self.peaceful = set()
        self.last_foe = None

    def load_priors(self):
        parts = [(ROOT / n).read_text().strip() for n in ("AGENT.md", "rules.md") if (ROOT / n).exists()]
        sk = ROOT / "skills"
        if sk.is_dir():
            parts += [p.read_text().strip() for p in sorted(sk.glob("*.md"))]
        return "\n\n".join(p for p in parts if p)

    def reset(self):
        self.prompt_builder.reset()
        self.chart = Chart()
        self.quest = Quest()
        self.peaceful = set()
        self.last_foe = None

    def legal(self):
        sp = getattr(self.prompt_builder, "system_prompt", None) or ""
        names = []
        for line in sp.split("In a moment")[0].splitlines():
            m = re.match(r"^\s*([a-z][a-z ]*?):\s", line)
            if m:
                names.append(m.group(1).strip())
        return names or list(STEP)

    def reply(self, action, why):
        try:
            from balrog.client import LLMResponse
            return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                               input_tokens=0, output_tokens=0, reasoning=why)
        except Exception:
            from collections import namedtuple
            R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
            return R("harness", action, "stop", 0, 0, why)

    def ask(self, legal, banned, menu):
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            tail = (self.priors + "\n\n" if self.priors else "") + TAIL
            if menu:
                tail += "\nOpen moves, least trodden first: " + ", ".join(menu) + "."
            if banned:
                tail += "\nClosed or deadly here: " + ", ".join(sorted(banned)) + "."
            messages[-1].content += "\n\n" + tail
        out = copy.deepcopy(self.client.generate(messages))
        pick = choose_action(out.completion, legal, banned)
        if pick is None:
            pick = menu[0] if menu else (choose_action(out.completion, legal)
                                         or re.sub(r"[^a-zA-Z\s:]", "", out.completion or ""))
        try:
            return out._replace(completion=pick, reasoning="model:consult")
        except Exception:
            return out._replace(completion=pick)

    # -- stages ----------------------------------------------------------------------------------
    def st_prompt(self, c):
        """Every modal frame is answered here and never reaches the geometry: `more` escapes all."""
        fr, legal = c["fr"], c["legal"]
        if not fr["modal"]:
            return None
        self.chart.pending = None
        if Quest.armed(legal):
            got = self.quest.answer(fr["message"], legal)
            if got:
                return self.reply(*got)
        if "really attack" in fr["message"].lower() and self.last_foe:
            self.peaceful.add(self.last_foe)
        for word in ("more", "esc"):
            if word in legal:
                return self.reply(word, "reflex:" + word)
        return self.ask(legal, set(), [])

    def st_orient(self, c):
        fr = c["fr"]
        self.chart.fold(fr)
        pos = fr["pos"]
        if pos is None or not c["dirs"]:
            return self.ask(c["legal"], set(), [])
        lethal = set()
        for noun, dirs in fr["adjacent"].items():
            if "lava" in noun or "water" in noun:
                lethal |= dirs
        lethal |= {d for d, (dx, dy) in STEP.items() if self.chart.terrain.get((pos[0] + dx, pos[1] + dy)) == "}"}
        c["banned"] = lethal | {d for d in c["dirs"] if (pos, d) in self.chart.blocked}
        c["open"] = [d for d in c["dirs"] if d not in c["banned"]]
        c["cell"] = {d: (pos[0] + STEP[d][0], pos[1] + STEP[d][1]) for d in c["open"]}
        c["field"], c["kind"] = self.chart.field(pos)
        c["quest"] = Quest.armed(c["legal"])
        if c["quest"]:
            self.quest.observe(fr)
        return None

    def st_stairs(self, c):
        under = "staircase down" in c["fr"]["message"].lower()
        if (self.chart.terrain.get(c["fr"]["pos"]) == ">" or under) and "down" in c["legal"]:
            return self.reply("down", "reflex:down")
        return None

    def st_swing(self, c):
        for d in c["open"]:
            ch = F.glyph(c["fr"], c["cell"][d])
            if FOE.fullmatch(ch) and ch not in self.peaceful:
                self.last_foe = ch
                return self.reply(d, "reflex:swing")
        return None

    # -- Quest regime (tools/quest.py); every stage abstains unless the verb list arms it ---------
    def st_cross(self, c):
        if not c["quest"]:
            return None
        got = self.quest.cross(self.chart, c["fr"]["pos"], c["legal"], c["kind"])
        return self.reply(*got) if got else None

    def st_hold(self, c):
        if not c["quest"]:
            return None
        got = self.quest.hold(c["fr"], self.chart, c["fr"]["pos"], c["legal"], c["kind"])
        return self.reply(*got) if got else None

    def st_kit(self, c):
        if not c["quest"]:
            return None
        got = self.quest.kit(self.chart, c["fr"]["pos"], c["legal"])
        return self.reply(*got) if got else None

    def st_fetch(self, c):
        if not c["quest"] or c["kind"] == "stairs":
            return None
        pos = c["fr"]["pos"]
        occupied = {(x, y) for y, row in enumerate(c["fr"]["rows"]) for x, ch in enumerate(row)
                    if FOE.fullmatch(ch) and (x, y) != pos}
        d, n = Quest.toward(self.chart, pos, self.quest.targets(self.chart, c["legal"], c["kind"]),
                            occupied)
        if d is None or d in c["banned"] or d not in c["legal"]:
            return None
        self.chart.pending = (pos, d)
        return self.reply(d, "quest:walk:%d" % n)

    def st_descend(self, c):
        field, visits = c["field"], self.chart.visits
        scored = sorted((field[c["cell"][d]] + 0.25 * visits.get(c["cell"][d], 0), i, d)
                        for i, d in enumerate(c["open"])
                        if c["cell"][d] in field and self.chart.edge(c["fr"]["pos"], d) is not None)
        if not scored:
            return None
        d = scored[0][2]
        self.chart.pending = (c["fr"]["pos"], d)
        return self.reply(d, "field:" + c["kind"])

    def st_consult(self, c):
        visits, cell = self.chart.visits, c["cell"]
        menu = sorted(c["open"], key=lambda d: (visits.get(cell[d], 0), c["dirs"].index(d)))
        if c["quest"]:
            pos = c["fr"]["pos"]
            menu = [d for d in menu if self.chart.edge(pos, d) is not None] or menu
            if not menu:
                word = "search" if "search" in c["legal"] else c["legal"][0]
                return self.reply(word, "quest:idle")
            self.chart.pending = (pos, menu[0])
            return self.reply(menu[0], "quest:menu")
        resp = self.ask(c["legal"], c["banned"], menu)
        if resp.completion in STEP:
            self.chart.pending = (c["fr"]["pos"], resp.completion)
        return resp

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        legal = self.legal()
        c = {"fr": F.parse_tty_frame(text), "legal": legal, "dirs": [d for d in STEP if d in legal],
             "banned": set(), "open": [], "cell": {}, "field": {}, "kind": None, "quest": False}
        for name in STAGES:
            got = getattr(self, "st_" + name)(c)
            if got is not None:
                return got
        return self.reply(c["dirs"][0] if c["dirs"] else legal[0], "fallback")


def build_agent(client_factory, prompt_builder, config=None):
    return GradientAgent(client_factory, prompt_builder, config)
