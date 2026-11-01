"""harness.py - ENTRYPOINT. A charted learner wired as an ordered STAGE PIPELINE.

Each step builds one per-step context `c` from the frame, then walks `STAGES` in order; every stage
either abstains (None) or answers (word, rung[, kind]). The first answer is emitted and its rung is
written to `reasoning`. Modules:
  tools/frame.py  - observation parser, glyph lookup, reply projection onto the legal vocabulary
  tools/chart.py  - per-level map memory + BFS (`route` to `>`/frontier, `toward` arbitrary goals)
  tools/quest.py  - lava crossing: pick up a horn/wand, walk to a bank, fire it across the lava
The TD value learner (the last stage) walks wherever the chart has no route."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_NS = "_hm_" + hashlib.md5(str(ROOT).encode()).hexdigest()[:10]


def _load(name):
    """Load tools/<name>.py under a per-candidate module name (no sys.path / cache collisions)."""
    key = "%s_%s" % (_NS, name)
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


F = _load("frame")
Chart = _load("chart").Chart
Quest = _load("quest").Quest
STEP, ORDER, LETHAL = F.STEP, F.ORDER, F.LETHAL

NAIVE_INSTRUCTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)
DOOR_NOUNS = ("doorway", "open door", "broken door")
CREATURE = re.compile(r"[a-zA-Z&;:']")
TAME = {"@"}

ALPHA, GAMMA, LAMBDA = 0.5, 0.9, 0.8
OPTIMISM, BONUS = 1.0, 0.6
BUMP_R, NEW_SQUARE_R, CELL_R, STAIR_R = -1.0, 0.5, 0.05, 0.4
TIE = 0.05

# the pipeline wiring: one ordered table; each stage abstains (None) or answers
STAGES = ("prompt", "orient", "stairs", "fight", "quest", "fetch", "route", "learner")


class LearnerAgent:
    """Hub: owns the Chart, the Quest stage and the value table; `act` runs STAGES in order."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.priors = self.compose_priors()
        self.chart = Chart(F)
        self.quest = Quest(F)
        self.forget()

    def compose_priors(self):
        parts = []
        for name in ["AGENT.md", "rules.md"]:
            p = ROOT / name
            if p.exists() and p.read_text().strip():
                parts.append(p.read_text().strip())
        sk = ROOT / "skills"
        if sk.is_dir():
            parts += [p.read_text().strip() for p in sorted(sk.glob("*.md")) if p.read_text().strip()]
        return "\n\n---\n\n".join(parts)

    def forget(self):
        self.chart.clear()
        self.quest.clear()
        self.q = {}
        self.tries = {}
        self.trace = {}
        self.pending = None
        self.level = None
        self.consulted = set()
        self.peaceful = set()
        self.mores = 0
        self.last_foe = None

    def reset(self):
        self.prompt_builder.reset()
        self.priors = self.compose_priors()
        self.forget()

    def legal_actions(self):
        sp = getattr(self.prompt_builder, "system_prompt", None) or ""
        names = []
        for line in sp.split("In a moment")[0].splitlines():
            m = re.match(r"^\s*([a-z][a-z ]*?):\s", line)
            if m:
                names.append(m.group(1).strip())
        return names or list(ORDER)

    def reply(self, action, why, template=None):
        if template is not None:
            return template._replace(completion=action, reasoning=why)
        try:
            from balrog.client import LLMResponse
            return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                               input_tokens=0, output_tokens=0, reasoning=why)
        except Exception:
            from collections import namedtuple
            R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
            return R("harness", action, "stop", 0, 0, why)

    # -- learner state ------------------------------------------------------------------------
    def stairs_gap(self, fr):
        pos = fr["pos"]
        best = None
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch == ">":
                    d = max(abs(x - pos[0]), abs(y - pos[1]))
                    best = d if best is None else min(best, d)
        return best

    def observe(self, fr):
        """Doors named in prose or just opened become passable; any other no-move closes an edge."""
        ch, pos = self.chart, fr["pos"]
        for noun, dirs in fr["adjacent"].items():
            if any(k in noun for k in DOOR_NOUNS):
                for d in dirs:
                    ch.doors.add((pos[0] + STEP[d][0], pos[1] + STEP[d][1]))
        ch.visits[pos] = ch.visits.get(pos, 0) + 1
        if self.pending is None:
            return
        s, a, kind, _ = self.pending
        if s != pos or kind != "move" or a not in STEP:
            return
        target = (s[0] + STEP[a][0], s[1] + STEP[a][1])
        if "door opens" in fr["message"].lower():
            ch.doors.add(target)
        else:
            ch.blocked.add((s, a))

    def admissible(self, pos, d):
        nxt = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
        return (pos, d) not in self.chart.blocked and (self.chart.passable(nxt) or self.chart.drawn.get(nxt) == "`")

    def value(self, s, a):
        return self.q.get((s, a), OPTIMISM)

    def score(self, s, a):
        n = self.tries.get((s, a), 0)
        return self.value(s, a) + BONUS / math.sqrt(1 + n)

    def credit(self, fr, moves):
        """Reward the previous action and push the TD error back along the trace."""
        pos = fr["pos"]
        new_cells = self.chart.fold(fr)
        gap = self.stairs_gap(fr)
        if self.pending is None or fr["prompt"]:
            self.chart.stood.add(pos)
            return gap
        s, a, kind, old_gap = self.pending
        r = CELL_R * min(new_cells, 10)
        opened = "door opens" in fr["message"].lower()
        bump = pos == s and kind == "move" and not opened
        if bump:
            r += BUMP_R
        elif pos not in self.chart.stood:
            r += NEW_SQUARE_R
        if old_gap is not None and gap is not None:
            r += STAIR_R * (old_gap - gap)
        self.chart.stood.add(pos)
        nxt = max((self.value(pos, b) for b in moves), default=0.0)
        if (s, a) in self.trace:
            delta = r + GAMMA * nxt - self.value(s, a)
            for key in list(self.trace):
                self.q[key] = self.q.get(key, OPTIMISM) + ALPHA * delta * self.trace[key]
                self.trace[key] *= GAMMA * LAMBDA
                if self.trace[key] < 0.01:
                    del self.trace[key]
        if bump:
            self.q[(s, a)] = min(self.q.get((s, a), OPTIMISM), BUMP_R)
        return gap

    def deadly(self, fr):
        bad = set()
        for noun, dirs in fr["adjacent"].items():
            if "lava" in noun or "water" in noun:
                bad |= dirs
        pos = fr["pos"]
        for d, (dx, dy) in STEP.items():
            if F.glyph(fr, (pos[0] + dx, pos[1] + dy)) in LETHAL:
                bad.add(d)
        return bad

    def consult(self, legal, allowed, note=""):
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            tail = NAIVE_INSTRUCTION
            if self.priors:
                tail = self.priors + "\n\n" + tail
            if note:
                tail += "\n" + note
            messages[-1].content += "\n\n" + tail
        resp = self.client.generate(messages)
        out = copy.deepcopy(resp)
        pick = F.decide_action(out.completion, legal, allowed)
        return out, pick

    # -- stages: each takes the step context `c`, returns None or (word, rung[, kind[, template]]) --
    def st_prompt(self, c):
        fr, legal, pos, msg = c["fr"], c["legal"], c["pos"], c["fr"]["message"]
        if not (fr["prompt"] or (pos is not None and pos[1] < 0)):
            self.mores = 0
            return None
        last = self.pending
        self.pending = None
        got = self.quest.answer(msg, legal)
        if got:
            return got[0], got[1], "prompt"
        if "[yn" in msg and "attack" in msg.lower():
            if last and last[2] == "swing" and self.last_foe:
                self.peaceful.add(self.last_foe)
        if "more" in legal and self.mores < 2:
            self.mores += 1
            return "more", "reflex:more", "prompt"
        if "esc" in legal:
            return "esc", "reflex:escape", "prompt"
        return None

    def st_orient(self, c):
        fr, pos = c["fr"], c["pos"]
        if pos is None or pos[1] < 0 or not c["moves"]:
            self.pending = None
            out, pick = self.consult(c["legal"], None)
            return pick or re.sub(r"[^a-zA-Z\s:]", "", out.completion or ""), "model:blind", "raw", out
        self.observe(fr)
        c["gap"] = self.credit(fr, c["moves"])
        self.quest.observe(fr, self.chart)
        c["bad"] = self.deadly(fr)
        c["hop"], c["kind"] = self.chart.route(pos, c["moves"], c["bad"])
        return None

    def st_stairs(self, c):
        under = "staircase down here" in c["fr"]["message"].lower()
        if (self.chart.drawn.get(c["pos"]) == ">" or under) and "down" in c["legal"]:
            return "down", "reflex:stairs", "verb"
        return None

    def st_fight(self, c):
        fr, pos = c["fr"], c["pos"]
        for d in c["moves"]:
            if d in c["bad"]:
                continue
            ch = F.glyph(fr, (pos[0] + STEP[d][0], pos[1] + STEP[d][1]))
            if CREATURE.fullmatch(ch or " ") and ch not in TAME and ch not in self.peaceful:
                self.last_foe = ch
                return d, "reflex:swing", "swing"
        return None

    def st_quest(self, c):
        if not Quest.armed(self.chart, c["legal"]):
            return None
        got = self.quest.step(self.chart, c["pos"], c["legal"], c["kind"])
        return (got[0], got[1], "verb") if got else None

    def st_fetch(self, c):
        if c["kind"] == "stairs" or not Quest.armed(self.chart, c["legal"]):
            return None
        goals = self.quest.targets(self.chart, c["legal"], c["kind"]) - {c["pos"]}
        if not goals:
            return None
        hop, n = self.chart.toward(c["pos"], c["moves"], c["bad"], goals)
        return (hop, "quest:walk:%d" % n, "move") if hop else None

    def st_route(self, c):
        return (c["hop"], "chart:route:" + c["kind"], "move") if c["hop"] else None

    def st_learner(self, c):
        pos, moves, bad = c["pos"], c["moves"], c["bad"]
        safe = [d for d in moves if d not in bad] or moves
        safe = [d for d in safe if self.admissible(pos, d)] or safe
        ranked = sorted(safe, key=lambda d: (-self.score(pos, d), ORDER.index(d)))
        top = self.score(pos, ranked[0])
        tied = [d for d in ranked if top - self.score(pos, d) <= TIE]
        if len(tied) > 1 and pos not in self.consulted:
            self.consulted.add(pos)
            out, pick = self.consult(c["legal"], set(tied),
                                     "Equally promising moves from this square: " + ", ".join(tied) + ".")
            if pick:
                return pick, "learner:consult", "move", out
        return ranked[0], "learner:argmax", "move"

    # -- the loop -------------------------------------------------------------------------------
    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        fr = F.parse_glyph_map(text)
        legal = self.legal_actions()
        if fr["dlvl"] != self.level:
            self.forget()
            self.level = fr["dlvl"]
        c = {"fr": fr, "legal": legal, "pos": fr["pos"], "moves": [d for d in ORDER if d in legal],
             "gap": None, "bad": set(), "hop": None, "kind": None}
        for name in STAGES:
            got = getattr(self, "st_" + name)(c)
            if got:
                return self.emit(c, *got)
        return self.emit(c, c["moves"][0] if c["moves"] else legal[0], "fallback", "move")

    def emit(self, c, word, rung, kind, template=None):
        pos = c["pos"]
        if kind in ("move", "swing", "verb") and pos is not None:
            if kind != "verb":
                self.tries[(pos, word)] = self.tries.get((pos, word), 0) + 1
                self.trace[(pos, word)] = 1.0
            self.pending = (pos, word, kind, c["gap"])
        elif kind == "raw":
            return template._replace(completion=word)
        return self.reply(word, rung, template)


def build_agent(client_factory, prompt_builder, config=None):
    return LearnerAgent(client_factory, prompt_builder, config)
