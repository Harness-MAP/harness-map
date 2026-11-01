"""harness.py - ENTRYPOINT. A chart pilot wired as an ordered STAGE PIPELINE over tools/ modules.

Each step builds one context `c` from the frame, then walks `STAGES` in order; every stage abstains
(None) or answers (word, rung, kind[, template]). The first answer is emitted with its rung written to
`reasoning`. The model is the LAST stage: it is asked only when no stage above it owns the step.
  tools/frame.py - observation parser, glyph lookup, reply projection onto the legal vocabulary
  tools/chart.py - per-level map memory (doors, bars, bump ledger) + BFS routing
  tools/quest.py - lava crossing: pick up a horn/wand, walk to a bank, fire it across the lava"""
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
CREATURE = re.compile(r"[a-zA-Z&;:']")
TAME = {"@"}
MORE_CAP = 3

# the pipeline wiring: one ordered table; each stage abstains (None) or answers
STAGES = ("prompt", "orient", "stairs", "fight", "quest", "fetch", "route", "model")


class GroundedAgent:
    """Hub: owns the Chart and the Quest stage; `act` runs STAGES in order."""

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
        self.pending = None
        self.level = None
        self.peaceful = set()
        self.last_foe = None
        self.mores = 0

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
            return template._replace(completion=action)
        try:
            from balrog.client import LLMResponse
            return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                               input_tokens=0, output_tokens=0, reasoning=why)
        except Exception:
            from collections import namedtuple
            R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
            return R("harness", action, "stop", 0, 0, why)

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

    def consult(self, legal, allowed=None, banned=()):
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            tail = NAIVE_INSTRUCTION
            if self.priors:
                tail = self.priors + "\n\n" + tail
            if banned:
                tail += "\nMoves that are closed or deadly from this square: " + ", ".join(sorted(banned)) + "."
            messages[-1].content += "\n\n" + tail
        resp = self.client.generate(messages)
        out = copy.deepcopy(resp)
        return out, F.decide_action(out.completion, legal, allowed, banned)

    # -- stages: each takes the step context `c`, returns None or (word, rung, kind[, template]) --
    def st_prompt(self, c):
        fr, legal, msg = c["fr"], c["legal"], c["fr"]["message"]
        if not fr["prompt"]:
            self.mores = 0
            return None
        last, self.pending = self.pending, None
        got = self.quest.answer(msg, legal)
        if got:
            return got[0], got[1], "prompt"
        if "[yn" in msg and "attack" in msg.lower() and last and last[2] == "swing" and self.last_foe:
            self.peaceful.add(self.last_foe)
        if "more" in legal and self.mores < MORE_CAP:
            self.mores += 1
            return "more", "reflex:more", "prompt"
        if "esc" in legal:
            return "esc", "reflex:escape", "prompt"
        return None

    def st_orient(self, c):
        fr, pos = c["fr"], c["pos"]
        if pos is None or pos[1] < 0 or not c["moves"]:
            self.pending = None
            out, pick = self.consult(c["legal"])
            return pick or re.sub(r"[^a-zA-Z\s:]", "", out.completion or ""), "model:blind", "raw", out
        if self.pending is not None and not fr["prompt"]:
            prev, move, kind = self.pending
            if kind == "move":
                self.chart.book(prev, move, pos, fr["message"])
        self.chart.fold(fr)
        self.quest.observe(fr, self.chart)
        c["bad"] = self.deadly(fr)
        c["closed"] = {d for d in c["moves"] if (pos, d) in self.chart.blocked}
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
            if d in c["bad"] or (pos, d) in self.chart.blocked:
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

    def st_model(self, c):
        """Fallback: the model picks among compass moves + search only (no item verb can open a prompt)."""
        pos, legal = c["pos"], c["legal"]
        banned = c["bad"] | c["closed"]
        allowed = set(c["moves"]) | ({"search"} & set(legal))
        out, pick = self.consult(legal, allowed, banned)
        if pick is None:
            safe = [d for d in c["moves"] if d not in banned] or c["moves"]
            pick = min(safe, key=lambda d: (self.chart.visits.get((pos[0] + STEP[d][0], pos[1] + STEP[d][1]), 0),
                                            ORDER.index(d)))
        return pick, "model:fallback", ("move" if pick in STEP else "verb"), out

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
             "bad": set(), "closed": set(), "hop": None, "kind": None}
        for name in STAGES:
            got = getattr(self, "st_" + name)(c)
            if got:
                return self.emit(c, *got)
        return self.emit(c, c["moves"][0] if c["moves"] else legal[0], "fallback", "move")

    def emit(self, c, word, rung, kind, template=None):
        pos = c["pos"]
        if kind == "raw":
            return template._replace(completion=word)
        if kind in ("move", "swing", "verb") and pos is not None:
            self.pending = (pos, word, kind)
        return self.reply(word, rung, template)


def build_agent(client_factory, prompt_builder, config=None):
    return GroundedAgent(client_factory, prompt_builder, config)
