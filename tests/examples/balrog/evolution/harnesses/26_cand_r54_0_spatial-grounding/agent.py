"""agent.py - ENTRYPOINT. The chart walker wired as an ordered STAGE PIPELINE over tools/ modules.

Each step builds a per-step context `c`, then walks `STAGES` in order; a stage abstains (None) or
answers (word, rung). The first answer is emitted and its rung is written to `reasoning`.
  tools/frame.py - observation parser, glyph tables, reply projection onto the legal vocabulary
  tools/chart.py - per-level map memory (regime-aware walls, doors) + BFS route / dark-room probe
The model is the last stage: it is asked only when no chart stage has an answer."""
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
STEP, ORDER = F.STEP, F.ORDER
parse_glyph_map, decide_action = F.parse_glyph_map, F.decide_action

NAIVE_INSTRUCTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)
PROMPT_CAP = 3

# the pipeline wiring: one ordered table; each stage abstains (None) or answers (word, rung)
STAGES = ("prompt", "orient", "stairs", "fight", "route", "probe", "model")


class GroundedAgent:
    """Hub: owns the Chart; `act` runs STAGES in order."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.priors = self.compose_priors()
        self.chart = Chart(F)
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
        self.last = None
        self.level = None
        self.stood = None
        self.prompt_run = 0

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
            return template._replace(completion=action, reasoning=why) \
                if "reasoning" in getattr(template, "_fields", ()) else template._replace(completion=action)
        try:
            from balrog.client import LLMResponse
            return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                               input_tokens=0, output_tokens=0, reasoning=why)
        except Exception:
            from collections import namedtuple
            R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
            return R("harness", action, "stop", 0, 0, why)

    # ---- stages -------------------------------------------------------------------------------

    def st_prompt(self, c):
        """An open prompt is answered with `more` (the one legal verb that closes both `[yn]` and
        the item-letter prompts). A `[yn]` raised by a move also closes that (square, bearing) edge."""
        fr = c["fr"]
        if not fr["prompt"]:
            self.prompt_run = 0
            return None
        if fr["yn"] and self.last and self.last[1] in STEP:
            self.chart.blocked.add((self.last[0], self.last[1]))
        self.last = None
        self.prompt_run += 1
        if "more" in c["legal"] and self.prompt_run <= PROMPT_CAP:
            return "more", "reflex:prompt-more"
        return None

    def st_orient(self, c):
        fr, chart = c["fr"], self.chart
        if fr["dlvl"] != self.level:
            chart.clear()
            self.level = fr["dlvl"]
        chart.set_regime(c["moves"])
        chart.fold(fr)
        pos = fr["pos"]
        if pos is not None:
            if self.last:
                prev, move, kind = self.last
                if "door opens" in fr["message"].lower() and move in STEP:
                    chart.mark_door(prev, move)
                elif prev == pos and move in STEP and kind != "swing":
                    chart.blocked.add((prev, move))
            chart.visits[pos] = chart.visits.get(pos, 0) + 1
            for noun, dirs in fr["adjacent"].items():
                if any(k in noun for k in F.DOOR_NOUNS):
                    for d in dirs:
                        chart.mark_door(pos, d)
        self.last = None
        c["pos"] = pos
        c["bad"] = self.deadly(fr) if pos is not None else set()
        return None

    def st_stairs(self, c):
        if c["pos"] is not None and self.chart.drawn.get(c["pos"]) == ">" and "down" in c["legal"]:
            return "down", "reflex:stairs"
        return None

    def st_fight(self, c):
        pos = c["pos"]
        if pos is None:
            return None
        for d in c["moves"]:
            if d in c["bad"] or (pos, d) in self.chart.blocked:
                continue
            ch = F.glyph(c["fr"], (pos[0] + STEP[d][0], pos[1] + STEP[d][1]))
            if F.CREATURE.fullmatch(ch or " ") and ch not in F.TAME:
                self.last = (pos, d, "swing")
                return d, "reflex:swing"
        return None

    def st_route(self, c):
        if c["pos"] is None:
            return None
        hop, kind = self.chart.route(c["pos"], c["moves"], c["bad"])
        if hop:
            self.last = (c["pos"], hop, "route")
            return hop, "chart:route-" + kind
        return None

    def st_probe(self, c):
        if c["pos"] is None:
            return None
        hop = self.chart.probe(c["pos"], c["moves"], c["bad"])
        if hop:
            self.last = (c["pos"], hop, "probe")
            return hop, "chart:probe"
        return None

    def st_model(self, c):
        pos = c["pos"]
        banned = set(c["bad"])
        if pos is not None:
            banned |= {d for d in c["moves"] if (pos, d) in self.chart.blocked}
        resp = self.ask_model(c["legal"], banned, pos, c["moves"])
        if resp.completion in STEP and pos is not None:
            self.last = (pos, resp.completion, "model")
        return resp

    # ---- helpers ------------------------------------------------------------------------------

    def deadly(self, fr):
        bad = set()
        for noun, dirs in fr["adjacent"].items():
            if "lava" in noun or "water" in noun:
                bad |= dirs
        pos = fr["pos"]
        for d, (dx, dy) in STEP.items():
            if F.glyph(fr, (pos[0] + dx, pos[1] + dy)) in F.LETHAL:
                bad.add(d)
        return bad

    def ask_model(self, legal, banned, pos, moves):
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
        pick = decide_action(out.completion, legal, banned)
        if pick is None:
            # never pass raw text through: fall back to the least-visited safe bearing
            safe = [d for d in moves if d not in banned] or list(moves)
            if safe and pos is not None:
                pick = min(safe, key=lambda d: (self.chart.visits.get((pos[0] + STEP[d][0], pos[1] + STEP[d][1]), 0),
                                                ORDER.index(d)))
            elif safe:
                pick = safe[0]
            else:
                pick = decide_action(out.completion, legal) or legal[0]
        return out._replace(completion=pick)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        legal = self.legal_actions()
        c = {"fr": parse_glyph_map(text), "legal": legal, "moves": [d for d in ORDER if d in legal],
             "pos": None, "bad": set()}
        for name in STAGES:
            ans = getattr(self, "st_" + name)(c)
            if ans is None:
                continue
            if type(ans) is tuple:  # a stage verdict (word, rung); an LLMResponse is a namedtuple
                return self.reply(ans[0], ans[1])
            return ans
        return self.reply(c["moves"][0] if c["moves"] else legal[0], "fallback")


def build_agent(client_factory, prompt_builder, config=None):
    return GroundedAgent(client_factory, prompt_builder, config)
