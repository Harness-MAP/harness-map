"""agent.py - gradient walker: code descends the field, the model is asked only where it is flat."""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from field import STEP, parse_tty_frame  # noqa: E402

TAIL = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)
FOE = re.compile(r"[a-zA-Z&']")


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
    """Hub: owns the chart, reads the field, falls back to one model call."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.priors = self.load_priors()
        self.chart = {}

    def load_priors(self):
        parts = [(ROOT / n).read_text().strip() for n in ("AGENT.md", "rules.md") if (ROOT / n).exists()]
        sk = ROOT / "skills"
        if sk.is_dir():
            parts += [p.read_text().strip() for p in sorted(sk.glob("*.md"))]
        return "\n\n".join(p for p in parts if p)

    def reset(self):
        self.prompt_builder.reset()
        self.chart = {}

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

    def consult(self, legal, banned, menu):
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
        return out._replace(completion=pick)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        fr = parse_tty_frame(text, self.chart)
        legal = self.legal()
        msg = fr["message"]
        if fr["modal"]:
            if "[yn" in msg and "y" in legal and "attack" in msg.lower():
                return self.reply("y", "reflex:confirm")
            if msg.endswith("--More--") and "more" in legal:
                return self.reply("more", "reflex:more")
            if "esc" in legal:
                return self.reply("esc", "reflex:esc")
        pos = fr["pos"]
        dirs = [d for d in STEP if d in legal]
        if pos is None or not dirs:
            return self.consult(legal, set(), [])
        terrain, visits, field = self.chart["terrain"], self.chart["visits"], self.chart["field"]
        if terrain.get(pos) == ">" and "down" in legal:
            return self.reply("down", "reflex:down")
        banned = set(fr["lethal"]) | {d for d in dirs if (pos, d) in self.chart["blocked"]}
        open_dirs = [d for d in dirs if d not in banned]
        cell = {d: (pos[0] + STEP[d][0], pos[1] + STEP[d][1]) for d in open_dirs}
        for d in open_dirs:
            y, x = cell[d][1], cell[d][0]
            ch = fr["rows"][y][x] if 0 <= y < len(fr["rows"]) and 0 <= x < len(fr["rows"][y]) else " "
            if FOE.fullmatch(ch):
                return self.reply(d, "reflex:swing")
        scored = sorted((field[cell[d]] + 0.25 * visits.get(cell[d], 0), i, d)
                        for i, d in enumerate(open_dirs) if cell[d] in field)
        if scored:
            d = scored[0][2]
            self.chart["pending"] = (pos, d)
            return self.reply(d, "field:descend")
        menu = sorted(open_dirs, key=lambda d: (visits.get(cell[d], 0), dirs.index(d)))
        resp = self.consult(legal, banned, menu)
        if resp.completion in STEP:
            self.chart["pending"] = (pos, resp.completion)
        return resp


def build_agent(client_factory, prompt_builder, config=None):
    return GradientAgent(client_factory, prompt_builder, config)
