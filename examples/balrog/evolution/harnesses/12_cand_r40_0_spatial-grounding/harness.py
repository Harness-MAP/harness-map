"""harness.py - grounded dungeon walker: a remembered chart descends a field; the model picks only where it is flat."""
from __future__ import annotations

import copy
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from grid import parse_glyph_map, fold_chart, flood_field, BEARINGS  # noqa: E402

NAIVE_INSTRUCTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)
ORDER = ["north", "east", "south", "west", "northeast", "southeast", "southwest", "northwest"]
RANK = {"stairs down": 0, "creature": 1, "open": 2, "boulder": 3, "dark": 4}


def extract_action(text, legal):
    """Whole-word projection of a reply onto the legal set."""
    t = re.sub(r"[^a-zA-Z\s:]", " ", text or "").lower()
    t = re.sub(r"\battack\s*:?\s*", "", t).strip()
    if t in legal:
        return t
    for name in sorted(legal, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", t):
            return name
    return ""


class GroundedAgent:
    """Model proposes; the decoded map grounds the prompt and vets lethal or stale moves."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.priors = self.load_priors()
        self.reset_state()

    def load_priors(self):
        parts = []
        for p in [ROOT / "AGENT.md", ROOT / "rules.md"] + sorted((ROOT / "skills").glob("*.md")):
            if p.exists() and p.read_text().strip():
                parts.append(p.read_text().strip())
        return "\n\n---\n\n".join(parts)

    def reset_state(self):
        self.visits = Counter()
        self.trail = []
        self.refused = set()
        self.last = None
        self.bars = False
        self.chart = {}
        self.field = {}

    def reset(self):
        self.prompt_builder.reset()
        self.reset_state()

    def legal(self):
        sp = getattr(self.prompt_builder, "system_prompt", None) or ""
        names = re.findall(r"(?m)^\s*([a-z][a-z ]*?)\s*:", sp)
        found = [n for n in names if n]
        return found or list(ORDER)

    def learn(self, frame, prev_action):
        pos = frame["pos"]
        if self.last and prev_action in BEARINGS and pos is not None:
            ppos, pring = self.last
            kind = pring.get(prev_action, ("dark",))[0]
            if pos == ppos and kind != "creature":
                self.refused.add((ppos, prev_action))
        if pos is not None:
            self.visits[pos] += 1
            self.trail.append(pos)
            self.last = (pos, frame["ring"])
        else:
            self.last = None

    def score(self, d, frame):
        kind, _, target = frame["ring"][d]
        if d in frame["foes"]:
            kind = "creature"
        if frame["down"] and target == frame["down"]:
            kind = "stairs down"
        return (RANK.get(kind, 5), self.visits[target], ORDER.index(d) if d in ORDER else 9)

    def forbidden(self, d, frame, soft=True):
        pos = frame["pos"]
        if d in frame["lava"]:
            return True
        if pos is None or d not in frame["ring"]:
            return False
        kind, _, target = frame["ring"][d]
        if kind in ("lava", "wall") or (pos, d) in self.refused:
            return True
        if kind == "dark" and target not in self.chart.get("terrain", {}):
            return True
        if not soft:
            return False
        stale = len(self.trail) >= 3 and target == self.trail[-2] and self.visits[target] >= 3
        return stale and d not in frame["foes"]

    def vet(self, choice, frame, legal):
        bearings = [d for d in ORDER if d in legal]
        if choice and choice not in BEARINGS:
            return choice, "model"
        if choice and not self.forbidden(choice, frame):
            return choice, "model"
        options = [d for d in bearings if not self.forbidden(d, frame)]
        if not options:
            options = [d for d in bearings if not self.forbidden(d, frame, soft=False)]
        if frame["pos"] is not None:
            options = [d for d in options if d in frame["ring"]]
            options.sort(key=lambda d: self.score(d, frame))
        if options:
            return options[0], "grounded"
        return (choice or (bearings[0] if bearings else "north")), "fallback"

    def ground(self, frame, legal):
        if frame["pos"] is None:
            return ""
        lines = ["Grounded surroundings (decoded from the map; trust these over your guess):"]
        for d in ORDER:
            if d not in legal or d not in frame["ring"]:
                continue
            kind, ch, target = frame["ring"][d]
            if d in frame["foes"]:
                kind = frame["foes"][d] + " (attack by moving " + d + ")"
            if d in frame["lava"]:
                kind = "lava - deadly"
            tag = "refused before" if (frame["pos"], d) in self.refused else "visited %dx" % self.visits[target]
            lines.append("  %s: %s [%s], %s" % (d, kind, ch.strip() or "blank", tag))
        if frame["down"]:
            dx, dy = frame["down"][0] - frame["pos"][0], frame["down"][1] - frame["pos"][1]
            lines.append("  stairs down `>` is %d east, %d south of you" % (dx, dy))
        mem = (ROOT / "memory.md").read_text() if (ROOT / "memory.md").exists() else ""
        if mem.strip():
            lines.append(mem.strip()
                         .replace("{steps}", str(len(self.trail)))
                         .replace("{squares}", str(len(self.visits)))
                         .replace("{here}", str(self.visits[frame["pos"]]))
                         .replace("{trail}", " ".join("%d,%d" % p for p in self.trail[-6:])))
        return "\n".join(lines)

    def reply(self, action, why):
        try:
            from balrog.client import LLMResponse
            return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                               input_tokens=0, output_tokens=0, reasoning=why)
        except Exception:
            from collections import namedtuple
            R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
            return R("harness", action, "stop", 0, 0, why)

    def reflex(self, text, frame, legal):
        """Code-owned rungs above the model: prompts, stairs, swings, field descent, dead-end search."""
        m = re.search(r"(?ms)^message:[ \t]*\n?(.*?)(?=^language observation:|^cursor:|^map:|\Z)", text)
        msg = (m.group(1) if m else "").strip()
        if "[yn" in msg and "y" in legal and "attack" in msg.lower():
            return "y", "reflex:confirm"
        if msg.endswith("--More--") and "more" in legal:
            return "more", "reflex:more"
        if ("[yn" in msg or "What do you want" in msg or "In what direction" in msg) and "esc" in legal:
            return "esc", "reflex:esc"
        pos = frame["pos"]
        if pos is None:
            return None
        terrain = self.chart.get("terrain", {})
        if terrain.get(pos) == ">" and "down" in legal:
            return "down", "reflex:down"
        dirs = [d for d in ORDER if d in legal and d in frame["ring"]]
        for d in dirs:
            if (d in frame["foes"] or frame["ring"][d][0] == "creature") and d not in frame["lava"]:
                return d, "reflex:swing"
        scored = []
        for i, d in enumerate(dirs):
            target = frame["ring"][d][2]
            if target in self.field and not self.forbidden(d, frame, soft=False):
                scored.append((self.field[target] + 0.25 * self.visits[target], i, d))
        if scored:
            return min(scored)[2], "field:descend"
        probes = self.chart.setdefault("probes", {})
        boulders = any(ch == "`" for ch in terrain.values())
        exits = [d for d in dirs if frame["ring"][d][0] in ("open", "stairs down")]
        if "search" in legal and not boulders and len(exits) <= 1 and probes.get(pos, 0) < 8:
            probes[pos] = probes.get(pos, 0) + 1
            return "search", "reflex:search"
        return None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = ""
        if isinstance(obs, dict):
            text = (obs.get("text") or {}).get("long_term_context", "") or ""
        if fold_chart(text, self.chart):
            self.visits, self.trail, self.refused, self.last = Counter(), [], set(), None
        frame = parse_glyph_map(text, self.bars)
        self.bars = frame["bars"]
        self.learn(frame, prev_action)
        legal = self.legal()
        self.field = flood_field(self.chart, self.visits, self.refused, self.bars)
        ruled = self.reflex(text, frame, legal)
        if ruled:
            return self.reply(*ruled)
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            block = "\n\n".join(b for b in (self.priors, self.ground(frame, legal)) if b)
            messages[-1].content += "\n\n" + (block + "\n\n" if block else "") + NAIVE_INSTRUCTION
        response = self.client.generate(messages)
        out = copy.deepcopy(response)
        action, owner = self.vet(extract_action(out.completion, legal), frame, legal)
        out = out._replace(completion=action)
        try:
            out = out._replace(reasoning=owner)
        except (ValueError, AttributeError):
            pass
        return out


def build_agent(client_factory, prompt_builder, config=None):
    return GroundedAgent(client_factory, prompt_builder, config)
