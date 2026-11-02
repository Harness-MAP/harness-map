"""agent.py - grounded dungeon walker: the model picks, the map vets."""
from __future__ import annotations

import copy
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from grid import parse_glyph_map, BEARINGS  # noqa: E402

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

    def forbidden(self, d, frame):
        pos = frame["pos"]
        if d in frame["lava"]:
            return True
        if pos is None or d not in frame["ring"]:
            return False
        kind, _, target = frame["ring"][d]
        if kind in ("lava", "wall") or (pos, d) in self.refused:
            return True
        stale = len(self.trail) >= 3 and target == self.trail[-2] and self.visits[target] >= 3
        return stale and d not in frame["foes"]

    def vet(self, choice, frame, legal):
        bearings = [d for d in ORDER if d in legal]
        if choice and choice not in BEARINGS:
            return choice, "model"
        if choice and not self.forbidden(choice, frame):
            return choice, "model"
        options = [d for d in bearings if not self.forbidden(d, frame)]
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

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = ""
        if isinstance(obs, dict):
            text = (obs.get("text") or {}).get("long_term_context", "") or ""
        frame = parse_glyph_map(text, self.bars)
        self.bars = frame["bars"]
        self.learn(frame, prev_action)
        legal = self.legal()
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
