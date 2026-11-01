"""orchestrator.py - chart-routed learner: a remembered map + BFS route decide; the value learner owns the no-route residue."""
from __future__ import annotations

import copy
import math
import re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NAIVE_INSTRUCTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)
STEP = {
    "north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1),
}
ORDER = list(STEP)
LETHAL = set("}")
WALL = set("|- ")
CREATURE = re.compile(r"[a-zA-Z&;:']")
TAME = {"@"}
SECTIONS = ["message", "language observation", "cursor", "map"]

ALPHA, GAMMA, LAMBDA = 0.5, 0.9, 0.8
OPTIMISM, BONUS = 1.0, 0.6
BUMP_R, NEW_SQUARE_R, CELL_R, STAIR_R = -1.0, 0.5, 0.05, 0.4
TIE = 0.05


def clause_index(lang):
    """Prose clauses `<noun> adjacent <dirs>` -> {noun: set(dirs)}."""
    out = {}
    for line in (lang or "").splitlines():
        m = re.match(r"^\s*(.+?)\s+adjacent\s+(.+?)\s*$", line.strip())
        if not m:
            continue
        dirs = re.findall(r"(northeast|northwest|southeast|southwest|north|south|east|west)", m.group(2))
        out.setdefault(m.group(1).strip().lower(), set()).update(dirs)
    return out


def parse_glyph_map(text):
    """Split message / prose / cursor / rows; row index = cursor.y - 1."""
    text = text or ""
    parts = {}
    for i, name in enumerate(SECTIONS):
        m = re.search(rf"(?m)^{name}:[ \t]*\n?", text)
        if not m:
            parts[name] = ""
            continue
        rest = text[m.end():]
        cut = len(rest)
        for other in SECTIONS[i + 1:]:
            k = re.search(rf"(?m)^{other}:", rest)
            if k:
                cut = min(cut, k.start())
        parts[name] = rest if name == "map" else rest[:cut]
    rows = parts["map"].split("\n")[:21]
    cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", parts["cursor"])
    pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
    dl = re.search(r"Dlvl:(\d+)", text)
    msg = parts["message"].strip()
    return {
        "message": msg,
        "adjacent": clause_index(parts["language observation"]),
        "rows": rows,
        "pos": pos,
        "dlvl": dl.group(1) if dl else "?",
        "prompt": ("[yn" in msg) or ("What do you want" in msg) or ("In what direction" in msg)
        or msg.endswith("--More--"),
    }


def decide_action(reply, legal, allowed=None):
    """Whole-word projection of a reply onto the legal vocabulary (optionally a subset of it)."""
    text = (reply or "").lower()
    pool = [a for a in legal if allowed is None or a in allowed]
    hits = []
    for name in pool:
        for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?![a-z])", text):
            hits.append((m.start(), -len(name), name))
    return sorted(hits)[0][2] if hits else None


class LearnerAgent:
    """Hub: value table, trace, reward signal, reflexes and the tie-break consult."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.priors = self.compose_priors()
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
        self.q = {}
        self.tries = {}
        self.trace = {}
        self.drawn = {}
        self.stood = set()
        self.pending = None
        self.level = None
        self.consulted = set()
        self.terrain = {}
        self.blocked = set()
        self.bars = False

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

    def glyph(self, fr, cell):
        x, y = cell
        rows = fr["rows"]
        if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
            return rows[y][x]
        return " "

    def stairs_gap(self, fr):
        pos = fr["pos"]
        best = None
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch == ">":
                    d = max(abs(x - pos[0]), abs(y - pos[1]))
                    best = d if best is None else min(best, d)
        return best

    def value(self, s, a):
        if (s, a) in self.blocked:
            return BUMP_R
        nxt = (s[0] + STEP[a][0], s[1] + STEP[a][1])
        if nxt in self.terrain and not self.passable(nxt):
            return self.q.get((s, a), BUMP_R)
        return self.q.get((s, a), OPTIMISM)

    def score(self, s, a):
        n = self.tries.get((s, a), 0)
        return self.value(s, a) + BONUS / math.sqrt(1 + n)

    def credit(self, fr, moves):
        """Reward the previous action and push the TD error back along the trace."""
        pos = fr["pos"]
        new_cells = 0
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@":
                    new_cells += (x, y) not in self.drawn
                    self.drawn[(x, y)] = ch
        gap = self.stairs_gap(fr)
        if self.pending is None or fr["prompt"]:
            self.stood.add(pos)
            return gap
        s, a, kind, old_gap = self.pending
        r = CELL_R * min(new_cells, 10)
        if pos == s and kind != "swing":
            r += BUMP_R
        elif pos not in self.stood:
            r += NEW_SQUARE_R
        if old_gap is not None and gap is not None:
            r += STAIR_R * (old_gap - gap)
        self.stood.add(pos)
        nxt = max((self.value(pos, b) for b in moves), default=0.0)
        delta = r + GAMMA * nxt - self.value(s, a)
        for key in list(self.trace):
            self.q[key] = self.q.get(key, OPTIMISM) + ALPHA * delta * self.trace[key]
            self.trace[key] *= GAMMA * LAMBDA
            if self.trace[key] < 0.01:
                del self.trace[key]
        if pos == s and kind != "swing":
            self.q[(s, a)] = min(self.q.get((s, a), OPTIMISM), BUMP_R)
        return gap

    def chart(self, fr):
        """Fold the frame into the remembered terrain; convict the (square, move) edge of a bump."""
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@":
                    self.terrain[(x, y)] = ch
        pos = fr["pos"]
        if self.terrain.get(pos, "@") == "@":
            self.terrain[pos] = "."
        low = fr["message"].lower()
        if "bars" in low:
            self.bars = True
        if self.pending is not None:
            s, a, kind, _ = self.pending
            if pos == s and kind != "swing" and not fr["prompt"]:
                self.blocked.add((s, a))
            if "really attack" in low:
                self.blocked.add((s, a))

    def passable(self, cell):
        ch = self.terrain.get(cell)
        if ch is None or ch in WALL or ch in LETHAL:
            return False
        return not (ch == "#" and self.bars)

    def frontier(self, cell):
        if cell in self.stood:
            return False
        x, y = cell
        return any((x + dx, y + dy) not in self.terrain for dx, dy in STEP.values())

    def route(self, fr, moves, bad):
        """BFS over charted floor to a drawn `>`, else the nearest unstood frontier; first hop or None."""
        start = fr["pos"]
        seen = {start: None}
        q = deque([start])
        best = None
        while q:
            cur = q.popleft()
            if cur != start:
                if self.terrain.get(cur) == ">":
                    best = cur
                    break
                if best is None and self.frontier(cur):
                    best = cur
                    if ">" not in self.terrain.values():
                        break
            for d in moves:
                if (cur, d) in self.blocked or (cur == start and d in bad):
                    continue
                nxt = (cur[0] + STEP[d][0], cur[1] + STEP[d][1])
                if nxt in seen or not self.passable(nxt):
                    continue
                seen[nxt] = (cur, d)
                q.append(nxt)
        if best is None:
            return None
        hop = None
        while seen[best] is not None:
            best, hop = seen[best]
        return hop

    def deadly(self, fr):
        bad = set()
        for noun, dirs in fr["adjacent"].items():
            if "lava" in noun or "water" in noun:
                bad |= dirs
        pos = fr["pos"]
        for d, (dx, dy) in STEP.items():
            if self.glyph(fr, (pos[0] + dx, pos[1] + dy)) in LETHAL:
                bad.add(d)
        return bad

    def foe_bearing(self, fr, moves, bad):
        pos = fr["pos"]
        for d in moves:
            if d in bad or (pos, d) in self.blocked:
                continue
            ch = self.glyph(fr, (pos[0] + STEP[d][0], pos[1] + STEP[d][1]))
            if CREATURE.fullmatch(ch or " ") and ch not in TAME:
                return d
        return None

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
        pick = decide_action(out.completion, legal, allowed)
        return out, pick

    def commit(self, s, a, kind, gap, why, template=None):
        self.tries[(s, a)] = self.tries.get((s, a), 0) + 1
        self.trace[(s, a)] = 1.0
        self.pending = (s, a, kind, gap)
        return self.reply(a, why, template)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        fr = parse_glyph_map(text)
        legal = self.legal_actions()
        moves = [d for d in ORDER if d in legal]
        pos = fr["pos"]
        if fr["dlvl"] != self.level:
            self.forget()
            self.level = fr["dlvl"]
        msg = fr["message"]
        if fr["prompt"]:
            if self.pending is not None and "really attack" in msg.lower():
                self.blocked.add(self.pending[:2])
            self.pending = None
            if "[yn" in msg and "y" in legal and "attack" in msg.lower():
                return self.reply("y", "reflex:confirm-attack")
            if msg.endswith("--More--") and "more" in legal:
                return self.reply("more", "reflex:more")
            if "esc" in legal:
                return self.reply("esc", "reflex:escape")
        if pos is None or not moves:
            self.pending = None
            out, pick = self.consult(legal, None)
            return out._replace(completion=pick or re.sub(r"[^a-zA-Z\s:]", "", out.completion or ""))
        self.chart(fr)
        gap = self.credit(fr, moves)
        under = "staircase down here" in fr["message"].lower()
        if (self.drawn.get(pos) == ">" or self.terrain.get(pos) == ">" or under) and "down" in legal:
            self.pending = None
            return self.reply("down", "reflex:stairs")
        bad = self.deadly(fr)
        foe = self.foe_bearing(fr, moves, bad)
        if foe:
            return self.commit(pos, foe, "swing", gap, "reflex:swing")
        hop = self.route(fr, moves, bad)
        if hop:
            return self.commit(pos, hop, "move", gap, "chart:route")
        safe = [d for d in moves if d not in bad] or moves
        ranked = sorted(safe, key=lambda d: (-self.score(pos, d), ORDER.index(d)))
        top = self.score(pos, ranked[0])
        tied = [d for d in ranked if top - self.score(pos, d) <= TIE]
        if len(tied) > 1 and pos not in self.consulted:
            self.consulted.add(pos)
            out, pick = self.consult(legal, set(tied),
                                     "Equally promising moves from this square: " + ", ".join(tied) + ".")
            if pick:
                return self.commit(pos, pick, "move", gap, "learner:consult", out)
        return self.commit(pos, ranked[0], "move", gap, "learner:argmax")


def build_agent(client_factory, prompt_builder, config=None):
    return LearnerAgent(client_factory, prompt_builder, config)
