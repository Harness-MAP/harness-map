"""harness.py - a charted learner wired as an ordered STAGE table.

`act()` builds one context `c` and runs STAGES in order; the first stage that answers is emitted
and its rung is written to `reasoning`. The chart routes, the value table walks where the chart has
no route, and tools/quest.py (armed only on the printed verb list {pickup, kick, zap}, i.e. the
Quest pair) owns the non-movement plan: kick-split the spawn pile, pick up horn/wand, fire across
the lava, hold the corridor mouth."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import re
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_TAG = hashlib.md5(str(ROOT).encode()).hexdigest()[:10]


def load_tool(name):
    """Import tools/<name>.py under a path-unique module key (no cross-candidate cache hits)."""
    key = "_hm_%s_%s" % (name, _TAG)
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / (name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


Quest = load_tool("quest").Quest
Sokoban = load_tool("sokoban").Sokoban
STAGES = ("prompt", "orient", "sokoban", "stairs", "swing", "cross", "hold", "kit", "fetch",
          "route", "learner")
FOE_GLYPH = re.compile(r"[a-zA-Z&']")

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
DOOR_NOUNS = ("doorway", "open door", "broken door")
REFUSALS = ("solid stone", "it's a wall", "bars", "diagonally", "boulder", "locked")
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
    prompt = (("[yn" in msg) or ("What do you want" in msg) or ("In what direction" in msg)
              or msg.endswith("--More--"))
    return {
        "message": msg,
        "lang": parts["language observation"],
        "modal": prompt or (pos is not None and pos[1] < 0),
        "adjacent": clause_index(parts["language observation"]),
        "rows": rows,
        "pos": pos,
        "dlvl": dl.group(1) if dl else "?",
        "prompt": prompt,
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


class ChartView:
    """The Quest module's view of the learner's chart: `terrain`, `passable`, `edge`."""

    def __init__(self, agent):
        self.agent = agent

    @property
    def terrain(self):
        return self.agent.drawn

    def passable(self, cell):
        return self.agent.passable(cell)

    def edge(self, a, d):
        b = (a[0] + STEP[d][0], a[1] + STEP[d][1])
        if (a, d) in self.agent.blocked or not self.agent.passable(b):
            return None
        if len(d) > 5 and (a in self.agent.doors or b in self.agent.doors):
            return None
        return b


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
        self.visits = {}
        self.blocked = set()
        self.doors = set()
        self.peaceful = set()
        self.mores = 0
        self.last_foe = None
        self.quest = Quest()
        self.soko = Sokoban()
        self.bars = False
        self.chart = ChartView(self)

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

    def observe(self, fr):
        """Chart memory: doors named in prose or just opened become passable; bumps close an edge."""
        pos = fr["pos"]
        if any("bars" in noun for noun in fr["adjacent"]) or "through the bars" in fr["message"]:
            self.bars = True                        # prose `bars` <=> `#` is iron bars, not corridor
        for noun, dirs in fr["adjacent"].items():
            if any(k in noun for k in DOOR_NOUNS):
                for d in dirs:
                    self.doors.add((pos[0] + STEP[d][0], pos[1] + STEP[d][1]))
        self.visits[pos] = self.visits.get(pos, 0) + 1
        if self.pending is None:
            return
        s, a, kind, _ = self.pending
        if s != pos or kind == "swing" or a not in STEP:
            return
        low = fr["message"].lower()
        target = (s[0] + STEP[a][0], s[1] + STEP[a][1])
        if "door opens" in low:
            self.doors.add(target)
        else:
            self.blocked.add((s, a))

    def passable(self, cell):
        if cell in self.doors:
            return True
        ch = self.drawn.get(cell)
        if ch == "#" and self.bars:
            return False
        return ch is not None and ch not in WALL and ch not in LETHAL

    def frontier(self, cell):
        if self.visits.get(cell, 0):
            return False
        x, y = cell
        return any((x + dx, y + dy) not in self.drawn for dx, dy in STEP.values())

    def route(self, fr, moves, bad):
        """BFS over charted floor to a drawn `>`, else the nearest unstood frontier; returns the first hop."""
        start = fr["pos"]
        stairs = {c for c, ch in self.drawn.items() if ch == ">"}
        seen = {start: None}
        q = deque([start])
        best = None
        while q:
            cur = q.popleft()
            if cur != start:
                if cur in stairs:
                    best = cur
                    break
                if best is None and self.frontier(cur):
                    best = cur
                    if not stairs:
                        break
            for d in moves:
                if (cur, d) in self.blocked or (cur == start and d in bad):
                    continue
                nxt = (cur[0] + STEP[d][0], cur[1] + STEP[d][1])
                if nxt in seen or not self.passable(nxt):
                    continue
                if len(d) > 5 and d not in ("north", "south") and (nxt in self.doors or cur in self.doors):
                    continue
                seen[nxt] = (cur, d)
                q.append(nxt)
        if best is None:
            return None
        hop = None
        while seen[best] is not None:
            best, hop = seen[best]
        return hop

    def admissible(self, pos, d):
        nxt = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
        return (pos, d) not in self.blocked and (self.passable(nxt) or self.drawn.get(nxt) == "`")

    def value(self, s, a):
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
        opened = "door opens" in fr["message"].lower()
        if pos == s and kind != "swing" and not opened:
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
        if pos == s and kind != "swing" and not opened:
            self.q[(s, a)] = min(self.q.get((s, a), OPTIMISM), BUMP_R)
        return gap

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
            if d in bad:
                continue
            ch = self.glyph(fr, (pos[0] + STEP[d][0], pos[1] + STEP[d][1]))
            if CREATURE.fullmatch(ch or " ") and ch not in TAME and ch not in self.peaceful:
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

    def goal_kind(self, pos):
        """'stairs' when a drawn `>` is reachable on the chart, else 'frontier' / None."""
        seen, q, front = {pos}, deque([pos]), False
        while q:
            cur = q.popleft()
            if self.drawn.get(cur) == ">":
                return "stairs"
            if cur != pos and self.frontier(cur):
                front = True
            for d in STEP:
                nxt = self.chart.edge(cur, d)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return "frontier" if front else None

    def verb(self, word, why):
        """A non-movement emission: never credited as a bump by the learner."""
        self.pending = None
        return self.reply(word, why)

    # -- stages: each returns a reply or None (abstain) ------------------------------------------
    def st_prompt(self, c):
        fr, legal, msg = c["fr"], c["legal"], c["fr"]["message"]
        if not fr["modal"]:
            self.mores = 0
            return None
        last = self.pending
        self.pending = None
        if c["quest"]:
            got = self.quest.answer(msg, legal)
            if got:
                return self.reply(*got)
        if "[yn" in msg and "attack" in msg.lower():
            if last and last[2] == "swing" and self.last_foe:
                self.peaceful.add(self.last_foe)
        if "more" in legal and self.mores < 2:
            self.mores += 1
            return self.reply("more", "reflex:more")
        if "esc" in legal:
            return self.reply("esc", "reflex:escape")
        self.mores = 0
        return None

    def st_orient(self, c):
        fr, legal = c["fr"], c["legal"]
        pos = fr["pos"]
        if pos is None or pos[1] < 0 or not c["moves"]:
            self.pending = None
            out, pick = self.consult(legal, None)
            return out._replace(completion=pick or re.sub(r"[^a-zA-Z\s:]", "", out.completion or ""))
        self.observe(fr)
        c["gap"] = self.credit(fr, c["moves"])
        c["bad"] = self.deadly(fr)
        if c["quest"]:
            self.quest.observe(fr)
            c["kind"] = self.goal_kind(pos)
        return None

    def st_sokoban(self, c):
        """Boxoban owner (armed on 4-cardinal vocab + drawn boulder and fountain): exact push plan."""
        fr = c["fr"]
        if not Sokoban.armed(c["moves"], fr["rows"]):
            return None
        got = self.soko.step(fr["rows"], fr["pos"], self.drawn)
        if got is None or got[0] not in c["legal"]:
            return None
        return self.commit(fr["pos"], got[0], "move", c["gap"], got[1])

    def st_stairs(self, c):
        under = "staircase down here" in c["fr"]["message"].lower()
        if (self.drawn.get(c["fr"]["pos"]) == ">" or under) and "down" in c["legal"]:
            return self.verb("down", "reflex:stairs")
        return None

    def st_swing(self, c):
        fr, pos = c["fr"], c["fr"]["pos"]
        foe = self.foe_bearing(fr, c["moves"], c["bad"])
        if foe:
            self.last_foe = self.glyph(fr, (pos[0] + STEP[foe][0], pos[1] + STEP[foe][1]))
            return self.commit(pos, foe, "swing", c["gap"], "reflex:swing")
        return None

    def st_cross(self, c):
        if not c["quest"]:
            return None
        got = self.quest.cross(self.chart, c["fr"]["pos"], c["legal"], c["kind"])
        return self.verb(*got) if got else None

    def st_hold(self, c):
        if not c["quest"]:
            return None
        got = self.quest.hold(c["fr"], self.chart, c["fr"]["pos"], c["legal"], c["kind"])
        return self.verb(*got) if got else None

    def st_kit(self, c):
        if not c["quest"]:
            return None
        got = self.quest.kit(self.chart, c["fr"]["pos"], c["legal"])
        return self.verb(*got) if got else None

    def st_fetch(self, c):
        if not c["quest"] or c["kind"] == "stairs":
            return None
        fr, pos = c["fr"], c["fr"]["pos"]
        occupied = {(x, y) for y, row in enumerate(fr["rows"]) for x, ch in enumerate(row)
                    if FOE_GLYPH.fullmatch(ch) and (x, y) != pos}
        d, n = Quest.toward(self.chart, pos, self.quest.targets(self.chart, c["legal"], c["kind"]),
                            occupied)
        if d is None or d in c["bad"] or d not in c["legal"]:
            return None
        return self.commit(pos, d, "move", c["gap"], "quest:walk:%d" % n)

    def st_route(self, c):
        hop = self.route(c["fr"], c["moves"], c["bad"])
        if hop:
            return self.commit(c["fr"]["pos"], hop, "move", c["gap"], "chart:route")
        return None

    def st_learner(self, c):
        pos, moves, bad, gap = c["fr"]["pos"], c["moves"], c["bad"], c["gap"]
        safe = [d for d in moves if d not in bad] or moves
        safe = [d for d in safe if self.admissible(pos, d)] or safe
        ranked = sorted(safe, key=lambda d: (-self.score(pos, d), ORDER.index(d)))
        top = self.score(pos, ranked[0])
        tied = [d for d in ranked if top - self.score(pos, d) <= TIE]
        if len(tied) > 1 and pos not in self.consulted and not c["quest"]:
            self.consulted.add(pos)
            out, pick = self.consult(c["legal"], set(tied),
                                     "Equally promising moves from this square: " + ", ".join(tied) + ".")
            if pick:
                return self.commit(pos, pick, "move", gap, "learner:consult", out)
        return self.commit(pos, ranked[0], "move", gap, "learner:argmax")

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        fr = parse_glyph_map(text)
        legal = self.legal_actions()
        if fr["dlvl"] != self.level:
            self.forget()
            self.level = fr["dlvl"]
        c = {"fr": fr, "legal": legal, "moves": [d for d in ORDER if d in legal],
             "quest": Quest.armed(legal), "gap": None, "bad": set(), "kind": None}
        for name in STAGES:
            got = getattr(self, "st_" + name)(c)
            if got is not None:
                return got
        return self.reply(c["moves"][0] if c["moves"] else legal[0], "fallback")


def build_agent(client_factory, prompt_builder, config=None):
    return LearnerAgent(client_factory, prompt_builder, config)
