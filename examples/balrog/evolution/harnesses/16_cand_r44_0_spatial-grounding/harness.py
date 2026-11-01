"""harness.py - one-module pilot: the chart walks, the model is asked only when the chart has no route."""
from __future__ import annotations

import copy
import importlib.util
import re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_tool(name):
    """Load tools/<name>.py under a path-unique module name (several harnesses share one process)."""
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_harness_{abs(hash(str(path)))}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


terrain = _load_tool("terrain")

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
    """Decode message / prose / cursor / map rows; map row = cursor.y - 1."""
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
    on_msg_line = pos is not None and pos[1] < 0
    if on_msg_line:
        pos = None
    dl = re.search(r"Dlvl:(\d+)", text)
    msg = parts["message"].strip()
    return {
        "message": msg,
        "adjacent": clause_index(parts["language observation"]),
        "rows": rows,
        "pos": pos,
        "dlvl": dl.group(1) if dl else "?",
        "prompt": ("[yn" in msg) or ("What do you want" in msg) or ("In what direction" in msg)
        or msg.endswith("--More--") or on_msg_line,
    }


def decide_action(reply, legal, banned=()):
    """Whole-word projection of the model reply onto the legal vocabulary, skipping banned moves."""
    text = (reply or "").lower()
    hits = []
    for name in legal:
        for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?![a-z])", text):
            hits.append((m.start(), -len(name), name))
    for _, _, name in sorted(hits):
        if name not in banned:
            return name
    return None


class GroundedAgent:
    """Hub: chart memory, frontier walk, modal reflex, hazard veto and the fallback model call."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.priors = self.compose_priors()
        self.reset_chart()

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

    def reset_chart(self):
        self.terrain = {}
        self.visits = {}
        self.blocked = set()
        self.last = None
        self.level = None
        self.facts = {}
        self.bars = False
        self.peaceful = set()
        self.swing_glyph = None
        self.mores = 0

    def reset(self):
        self.prompt_builder.reset()
        self.priors = self.compose_priors()
        self.reset_chart()

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

    def observe(self, fr):
        if fr["dlvl"] != self.level:
            self.reset_chart()
            self.level = fr["dlvl"]
        for y, row in enumerate(fr["rows"]):
            for x, ch in enumerate(row):
                if ch != " " and ch != "@":
                    self.terrain[(x, y)] = ch
        pos = fr["pos"]
        if pos is None:
            return
        if terrain.level_has_bars(fr["adjacent"]):
            self.bars = True
        for cell, ch in self.terrain.items():
            if ch == "+" and self.facts.get(cell) is None:
                self.facts[cell] = terrain.DOOR
        self.facts.update(terrain.prose_cell_facts(fr["adjacent"], pos))
        if pos not in self.terrain or self.terrain[pos] == "@":
            self.terrain[pos] = "."
        if self.last and not fr["prompt"]:
            prev, move, kind = self.last
            if prev == pos and move in STEP and kind != "swing":
                target = (prev[0] + STEP[move][0], prev[1] + STEP[move][1])
                if "door opens" in fr["message"].lower():
                    self.facts[target] = terrain.DOOR
                else:
                    self.blocked.add((prev, move))
        self.visits[pos] = self.visits.get(pos, 0) + 1

    def glyph(self, fr, cell):
        x, y = cell
        rows = fr["rows"]
        if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
            return rows[y][x]
        return " "

    def lethal_moves(self, fr):
        bad = set()
        for noun, dirs in fr["adjacent"].items():
            if "lava" in noun or "water" in noun:
                bad |= dirs
        pos = fr["pos"]
        if pos:
            for d, (dx, dy) in STEP.items():
                if self.glyph(fr, (pos[0] + dx, pos[1] + dy)) in LETHAL:
                    bad.add(d)
        return bad

    def passable(self, cell):
        if cell not in self.terrain and cell not in self.facts:
            return False
        return terrain.passable(self.terrain.get(cell), self.facts.get(cell), self.bars)

    def edge_ok(self, cur, nxt, move):
        return terrain.edge_ok(cur, nxt, move, self.facts, self.terrain.get)

    def frontier(self, cell):
        x, y = cell
        if self.visits.get(cell, 0):
            return False
        return any((x + dx, y + dy) not in self.terrain for dx, dy in STEP.values())

    def route(self, fr, moves, bad=()):
        """BFS over charted floor to a drawn `>`, else the nearest frontier; returns the first hop."""
        start = fr["pos"]
        stairs = [c for c, ch in self.terrain.items() if ch == ">"]
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
                if nxt in seen or not self.passable(nxt) or not self.edge_ok(cur, nxt, d):
                    continue
                seen[nxt] = (cur, d)
                q.append(nxt)
        if best is None:
            return None
        hop = None
        while seen[best] is not None:
            best, hop = seen[best]
        return hop

    def adjacent_foe(self, fr, moves, bad):
        pos = fr["pos"]
        for d in moves:
            if d in bad or (pos, d) in self.blocked:
                continue
            nxt = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
            ch = self.glyph(fr, nxt)
            if CREATURE.fullmatch(ch or " ") and ch not in TAME and ch not in self.peaceful \
                    and self.edge_ok(pos, nxt, d):
                self.swing_glyph = ch
                return d
        return None

    def ask_model(self, legal, banned, pos=None):
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
        if pick is None and decide_action(out.completion, legal) in STEP and pos is not None:
            safe = [d for d in ORDER if d in legal and d not in banned]
            if safe:
                pick = min(safe, key=lambda d: (self.visits.get((pos[0] + STEP[d][0], pos[1] + STEP[d][1]), 0),
                                                ORDER.index(d)))
        if pick is None:
            pick = decide_action(out.completion, legal) or re.sub(r"[^a-zA-Z\s:]", "", out.completion or "")
        return out._replace(completion=pick)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        fr = parse_glyph_map(text)
        legal = self.legal_actions()
        moves = [d for d in ORDER if d in legal]
        self.observe(fr)
        self.last = None
        msg = fr["message"]
        if fr["prompt"]:
            if "[yn" in msg and "attack" in msg.lower() and self.swing_glyph:
                # a `Really attack?` prompt names a peaceful: stop swinging at that glyph
                self.peaceful.add(self.swing_glyph)
            if "more" in legal and self.mores < 3:
                # `more` is the one legal verb that dismisses --More--, [yn] and item prompts alike
                self.mores += 1
                return self.reply("more", "reflex:more")
        else:
            self.mores = 0
        pos = fr["pos"]
        if pos is None or not moves:
            return self.ask_model(legal, set())
        if self.terrain.get(pos) == ">" and "down" in legal:
            return self.reply("down", "reflex:stairs")
        bad = self.lethal_moves(fr)
        closed = {d for d in moves if (pos, d) in self.blocked}
        foe = self.adjacent_foe(fr, moves, bad)
        if foe:
            self.last = (pos, foe, "swing")
            return self.reply(foe, "reflex:swing")
        hop = self.route(fr, moves, bad)
        if hop:
            self.last = (pos, hop, "route")
            return self.reply(hop, "chart:route")
        resp = self.ask_model(legal, bad | closed, pos)
        if resp.completion in STEP:
            self.last = (pos, resp.completion, "model")
        return resp


def build_agent(client_factory, prompt_builder, config=None):
    return GroundedAgent(client_factory, prompt_builder, config)
