"""harness.py — ENTRYPOINT. Naive transcript + a code-kept NEIGHBOUR LEDGER + legal projection.

The model still makes every decision, one call per step, but it decides from facts code has
grounded: the ledger records, per level, how often each square was stood on and which (square,
direction) moves produced no displacement. Each turn the eight compass neighbours of the cursor are
rendered as a table (tools/prompting.render_ledger). The reply is projected onto the legal action
names; a direction the ledger marks deadly (lava/water named adjacent) or already failed from this
square is vetoed (tools/action_io.extract_action).
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages          # noqa: E402
from action_io import extract_action, legal_actions   # noqa: E402

STEP = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
        "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
DIRS = "northeast|northwest|southeast|southwest|north|south|east|west"


class Ledger:
    """Self-derived per-level spatial record: visits per square, failed (square, direction) moves."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.level, self.visits, self.failed, self.last = None, {}, set(), None

    def observe(self, text):
        text = text or ""
        cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", text)
        pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
        dl = re.search(r"Dlvl:(\d+)", text)
        lvl = dl.group(1) if dl else "?"
        if lvl != self.level:
            self.level, self.visits, self.failed, self.last = lvl, {}, set(), None
        if pos is not None:
            if self.last and self.last[0] == pos:
                self.failed.add(self.last)
            self.visits[pos] = self.visits.get(pos, 0) + 1
        mp = text.split("map:", 1)[1].split("\n")[1:22] if "map:" in text else []
        hazard = set()
        for line in text.splitlines():
            if ("lava" in line or "water" in line) and "adjacent" in line:
                hazard |= set(re.findall(DIRS, line.split("adjacent", 1)[1]))
        return pos, mp, hazard

    def rows(self, pos, mp, hazard):
        if pos is None:
            return []
        out = []
        for d, (dx, dy) in STEP.items():
            x, y = pos[0] + dx, pos[1] + dy
            g = mp[y][x] if 0 <= y < len(mp) and 0 <= x < len(mp[y]) else " "
            out.append((d, g, self.visits.get((x, y), 0), (pos, d) in self.failed, d in hazard))
        return out

    def commit(self, pos, action):
        self.last = (pos, action) if pos is not None and action in STEP else None


class LedgerHarness:
    """Naive agent whose single call is grounded by, and filtered through, the Ledger."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.ledger = Ledger()

    def reset(self):
        self.prompt_builder.reset()
        self.ledger.reset()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        pos, mp, hazard = self.ledger.observe(text)
        rows = self.ledger.rows(pos, mp, hazard)
        legal = legal_actions(getattr(self.prompt_builder, "system_prompt", "") or "")
        vetoed = {d for d, _, _, f, h in rows if f or h}
        open_rows = sorted((g in "|- ", v, i, d) for i, (d, g, v, f, h) in enumerate(rows)
                           if d not in vetoed and d in legal)
        fallback = open_rows[0][3] if open_rows else None
        messages = build_messages(self.prompt_builder, rows)
        response = self.client.generate(messages)
        out = copy.deepcopy(response)
        action = extract_action(out.completion, legal, vetoed, fallback)
        self.ledger.commit(pos, action)
        return out._replace(completion=action)


def build_agent(client_factory, prompt_builder, config=None):
    return LedgerHarness(client_factory, prompt_builder, config)
