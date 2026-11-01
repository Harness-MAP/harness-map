"""harness.py — ENTRYPOINT. Gradient walker over a prose-grounded chart; the model breaks FLAT ground only.

Decision order per step (the deciding rule is written to `response.reasoning`):
  perceive  tools/ledger.py      cursor, map, Dlvl, "<noun> adjacent <dirs>" prose -> 8 neighbour rows
  chart     tools/field.py       per-level terrain chart + reverse Dijkstra field (re-elected every step)
  modal     harness.modal_reflex an open `[yn]` / `--More--` prompt -> `more` (never a compass move)
  down      standing on `>` with `down` legal
  swing     a drawn LETTER on a neighbour the prose does not name as terrain
  field     step to the open neighbour with the lowest potential (deadly/blocked/failed excluded)
  model     only when no neighbour carries a potential: one call on the naive transcript + table,
            projected by tools/action_io.arbitrate (legal set, veto, orbit break)
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages                  # noqa: E402
from action_io import arbitrate, legal_actions        # noqa: E402
from ledger import Ledger, STEP                        # noqa: E402
from field import Chart                                # noqa: E402

FOE = re.compile(r"[a-zA-Z&']")


def modal_reflex(text: str, legal: list):
    """An open modal prompt freezes the frame; answer it with a legal verb, never a move.
    `y` is not a MiniHack action; `more` escapes both `[yn]` and `--More--`."""
    head = (text or "").split("language observation", 1)[0]
    if ("[yn" in head or "--More--" in head or "What do you want" in head) and "more" in legal:
        return "more"
    return None


class GroundedHarness:
    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.ledger = Ledger()
        self.chart = Chart()

    def reset(self):
        self.prompt_builder.reset()
        self.ledger.reset()
        self.chart.reset()

    def reply(self, action, why):
        try:
            from balrog.client import LLMResponse
            return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                               input_tokens=0, output_tokens=0, reasoning=why)
        except Exception:
            from collections import namedtuple
            R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
            return R("harness", action, "stop", 0, 0, why)

    def decide_in_code(self, text, legal, pos, mp, rows):
        """-> (action, rule) or None when the ground is flat and the model must decide."""
        reflex = modal_reflex(text, legal)
        if reflex:
            if self.ledger.last:
                self.ledger.failed.add(self.ledger.last)   # do not re-bump what opened the prompt
            return reflex, "modal"
        if pos is None:
            return None
        if self.chart.terrain.get(pos) == ">" and "down" in legal:
            return "down", "down"
        for d, g, v, f, kind, back in rows:
            if d in legal and FOE.fullmatch(g or " ") and g != "@" and kind in ("creature", ""):
                return d, "swing"
        field = self.chart.flood(self.ledger.visits, self.ledger.failed)
        scored = []
        for i, (d, g, v, f, kind, back) in enumerate(rows):
            c = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
            if d not in legal or f or kind in ("deadly", "blocked") or c not in field:
                continue
            scored.append((field[c], v, back, i, d))
        if scored:
            return min(scored)[-1], "field"
        return None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        try:
            text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        except (KeyError, TypeError):
            text = str(obs)
        pos, mp, adj = self.ledger.observe(text)
        rows = self.ledger.rows(pos, mp, adj)
        self.chart.fold(self.ledger.level, mp, pos, rows)
        legal = legal_actions(getattr(self.prompt_builder, "system_prompt", "") or "")
        coded = self.decide_in_code(text, legal, pos, mp, rows) if legal else None
        if coded:
            action, rule = coded
            self.ledger.commit(pos, action)
            return self.reply(action, f"[{rule}]")
        messages = build_messages(self.prompt_builder, rows)
        out = copy.deepcopy(self.client.generate(messages))
        action, rule = arbitrate(out.completion, legal, rows)
        self.ledger.commit(pos, action)
        out = out._replace(completion=action)
        if "reasoning" in getattr(out, "_fields", ()):
            out = out._replace(reasoning=f"[{rule}] {getattr(out, 'reasoning', '') or ''}".strip())
        return out


def build_agent(client_factory, prompt_builder, config=None):
    return GroundedHarness(client_factory, prompt_builder, config)
