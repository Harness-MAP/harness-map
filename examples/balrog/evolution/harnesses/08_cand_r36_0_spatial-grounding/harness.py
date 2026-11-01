"""harness.py - ENTRYPOINT. A perception -> chart -> reflex -> planner -> arbiter pipeline.

Each step:

    obs --> tools/perception.parse_glyph_map --> frame (message, prose nouns, door bearings, rows, pos)
                 |
                 v
          tools/chart.Chart.observe      (terrain, visits, (square,bearing) ledger, door identity)
                 |
                 +-> tools/reflex.modal   --(open prompt)--> `more` / `northwest`, NO model call
                 +-> stairs `down`, tools/reflex.adjacent_foe swing
                 v
          Chart.route      --> first hop to `>` or the nearest frontier (doors seal diagonals)
          Chart.search_plan --> exhausted chart: walk to a dead end and `search` (per-cell budget)
                 |
                 v  (only when every rung above is silent)
          tools/prompting.build_messages --> ONE client.generate --> tools/action_io.arbitrate
                 |
                 v
          action (+ `reasoning` = the rung that decided)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from perception import ORDER, STEP, parse_glyph_map          # noqa: E402
from chart import Chart                                        # noqa: E402
from reflex import modal, adjacent_foe, lethal_moves           # noqa: E402
from action_io import legal_actions, arbitrate                 # noqa: E402
from prompting import compose_priors, build_messages           # noqa: E402


class GroundedAgent:
    """Orchestrator: owns the chart and routes each step through the rung ladder."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.priors = compose_priors(ROOT)
        self.chart = Chart()
        self._sys = None

    def reset(self):
        self.prompt_builder.reset()
        self.priors = compose_priors(ROOT)
        self.chart.reset()
        self._sys = None

    def legal_actions(self):
        sp = getattr(self.prompt_builder, "system_prompt", None) or ""
        if sp != self._sys:
            # the agent is reused across tasks: a new system prompt is a new map
            self._sys = sp
            self.chart.reset()
        return legal_actions(sp)

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

    def ask_model(self, legal, banned, pos=None, foe=None):
        messages = build_messages(self.prompt_builder, self.priors, banned)
        resp = self.client.generate(messages)
        out = copy.deepcopy(resp)
        pick, source = arbitrate(out.completion, legal, banned, pos, self.chart.visits, foe)
        try:
            return out._replace(completion=pick, reasoning=f"{source}:{pick}")
        except Exception:
            return out._replace(completion=pick)

    def _move(self, pos, d, kind, why):
        self.chart.last = (pos, d, kind)
        return self.reply(d, why)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        fr = parse_glyph_map(text)
        legal = self.legal_actions()
        moves = [d for d in ORDER if d in legal]
        chart = self.chart
        chart.observe(fr)
        chart.last = None

        forced = modal(fr, legal)
        if forced:
            return self.reply(*forced)
        pos = fr["pos"]
        if pos is None or not moves:
            return self.ask_model(legal, set())
        if chart.terrain.get(pos) == ">" and "down" in legal:
            return self.reply("down", "reflex:stairs")
        bad = lethal_moves(fr)
        closed = {d for d in moves if (pos, d) in chart.blocked}
        foe = adjacent_foe(fr, moves, bad, chart.blocked)
        if foe:
            return self._move(pos, foe, "swing", "reflex:swing")
        hop = chart.route(pos, moves, bad)
        if hop:
            return self._move(pos, hop, "route", "chart:route")
        if "search" in legal and not chart.stairs_charted():
            plan = chart.search_plan(pos, moves, bad)
            if plan:
                kind, hop = plan
                if kind == "search":
                    return self.reply("search", "chart:search")
                return self._move(pos, hop, "route", "chart:to-dead-end")
        resp = self.ask_model(legal, bad | closed, pos, foe)
        if resp.completion in STEP:
            chart.last = (pos, resp.completion, "model")
        return resp


def build_agent(client_factory, prompt_builder, config=None):
    return GroundedAgent(client_factory, prompt_builder, config)
