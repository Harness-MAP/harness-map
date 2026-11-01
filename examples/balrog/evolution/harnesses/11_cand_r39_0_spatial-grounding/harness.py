"""harness.py — ENTRYPOINT. Naive transcript + a code-kept SPATIAL CHART with a BFS route hint.

The model still makes every decision, with one call per step. Before the call, tools/chart.Chart
folds the observation into a per-level map memory: last-seen glyphs, visit counts, blocked (square,
direction) moves and lava/water named adjacent. It then runs a BFS to a charted `>`, or else to the
nearest unexplored square. The prompt gets the route's first hop plus an eight-row neighbour table
(tools/prompting.render_chart). The reply is projected onto the legal action names. A deadly or
already-blocked direction is vetoed, and the BFS first hop is the fallback
(tools/action_io.extract_action).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages, render_chart      # noqa: E402
from action_io import extract_action, legal_actions     # noqa: E402
from chart import Chart, STEP                           # noqa: E402


class ChartHarness:
    """Naive agent whose single call is grounded by, and filtered through, a map-memory Chart."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.chart = Chart()

    def reset(self):
        self.prompt_builder.reset()
        self.chart.reset()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        pos = self.chart.observe(text)
        legal = legal_actions(getattr(self.prompt_builder, "system_prompt", "") or "")
        neigh = self.chart.neighbours()
        route = self.chart.plan(legal)
        vetoed = {d for d, _, _, b, h in neigh if b or h}
        fallback = route[0]
        if fallback is None:
            open_rows = sorted((g in "|- ", v, i, d) for i, (d, g, v, b, h) in enumerate(neigh)
                               if d not in vetoed and d in legal)
            fallback = open_rows[0][3] if open_rows else None
        messages = build_messages(self.prompt_builder, render_chart(neigh, route))
        response = self.client.generate(messages)
        out = copy.deepcopy(response)
        action = extract_action(out.completion, legal, vetoed, fallback)
        self.chart.last = (pos, action) if pos is not None and action in STEP else None
        reason = f"route={route[1]}:{route[0]}@{route[2]} picked={action}"
        try:
            return out._replace(completion=action, reasoning=reason)
        except (TypeError, ValueError):
            return out._replace(completion=action)


def build_agent(client_factory, prompt_builder, config=None):
    return ChartHarness(client_factory, prompt_builder, config)
