"""harness.py — ENTRYPOINT. Naive single call wrapped in a MIDDLEWARE STACK over a map-memory Chart.

Per step: tools/chart.Chart folds the observation into a per-level map memory (glyphs, visits,
prose-named walls, terrain-refused edges) and plans a BFS route to a charted `>` or the nearest
frontier. tools/stages.default_stack() then runs pre hooks (outer->inner: Veto, Swing, Route) that
add vetoes and prompt notes, the model is called ONCE, its reply is projected onto the legal set
(tools/action_io.extract_action), and post hooks (inner->outer) may rewrite the proposal: Route
replaces a move onto a visited square with the route hop, Swing protects an attack on an adjacent
creature, Veto swaps a vetoed pick for the fallback. The deciding trail goes to `reasoning`.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages          # noqa: E402
from action_io import extract_action, legal_actions   # noqa: E402
from chart import Chart, STEP                  # noqa: E402
from stages import StepContext, default_stack  # noqa: E402


class PipelineHarness:
    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.chart = Chart()
        self.stack = default_stack()
        self._sp = None

    def reset(self):
        self.prompt_builder.reset()
        self.chart.reset()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        sp = getattr(self.prompt_builder, "system_prompt", "") or ""
        if sp != self._sp:  # the agent object is reused across tasks: new task, fresh chart
            self._sp = sp
            self.chart.reset()
        text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        pos = self.chart.observe(text)
        legal = legal_actions(sp)
        ctx = StepContext(self.chart, legal, self.chart.neighbours(), self.chart.plan(legal))
        for st in self.stack:
            st.pre(ctx)
        if ctx.fallback is None:
            open_rows = sorted((g in "|- ", v, i, d) for i, (d, g, v, b, h, w) in enumerate(ctx.neigh)
                               if d not in ctx.vetoed and d in legal)
            ctx.fallback = open_rows[0][3] if open_rows else None
        messages = build_messages(self.prompt_builder, ctx.neigh, notes=ctx.notes)
        response = self.client.generate(messages)
        out = copy.deepcopy(response)
        action = extract_action(out.completion, legal, ctx.vetoed, ctx.fallback)
        ctx.trail.append(f"model:{action}")
        for st in reversed(self.stack):
            action = st.post(ctx, action)
        self.chart.last = (pos, action) if pos is not None and action in STEP else None
        reason = f"route={ctx.route[1]}:{ctx.route[0]}@{ctx.route[2]} " + " ".join(ctx.trail) + f" picked={action}"
        try:
            return out._replace(completion=action, reasoning=reason)
        except (TypeError, ValueError):
            return out._replace(completion=action)


def build_agent(client_factory, prompt_builder, config=None):
    return PipelineHarness(client_factory, prompt_builder, config)
