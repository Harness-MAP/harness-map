"""orchestrator.py — ENTRYPOINT. A MIDDLEWARE PIPELINE whose decision locus for movement is a CHART.

Component graph (tools/stages.py owns the stages, tools/context.py the per-step object):
    obs --context.build--> StepContext
        --stages[*].pre (outer->inner)--> short-circuit?   Modal: `[yn]`/`--More--`
                                                           Route: BFS hop to `>` / nearest frontier
        --prompting.build_messages(notes)--> ONE model call --action_io.snap-->
        --stages[*].post (inner->outer)--> action          Bump veto, Hazard veto, Legal audit
The model is called only when code has no route (a creature adjacent, a sealed or premapped level).
The deciding stage is written to `response.reasoning`.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages                  # noqa: E402
from action_io import legal_actions, snap             # noqa: E402
from context import build as build_context            # noqa: E402
from stages import default_stack                      # noqa: E402


def _reply(action, why, template=None):
    if template is not None:
        try:
            return template._replace(completion=action, reasoning=why)
        except (TypeError, ValueError):
            return template._replace(completion=action)
    try:
        from balrog.client import LLMResponse
        return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                           input_tokens=0, output_tokens=0, reasoning=why)
    except Exception:
        from collections import namedtuple
        R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
        return R("harness", action, "stop", 0, 0, why)


class PipelineHarness:
    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.stack = default_stack()
        self._system = None
        self.legal = []
        self._reset_stages()

    def _reset_stages(self):
        for st in self.stack:
            st.reset()
        self.route = next(st for st in self.stack if hasattr(st, "chart"))

    def reset(self):
        self.prompt_builder.reset()
        self._reset_stages()

    def _sync_legal(self):
        sp = getattr(self.prompt_builder, "system_prompt", "") or ""
        if sp != self._system:                 # the agent object is reused across tasks
            self._system = sp
            self.legal = legal_actions(sp)
            self._reset_stages()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        self._sync_legal()
        ctx = build_context(obs, self.legal)

        action, template = None, None
        for st in self.stack:
            action = st.pre(ctx)
            if action is not None:
                break
        if action is None:
            response = self.client.generate(build_messages(self.prompt_builder, notes=ctx.notes))
            template = copy.deepcopy(response)
            action = snap(response.completion, self.legal)
            ctx.trail.append(f"model:{action}")
            for st in reversed(self.stack):
                action = st.post(ctx, action)
        self.route.chart.commit(ctx.pos, action)
        return _reply(action, " | ".join(ctx.trail), template)


def build_agent(client_factory, prompt_builder, config=None):
    return PipelineHarness(client_factory, prompt_builder, config)
