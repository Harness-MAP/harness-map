"""harness.py — ENTRYPOINT. A MIDDLEWARE PIPELINE around one model call, with a BINDING chart route.

Component graph:
    obs --context.build--> StepContext --ledger.observe--> per-level memory
        --stages[*].pre (outer->inner)--> [short-circuit: modal `more`]
        --prompting.build_messages(notes)--> model (one call) --action_io.snap-->
        --stages[*].post (inner->outer): Route -> Hazard -> Legal--> action
The model proposes; RouteStage (tools/route.py BFS over the drawn map: `>` first, else the best
unprobed frontier) keeps the proposal only if it is the route's first hop or a swing at an adjacent
creature; HazardStage vetoes deadly/blocked/wall/failed moves last. The provenance trail is written
to `response.reasoning`.
"""
from __future__ import annotations

import copy
import sys
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages                    # noqa: E402
from action_io import legal_actions, snap                # noqa: E402
from context import build as build_context              # noqa: E402
from ledger import Ledger                                # noqa: E402
from stages import default_stack                         # noqa: E402

_Resp = namedtuple("_Resp", ["completion", "reasoning", "input_tokens", "output_tokens"],
                   defaults=("", "", 0, 0))


class PipelineHarness:
    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.ledger = Ledger()
        self.stack = default_stack(self.ledger)
        self._system = None
        self.legal = []

    def _reset_state(self):
        self.ledger.reset()
        for st in self.stack:
            st.reset()

    def reset(self):
        self.prompt_builder.reset()
        self._reset_state()

    def _sync_legal(self):
        sp = getattr(self.prompt_builder, "system_prompt", None)
        if not sp:
            try:
                msgs = self.prompt_builder.get_prompt()
                sp = next((m.content for m in msgs if getattr(m, "role", "") == "system"), "")
            except Exception:
                sp = ""
        if sp != self._system:                           # the agent object is reused across tasks
            self._system = sp
            self.legal = legal_actions(sp or "")
            self._reset_state()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        self._sync_legal()
        ctx = build_context(obs, self.legal)
        self.ledger.observe(ctx)

        for st in self.stack:
            early = st.pre(ctx)
            if early is not None:
                self.ledger.commit(ctx.pos, early)
                return _Resp(early, " | ".join(ctx.trail))

        response = self.client.generate(build_messages(self.prompt_builder, notes=ctx.notes))
        action = snap(response.completion, self.legal)
        ctx.trail.append(f"model:{action}")
        for st in reversed(self.stack):
            action = st.post(ctx, action)
        self.ledger.commit(ctx.pos, action)
        out = copy.deepcopy(response)
        try:
            return out._replace(completion=action, reasoning=" | ".join(ctx.trail))
        except (TypeError, ValueError):
            return out._replace(completion=action)


def build_agent(client_factory, prompt_builder, config=None):
    return PipelineHarness(client_factory, prompt_builder, config)
