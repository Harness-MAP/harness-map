"""harness.py — ENTRYPOINT. A MIDDLEWARE PIPELINE around one model call.

Component graph:
    obs --context.build--> StepContext
        --stages[*].pre (outer->inner)--> [short-circuit action?]
        --prompting.build_messages(notes)--> model --action_io.snap-->
        --stages[*].post (inner->outer)--> action
FieldStage (innermost) now OWNS movement: it descends a visit-weighted potential field over a
persistent chart and short-circuits; the model is consulted only where the field is flat.
Otherwise: the model PROPOSES, each stage (tools/stages.py) may
answer before the model is asked (modal prompts) or rewrite its proposal afterwards (bump memory,
lava veto, orbit breaking, legality). Stages share nothing but the StepContext.
"""
from __future__ import annotations

import copy
import sys
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages                  # noqa: E402
from action_io import extract_action, legal_actions, snap  # noqa: E402
from context import build as build_context            # noqa: E402
from stages import default_stack                      # noqa: E402

_Resp = namedtuple("_Resp", ["completion", "reasoning", "input_tokens", "output_tokens"],
                   defaults=("", "", 0, 0))


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

    def reset(self):
        self.prompt_builder.reset()
        self._reset_stages()

    def _sync_legal(self):
        sp = getattr(self.prompt_builder, "system_prompt", None)
        if not sp:
            msgs = self.prompt_builder.get_prompt()
            sp = next((m.content for m in msgs if getattr(m, "role", "") == "system"), "")
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

        early, depth = None, len(self.stack)
        for i, st in enumerate(self.stack):
            early = st.pre(ctx)
            if early is not None:
                depth = i
                break

        if early is not None:
            # short-circuit: the stages OUTSIDE the answering one still get their post() veto
            action = early
            for st in reversed(self.stack[:depth]):
                action = st.post(ctx, action)
            ctx.action = action
            return _Resp(action, " | ".join(ctx.trail))

        response = self.client.generate(build_messages(self.prompt_builder, notes=ctx.notes))
        action = snap(response.completion, self.legal) if self.legal else extract_action(response.completion)
        ctx.trail.append(f"model:{action}")
        for st in reversed(self.stack):
            action = st.post(ctx, action)
        ctx.action = action
        out = copy.deepcopy(response)
        try:
            return out._replace(completion=action, reasoning=" | ".join(ctx.trail))
        except (TypeError, ValueError):
            return out._replace(completion=action)


def build_agent(client_factory, prompt_builder, config=None):
    return PipelineHarness(client_factory, prompt_builder, config)
