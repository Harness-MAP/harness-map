"""agent.py — ENTRYPOINT. A MIDDLEWARE PIPELINE whose innermost decision stage is a code planner.

Component graph:
    obs --context.build--> StepContext
        --stages[*].pre (outer->inner, ALL run; first answer wins)--> [answer?]
        --(no answer) prompting.build_messages(notes) --> model --action_io.snap-->
        --stages[*].post (inner->outer)--> action --reasoning = provenance trail
Stage order (tools/stages.py:default_stack): Legal, Modal ([yn]/--More--), Door (locked -> kick),
Bump (message-classified refusal memory), Hazard (lava/water), Foe (swing at an adjacent drawn
creature), Nav (BFS over the remembered chart to `>` / nearest unstood frontier), Orbit (model
fallback only). The model is consulted only when no stage answers, and then only proposes.
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
        self.shared = {}
        self._reset_stages()

    def _reset_stages(self):
        self.shared = {}
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
        ctx = build_context(obs, self.legal, self.shared)

        answer = None
        for st in self.stack:
            try:
                early = st.pre(ctx)
            except Exception as exc:           # a stage bug must never cost the episode
                ctx.trail.append(f"{type(st).__name__}!{exc!r}")
                early = None
            if answer is None and early is not None and (not self.legal or early in self.legal):
                answer, ctx.source = early, type(st).__name__

        response = None
        if answer is None:
            ctx.source = "model"
            response = self.client.generate(build_messages(self.prompt_builder, notes=ctx.notes))
            answer = snap(response.completion, self.legal) if self.legal else extract_action(
                response.completion)
            ctx.trail.append(f"model:{answer}")
        action = answer
        for st in reversed(self.stack):
            try:
                action = st.post(ctx, action)
            except Exception as exc:
                ctx.trail.append(f"{type(st).__name__}.post!{exc!r}")
        why = " | ".join(ctx.trail)
        if response is None:
            return _Resp(action, why)
        out = copy.deepcopy(response)
        try:
            return out._replace(completion=action, reasoning=why)
        except (TypeError, ValueError):
            return out._replace(completion=action)


def build_agent(client_factory, prompt_builder, config=None):
    return PipelineHarness(client_factory, prompt_builder, config)
