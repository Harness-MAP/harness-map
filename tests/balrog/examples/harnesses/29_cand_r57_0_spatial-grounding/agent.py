"""agent.py — ENTRYPOINT. A STAGE PIPELINE whose route stage owns navigation; the model owns the rest.

Component graph:
    obs --tools/ledger.Ledger.observe--> ctx {pos, mp, adj, rows, legal, text}
        --stages[*].pre (outer->inner)--> [short-circuit: modal verb | BFS route hop]  (no model call)
        --tools/prompting.build_messages(rows)--> model
        --stages[*].post (inner->outer)--> action     (legal projection, veto, orbit break)
Stages live in tools/stages.py: Modal, Melee, Route, Arbitrate. The ledger is combat-aware (a swing is
never a failed move) and grounds each neighbour by glyph AND prose noun. Every step's deciding stage is
written to `response.reasoning` so the trace shows who chose each action.
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
from action_io import legal_actions                   # noqa: E402
from ledger import Ledger                              # noqa: E402
from stages import default_stack                       # noqa: E402

_Resp = namedtuple("_Resp", ["completion", "reasoning", "input_tokens", "output_tokens"],
                   defaults=("", "", 0, 0))


class PipelineHarness:
    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.stack = default_stack()
        self.ledger = Ledger()
        self._system = None
        self.legal = []

    def reset(self):
        self.prompt_builder.reset()
        self.ledger.reset()

    def _sync_legal(self):
        sp = getattr(self.prompt_builder, "system_prompt", "") or ""
        if sp != self._system:                  # the agent object is reused across tasks
            self._system = sp
            self.legal = legal_actions(sp)
            self.ledger.reset()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        self._sync_legal()
        try:
            text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        except (KeyError, TypeError):
            text = str(obs)
        pos, mp, adj = self.ledger.observe(text)
        rows = self.ledger.rows(pos, mp, adj)
        ctx = {"legal": self.legal, "text": text or "", "pos": pos, "mp": mp, "adj": adj,
               "rows": rows, "ledger": self.ledger, "trail": []}

        for st in self.stack:
            try:
                early = st.pre(ctx)
            except Exception as e:              # a stage bug must never cost the episode
                ctx["trail"].append(f"{type(st).__name__}!{type(e).__name__}")
                early = None
            if early is not None and (not self.legal or early in self.legal):
                ctx["trail"].append(f"pre:{type(st).__name__}->{early}")
                self.ledger.commit(pos, early)
                return _Resp(early, " | ".join(ctx["trail"]))

        response = self.client.generate(build_messages(self.prompt_builder, rows))
        action = response.completion
        for st in reversed(self.stack):
            action = st.post(ctx, action)
        self.ledger.commit(pos, action)
        out = copy.deepcopy(response)
        try:
            return out._replace(completion=action, reasoning=" | ".join(ctx["trail"]))
        except (TypeError, ValueError, AttributeError):
            return out._replace(completion=action)


def build_agent(client_factory, prompt_builder, config=None):
    return PipelineHarness(client_factory, prompt_builder, config)
