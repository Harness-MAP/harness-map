"""harness.py — ENTRYPOINT. SPATIALLY-GROUNDED ARBITER.

Component graph (each step):

    obs ──► tools/perception.parse ──► Percept(message, pos, grid, adjacent nouns)
                     │
                     ▼
            tools/spatial_memory.observe   (visits, refused (cell,dir), persistent seen-map)
                     │
                     ├─► tools/action_io.modal_response ──(open prompt)──► answer it, NO model call
                     ▼
            tools/grounding.ground  ──► move menu with vetoes + BFS route (`>` or frontier)
                     │
                     ▼
            tools/prompting.build_messages (BALROG history + priors + rendered grounding block)
                     │
                     ▼
            ONE client.generate  ──► tools/action_io.arbitrate (legal snap, veto override, fallback)
                     │
                     ▼
            action (+ `reasoning` = which component decided)

The decision is made by the ARBITER, not by the raw reply: the model proposes, code disposes.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages                               # noqa: E402
from action_io import legal_actions, modal_response, arbitrate     # noqa: E402
from perception import parse                                       # noqa: E402
from spatial_memory import SpatialMemory                           # noqa: E402
from grounding import ground, render                               # noqa: E402


class _Resp:
    """Minimal LLMResponse-compatible object for steps decided without a model call."""

    def __init__(self, completion, reasoning):
        self.model_id = "harness"
        self.completion = completion
        self.stop_reason = "stop"
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning = reasoning

    def _replace(self, **kw):
        r = copy.copy(self)
        for k, v in kw.items():
            setattr(r, k, v)
        return r


class GroundedHarness:
    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.memory = SpatialMemory()
        self._sys = None
        self._legal = []
        self._dlvl = None

    def reset(self):
        self.prompt_builder.reset()
        self.memory.reset()
        self._sys = None
        self._dlvl = None

    def _refresh_legal(self, messages):
        sp = next((m.content for m in messages if getattr(m, "role", "") == "system"), None)
        if sp is None:
            sp = getattr(self.prompt_builder, "system_prompt", "") or ""
        if sp != self._sys:
            self._sys = sp
            self._legal = legal_actions(sp)
            self.memory.reset()          # a new task on a reused agent: forget the old map

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        percept = parse(obs)
        dl = re.search(r"Dlvl:(\d+)", percept.raw)
        if dl and dl.group(1) != self._dlvl:
            if self._dlvl is not None:
                self.memory.reset()
            self._dlvl = dl.group(1)

        messages = self.prompt_builder.get_prompt()
        self._refresh_legal(messages)
        self.memory.observe(percept)

        forced = modal_response(percept.message, self._legal)
        if forced:
            self.memory.note_action(forced)
            return _Resp(forced, f"modal: {percept.message[:60]}")

        g = ground(percept, self.memory, self._legal)
        block = render(g, percept, self.memory)
        messages = build_messages(self.prompt_builder, grounding=block, messages=messages)
        try:
            response = self.client.generate(messages)
        except Exception as e:                                   # noqa: BLE001
            if g.recommended:
                self.memory.note_action(g.recommended)
                return _Resp(g.recommended, f"fallback-route (model error {type(e).__name__})")
            raise
        action, source = arbitrate(response.completion, g, self._legal)
        self.memory.note_action(action)
        out = copy.deepcopy(response)
        return out._replace(completion=action,
                            reasoning=f"{source}: reply={str(response.completion)[:40]!r} rec={g.recommended}")


def build_agent(client_factory, prompt_builder, config=None):
    return GroundedHarness(client_factory, prompt_builder, config)
