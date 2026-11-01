"""orchestrator.py — ENTRYPOINT. Naive transcript, grounded by a prose-noun NEIGHBOUR LEDGER and arbitrated in code.

The model still answers once per step. Code grounds what reaches it and arbitrates what leaves it:
  perceive  tools/ledger.py     cursor, map, Dlvl, "<noun> adjacent <dirs>" prose -> 8 neighbour rows
  modal     harness.modal_reflex an open `[yn]` / `--More--` prompt is answered in code (legal verbs only)
  decide    one model call on the naive transcript + tools/prompting.render_table
  arbitrate tools/action_io.arbitrate  legal projection, deadly/blocked/failed veto, orbit break
The deciding rule is written to `response.reasoning` so the trace shows who chose each action.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages                  # noqa: E402
from action_io import arbitrate, legal_actions        # noqa: E402
from ledger import Ledger                              # noqa: E402


def modal_reflex(text: str, legal: list):
    """An open modal prompt freezes the frame; answer it with a legal verb, never a move."""
    if "[yn" in text and "y" in legal:
        return "y"
    if "--More--" in text and "more" in legal:
        return "more"
    return None


class GroundedHarness:
    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.ledger = Ledger()

    def reset(self):
        self.prompt_builder.reset()
        self.ledger.reset()

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
        legal = legal_actions(getattr(self.prompt_builder, "system_prompt", "") or "")
        messages = build_messages(self.prompt_builder, rows)
        response = self.client.generate(messages)
        out = copy.deepcopy(response)
        reflex = modal_reflex(text, legal)
        if reflex:
            action, rule = reflex, "modal"
        else:
            action, rule = arbitrate(out.completion, legal, rows)
        self.ledger.commit(pos, action)
        out = out._replace(completion=action)
        if "reasoning" in getattr(out, "_fields", ()):
            out = out._replace(reasoning=f"[{rule}] {getattr(out, 'reasoning', '') or ''}".strip())
        return out


def build_agent(client_factory, prompt_builder, config=None):
    return GroundedHarness(client_factory, prompt_builder, config)
