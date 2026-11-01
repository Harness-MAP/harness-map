"""harness.py — ENTRYPOINT. BEHAVIOURALLY IDENTICAL to BALROG's stock NaiveAgent.

WHY THIS IS THE ANCHOR. The anchor is the starting point every arm inherits, so anything hand-built
into it is credited to the search that did not earn it. A hand-engineered anchor measured 7.5% on
the official 8 against BALROG's published naive agent at 2.5% — those 5 points were the author's,
not the method's, and worse, they came from priors written against measured failures of the
evaluation set itself. So the anchor is the PUBLISHED BASELINE, and every point above 2.5% is
attributable to the search.

STRUCTURALLY DECOMPOSED, BEHAVIOURALLY NAIVE. A faithful naive agent is one class, and a
single-component anchor collapses phi (coverage-init raises CoverageError because variants cannot
separate). So naive's logic is split into real modules — prompt assembly, action extraction — that
do EXACTLY what NaiveAgent does and nothing more. The structure gives the proposer separable
surfaces to edit and phi something to measure; the behaviour adds zero capability.

EQUIVALENCE CONTRACT (balrog/agents/naive.py):
  · append the same naive_instruction to the last user message
  · no extra priors, no memory, no observation parsing
  · extract with the same filter: re.sub(r"[^a-zA-Z\\s:]", "", completion)
Verified by running both on the official 8 and comparing.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages          # noqa: E402
from action_io import extract_action          # noqa: E402


class NaiveHarness:
    """BALROG NaiveAgent, in the harness_map component skeleton."""

    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})

    def reset(self):
        self.prompt_builder.reset()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = build_messages(self.prompt_builder)
        response = self.client.generate(messages)
        out = copy.deepcopy(response)
        return out._replace(completion=extract_action(out.completion))


def build_agent(client_factory, prompt_builder, config=None):
    return NaiveHarness(client_factory, prompt_builder, config)
