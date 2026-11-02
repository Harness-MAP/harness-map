"""agent.py — ENTRYPOINT. BEHAVIOURALLY IDENTICAL to BALROG's stock NaiveAgent.

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


# --- inlined from tools/action_io.py ---
def extract_action(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s:]", "", text or "")

# --- inlined from tools/prompting.py ---
ROOT = Path(__file__).resolve().parent.parent

NAIVE_INSTRUCTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)

PROSE_ORDER = ["AGENT.md", "rules.md"]

SKILL_DIR = "skills"

def compose_priors(root: Path = ROOT) -> str:
    """Concatenate the prose levers. Empty in the anchor -> returns ''."""
    parts = []
    for name in PROSE_ORDER:
        p = root / name
        if p.exists() and p.read_text().strip():
            parts.append(p.read_text().strip())
    sk = root / SKILL_DIR
    if sk.is_dir():
        for p in sorted(sk.glob("*.md")):
            if p.read_text().strip():
                parts.append(p.read_text().strip())
    return "\n\n---\n\n".join(parts)

def build_messages(prompt_builder, priors: str | None = None):
    """NaiveAgent's assembly + an (empty in the anchor) priors block."""
    messages = prompt_builder.get_prompt()
    if messages and messages[-1].role == "user":
        block = priors if priors is not None else compose_priors()
        tail = (block + "\n\n" + NAIVE_INSTRUCTION) if block else NAIVE_INSTRUCTION
        messages[-1].content += "\n\n" + tail
    return messages
