"""action_io.py — turn the model's reply into the action string the env receives.

LEVER (code). Reproduces BALROG NaiveAgent's `_extract_final_answer` exactly: keep only letters,
whitespace and colons. It is deliberately NOT robust — the stock agent is not, and making it robust
here would be an unearned improvement credited to the search.
"""
from __future__ import annotations

import re


def extract_action(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s:]", "", text or "")
