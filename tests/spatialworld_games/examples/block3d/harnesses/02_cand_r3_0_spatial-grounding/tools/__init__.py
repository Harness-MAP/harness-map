"""Executable CODE helpers this harness CALLS (not model-facing function tools). Each one reuses the
evaluator's already-built engine surface — the frame it is handed, the pristine per-step VLM fn, the
engine's own clamped cursor update — and adds no new rendering or engine path:

    frame.py    the frame array, its viewport digest, and the cursor inverted from the highlight ink
    panels.py   the three target panels, read by FITTING the lattice the strip itself draws
    layers.py   THE PLANNER: per-height minimum bipartite edge covers -> the smallest admitted shape
    serpent.py  THE SCHEDULE: a boustrophedon press tape whose every prefix is a candidate
    press.py    the press algebra: the clamp, the L1 metric, and which keys are free to hand away
    ledger.py   the dead-reckoned cursor / placed set, audited against the digest and the ink
    vlm.py      the decision loop and the ask-only-when-it-can-change-the-key arbiter
"""
