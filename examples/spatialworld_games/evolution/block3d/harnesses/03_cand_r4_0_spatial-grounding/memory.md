<!-- memory.md — the per-step scratchpad TEMPLATE (an evolvable lever).

     This harness maintains every field below in tools/book.py::BuildState and tools/optics.py::
     Gauge, and writes them to its own trace, but it does NOT inject them into the agent's prompt.
     That is a deliberate choice, not an omission: `evaluator.current_prompt` is consumed on the
     first turn only, so a per-turn channel means rewriting conversation_history[0] every turn, and
     this harness prices the model's answer against the plan instead of steering the model toward
     the plan's answer. A proposer that wants the state in the prompt wires that rewrite in
     tools/vlm.py::decide and renders this template; nothing else has to change.

     Everything here is SELF-DERIVED — from the frame and from the actions the harness applied.
     Never put privileged state here (the target's coordinates, the level's layout file, the true
     distance): that is the one thing this file must not become. -->

## working memory (self-derived only)
- cursor:            # dead-reckoned from applied actions, corrected whenever the optics read it
- placed:            # the harness's own toggles, maintained as a set
- booked:            # structures already shown to the engine and therefore already refuted
- goal / phase:      # the tape cell this turn is for, and whether we are building or trimming
- axes_measured:     # which of X/Y/Z the projection has calibrated so far
- slack:             # the containment margin of the last cursor readout
- frozen:            # moves whose frame did not change: clamps, each one an exact coordinate
- fixes:             # times the measured cursor disagreed with the dead-reckoned one
