<!-- memory.md — a per-step scratchpad TEMPLATE (evolvable lever).
     harness.py can fill this per step and inject it into the prompt to give the frozen agent
     self-derived working state. This candidate does NOT inject it: the same state (head cell,
     heading, the recent-cell trail and a per-cell visit count) is now maintained in code by
     tools/vlm.py, where it gates the key that is actually pressed instead of being advice the
     agent may ignore. The template is kept as the live lever for a later candidate that wants
     the frozen model to SEE the ledger and participate in the ranking rather than be overridden.
     Put self-derived state ONLY here; never oracle/privileged state (the food's cell, the body's
     full coordinate list, the level layout, the seed). -->

## working memory (self-derived only)
- last_pos:
- last_dir:
- recent_cells:
- visit_counts:
