<!-- memory.md — a per-step scratchpad TEMPLATE (evolvable lever).
     orchestrator.py can fill this per step and inject it into the prompt to give the frozen agent
     self-derived working state (last POS/DIR, recent actions, a stuck counter). The ANCHOR does
     NOT inject it — keeping the composed prompt equal to the baseline — so it starts empty.
     A proposer that wants working memory wires compose_system_prompt (or tools/vlm.py) to render
     this template each turn. Put self-derived state ONLY here; never oracle/privileged state (the
     exit's coordinates, the level's layout file, the true distance). -->

## working memory (self-derived only)
- last_pos:
- last_dir:
- recent_actions:
- stuck_counter:
