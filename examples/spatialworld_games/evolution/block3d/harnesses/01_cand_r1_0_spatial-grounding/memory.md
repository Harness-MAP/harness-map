<!-- memory.md — the per-step working-state schema (evolvable lever).
     This is LIVE: tools/reckon.py:Ledger maintains exactly these fields and
     tools/vlm.py:Arbiter.brief() renders them into the system turn every step after the first
     (it rebuilds message 0 as composed-text + brief, so the engine's own response contract is
     never truncated away). Everything below is SELF-DERIVED — the cursor is read off the
     highlight ink and cross-checked against a replay of the actions this harness itself
     returned, the structure is read off the three rendered panels — so no engine state is
     consulted anywhere. A proposer adding a field should add it to Ledger and to brief()
     together, and keep it self-derived. -->

## working memory (self-derived only)
- cursor:          (x,y,z), from the highlight hexagon, else from the booked replay
- placed:          every cell this harness has toggled on and not toggled back off
- admitted:        the cell set all three decoded panels jointly allow
- owed:            the panel squares the placed cells do not yet explain
- forced:          admitted cells that are the only possible explanation of some lit square
- goal:            the cell this decision is walking toward, re-derived from owed each turn
- dead steps:      the moves that cannot change the cursor from here
- decisions left:  of the wallet an episode allows
- nulls / cancels: clamped steps and self-cancelling toggles that still got through
- corrections:     turns where the picture's cursor disagreed with the booked one
