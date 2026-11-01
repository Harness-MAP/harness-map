<!-- memory.md — the agent-facing PRIMER for the per-step measured-map block (evolvable lever).
     tools/vlm.py reads this file at import, keeps everything after this comment, and emits it as
     the first line of the MEASURED MAP block. Self-derived state ONLY — never the exit's
     coordinates, the layout file or the true distance. -->

How to read it: the four indented lines are the four sides of the cell you stand in, named
relative to the way you face. `WALL` means the frame shows that side solid, or one of your own
forward moves was stopped by it; `OPEN` means it can be entered; `UNTESTED` means it has never
been in view. `recent` is your own last few decisions, and says where one of them was corrected.
