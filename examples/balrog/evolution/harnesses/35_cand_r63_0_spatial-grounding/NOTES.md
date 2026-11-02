# NOTES — cand_r063_0_spatial-grounding (parent r053 one-hop menu; steered toward r027's gradient walker)

## module map
- `agent.py` — entry (`build_agent` → `MenuHarness`). `Ledger` keeps visits, failed (square,dir) edges, and now a
  terrain chart + door memory + potential field. `MenuHarness.act`: modal/stairs reflexes → field-ranked menu →
  code emits option 1 when the field (or a `>`/creature neighbour) decides → otherwise ONE model call with the menu.
- `tools/field.py` (new) — `flood`: reverse multi-source BFS from drawn `>` (else the unvisited frontier), visit-weighted.
- `tools/prompting.py`, `tools/action_io.py` — unchanged (menu rendering; whole-word legal projection + veto).
- `AGENT.md`, `memory.md` — unchanged prose.

## axis
Code: where the decision is made (control flow + a new planning tool). Multi-hop potential field decides; the model is
consulted only where the field is flat (r027's structure), grafted onto r053's ledger/menu.

## change
Added `tools/field.py` and a per-level terrain chart in `Ledger` (`chart`, `potential`). The menu key is now
(tier: `>`/creature/other, no-field, field value + 0.25·visits, wallish, just-left, visits, index), so a `>` 5 hops
away or a frontier 12 hops away is routed to, instead of the fixed-index N/E tie-break. When option 1 is a `>`, a
creature, or has a finite field value, the step is emitted by code with no model call (`reasoning` = rung name);
reflexes answer `Really attack? [yn]` with `y`, `--More--` with `more`, and emit `down` on a charted `>`. Unlike
r027, a `-`/`|` once charted as `+` (or stood on) stays passable, so an opened door no longer cuts the route
(r027's R3 loss mode). `dark area` added to the adjacent-noun veto (solid stone 54/55 in r053 traces).

## why
r053 diagnosis: binding constraint is the ONE-HOP menu — drawn `>` ≥2 hops never routed (R3 run_02), exhausted
neighbourhood stalls (MW15 run_00/04, R3 00/01/03), and the model already follows option 1 86–99% of the time so its
call is pure cost (308k tok/ep). r027's field walker got MW 10/10 at 0 calls. Expect MW15 3→5, R3 1→3-4, big cost cut.
Quest/Boxoban unaddressed (fall back to the model menu when the field is flat).

## files
agent.py, tools/field.py (new), NOTES.md
