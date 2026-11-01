# Diagnosis: minihack iter 27 / cand_r027_0 (gradient walker, `tools/field.py`)

**Headline:** primary=0.325 (13/40, SE 0.074), progression 0.275, survival 0.800, step_eff 0.250,
cost 87.5k tok/ep. The anchor scored 0.100, 0.700 and 309k.
Per task: MazeWalk-9x9 5/5, MazeWalk-15x15 5/5, Corridor-R3 2/5, CB-Dark 1/5, and 0/5 on Boxoban-Hard,
Boxoban-Medium, Quest-Easy and Quest-Medium.

**Binding constraint:** the field fails in two ways. It is **wrong** on Boxoban, where it treats bars (`#`) as
open floor. It goes **flat** (no neighbour has a value) whenever the only seeds sit behind something `SOLID`:
an open door drawn `-`/`|`, or a lava `}` river. Once the field is flat, the model takes over and an
unfiltered menu is all it has. It then bumps walls until the timeout. Every one of the 27 losses traces to one
of these, except the 5 melee deaths (4 CB-Dark, 4 Quest-Medium, with overlap in cause). This is mostly a
**HARNESS-BUG** (door and doorway handling, menu filter, `[yn]` handling, flat-field fallback), plus a
**TASK-GAP** on Boxoban and Quest.

## 1. What this harness does (mechanism)
`orchestrator.py:GradientAgent.act` runs these steps in order:
1. `tools/field.py:parse_tty_frame` folds the frame into `chart` (terrain, visits, blocked (pos,dir)).
2. It runs `flood_field`, a reverse Dijkstra with a visit penalty. The seeds are every drawn `>`, or, only when
   no `>` is drawn, the unvisited open cells that border undrawn space.
3. It handles modal prompts: `y` on "Really attack", `more` on `--More--`, otherwise `esc`.
4. It emits `down` when standing on `>`.
5. It emits `reflex:swing` when a letter is adjacent.
6. Otherwise it takes `field:descend`, the neighbour with the minimum field value.
7. Only if no neighbour has a value does it `consult` the VLM. The VLM gets a menu of `open_dirs` sorted by
   visits.

`SOLID = "|- }"`. Any other drawn glyph, including `#`, counts as open.

## 2. What actually happened (causal chains)

**A. Corridor-R3 (3/3 losses): an open door is drawn `-`/`|`, so the field treats it as a wall and goes flat.**
- *R3/00:* step 38 `east` gives "The door opens." At step 39 the map shows `@-.......` with `>` two rows up
  in that room (`....>|`). The door cell is now `-`, which is `SOLID`, so the `>` seed cannot reach `@`.
  Because a `>` is drawn, `flood_field` never falls back to frontier seeds, and the field is empty
  around `@`. All 61 remaining steps are VLM consults (Reasoning column empty). They include step 45
  "You can't move diagonally into an intact doorway." and 30 "It's solid stone." bumps. The agent times out
  with `>` in view.
- *R3/01:* step 53 "The door opens." puts `>` at `|>....-@`, only 6 squares west. The next 46 steps are all
  consults, and the agent never goes through.
- *R3/03:* step 9 opens the NW door `-`. From step 10 the field is flat, because every drawn frontier cell has
  been visited and the corridor stub at (25,6)/(25,7) is a dead end. 90/100 steps are consults, with 24 "It's a
  wall." and 9 "solid stone" bumps.
- The 2 wins (R3/02 in 24 steps, R3/04 in 40 steps) never opened a door. They were almost pure `field:descend`.

**B. Quest-Easy (5/5) and Quest-Medium (5/5): the `>` is behind lava, so the field is flat from the first step.**
- In Quest-Easy all 5 episodes show 99–100/100 steps as consults. Each episode costs 415k–449k tokens, and
  Quest-Easy accounts for most of the bill.
- The room west of the `}` column is fully walled, so it has no frontier cell. Every seed is across the lava.
- The model wanders and bumps: 29–33 "It's a wall." per episode, because the menu offers directions into
  drawn walls.
- The horn `(` sits in the start room ("horn very near westnorthwest"), and nothing ever goes to it.
- *Quest-Easy/04:* at step 6, `reflex:swing` south into a peaceful goblin raises "Really attack the goblin?
  [yn] (n)". `y` and `esc` are **not in BALROG's MiniHack vocabulary**, so the modal branch falls through.
  While the prompt is up the cursor sits on the message line `(x=35,y=0)`, so `pos=(35,-1)`. The field is empty
  there, so the model is consulted, and it answers `south`. That declines the prompt and immediately walks into
  the goblin again. This repeats for **93 consecutive frames** with the same prompt, ending in a timeout.
- *Quest-Medium 00/01/02/04 (deaths at steps 21–25):* 6 `field:descend` steps take the agent out of the start
  room into the dark corridor. Then the field goes flat (the `>` is behind `}`). The model walks `east` 10×
  into the open lit hall toward the drawn `r r r r` pack. `reflex:swing` then fights 3–4 giant rats at once in
  open floor, and HP drops 16→14→12→10→9→7→4→dead (QM/00 steps 15–22).
- *QM/03:* the agent survived and timed out, with 78 consults.

**C. Boxoban (10/10 timeouts, 0 VLM calls): `#` is BARS, and the field treats it as open.**
- Every step is `field:descend`. The outer ring of `#` borders undrawn space, so every bar cell is a
  frontier seed. The agent walks toward them, gets "You cannot pass through the bars." (17–34 per episode), and
  `blocked` closes one (pos,dir) at a time.
- 20–37% of steps are no-moves.
- Boxoban has no `>`: the goal is to push boulders `` ` `` onto fountains `{`. The field objective has no
  meaning here.
- The pushes that do happen ("With great effort you move the boulder.", 1–6 per episode) are incidental. Several
  are dead pushes into corners ("in vain" 2–10 per episode).

**D. CB-Dark (4/5 deaths) is the known dice fight.**
- The agent walks east along the corridor into the room doorway and fights 2 rats in the doorway (e.g. CB-Dark/01
  at `|....#######@r`, where HP drops 14→12→6→3).
- At 3 HP it steps east (`field:descend`, step 23) into the room and dies at step 26.

**E. MazeWalk (10/10) is solved by the field alone,** with 0 no-moves and 0 VLM calls. This part of the design
works and must be preserved.

## 3. Failure breakdown (27 losses)
| cause | eps | fix class |
|---|---|---|
| Open door drawn `-`/`|` makes the field flat, and the VLM bumps until timeout | 3 (all R3 losses) | HARNESS-BUG |
| `>` behind lava makes the field flat; no crossing lever exists | 10 (all Quest; 4 of them end in rat-pack deaths, 1 in the 93-frame `[yn]` freeze) | TASK-GAP + HARNESS-BUG |
| Boxoban: bars counted as open, and there is no push planner | 10 | TASK-GAP |
| CB-Dark melee death | 4 | dice (see §6) |

Flat-field consults make up 100% of the token cost: R3 00/01/03 plus all 10 Quest episodes. The wall-bump rate
in consult mode is high because `open_dirs` never filters on terrain.

## 4. Binding constraint
The binding constraint is a **HARNESS-BUG** in `tools/field.py` passability and in the flat-field fallback. It
costs up to 3 R3 episodes (+0.075) at near-zero risk, since MazeWalk never meets doors or lava.

The larger pools are **TASK-GAP**. Boxoban needs a push planner (+0.25 ceiling). Quest needs a lava-crossing
lever. Neither is a VLM ceiling: the model is not the deciding component on any win.

## 5. Concrete changes (ranked by leverage and risk)

**1. `tools/field.py`: make opened doors passable, orthogonal only (R3 +1–3 eps).**
- In `parse_tty_frame`, handle `"The door opens."` (or no message, when `pend` moved into a `-`/`|` cell). Take
  the target cell `pend[0]+STEP[pend[1]]` and record it in `chart["doors"]`.
- In `flood_field`, `open_cell(c)` should return True for `c in chart["doors"]`.
- Forbid any diagonal step into or out of a door cell (the `dx and dy` check). This matches the
  "can't move diagonally into an intact doorway" message.
- Also let a `-`/`|` cell count as a door when it lies between two drawn `.` squares on the line
  perpendicular to its wall. That is cheap and catches doors other agents opened.
- The prior finding is that door-passable plus a one-shot refusal gave +89/−6 over 320 levels.

**2. `tools/field.py:flood_field`: fall back when the `>` is unreachable.**
- If `pos` is not in `pot` after seeding from `>`, reseed from the frontier set and rerun.
- If the frontier set is empty or unreachable too, seed from visited dead-end cells (corridor `#` with ≤1
  open neighbour) and emit `search` there, budgeted per probed cell (~10). R3 hides corridors in drawn walls.
- This removes almost every consult on R3 and part of the Quest consults. That cuts cost by roughly 3–4×, and
  cost is a scored key.

**3. `orchestrator.py:act`: fix the `[yn]` prompt and the modal-cursor pollution.**
- Detect modal as `cursor y==0` or the current message checks.
- In that state, do **not** fold `pos` into `chart` (no visits, and don't consume `pending`).
- `y` and `esc` are not legal, so answer "Really attack …? [yn]" with `more`, which escapes the prompt.
- Then add `(pos, swing_dir)` to a per-level `peaceful` ban for about 20 turns, so `reflex:swing` stops
  re-attacking.
- This removes the QE/04 93-frame freeze.

**4. `orchestrator.py:act`: filter the consult menu by terrain.**
- `open_dirs` must drop any neighbour whose drawn glyph is in `SOLID` (unless it is a known door), and any
  undrawn neighbour already bumped.
- Offer diagonals only when the level has them.
- Better: when the field is flat and not on Boxoban, skip the VLM and take `menu[0]` deterministically. No win
  in this eval came from a consult.

**5. New `tools/sokoban.py` plus a Boxoban mode (Boxoban +up to 10 eps, the largest pool).**
- Detect the mode when the map has `` ` `` and `{`, no `>`, and a `#` perimeter.
- In this mode `#` means bars, so it is impassable, and only the 4 cardinals are legal.
- Run a push-space BFS/A* over (player region, boulder set) to the fountain targets, then commit the move tape.
- Prior evidence: one push-space A* solved 19/20 rooms model-free. The harness must prune on walls only.
- Win iff the optimal plan is ≤99 steps. Keep the existing field path for every non-Boxoban task.

**6. Quest lava-crossing lever (the Quest-Easy/Quest-Medium TASK-GAP).**
- Trigger when the `>` is drawn, `flood_field` cannot reach it, and `}` cells lie on the straight boundary.
  Then route to the drawn item `(` or `/` in the reachable region ("horn very near …" in the prose), `pickup`,
  `apply`/`zap`, and give the bearing toward the lava.
- Prior evidence: that sequence freezes the lava (5/5 QM, 2/5 QE on CRN seeds). A double kick re-piles the items,
  so kick once at most.
- Until this exists, Quest should at least not spend 100 VLM calls: fix 2 and fix 4 make it model-free.

**7. Quest-Medium / CB-Dark survival (low priority, dice).**
- When ≥2 letters are drawn within 3 squares and the agent is in a corridor/doorway square with ≤2 open
  neighbours, `wait` there instead of `field:descend` into open floor. That is the "committed fight post" plus
  the `wait` hold.
- QM/00 and QM/01 died after walking into an open hall toward a visible rat pack.
- Price this on ≥200 CB-Dark episodes before trusting it.

Expected effect: fixes 1–4 give R3 about 4/5, Quest drops to near-zero tokens, and survival rises slightly.
Primary lands around 0.35–0.40, with cost down about 3×. Fix 5 is the step change (up to +0.25).

## 6. Exhausted: do NOT repeat
- **Melee re-parameterisation on CB-Dark:** HP-triggered retreat, disengage dose-response, hop-lookahead retreat,
  creature-as-refusal. All are refuted or dice; the melee axis is closed.
- **Reworking the ranking/frontier objective.** Information-gain ranking and nub demotion were refuted, and
  MazeWalk is already 10/10 with the current visit-penalised field. Do not touch its cost function.
- **Prompt-only changes to AGENT.md/rules.md.** The VLM decides no winning step here. Levers have to be in the
  code path.
