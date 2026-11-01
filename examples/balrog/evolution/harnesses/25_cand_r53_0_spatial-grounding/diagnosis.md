# Diagnosis — minihack iter 53 / cand_r053_0_spatial-grounding (code-ranked move menu)

**Headline:** success **0.225 ± 0.066** (9/40), progression 0.200, survival 0.775, step_efficiency 0.190,
~308k tokens/episode. Per task: MazeWalk-9x9 5/5, MazeWalk-15x15 3/5, Corridor-R3 1/5, all others 0/5.
This beats its region parent r029 (0.150) but sits **below** the chain's naive-transcript cluster
(r1/r36/r41/r44: 0.35–0.375, which reach MW15 5/5 and R3 4/5). It is far below the chart-walker line
(r052 0.600 → r068 0.900).

**Binding constraint:** the menu succeeded in taking the decision away from the model. It is now the
policy: menu option 1 is emitted on **68–99%** of steps. But the ranking it enforces is a **one-hop,
least-visited, north-first local rule**. It has no route to a drawn `>`, no frontier target, and no
non-movement verbs. So the harness walks away from a `>` it has seen, sweeps small rooms for 100 steps,
and can never do the Quest item chain or a Boxoban plan.
**Classification: TASK-GAP.** Legal moves are executed faithfully, but no goal-directed planning lever
exists. There are also **HARNESS-BUG** details inside the ranking (the tie-break, `<` and unseen squares
offered as fresh, the reasoning column left empty).

> **Trace alignment note (replay method):** each CSV row is (action, observation AFTER it). The decision
> for row i was made on row i−1's observation. With that alignment, replaying `Ledger` offline reproduces
> the menus. Without it, "model follows option 1" reads as 31–53% instead of 68–99%. Re-use this alignment.

## 1. What this harness does (mechanism)
`orchestrator.py:MenuHarness.act`, once per step:
1. `Ledger.observe` parses the cursor `(x, y-1)`, `Dlvl`, and the 21 `map:` rows. From prose lines containing
   `adjacent` it builds `deadly` (lava/water) and `blocked` (wall/bars, not door). It increments `visits[pos]`.
   It books `(pos, dir)` as failed on a no-move, unless the target glyph was a letter or the message has combat text.
2. `Ledger.menu` keeps every legal compass move that is not deadly, blocked, failed, or `}`. It sorts them by
   `(tier[> =0, letter =1, else 2], glyph in "|- ", square just left, visits, fixed STEP index)`.
3. `tools/prompting.py:render_menu` appends the numbered menu after AGENT.md + memory.md. The model is called once.
   `extract_action` takes the first non-vetoed legal mention, falling back to option 1.
4. Orbit override: if the reply is the square just left while option 1 is never-stood and not a creature,
   option 1 is emitted instead.
5. The chosen action replaces `completion`. Nothing is written to `reasoning`, so the CSV `Reasoning`
   column is **empty on every row**.

## 2. What actually happened (causal chains, with evidence)

**A. The model is now inert, and the menu is the policy.** Replayed agreement with option 1 by task:
Boxoban-H 0.87, Boxoban-M 0.91, R3 0.68, CB-Dark 0.99, MW15 0.87, MW9 0.93, QE 0.77, QM 0.94.
Deviations rarely help. In R3 they go to already-visited squares 15% of the time and unseen squares 8%.
In Quest-Easy they go to visited squares 21% of the time. The r029 problem ("model ignores the table") is
solved, so the score now measures the **ranking rule** itself.

**B. A drawn `>` is ignored unless it is adjacent. This is the clearest loss.** *Corridor-R3/02:*
the first frame already draws `|.>.........` at x=3 on row 4, with "stairs down very far west" in the prose.
The hero starts at (25,4). `>` gets tier 0 only as a direct neighbour, so the menu ranks the fresh
neighbours by the fixed STEP index: north first. Steps 0–28 sweep the 8×4 start room square by square
(e.g. step 4 at (22,4): option 1 = `north`, fresh). At step 30 the hero leaves by the SW corridor and wanders the
`#` network (steps 32–40 at x=14–15). The episode ends at 100 steps, 0 progress, with a `>` it saw at t=0.

**C. The one-hop least-visited rule traps the agent in small rooms.** *Corridor-R3/03:* only 24 unique squares
in 100 steps, max 19 visits on one square. The hero is in a 4×5 room with an east doorway `.#` at (24,5).
At steps 62–66 it steps onto the corridor `#`. The next corridor squares are unseen (`' '`), which the rank
puts in the wall class, so they sort last. The room squares, at 3–4 visits, then win option 1, and the agent
walks back in. From step 70 to step 99 every option-1 row is an interior square with 3–5 visits. The rule
never has a *target*, so the visit counts level off and it drifts. It is a random walk with a memory.
*MazeWalk-15x15/00 and /04:* 49 and 57 unique squares in 100 steps, `>` never drawn, repeated "It's solid stone." bumps.
The same one-hop sweep spends its budget on dead ends. The naive cluster reaches MW15 5/5.

**D. Quest is unreachable by construction, because the menu offers compass moves only.** *Quest-Easy/00:* the start room
(x=25–29) is closed off by a lava column `}` at x=30 (map at row 60). The horn `(` and potion `!` lie in the
room, and "stood 7" on the `(` square is the spawn pile. The menu drops `}` squares, so it has nothing to offer but
the 26 room squares. All 5 QE episodes have exactly **26–27 unique squares over 100 steps**. `pickup`/`apply`/`zap`
are never emitted: 0 non-movement actions in all 40 episodes. Winning needs pickup → apply/zap across the lava. The
r052 line does this and gets QE 0.8–1.0.

**E. Melee: the swing fix works, but "creature first" walks the hero into open-room swarms.** The r029 bug is gone.
*CB-Dark/00* swings `east` 11 times in a row at (38,10) (steps 14–24, "You kill the giant rat!" ×3).
But tier 1 = adjacent creature, and a fresh floor square beats the corridor. So the hero leaves the 1-lane corridor
(x=26–33) and engages from (38,10), open floor with rats at E/SE/S. It dies at step 26. *Quest-Medium/01* is the
same pattern: it enters the rat room through the doorway (32,12), stands at (35,11) with 3 adjacent `r`, and dies at step 28.
Deaths: CB-Dark 5/5, QM 4/5. That is 9 of the 9 deaths, and all are multi-rat open-floor fights.

**F. Boxoban:** 4 cardinals, and the menu ranks boulder pushes like floor steps. There are 13–38 unique squares per episode
and "You try to move the boulder, but in vain" is a top message. The one-hop rule has no push model, and it
jams boulders into corners early. This is out of reach for any menu rule.

## 3. Failure breakdown (40 episodes)
| category | episodes | evidence |
|---|---|---|
| wins (MazeWalk short-spawn + R3/04 + MW15/03) | 9 | 6 of them finish in ≤6 steps, with `>` near spawn |
| timeout: one-hop sweep, no goal/frontier target (R3 ×4, MW15 ×2) | 6 | R3/02 saw `>` at t=0; R3/03 room trap; MW15 49–57 uniq |
| timeout: Quest-Easy lava wall, no item verbs | 5 | 26–27 unique squares each, 0 non-move actions |
| death: open-floor multi-rat fight (CB-Dark ×5, QM ×4) | 9 | death at steps 22–31, 2–3 `r` adjacent |
| timeout: Boxoban, no push planning | 10 | boulder "in vain" loops |
| timeout: Quest-Medium/00 | 1 | same as QE after the rats |
Note on the scalar: 6 of the 9 wins are MazeWalk spawns where `>` is within about 6 squares. On every other
task the harness's ranking contributes nothing positive.

## 4. Binding constraint
**TASK-GAP**, with embedded HARNESS-BUGs. The model has been removed as a factor: it copies option 1. What
remains is a greedy one-hop rule that (i) cannot aim at a drawn `>` or an unexplored frontier more than
one step away, (ii) cannot emit item or verb actions, and (iii) prefers open floor next to monsters.
Re-wording the prompt or menu cannot fix any of these. The fix is **code that plans over the decoded map**:
BFS to `>`, BFS to the nearest frontier, and a Quest item module. The chain has already proven this lever:
r052's potential-field / chart walker, then the r064/r068 Boxoban solver.

## 5. Concrete changes to try (ranked by leverage)
1. **Replace the one-hop rank with a charted BFS target** (new `tools/chart.py`; `Ledger.menu` → option 1 =
   first hop of a path). Keep a persistent per-level map memory, merging each `map:` frame so squares seen once stay known.
   Treat `. # > < + ( ! % ` and letters as passable, and `| - ' ' }` as not passable (a `|`/`-` named an open door in the prose is passable).
   Order the targets: (a) a **drawn `>`** reachable over known passable squares → shortest path; (b) otherwise the
   **nearest frontier**, i.e. a known passable square with an unseen `' '` neighbour, using BFS distance and a
   *total-order* tie-break (distance, then y, x in level frame). Do not use the fixed STEP index; that is a north bias.
   Re-elect the target every step (a fixed point, no stale routes).
   *Predicted:* R3/02 wins, because `>` is drawn at t=0 and ~22 steps away. R3/03 leaves the room because the corridor
   frontier becomes the target. MW15 goes toward 5/5. So R3 0.2 → ~0.6–0.8 and MW15 0.6 → ~1.0: **≈ +0.10 success**.
   *Cause:* B and C are both "no target beyond one hop".
2. **Add the Quest module, armed only when `{pickup, kick, zap} ⊆ legal`** (port r052 `tools/quest.py`): pickup at
   spawn; one kick of the pile from a wall-backed post; horn before wand; fire `apply`/`zap` across the adjacent lava run;
   answer `In what direction?` and item prompts. The other 30 episodes are untouched.
   *Predicted:* QE 0 → 0.8–1.0 and QM 0 → 0.8–1.0, as r052/r058 show. **≈ +0.20.** *Cause:* D. The menu cannot emit a verb.
3. **Fight from the corridor mouth, not open floor** (`Ledger.menu` tier rule + a `wait` stage): when a hostile is named
   in the prose and the hero stands on a square with ≤2 passable neighbours, emit `wait` rather than advance.
   Demote creature-tier to "swing only if already adjacent". Never step *toward* a creature from a 1-lane square.
   *Predicted:* QM deaths 4/5 fall, so survival rises, and Quest wins become possible once item 2 lands. For CB-Dark, the
   chain history says it is dice at n=5. Expect ≤ +1 episode. *Cause:* E.
4. **Boxoban: a code push-space planner** armed on "4 cardinals legal + boulder glyphs" (port the r064/r068 solver).
   Boxoban is premapped and lit, and memory notes an exact A* over push-space reaching 19/20. *Predicted:* Boxoban 0 → ~1.0, **≈ +0.25**.
5. **Hygiene:** (a) write the chosen stage/menu row to `response.reasoning` (currently empty on all 2,573 rows, which hides
   which rule fired). (b) Do not offer `<` as a fresh square; it is a dead end on every task. (c) Trim the replayed transcript.
   At 308k tokens/episode, ~90% is history, and the model now contributes nothing a 1-frame prompt could not.

## 6. Exhausted — do NOT repeat
- Advisory neighbour/visit **tables** for the model to read (r029: the model ignores them, 65% visited-over-fresh).
- **Menu order via prose** ("take option 1"): it already works (68–99% agreement), so more prose buys nothing.
- **One-hop re-rankings** (visit counts, just-left demotion, wall class, fixed-index tie-break): this candidate is that
  family at its best, and it sits below the plain naive cluster.
- Booking a melee swing as a failed move: fixed here. Keep the fight-aware `failed` rule.
- Retreat / HP-gated disengagement on CB-Dark: refuted repeatedly in chain memory (dose-response: more disengagement,
  more deaths). Use only the *hold at a corridor mouth* form in item 3.
