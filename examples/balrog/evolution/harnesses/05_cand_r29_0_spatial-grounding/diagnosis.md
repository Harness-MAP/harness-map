# Diagnosis — minihack iter 29 / cand_r029_0_spatial-grounding (neighbour ledger)

**Headline:** primary success = **0.150 ± 0.056** (6/40), survival 0.750, step_efficiency 0.134,
~340k tokens/episode. The anchor scored 0.100 / 0.700 / 0.097 / 309k. All 6 wins are MazeWalk
(9x9 4/5, 15x15 2/5), and 4 of them finish in ≤4 steps, where the `>` is next to spawn. The other
6 tasks score 0/5. The +2 wins over the anchor are within one SE.
**Binding constraint:** the ledger reaches the model as a **table the model does not read**. When an
unvisited passable neighbour exists, the model picks an already-visited one **65% of the time**
(1133/1732). Period-2 orbits therefore survive intact. The only part of the ledger that changes
behaviour is the code veto, and that veto has two bugs. (a) It fires only *after* a bump, although
**every one of the 356 no-move steps targeted a square the map already showed as impassable**.
(b) It records a melee swing as a "failed move", so the agent loses the ability to attack whatever
is next to it. Classification: **VLM-CEILING** on reading the table, plus a **HARNESS-BUG** in the
veto. The fix is to move the visit-count decision into code, not to re-word the table.

## 1. What this harness does (mechanism)
- `harness.py:LedgerHarness.act` does the following each step:
  1. `Ledger.observe(text)` parses the cursor `(x,y)` (stored as `y-1`), the `map:` block (rows 1–21), `Dlvl`, and
     prose lines that contain `lava|water … adjacent <dirs>`.
  2. It increments `visits[pos]`. If the previous committed `(pos, dir)` left the cursor unmoved, it adds that pair to `failed`.
  3. `Ledger.rows` builds 8 rows `(dir, glyph, visits, failed, hazard)`. `vetoed` = failed ∪ hazard.
  4. `fallback` = the least-visited non-vetoed direction, with walls (`|`, `-`, `' '`) sorted last.
  5. `tools/prompting.py:build_messages` appends AGENT.md + memory.md + `render_ledger` table + NAIVE_INSTRUCTION
     ("output one action and no other text").
  6. One model call follows. `tools/action_io.py:extract_action` takes the earliest whole-word legal mention that is not vetoed,
     falls back to `fallback`, and returns the first mention if every direction is vetoed.
- The model emits about 2 output tokens per step (e.g. MazeWalk-9x9/00: 234 output tokens over 100 steps). There is no
  reasoning, so the table's only route to the decision is the model's one-token pick.
- Replaying `Ledger` offline over the CSVs reproduces the harness exactly. A vetoed action was emitted only
  on the 3 steps where all 8 directions were vetoed. The numbers below come from that replay. Note that the CSV row is
  (action, obs *after* it), so the decision at step i uses row i−1's observation.

## 2. What actually happened (causal chains, with evidence)

**A. The visit column is ignored → period-2 orbits persist (MazeWalk, R3, Boxoban, CB-Dark).**
- *Corridor-R3/03:* `south`/`north` between (22,3) and (22,4) for 97/100 steps, with 3 unique squares. The table shows both
  squares' visit counts climbing into the dozens while other floor neighbours stay at 0. The veto set is empty on
  every step, because both moves succeed. The ledger has no way to intervene.
- *MazeWalk-15x15/00* (75 A-B-A steps, 6 squares): the agent orbits the spawn `<` at (44,6)↔(43,5)/(44,5)/(44,7)
  (steps 20–34). The same pattern appears in *Boxoban-Medium/02* (93 A-B-A steps, 3 squares) and *R3/00* (60).
- Census over all 2946 compass moves: the model chose the **minimum-visit** passable neighbour on only
  27% of moves. It chose a strictly more-visited one on 1751 moves.
- **The `<` attractor:** 103/307 MazeWalk-15x15 moves, 41/146 MazeWalk-9x9 moves, 99/495 R3 moves and 65/248 CB-Dark moves
  step *onto the up-staircase* (the spawn square). The table labels it "stairs up", and the model keeps returning to it.

**B. Every no-move was predictable from the map; the veto only learns after the bump.**
No-move steps grouped by the map glyph of the target square at decision time:
| task | no-moves | wall `|-` | unseen `' '` | `#` | boulder | creature |
|---|---|---|---|---|---|---|
| Boxoban-H / M | 56 / 60 | – | – | 46 / 42 (`#` = **iron bars**, prose `bars adjacent`) | 10 / 18 | – |
| Corridor-R3 | 56 | 13 (13/13 targets) | 43 (43/43 targets) | 0 of 88 | – | – |
| MazeWalk 15 / 9 | 27 / 21 | – | 27/27, 21/21 (blank = solid stone) | – | – | – |
| Quest-Easy | 44 | 34 (34/34) | 3 of 25 | – | – | 5 |
| Quest-Medium | 58 | 39 (39/39) | – | – | – | 19 |
| CB-Dark | 34 | 19 (19/19) | – | – | – | 15 |
Each wall target was a no-move 100% of the time. On MazeWalk/R3, each unseen target was a no-move 100% of the time.
`failed` is keyed on (square, direction), so each wall face has to be bumped once per square before it is
vetoed. Example: *MW15/00* step 27 bumps northeast ("It's solid stone."), then step 32 bumps east from the same square.
The no-move rate fell from 28% (anchor) to 12%, but the remaining 12% is avoidable.

**C. A melee swing is booked as a "failed move" → the agent disarms itself (Quest-Medium, CB-Dark).**
*Quest-Medium/03* steps 41–57 at (21,10)/(21,11):
- Step 41: `east` swings at a rat ("You miss the giant rat."). The cursor does not move, so `failed` gets `((21,11),'east')`.
- Step 42: east is now vetoed with the rat still adjacent east. The agent retreats `north` and takes a bite.
- Steps 49–52: `east`, `south`, `southeast` are swings (two of them are "You kill the giant rat!") and each one is booked as failed.
- Steps 53–56: the veto set grows to 6, then 7, then **8/8**. The fallback walks into `north`/`northeast`/`west` walls that the map draws as
  `-`/`|` ("It's a wall." ×4) while two rats sit adjacent east and south.
- The agent dies at step 57.
The replay counts 4–14 decisions per Quest-Medium episode and 6–9 per CB-Dark episode where an adjacent creature's
direction was vetoed. Kills do not move you either, so a successful fight is also booked as failed.

**D. The DEADLY veto never fired.** `hazard` was non-empty on **0/2946** decisions. On Quest-Easy the lava column stays
"near"/"far" and is never adjacent (e.g. QE/00 step 5, lava 3 columns east at x=30). The Quest-Easy survival gain
(0 deaths vs anchor 2–3) is behavioural drift, not the veto. The agent also never emits a non-movement verb
(0 non-compass actions in 40 episodes), so it never picks up or applies the horn it stands next to (QE/00 step 5:
`horn adjacent west`).

**E. The legal projection works.** There were 0 illegal emissions. Anchor Quest-Easy/04 had 95 coerced `attack`s; that failure is gone.

## 3. Failure breakdown (34 failed episodes)
| mode | episodes | evidence |
|---|---|---|
| period-2 / low-coverage orbit (≤12 unique squares or ≥40 A-B-A) | ~13 | R3 00/03, MW15 00/03, MW9 00, BoxM 02/03, BoxH 00–03, CBD 04 |
| wandering without reaching the goal (visits ignored, more coverage) | ~6 | R3 01/02/04 (28–51 squares), MW15 04, BoxH 04, BoxM 00/01/04 |
| death by monster (swing veto + orbiting in contact) | 10 | Quest-Medium 5/5, CB-Dark 5/5 |
| Quest-Easy timeout (never approaches lava / never uses the horn) | 5 | QE 00–04 |
| Boxoban (needs push planning; no mechanism for it) | 10 | 0/10, 10–18% of steps push into bars or stuck boulders |

## 4. Binding constraint — classification
- **VLM-CEILING (primary):** Qwen3-VL-30B-A3B-Instruct, answering in 2 tokens, does not use a visits column. 65% of
  unvisited-available decisions skip the unvisited square. Re-wording the table or AGENT.md will not change this,
  because the one-token pick never consults it. The decision about *which neighbour* has to be scaffolded in code.
- **HARNESS-BUG (secondary, directly fixable):** (i) `Ledger.observe` books attacks and kills as failed moves.
  (ii) Impassable glyphs are not vetoed ex ante. (iii) The fallback falls onto walls when the only non-vetoed options are
  walls.
- **TASK-GAP:** there is no `>` seeking and no non-movement verbs (Quest horn, Boxoban planning).

## 5. Concrete changes to try (ranked by leverage)

1. **Make visit counts binding in code (orbit breaker).** `harness.py:LedgerHarness.act` + `tools/action_io.py:extract_action`.
   - Compute `passable = {d: visits for rows if glyph not impassable and d not vetoed}` and `m = min(passable.values())`.
   - Accept the model's first mention only if it is in `{d : visits[d] == m}`, or if it targets an adjacent creature.
   - Otherwise return the min-visit passable direction, with a *fixed* tie order. Per the memory notes, a total order avoids 2-cycles.
     Break ties toward the direction that continues the previous heading.
   - *Predicted:* removes the A-B-A orbits on R3/03 (97 steps), MW15/00 (75) and BoxM/02 (93), and turns MazeWalk into a DFS.
     MazeWalk is where the wins are, so expect MW15 2/5→3–4/5 and MW9 4/5→5/5. The 65% "skip an unvisited neighbour" rate
     should drop to ~0.
   - Keep one model call per step so the model still arbitrates among equal-visit options, which preserves the naive decision locus.
2. **Stop booking melee as a failed move.** `harness.py:Ledger.observe`:
   - Skip `self.failed.add(...)` when the last target glyph was a letter (creature) or when the new message matches
     `You (miss|hit|kill)|bites|misses|The .* (is killed|dies)`.
   - Clear `failed` entries for `pos` when a creature newly appears there.
   - *Predicted:* removes the 8/8-vetoed lockup in QM/03 steps 49–56 and the retreat-on-veto at QM/03 step 42.
   - The effect on survival is modest. The memory notes say CB-Dark is dice at n=5, so price any CB change on ≥200 episodes.
     Still, it removes wasted wall bumps under fire and the forced retreats that give free bites.
3. **Veto impassable glyphs ex ante.** `harness.py:LedgerHarness.act`, `vetoed |= {d : glyph in '|-'}`. Also:
   - `' '` (unseen): veto it once the level has produced one `It's solid stone.`, or when the neighbour is unseen and no prose noun
     names it. On MazeWalk/R3 this was a no-move 91/91 times, on Quest-Easy 3/25.
   - `#`: veto only when a prose line says `bars adjacent <d>`. `#` is a corridor on R3, where it was 0/88 no-moves, and bars on Boxoban.
   - Make `render_ledger` print "wall — blocked" instead of letting the model choose it.
   - *Predicted:* no-moves drop from 356 to about 30 (boulders plus the few Quest unseen cases). That frees ~10% of the step budget on every task.
4. **Seek a drawn `>`.** When the `map:` block contains `>`, run a BFS over passable glyphs from the cursor and emit the first hop,
   using the observation only.
   - *Predicted:* converts MazeWalk/R3 episodes where the `>` is charted but the orbit never reaches it. The memory notes
     report that goal-attracted seeking adds wins on R3.
5. **Label the spawn `<` as "start square (up stairs — not the exit)"** in `GLYPH_NAME`, and count it in `visits` from step 0.
   *Predicted:* it cuts the 25–33% of MazeWalk moves that step onto `<`. This is cheap, but it matters much less once #1 makes visits binding.
6. **Trim cost.** The ledger table plus priors add about 10% tokens per episode (340k vs 309k). If #1 moves the decision into code,
   print only the non-vetoed rows.

## 6. Exhausted — do NOT repeat
- **Prose/table-only grounding of visit counts:** this candidate shows the model ignores it (27% min-visit picks).
  Do not add more columns, rewording or AGENT.md exhortations ("prefer directions you have stood on least").
- **The DEADLY lava veto as a survival lever:** it fired on 0/2946 decisions. The Quest-Easy deaths it targeted did not recur,
  and they did not recur because of this veto.
- **Keying the failed-move veto on raw no-displacement:** it conflates combat with walls (see §2C). Classify the no-move message first.
