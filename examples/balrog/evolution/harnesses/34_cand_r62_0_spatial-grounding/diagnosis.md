# Diagnosis — minihack evolve r062 / cand_r062_0_spatial-grounding (potential-field walker, parent r047)

**Headline:** primary = **0.225** (9/40, se 0.066). Survival 0.875, step_eff 0.201, cost 45.4k tok/ep.
**10 of 40 episodes CRASHED** (end_status −99): all of Corridor-R3 and all of Quest-Medium. Each crash is the same `KeyError: 'unknown'` in `tools/prompting.py:48`.

| Task | Wins | Who decided | Why it lost |
|---|---|---|---|
| MazeWalk-9x9 | 5/5 | `[field]` 100%, 0 tok | — |
| MazeWalk-15x15 | 4/5 | `[field]` 100%, 0 tok | ep03 hits the 100-step cap while exploring |
| Corridor-R3 | 0/5 | `[field]`, then CRASH | the field goes flat once `>` is drawn but not charted-reachable, the model path is taken, and it crashes |
| Quest-Medium | 0/5 | `[field]` for 6 steps, then CRASH | the same flat-field → crash at step 6 (`>` drawn across lava) |
| Quest-Easy | 0/5 | `[model]` 74–90% of steps | lava seals every frontier, so the field is flat from step 0; ep04 is a 95-step swing↔modal loop |
| CorridorBattle-Dark | 0/5 (5 deaths) | `[field]` then `[swing]` | the field walks into the room mouth and fights 2–3 rats at once |
| Boxoban-Hard / Medium | 0/10 | `[field]` 100% | no push model (known TASK-GAP) |

**Binding constraint: HARNESS-BUG.** There are two composed bugs:
- The r062 change added a new ledger kind, `unknown`, but `KIND_NAME` has no entry for it.
- `Chart.flood` seeds ONLY `>` whenever one is drawn. So a drawn but unreachable `>` makes the field flat, which forces the model path, which crashes.

Fixing only the `KeyError` restores 10 episodes that can currently score only 0. Fixing the seeding means those episodes never need the model at all.

## 1. What this harness does (mechanism)
`orchestrator.py:GroundedHarness.act` → `ledger.observe` (cursor, `map:`, Dlvl, adjacent prose) → `ledger.rows` (8 neighbours, each labelled `deadly|blocked|unknown|creature|open|''`) → `chart.fold` (per-level terrain; deadly prose → `}`; prose-`open` blank → virtual cell) → `decide_in_code`. `decide_in_code` checks, in order:
1. `modal` → `more`, and books `ledger.last` as failed;
2. `down` on a charted `>`;
3. `swing` at any drawn letter on a neighbour whose kind is `creature` or `''`;
4. `field`: a reverse Dijkstra from the seeds. The seeds are **every `>` if any is drawn, else** the unvisited passable cells that border an undrawn cell. `SOLID = "|- }"`, and the visit surcharge is `0.5*min(v,6)`.

Only when every neighbour lacks a potential does the model get called: `build_messages` → `render_table` → `arbitrate`. `AGENT.md` and `memory.md` are consulted only on that path, and `rules.md` and `skills/` are empty. On 7 of 8 tasks the model is never consulted except at the moment of the crash, so this harness is effectively model-free.

## 2. What actually happened (causal chain, with evidence)

### 2a. The crash (R3 5/5, QM 5/5): the table has no key for `unknown`
`ledger.classify` now returns `'unknown'` for `dark area` (the r062 fix). `prompting.KIND_NAME` has keys only for `deadly, blocked, creature, open, ''`. Any model call with a dark-area neighbour therefore raises `KeyError` at `render_table` (eval.log:165–179, repeated for all 10 episodes). The Quest-Easy episodes survive only because their rooms are lit, so no neighbour is ever `dark area`.

### 2b. Why the model path is taken at all: the field goes flat as soon as `>` is drawn but not reachable
`Chart.flood`: `seeds = [>]` if any `>` is charted; frontier cells are used only when no `>` exists. When the `>` is visible but the chart has no passable path to it, no neighbour gets a potential, `decide_in_code` returns None, the model path runs, and the episode crashes (2a). Every R3/QM crash follows this pattern:
- **R3/00 step 63** at (12,17): `>` is drawn in the room to the NE. The only link is `vertical open door adjacent east`, drawn `-`, and `-` ∈ `SOLID`. The field goes flat and the episode crashes.
- **R3/01 step 54** at (20,3): `The door opens.`, then `vertical open door adjacent west`, drawn `-`, and the `>` room lies directly behind it. That is the same `SOLID` open door. The field goes flat and the episode crashes, **one step from the goal room**.
- **R3/02 step 12** at (15,5): the `>` room row is `|.>......... #@#`. Column 13 is undrawn (dark), so there is no charted path. The field goes flat and the episode crashes.
- **R3/04 step 23**: `>` is drawn NE, behind undrawn dark floor. The field goes flat and the episode crashes.
- **QM/00–04 step 5–6**: `>` is drawn beyond the lava river. `}` is SOLID. The field goes flat and the episodes crash (all 5 at step 6).
- **R3/03 step 9**: the field goes flat in a lit room whose only exit is a dark corridor. Since the r062 `dark area → unknown` fix, dark neighbours no longer become `virtual` cells, so the stood-on corridor tip is not a seed. That leaves 12 model steps, and the episode crashes when a `dark area` neighbour appears.

Two separate defects are therefore composed here:
- the `>`-only seeding;
- no optimistic edge into undrawn space. The r062 fix correctly stopped calling dark area *open*, but nothing replaced it with a *priced unknown* edge.

Also, `SOLID` treats every `-`/`|` as wall, even where the prose explicitly says `open door`. This is the known "an open door is drawn `-`/`|`" trap.

### 2c. CorridorBattle-Dark (0/5, all deaths at steps 22–31): the field chooses the fight square
The field walks the corridor straight to the room mouth. In CB/00 the hero is at (38,11) from step 14 onward, and the prose reads `giant rat adjacent east and southeast`, then `east, southeast, and south`. The `swing` rule fires 11 times in a row while HP drops 16 → 13 → 5 → 1 → dead. CB/04 follows the same pattern at (40,11), with 15 swings and up to 3 attackers. Nothing in `decide_in_code` looks at exposure. The swing rule wins every time a letter is adjacent, and the field had already advanced the hero into the open room. The prior lineage's CB-Dark wins (r058/r061: 1–2/5) came from holding on a ≤2-lane corridor square.

### 2d. Quest-Easy (0/5): no reachable seed from step 0, and the ep04 modal 3-cycle
- **QE/01**: the lava column `}` at x=30 walls the spawn pocket, and every frontier lies beyond it. `flood` returns no potentials at the hero, so the model decides 90/100 steps inside a 4×5 pocket (orbit-break fires 10×). There is no acquisition or crossing mechanism.
- **QE/04 steps 5–99**: `swing` southeast at a peaceful goblin, then `Really attack the goblin? [yn] (n)`, then `modal → more` (declines), then the same frame again, and `swing` again. This repeats 32 times (95 steps).
  - `modal` does book `(pos,'southeast')` as failed, but the `swing` loop in `decide_in_code` **never checks `f`** (the failed flag in the row tuple).
  - The prompt frame's cursor also reads `(35,0)`, because a modal prompt puts the cursor at y=0. As a result, `ledger.observe` counts a visit at a garbage square on every prompt frame.

### 2e. MazeWalk-15x15/03 (timeout)
All 100 steps are `[field]`. The trace is full of one-step probe-and-return nubs (`southwest→north`, `east→west` at (36,10), (38,11)). The visit surcharge (+0.5/visit) is smaller than a 1-hop detour, so the walker keeps re-entering explored corridor spines. This is a minor efficiency issue; r061's BFS route won 5/5 on the same seeds.

## 3. Failure breakdown (40 eps)
| Mode | Eps | Class |
|---|---|---|
| Crash `KeyError: 'unknown'` after the field goes flat on a drawn but unreachable `>` | 10 (R3 5, QM 5) | HARNESS-BUG |
| No push planner | 10 (Boxoban) | TASK-GAP (known) |
| No lava acquisition/crossing; model wanders the pocket | 4 (QE 00–03) | TASK-GAP (r058 `tools/quest.py` solves it) |
| Swing↔`[yn]` 3-cycle on a peaceful monster | 1 (QE/04) | HARNESS-BUG |
| Fights at the room mouth against 2–3 rats | 5 (CB-Dark) | TASK-GAP / fight-square |
| Exploration timeout | 1 (MW15/03) | minor |

Tokens are spent only on Quest-Easy (≈450k/ep on 4 episodes). Every other task costs 0 tokens.

## 4. Binding constraint
**HARNESS-BUG, fixable in code.**
- The dominant loss (10/40 eps, 25% of the eval) is a missing dict key.
- It is exposed by a seeding rule that returns a flat field exactly when the goal is in view.

Corridor-R3 scored 0.4–0.8 on r055–r061. The same route-to-frontier mechanism, minus these two bugs, should recover **+2 to +4 R3 wins**. Quest-Medium stays at 0 until a Quest mechanism is grafted.

## 5. Concrete changes to try (ranked by leverage)

1. **`tools/prompting.py:KIND_NAME` — add `"unknown": "unseen (dark: stone or floor)"`.** Also make `render_table` use `KIND_NAME.get(kind, kind)` so a new ledger kind can never crash an episode again.
   - *Effect:* 10 crashed episodes become real episodes, and survival/efficiency are counted properly.
   - *Why:* this is the exact traceback in eval.log:179.

2. **`tools/field.py:Chart.flood` — never let a drawn `>` starve the frontier.** Seed `>` at potential 0 **and** the frontier cells at potential `F` (e.g. 0 for frontier cells, with `>` at −20, or two floods). Alternatively, when the `>`-seeded flood leaves the hero with no potential, rerun it with frontier seeds.
   - *Effect:* R3/00, 01, 02, 04 keep walking toward the gap (the dark column 13 in R3/02, the dark floor in R3/04) instead of going flat.
   - *Predicted:* R3 0/5 → 2–4/5.

3. **`tools/field.py:Chart.fold` + `SOLID` — an open door is passable.**
   - When a row's prose includes `open door` or `doorway` and its glyph is `-`/`|`, write `terrain[c] = "+"`, and make `+` passable with **orthogonal edges only**. NetHack forbids diagonal moves through a door, so in `flood` skip a diagonal `d` when either end is a door.
   - `The door opens.` already appears in R3/01's message.
   - *Predicted:* R3/00 and R3/01 walk straight into the `>` room (both were ≤10 steps from `>`).

4. **`tools/field.py:passable` — a priced optimistic edge into undrawn space.**
   - Replace the dropped `dark area → virtual` with: an undrawn cell 4-adjacent to a drawn `.`/`#`/door is enterable at a surcharge (e.g. +3).
   - A bump books the edge in `ledger.failed`; that bookkeeping already exists and closes the edge in `flood`.
   - This keeps the r062 fix: no 310 stone bumps from treating dark area as *open*. The dark doorway gaps in R3/02 and R3/04 still get a finite potential.
   - Verify on MazeWalk-15 that the bump count stays low. MazeWalk is lit, so undrawn cells there are stone, and the surcharge plus the ledger should bound it.

5. **`orchestrator.py:decide_in_code` swing rule — respect `f`, and do not count prompt frames.**
   - Change the swing condition to `... and not f`. Once the `[yn]` refusal books `(pos,dir)`, the peaceful goblin is no longer swung at, and the field takes over.
   - In `ledger.observe`, skip the visit/failed update when `cursor.y == 0` (modal frame), so that `(35,0)`-type garbage is not charted.
   - *Predicted:* QE/04's 95-step loop disappears. This alone wins nothing on QE (the lava gap remains), but it stops the loop and the tokens.

6. **Graft r058's `tools/quest.py` (Quest pair only, armed on `{pickup,kick,zap} ⊆ legal`).**
   - It covers acquisition, one kick toward a wall-terminated ray, `pickup`, zap/apply across the lava, and chart update on `solidifies`.
   - r058 scores QE 5/5 and QM 5/5 with it. The `field` then routes across the cooled lava because `}` → `.` on the chart.
   - Wire it into `decide_in_code` ahead of `field`.
   - *Predicted:* +8 to +10 episodes. This is the single biggest point source left, and it transfers because it reads only the observation.

7. **CB-Dark fight-square hold (`orchestrator.py:decide_in_code`, before `field`).**
   - When a hostile is named in the prose (not adjacent) and the hero stands on a square with ≤2 open neighbours (a corridor `#`), emit `search` instead of advancing along the field.
   - When the hero is adjacent to a foe on an open room square with ≥3 open neighbours, step back one hop to the last corridor cell instead of swinging.
   - Memory says CB-Dark is noisy: price it on ≥200 episodes before trusting it. r058/r061's hold gave 1–2/5 against 0/5 here.

8. *(minor)* **`flood` visit surcharge.** Raise it to `1.0*min(v,6)`, or rank the frontier by nearest-unstood BFS as r061 did (MW15 5/5), to cut MW15 nub revisits.

## 6. Exhausted — do NOT repeat
- Treating `dark area` as *open*: this caused 310 stone bumps (r047). Keep it `unknown`, and add a **priced** edge instead (change 4).
- Answering `[yn]` with `y`: `y` is not a MiniHack action. `more` is correct, but the swing must then respect the booked refusal.
- A one-hop visit-count arbiter as the whole policy (r029/r047: 0.15–0.20). The field/route structure is right; what failed here was its seeding.
- Prompt/prose rewording for the model: the model decides <1% of steps outside Quest-Easy, so prose levers are inert.
- Boxoban via frontier walking: 0/10 across every lineage. Only an exact push-space planner moves it.
