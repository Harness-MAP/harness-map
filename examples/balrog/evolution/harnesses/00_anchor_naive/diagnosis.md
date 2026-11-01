# Diagnosis — minihack iter 0 / anchor_naive (BALROG NaiveAgent, behaviourally unchanged)

**Headline:** primary=0.100 (4/40, SE 0.047), progression 0.100, survival 0.700, step_efficiency 0.097,
cost 309k tokens/episode. All 4 wins are MazeWalk episodes that end in 2–4 steps (9x9 runs 01/02/04,
15x15 run 02). No other task scores a single point.
**Binding constraint:** the policy has no state. It is a memoryless 1-step reflex over a 16-turn text
history, with no position memory, no chart and no hazard filter. It falls into deterministic
period-2 orbits (30% of all steps), walks into lava it has just been told about, and emits illegal
verbs that are silently coerced to `north`. This is **HARNESS-BUG / TASK-GAP**, not a VLM ceiling.
The observation already carries everything needed: an exact `map:` block, `cursor (x,y)`, and a
prose adjacency list.

## 1. What this harness does (mechanism)
- `orchestrator.py:NaiveHarness.act` → `prompt_builder.update_observation(obs)` →
  `tools/prompting.py:build_messages` appends `compose_priors()` (EMPTY: AGENT.md, rules.md, memory.md
  and skills/ are all empty) plus `NAIVE_INSTRUCTION` → one VLM call → `tools/action_io.py:extract_action`,
  which is only `re.sub(r"[^a-zA-Z\s:]", "", text)`.
- No reasoning is requested. The trace `Reasoning` column is empty on 2914/2914 rows.
- The model sees its last ≤16 (obs, action) turns as raw text. Nothing summarises where it has been.
- An action BALROG rejects is replaced with `north` by `check_action_validity`, and the step is spent.

## 2. What actually happened (causal chains, with evidence)
**A. Period-2 orbit (the dominant failure).** In 876/2914 steps (30%), the action equals the action two
steps earlier and differs from the previous one. Nine episodes are pure orbits with only 2–4 unique
cells: MazeWalk-9x9/00, 9x9/03, 15x15/00, Corridor-R3/01, R3/03 (`north 50 / south 50`),
R3/04 (`northeast 50 / southwest 49`), CB-Dark/02, and Boxoban-Med/02 and /04.
- MazeWalk-9x9/00 step 0: the map shows `@......` with a corridor running EAST from row 8 and a
  corridor going south below `<`. The agent steps north to (36,8), then south to (36,9), then north to
  (36,8), 50 times over. It never emits `east`, even though the whole unexplored corridor is east.
  The two alternating frames differ only in which square is "adjacent", and the model answers each
  frame freshly, so it bounces back and forth.
- MazeWalk-15x15/03: `north` ×100 against a wall, 2 unique positions. The blocked move advances no
  game turn and nothing in the prompt says the move failed.

**B. Walking into announced lava (Quest-Easy).** Runs 01, 02 and 03 die with "Burned by molten
lava" at steps 25, 4 and 5. In run 02 the step-0 prose says `lava near eastnortheast, east, and
southeast`, and the agent emits `east` ×4 into the `}` column. The model reads "the goal is east"
and ignores the hazard noun in the same message.

**C. Illegal verb → silent `north` (Quest).** In Quest-Easy/04, 95 of 100 actions were a bare
`attack`, which is not in the legal set (`failed_candidates` = 95). Every one was coerced to `north`,
giving `north` ×95 against a wall. Quest-Easy/00 has 39 such failures. Quest-Medium/00 emits
`attack: east` 9 times. The `:` survives the filter in `action_io.py`, so the string is not legal
and becomes `north`.

**D. Melee deaths (Quest-Medium 5/5, CB-Dark 4/5).** Quest-Medium/03 steps 28–31: `giant rat
adjacent northeast and east`, and the agent repeats `east` (attack by moving) with "You miss" and
"The giant rat bites!" until it dies at HP 0. There is no HP read and no retreat or positioning. In
CB-Dark, runs 00 and 02 orbit north/south in a dark corridor until a monster kills them.

**E. Boxoban.** 0/10. Each run has 3–12 unique cells and a 4-way random walk. No box is ever pushed
to a goal. A one-call-per-step reflex cannot plan a Sokoban solution.

## 3. Failure breakdown (40 episodes)
| category | episodes | tasks |
|---|---|---|
| orbit / wall-jam timeout (≤7 unique cells, 100 steps) | 18 | MazeWalk ×6, R3 ×4 (R3/02 is a `south`×82 jam), Boxoban ×10 partly |
| melee death | 9 | Quest-Medium ×5, CB-Dark ×4 |
| lava death | 3 | Quest-Easy 01/02/03 |
| illegal-verb coercion dominating the episode | 2 | Quest-Easy 00, 04 |
| Boxoban no-plan timeout | 10 | Boxoban-Hard/Medium |
| win (all ≤4 steps: the spawn is next to `>`) | 4 | MazeWalk |

The wins are not policy wins: every one ends in ≤4 steps. **On any episode that needs exploration,
the policy scores 0.**

## 4. Binding constraint — classification
**HARNESS-BUG (primary) + TASK-GAP.** The obs contains an exact ASCII `map:` block, the `cursor`
(x,y), and prose that names hazards and adjacency. The harness throws all of it at a stateless
reflex and keeps nothing between steps. The failures (orbits, wall jams, lava steps, illegal verbs)
are mechanical, and code-level state fixes them without any change to the model. Treat it as a
VLM-ceiling only for Boxoban planning. Even there, the level is fully drawn and lit, so a tool
can compute the plan.

## 5. Concrete changes to try (ranked by leverage)
1. **Chart + frontier explorer in `tools/` (new `chart.py`), called from `orchestrator.py:act`.** Parse
   the `map:` block (row index = cursor.y − 1, col = x) into a persistent grid. Record visited cells,
   and record squares where a move produced no position change. Each step, BFS to the nearest
   unvisited, passable frontier cell, or straight to `>` if it is drawn. Offer that step as a
   recommended action, or emit it directly with the VLM as a fallback.
   *Predicted:* MazeWalk 0.4→≥0.8 and R3 0→0.4+. This removes class A (30% of steps). Cause: the
   orbit exists only because nothing remembers the square just left.
2. **Hazard veto before emission (`tools/action_io.py` or a new `safety.py`).** If the prose
   adjacency line or the decoded map puts lava (`}`) or water in the target square of a move, drop
   that move and take the next-ranked one.
   *Predicted:* Quest-Easy deaths 3/5→≈0/5 and survival +0.075. It needs no model change, because
   the hazard noun is already in the observation.
3. **Legal-set projection in `extract_action`.** Map `attack: east` / `attack east` → the legal
   direction `east` (moving into a monster attacks it). Map a bare `attack` → the direction of the
   adjacent monster taken from the prose. Otherwise fall back to the explorer's move, never to
   BALROG's silent `north`.
   *Predicted:* removes the 134 coerced steps. Keep `far <dir>` out unless it is in the legal list.
4. **Only then prompt levers (AGENT.md/skills):** "`>` is the goal; `}` is lava, never step onto it;
   moving into a monster attacks it". These are cheap, but prose alone does not break orbits:
   the model already had the hazard noun and ignored it. Put them after 1–3.
5. **Boxoban solver tool (later round).** The map is premapped and lit, so a push-space A*/BFS over
   the decoded grid can compile a full action tape. Cross-lineage evidence says this works (see §6).
6. **Cost:** 309k tokens/episode is mostly replayed history (16 full 80×21 maps). Once a chart holds
   the state, cut `max_text_history`, or elide older observations to the action trace. The goal is
   roughly 3× fewer tokens with no loss of information.

## 6. Exhausted / known from other lineages (hints, re-verify here)
This is the first diagnosis in this chain, so no axis is exhausted locally. Priors from sibling
BALROG chains: a goal-following stub with no VLM reproduces this anchor's primary exactly. The
anchor emits only compass directions. An exact Boxoban push-solver reaches ~19/20. CB-Dark is
largely dice (price it on ≥200 episodes; melee retreat variants were refuted). The Quest lava can be
crossed (item pickup, then zap/apply, solidifies the lava), but it is expensive to engineer. Do the
explorer, the hazard veto and legal projection first.
