# Diagnosis — MiniHack (BALROG official 8, evolve split) iter 1 / cand_r001_0_spatial-grounding

**Headline:** primary success **0.350** ±0.075 (anchor 0.100), progression 0.325, survival 0.850 (anchor 0.70),
step_efficiency 0.258 (anchor 0.097), cost 301k tok/ep (anchor 309k). Per task: MazeWalk-9x9 5/5,
MazeWalk-15x15 5/5, Corridor-R3 4/5; **Boxoban-Hard, Boxoban-Medium, CB-Dark, Quest-Easy and Quest-Medium are all 0/5.**

**Binding constraint:** the VLM is inert, so the harness's BFS explorer effectively IS the policy. The explorer has
three code bugs: (a) its frontier set keeps squares it has already stood on, (b) `#` and boulders count as
passable, and (c) the modal handler cannot answer `[yn]`. Together these cause period-2 orbits on
**41–60 % of steps** on Boxoban, Quest-Medium and the failed R3 episode. The explorer also cannot use any
non-movement verb (`pickup`, `apply`, `kick`, `search`), and it has no push planner. **Mostly HARNESS-BUG,
plus a TASK-GAP on Quest and Boxoban.**

---

## 1. What this harness does (mechanism)

`harness.py:GroundedHarness.act` runs this loop:

1. `perception.parse` extracts the message, the cursor, a map grid aligned on `@`, and per-direction
   `X adjacent <dirs>` nouns.
2. `spatial_memory.observe` updates visits, the seen-map, and the refused `(cell,dir)` ledger. A move is
   convicted when the position did not change and the message was empty or a REFUSAL_MARKER.
3. `action_io.modal_response` answers `--More--`, `[yn]` and "what do you want to…" prompts with
   `esc`/`more`, without a model call.
4. `grounding.ground` builds the move menu with vetoes (lethal nouns and `}`, wall nouns, refused
   moves, bare `-|`). `plan_route` then runs a BFS to a drawn `>`, or else to the "nearest unexplored edge",
   meaning a passable seen cell with ANY undrawn 8-neighbour, keyed `(visits>0, dist, bfs_index)`. If that
   first hop is vetoed, the fallback is the least-visited allowed neighbour.
5. `render` produces a SPATIAL GROUNDING block ending in `recommended: <dir>`. One `client.generate`
   call follows, then `arbitrate` (legal snap, veto override, route fallback).

`skills/spatial.md` tells the model to "follow the `recommended` move". AGENT.md, rules.md, memory.md and
system_prompt.txt are empty.

## 2. What actually happened (causal chain, with evidence)

**The model copies the recommendation.** Parsed from the `Reasoning` column (`model: reply=… rec=…`):
**2468 / 2487 model steps (99.2 %) have reply == rec**. That is 500/500 Boxoban-Hard, 133/133 CB-Dark,
155/155 MW15 and 428/441 Quest-Medium. `override-veto` and `fallback-route` fired **0 times** in the live
eval; the replay smoke test predicted 919 overrides. So the +0.25 over the anchor comes entirely from
`grounding.plan_route`, and every loss below is a property of that planner. The model adds nothing except
~5 k input tokens per step.

**Failure A — orbits between stood frontier cells (R3, Quest-Medium, Quest-Easy).**
- *Corridor-R3 ep03* (100 steps, fail). At step 15 the agent reaches the doorway (25,6), then the
  dead-end corridor stub `#` at (25,7). From (25,7), `plan_route` excludes `pos` and picks the nearest
  other frontier, (25,6), because the doorway still has undrawn neighbours (26,5..7), which are rock.
  From (25,6) it picks (25,7) the same way. The result is **north/south × 42 from step 16 to 99**,
  83 of 100 steps in a 2-cycle. The undrawn squares next to a cell you have stood on will never be drawn:
  in NetHack they are solid rock. What R3 needed here was `search` for a hidden corridor.
- *Quest-Medium ep00/01/02/04*. After the rats die the agent sits at the corridor mouth (31,13) and
  orbits east/west to (32,13) until the 100-step cap (ep00 steps 42–99). The corridor mouth touches dark
  squares N/S ("dark area adjacent north, south"), so it stays a frontier forever. The room beyond is lit
  and fully drawn, and `>` is drawn at (53,10) behind the `}` column. BFS correctly refuses `}`, so no
  stairs route exists, and the planner falls back to the stood frontier. **2-cycle steps: 65, 62, 65, 66.**
- Quest-Easy ep00 is a milder version: 26 distinct cells in 100 steps, wandering the 5-wide west strip
  whose east side is solid `}`.

**Failure B — `#` and boulders are passable in the BFS (Boxoban).** `BLOCK_GLYPHS = "-|} "`, so Boxoban's
`#` (iron BARS, per the prose "bars adjacent north…") is a walkable frontier, and so is every border `#`
next to the blank outside. *Boxoban-Hard ep00*: at (36,16) the route's first hop is north to (36,15). From
(36,15) the next hop is north into `#`. The wall-noun veto rules that out (`bars adjacent north`), so the
least-visited fallback returns south. Then the same thing repeats. This gives **north/south orbits on
41 % (Hard) and 46 % (Medium) of Boxoban steps**. Boulders (`` ` ``) are also passable, so the route
pushes boulders incidentally. In ep00, steps 3/6/9/13 push the row-16 boulder east until it jams
("You try to move the boulder, but in vain", step 14), which is an irreversible deadlock. Boxoban has no
push/goal model at all; the fountains `{` never enter the planner.

**Failure C — the `[yn]` handler cannot close the prompt (Quest-Easy ep04, 54 of 100 steps).** A
peaceful goblin stands north of (29,12). Moving north gives `Really attack the goblin? [yn] (n)`.
`modal_response` tries `esc` then `more`. **`esc` is not in MiniHack's action set**
(BALROG `environments/minihack/__init__.py` filters to its own `ACTIONS` dict), so it sends `more` (^M).
That produces `Unknown command '^M'` and a second `more`. The position did not change, but the message was
non-empty and not a refusal marker, so `spatial_memory.observe` never convicts `((29,12), north)`. The
route re-recommends north. **The same 3-step cycle ran 30 times, from step 21 to step 98.**

**Failure D — no item acquisition (all 10 Quest episodes).** Every Quest frame shows the horn `(` (and in
Quest-Medium the drawn `>` beyond the lava). The explorer never paths to an item and never emits `pickup`,
`kick`, `apply` or `zap`. The lava partition is real (no path around it), so without the item chain Quest
is 0 by construction. The harness also actively blocks the item chain: `modal_response` answers every
"What do you want to …" prompt with `more`/`esc`, which would cancel `apply`/`zap` item selection.

**Failure E — CB-Dark (5/5 deaths, steps 23–30).** The explorer walks east through the dark corridor into
the rat pack and trades blows. In ep01 the agent kills 2 rats at (34,12) and dies at step 22. Quest-Medium
ep03 dies the same way (HP 16→1 while swinging east at (31,13), 16 swings). The explorer has no fight
position or hold logic, but the melee axis has been measured as largely dice (see §6).

## 3. Failure breakdown (40 episodes)

| Category | Episodes lost | Evidence |
|---|---|---|
| Boxoban: no push planner, and `#`/boulder treated as passable, causing orbits | 10 (Box H+M) | 41–46 % 2-cycle steps, 500/500 steps rec-followed |
| Quest: no item chain (pickup/kick/apply/zap), then an orbit at the lava edge | 10 (QE+QM) | QM 60 % 2-cycle steps; QE 26 cells explored in 100 steps |
| `[yn]` peaceful-blocker loop | (inside QE ep04) | 54 modal steps, `Unknown command '^M'` ×27 |
| Stood-frontier orbit at a dead end where `search` was needed | 1 (R3 ep03) | 83/100 2-cycle steps |
| Melee death in the dark | 5 (CB-Dark) + QM ep03 death | all within 23–41 steps |

Wins: MazeWalk 10/10 (2-cycle ≤ 6 %), R3 4/5. The anchor's MazeWalk rate was 4/10.

## 4. Binding constraint — classification

- **HARNESS-BUG (primary, cheap):** the frontier definition, the `#`/boulder passability, the `[yn]`
  handling, and the modal handler cancelling item prompts. These are deterministic code errors,
  reproducible from the traces with no model involvement.
- **TASK-GAP (Quest, Boxoban):** the agent moves legally but has no lever that achieves the goal. Boxoban
  needs a sokoban planner. Quest needs the pickup → apply/zap → bearing chain. Both are fully decodable
  from the observation (see §5).
- **Not VLM-CEILING in the useful sense:** the VLM is inert (99.2 % copy). Stop spending design effort on
  the prompt. The model call could be skipped whenever `recommended` exists, which would cut cost roughly
  10× at an unchanged primary, if cost is a tie-break key.

## 5. Concrete changes to try (ranked by leverage)

1. **`tools/grounding.py:plan_route`: exclude stood cells from the frontier, then fall back to `search`.**
   Change `frontier(c)` to `seen.get(n) is None for some n` **AND** `visits[c] == 0`. Drop the
   `(visits>0, …)` key term, which is now redundant. When no route exists (no `>` and no unstood
   frontier) and `search` is legal, recommend `search` at the dead-end square. That square is the stood
   cell with the most undrawn neighbours; walk to it first. Budget searches per probed cell (~8–10), not
   globally. Add `search` to the action options/`arbitrate` so it is not snapped away.
   *Predicted:* R3 ep03 → win (it was a dead-end stub); the Quest orbits stop (freeing budget for fix 3);
   MazeWalk unchanged or faster. Prior measurement of this exact fix: MW9 ep0 went from 100 steps/fail to
   18 steps/win.

2. **`tools/grounding.py` + a new `tools/sokoban.py`: a Boxoban push planner.** Detect Boxoban from the
   legal set: exactly 4 cardinals, and `{` plus `` ` `` drawn. Decode the map with `#` and `' '` as
   walls; `.`, `{` and `<` as floor; `` ` `` as boulders; `{` as goals. A boulder may start on a fountain
   and hide it (fewer `{` than boulders), so union the fountains across frames. Run A*/BFS in push-space
   (state = boulder set + player reachable region) and cache the move sequence. Advance it one move per
   step, and replan whenever the observed boulder set differs from the prediction. The prior measurement
   decodes the push model exactly on 2746/2746 transitions: the median optimum is 53 steps under the
   100-step cap and a one-A* solver solves 19–29 of 30 levels. Emit the plan directly; the model is inert
   anyway. Separately, add `#` and boulder glyphs to BLOCK_GLYPHS in the non-Boxoban BFS. On
   Corridor/MazeWalk `#` is corridor, so gate this on the prose noun: a `#` whose adjacent noun says
   `bars` is blocked.
   *Predicted:* **+0.15 to +0.25 primary** (Boxoban 0/10 → 6–10/10). This is the largest single lever.

3. **`tools/action_io.py:modal_response` + a new `tools/quest.py`: the item chain.** When a `}` column
   separates the agent from the frontier or `>` (BFS reaches neither) and an item glyph (`(` or `/`) is
   reachable:
   - Route onto the pile and `kick` it apart. A multi-object pile silently refuses `pickup` under NLE:
     east, `kick`, west.
   - `pickup` each item and read the `x - a horn.` / `y - a … wand.` letter from the message.
   - At the lava edge, `apply` (horn) or `zap` (wand). Answer the `[xy or ?*]` letter prompt with the
     legal verb whose keystroke equals the item letter: `h j k l y u b n` are directions,
     `a c e o q s z` are verbs. Answer `Improvise? [ynq]` with `northwest` (= `y`). Then answer the
     direction prompt with the bearing toward the lava.
   - Repeat until `The lava cools and solidifies`, then resume the `>` route.

   Pick up the wand FIRST so its letter lands on a reachable key. **This requires removing the blanket
   `"what do you want to"` → `more` rule and the `"in what direction"` → `more` rule** from
   `modal_response`: today they cancel the chain. Prior probe: Quest-Medium 5/5 and Quest-Easy 2/5 with
   an observation-only policy. On Quest-Medium, hold the 1-wide corridor mouth (x=31) while the rats
   come, instead of stepping into the room (ep03 died stepping in).
   *Predicted:* +0.10 to +0.15 primary.

4. **`tools/action_io.py:modal_response` + `spatial_memory.observe`: the `[yn]` peaceful loop.** `esc` is
   not legal in MiniHack, and `more` is ^M, which does not answer the prompt. Add `"really attack"` to the
   conviction logic: a same-position step with that message convicts `(pos, dir)`, and the conviction
   expires after about 5 steps so a displaced peaceful does not seal a corridor. Keep answering with
   `more` (the default is `n`), but the route must then pick another neighbour. Fix 3 needs `y`
   (= `northwest`) for `Improvise?`, so key those two cases separately.
   *Predicted:* frees up to 54 steps/episode on Quest-Easy. This is a precondition for fix 3.

5. **`harness.py:act`: skip the model call when `g.recommended` is set.** Use `_Resp(g.recommended,
   "route")` and call the model only when the planner has no recommendation. Evidence: 99.2 % copy rate
   and 0 live overrides. *Predicted:* primary unchanged, cost down about 90 %, lower variance (temp=1.0
   noise removed from the 19 divergent steps).

6. **(Low priority) CB-Dark.** Stop advancing when the message shows `bites`/`hits` and hold the current
   corridor square, swinging at the attacker's direction. Do not retreat, because retreat was refuted
   multiple times. Expect at most +1/5, and verify it on ≥200 episodes before believing it.

## 6. Exhausted — do NOT repeat

- Prompt rewording aimed at the VLM: the model copies `recommended` 99.2 % of the time, so prose
  changes are close to inert.
- Veto-heavy grounding tweaks on their own: the veto never fired live (0 overrides), because the planner
  already avoids vetoed moves.
- HP-triggered retreat and disengagement on CB-Dark: refuted before (every degree of disengagement
  caused more deaths). CB-Dark outcomes are mostly dice. Use it only as a tie-break, never as the
  headline axis.
- Treating a creature as a hard refusal (blocking its square permanently): refuted before. Keep any
  peaceful-blocker conviction short-lived (fix 4).
