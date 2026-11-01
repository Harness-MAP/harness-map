# block3d harness — the full offline measurement log

Companion to `NOTES.md`. Everything here was measured by a scratch rig that synthesises frames from the
documented geometry (three congruent panel outlines in `rgb(70,70,75)`, shadow fill `rgb(80,80,80)` on
`rgb(20,20,20)`, the highlight footprint at the affine centroid) and drives the REAL `orchestrator.py` through
a stub evaluator that owns the engine state and the scripted model reply. Three banks of 20 levels:
**structured** (unions of small slabs — the shape family a level actually is), **hull-closed** (random
targets whose admitted hull equals the target, the common real case), and **hull-over** (random targets
the hull strictly over-approximates). This file is descriptor-free scaffolding; the live harness never
reads it.

## what was measured, and on what
Everything below was measured offline against a rig that synthesises frames from the documented
geometry (three congruent panel outlines in `rgb(70,70,75)`, shadow fill `rgb(80,80,80)` on
`rgb(20,20,20)`, the highlight footprint at the affine centroid) and drives the REAL `orchestrator.py`
through a stub evaluator that owns the engine state. Three banks of 20 levels: **structured** (unions of
small slabs, the shape family a level actually is), **hull-closed** (random targets whose hull equals the
target, the common real case), **hull-over** (random targets the hull strictly over-approximates).

### the decode is a measurement, not a constant table
| check | result |
|---|---|
| masks reproduce the true three projections | **121/121** synthetic levels, exactly |
| blank strip | refuses (`strip-unreadable`) |
| all 36 squares lit in all three panels | still fits and decodes |
| strip moved (`left` 40 -> 70 / 100) | fit follows: `left` measured as 70 / 100, decode exact |
| pitch rescaled (15 -> 12 / 20 / 25) | fit follows: `pitch` measured as 12 / 20 / 25, decode exact |
| stride changed (150 -> 100 / 170) | fit follows: `tops` measured, decode exact |
| cursor inverted from the highlight ink | **216/216** poses on an empty grid, never wrong |
| ink with cubes in front of the cursor | refuses (clipped footprint), ledger stands |

The fit has to be occlusion-proof, and that is the one thing a naive census gets wrong: a lit square at a
panel border paints over part of the outline, so counting ink per ROW can lose a panel's top row
entirely. Two readings are taken — the maximal runs of outline ink down the tallest column, and the row
census as an arithmetic progression of three congruent panels — and the second answers when the first is
eroded. With only the row census, 2 of 44 random levels failed to decode; with both, 0 of 121.

### the tier key: three candidates kept, four refused
Prefix rate = how often the true target is a PREFIX of the tape (so the engine can end the episode on
it); fits = prefix AND the prefix costs <= 50 presses. 60 levels.

| tier key | prefix | fits | total toll |
|---|---|---|---|
| **(forced, height)** — shipped | **53/60** | **45/60** | **2390** |
| (support, height) | 53/60 | 45/60 | 2393 |
| (forced, support, height) | 53/60 | 45/60 | 2393 |
| (forced, height, rim-from-hull-centroid) | 53/60 | 44/60 | 2579 |
| (forced, height, minimum-cover membership) | 53/60 | 42/60 | 2494 |
| (height, forced, ...) — height outranks forced | 42/60 | 41/60 | 2422 |
| (height, minimum-cover, forced) | 42/60 | 41/60 | 2422 |

Two readings of that table matter. **Forced-first is load-bearing**: demote it below the height key and
on the levels where the hull over-approximates the prefix rate collapses from 14/20 to 3/20. And **the
rim-distance key is refuted here** — it is prefix-neutral and costs 8% more travel. Under a boustrophedon
in-layer sweep the lane order already encodes position, so the rim key only adds tier boundaries to walk
across. Minimum-cover membership as a third key is refuted the same way (4% more travel, no prefix gain),
which is why `thin_shape` is kept as the plan's lower bound and acceptance test rather than as a ranking.

### the affordability switch is refuted
If the hull tape does not fit the wallet, aim at the minimal shape instead — it is cheaper and is itself
an admitted structure. Measured: wins under the hull tape 19 / 15 / 11 on the three banks; under the thin
tape 15 / 15 / 11; **under either, 19 / 15 / 11.** The switch never adds a level. Recorded so the next
iteration does not spend a round on it.

### end-to-end, driving the real harness
| bank | wins | clamped presses | cancelled work | stand-downs |
|---|---|---|---|---|
| structured (realistic shapes) | **19/20** | 0 | 0 | 0 |
| hull-closed (random, hull == target) | 15/20 | 0 | 0 | 0 |
| hull-over (random, hull > target) | 11/20 | 0 | 0 | 0 |
| structured, model replying at RANDOM | **19/20** | 0 | 0 | 0 |
| structured, no highlight ink at all | 19/20 | 0 | 0 | 0 |
| structured, cubes occluding the cursor | 19/20 | 0 | 0 | 0 |
| a blank, undecodable frame | falls through to the parent's loop every press, 1 model call each |

The parent, on the real 20 levels, won 1/20 with **42.5%** of its moves clamped at a grid face and
**21.0%** of its toggles cancelling its own work. Both are zero here by construction: `tools/press.py`
knows which presses cannot move the cursor and `tools/serpent.py` knows which cells the plan still wants.
The five hull-closed losses are all BUDGET losses on random 11-to-18-cell targets scattered through the
box, which is an adversarial shape family for a lane sweep; the real levels' optimal tapes are at most 33
presses and the structured bank is the representative one.

The spawn gift is taken by PLAN, not by luck: on a level whose only target cell is `(2,2,0)` the tape's
first cell is the spawn, so the first decision emits `place_block` and wins at press 1 — measured even
when the model is scripted to say `move_left`. The parent took that gift on only 4 of the 9 levels where
the spawn is a target cell.

### the ask policy
One rule: ask only when the answer could change the emitted key. The first decision is always the
pristine, uninjected pass-through — it is the one decision made before a plan exists, it is what keeps
the episode's model-call count honest, and injecting on the first turn has cost the level-1 gift win
before. After that the model is asked when two or more axes close the gap equally, or when two or more
surplus blocks are equally droppable. Standing on the objective, the key is the toggle and nothing could
change it, so nothing is asked. Measured on the structured bank: **5 model calls per episode** instead of
the parent's 50, with the same wins as asking every press.


## honest limits
- The three panels are a lossy channel. On a level whose hull strictly over-approximates the target, the
  win depends on the target being a prefix of the tape; measured at 14/20 on an adversarial random bank.
- The ink read answers only when the footprint is whole. Once cubes stand in front of the cursor it
  refuses and the dead-reckoned ledger carries the episode. That reckoning is exact as long as every key
  this harness returns is the key the engine executes.
- `support` is O(|hull|^2) in the number of admitted cells and the hull can reach 216; at the sizes a
  level reaches (hull <= 20 in every bank) it is microseconds, but it is not a free call.
- The lane sweep is 10-15% more travel than a nearest-first walk on randomly scattered targets. It is
  within 1% on the structured bank, and it is what makes the rim key unnecessary.
- No part of this reads engine state. The only inputs are `frame_data.image` (with
  `evaluator.capture_game_screenshot` as the fallback channel) and the keys this harness itself returned.

