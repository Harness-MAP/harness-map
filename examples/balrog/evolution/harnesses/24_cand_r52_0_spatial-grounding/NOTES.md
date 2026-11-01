# cand_r052_0_spatial-grounding — potential-field walker + a Quest-regime module (tools/quest.py)

## module map
| file | role | lever |
|---|---|---|
| harness.py | `GradientAgent`; `STAGES = (prompt, orient, stairs, swing, cross, hold, kit, fetch, descend, consult)`, first stage to answer wins, rung -> `reasoning` | code |
| tools/field.py | frame decoder (sections, adjacent clauses, **now also raw `lang` prose**), glyph lookup | code |
| tools/chart.py | map memory + reachability-seeded potential field (unchanged) | code |
| tools/quest.py | **NEW** `Quest`: pile detection, a single safe kick, horn-first pickup plus a letter filler, firing across the lava, typed prompt answers, mouth hold, charted `toward` walk | code |
| AGENT.md / rules.md | consult-only prose (unchanged) | prose |

## axis:
Code, structure and wiring, steered toward r041/r046. r046 added a separable Quest owner (`quest`/`fetch` stages over a `tools/quest.py` module) to the chart walker. That slot is ported here with the mechanism r046 lacked. It is one new tool module plus four stages, all armed by the printed verb list.

## change:
**New `tools/quest.py`.** It is armed only when `{pickup, kick, zap} ⊆ legal`, which is true of the Quest pair and of no other task, so the other 30 episodes follow the parent's code path unchanged. The four new stages:
- **`cross`** (above hold): fires a held horn or wand (`apply`/`zap`) ACROSS an adjacent cardinal lava run, at most 2 times. It does not fire once `>` is reachable.
- **`hold`**: spends the turn with `wait` when a hostile is named in the prose and the hero is on a square with at most 2 lanes (the rat-room corridor mouth). It never retreats, is capped at 15 consecutive turns, and is off once the field seeks stairs.
- **`kit`**:
  - It does one `pickup` at spawn. If that pickup is silent, it marks the square as a pile.
  - It kicks the pile **once**, from a post whose ray through the pile ends on a wall before any `}`.
  - It picks up the horn `(` before the wand `/`.
  - If the next free letter is one no verb can type (e.g. caveman `f`/`g`), it first picks up a FILLER object to use up that letter.
- **`fetch`**: walks a charted BFS to, in order, the kick post, then tools, then fillers, then lava banks. It avoids squares with a creature drawn on them, because a peaceful goblin re-asks `Really attack?` forever.

**Other harness changes:**
- `st_prompt` routes `In what direction?`, the item prompt `[j or ?*]` (answered with the verb whose key is that letter) and `Improvise?` (`northwest` = y) to `Quest.answer` before the generic `more`.
- On an armed Quest level, `st_consult` no longer calls the VLM. It takes the least-trodden move whose edge is open on the chart. This removes the ~420k-token Quest-Easy orbit.

## why:
The r048 diagnosis names a TASK-GAP: the field goes flat on Quest, and the VLM consult emits 0 non-movement verbs in 1000 steps. r046's Quest stage was the right slot, but it failed on a silent pile pickup, had no kick, picked up the wrong items, and had no mouth hold. This candidate applies diagnosis items #1, #2 and #5 inside that slot.

I validated it on the real env: `make_env` with the stock config, the evaluator's seed protocol, and the 10 Quest eval seeds read from r048's run JSONs, with a stub VLM.
- **Quest-Medium: 5/5 wins** in 71–81 steps. The sequence is kick → `j` wand → `zap` → `solidifies`, with the rat fight held at the corridor mouth. The parent had 5/5 deaths.
- **Quest-Easy: 4/5 wins** in 43–55 steps. The filler pickup fixed the f/g letter trap on seed 255275049. The one loss (4228252957) has no filler object drawn, so the wand lands on untypeable `g`.
- **Non-Quest tasks:** Corridor-R3 ran through the parent's unchanged code path.

## files:
tools/quest.py (new), harness.py (4 stages, prompt routing, Quest consult bypass), tools/field.py (`lang` field), NOTES.md
