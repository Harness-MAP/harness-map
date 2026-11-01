# Offline measurements — block3d, 20 levels, real engine, stub model

Every number here comes from the real SpatialWorld block3d engine running headless
(`SDL_VIDEODRIVER=dummy`, `ANTICHEAT_GUARD=1`), one episode per process, with
`block3d_openai_evaluation.create_openai_response` replaced by a stub that returns the SAME action
every turn. A stub prices the HARNESS and nothing else: whatever the frozen model would have
contributed is held constant, so a difference between two arms is a difference between two
harnesses. All arms report `oracle_access: null` and `valid: true`.

This directory is φ-free (`traces` is a `DEFAULT_SKIP_PREFIXES` entry in
`harness_map/qd/descriptors.py`), so these tables can be as long as they need to be without
moving the candidate's descriptor. `NOTES.md` carries the conclusions; this file carries the
evidence.

## headline

| arm | stub answers | won | note |
|---|---|---|---|
| `staged_anchor` (parent) | `place_block` | **1/20** | its one win is L1, whose only target cell IS the spawn |
| `staged_anchor` (parent) | `move_right` | **0/20** | |
| this candidate | `place_block` | **20/20** | |
| this candidate | `move_right` | **20/20** | |
| this candidate | `move_up` | **20/20** | |

The three reply arms are identical in outcome, which is the point: the harness does not depend on
what the model happens to say. Earlier revisions were 20 / 19 / 19 and the two losses were both
the readout defects described below — a stub rig with ONE reply would have shipped them.

## presses per level (shipped arm)

| L | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| presses | 1 | 5 | 3 | 3 | 6 | 8 | 9 | 8 | 9 | 8 | 9 | 15 | 10 | 15 | 20 | 21 | 25 | 22 | 41 | 45 |
| target cells | 1 | 1 | 1 | 2 | 3 | 3 | 3 | 4 | 4 | 4 | 4 | 5 | 5 | 8 | 8 | 9 | 10 | 10 | 15 | 16 |

283 presses over 20 levels. A press costs two engine steps and the budget is 120 steps, so the
wallet is 60 presses per episode; the worst level uses 45 of them. The frozen model's own answer
was applied on 116 of 283 turns (41.0%).

`frozen = 0` and `fixes = 0` on every one of the 20 levels: on a clean run the cursor never
clamps and the measured cursor never disagrees with the dead-reckoned one. That is what makes the
perception ablations below come out neutral, and it is also the honest reason to keep them.

## ablations on the clean split — all win-neutral

| arm | what was removed | won |
|---|---|---|
| `nolocate` | the measured cursor is computed but never adopted | 20/20 |
| `nofence` | a frozen frame never pins a coordinate | 20/20 |
| `assertruler` | the ruler is asserted (40, 90, (30,180,330)) instead of fitted | 20/20 |
| `volmid` | the rim is measured from the volume centroid, not the layer's | 20/20 |

So on these 20 levels the PLAN carries the result and every piece of the new perception is
dormant. Reporting that honestly matters more than the 20/20: a reader who only saw the headline
would conclude the calibrated projection is what wins, and it is not.

An earlier revision DID show `volmid` losing L17 — but that revision also had the broken readout,
and with the readout fixed both centroid choices win 20/20. The layer centroid stays because it is
the principled choice (the surplus a three-panel read invents sits outside the shape THAT SLICE
admitted), not because it is worth a level here.

## fault injection — where the closed loop actually pays

The clean split cannot price a mechanism whose whole job is to catch a broken premise. So break
the premise: perturb the LEDGER (never the engine) and re-run the same 20 levels.

| fault | perception loop | won |
|---|---|---|
| ledger starts at the wrong cell, `(0,0,0)` instead of the published spawn | on | **3/20** |
| ledger starts at the wrong cell, `(0,0,0)` instead of the published spawn | blinded | **0/20** |
| ledger silently fails to advance on every 4th move | on | **8/20** |
| ledger silently fails to advance on every 4th move | blinded | **6/20** |

"Blinded" means `BuildState.correct` is never called and `BuildState.grade` always sees an
unchanged frame as "no information" — i.e. pure dead reckoning, which is what the parent and the
cell-17 founder both do. The loop converts 3 of 20 and 2 of 20 on the two fault classes, and it
cannot convert more because a wrong cursor also puts the wrong cell in the placed set, which no
amount of later correction un-presses.

## the two defects the first sweep found

Both are in the new mechanism, both were found by running the real engine, and neither is visible
in a synthetic rig that draws its own frames.

1. **`locate(None)` on an occluded cursor.** The highlight is drawn in the scene, so a cube
   standing nearer the camera can hide it completely and `spot` then returns None. `sight` passed
   that None straight into `locate`, which subscripts it. On L15 it fired on the press that would
   have won the episode: 7 of 8 target blocks placed, then `'NoneType' object is not subscriptable`
   for the remaining 100 turns. Fixed by `_LOOKABLE`, which treats "the cursor is not in the
   picture" as one of the two ordinary reasons a readout is unavailable.
2. **A basis learned from a clipped highlight.** `learn` accepted any non-null displacement, so a
   partially occluded highlight taught a basis vector that has nothing to do with the grid. Every
   later readout was then confidently wrong, and because the original `locate` allowed a few
   pixels of containment slack, a wrong cell could still come back UNIQUE and be adopted. On L13
   (stub `move_up`) that corrupted the ledger at turn 10 and the episode stood down at turn 13; on
   L19 (stub `move_right`) the ledger and the readout fought for 41 clamped moves. Fixed two ways
   at once: `learn` only accepts displacements between WHOLE highlights (pixel count equal to the
   template's), and `locate` requires strict containment plus a `margin`-pixel win over the
   runner-up.

The lesson for the next proposer: a loosened test is worse than a refusal. Both defects were the
mechanism answering when it should have said "I cannot see".

## the ruler, measured on the real frame

The panel strip's exact-constant census, read off a live frame (level 7):

- border colour `(70,70,75)`: 3474 px on the frame, cols 40..878, rows 30..502
- tile fill `(80,80,80)`: 1372 px, cols 40..128, rows 90..418
- cursor highlight `(100,200,250)`: 1170 px total — **1024 in the viewport (one whole cube
  footprint, centroid (352.24, 655.10) in (row, col)) and 146 in the panel strip, which is a text
  label**. Cropping the highlight search to the viewport is therefore load-bearing.
- GRID|INK columns in 130..299: **none**. The panel strip and the viewport do not overlap, and the
  viewport's largest per-column count of either constant is 10 against the border's 270.

Per-axis census of GRID|INK inside the strip, which is what `fit_gauge` reads:

- full-span columns (count 270): `[40, 129]` → `x0 = 40`, `span = 90`, `pitch = 15`, `tile = 14`
- full-span rows (count 90): `[30, 119, 180, 269, 330, 419]` → `tops = (30, 180, 330)`
- identical on every frame tried, and `gauge_audit` returns True on all of them.

**Why the union and not the border colour alone.** A lit tile is painted right up to the border,
so a tile in the first row or column erases part of it. Measured on L7: border column 40 carries
270 border-coloured pixels on an empty panel and **228** when the first column is lit. A
border-colour-only census of that frame fits `pitch = 89` and `tops = [31]`, `gauge_audit`
refuses, and the whole harness stands down for the episode — which is exactly what the first
version of this file did, and the stand-down is in the trace. The union census is 270 either way,
because a border pixel is always one colour or the other.

## the projection, measured on the real frame

Driven over a 52-frame scripted tour of real frames (`move_left`×2, `move_up`×2, then a
serpentine over three z layers), replaying the candidate's own `Gauge` offline:

- basis learned, in screen (row, column) pixels per +1 cell:
  `x = (+20, +40)`, `y = (+17, −34)`, `z = (−40, 0)`
- complete from turn 40, which is the first turn after the tour's first `move_forward` — the third
  axis cannot be measured before the episode walks it, and `step_toward(first=...)` exists to make
  that walk happen as early as the tape allows.
- `locate` offered 12 readouts and agreed with the dead-reckoned cursor on all 12, with no wrong
  answers and no refusals.

Synthetic cross-check (a scratch renderer that draws the documented layout, so it tests the
candidate's arithmetic rather than the real renderer): the ruler fits exactly, the panels decode,
the envelope contains the truth, the calibration closes from three moves, and `locate` is
**216/216 exact** both on an empty board and with six cubes down. The synthetic cursor is drawn
last and so is never occluded, which is precisely why the synthetic rig could not find defect 1.

## rig recipe

```
stub_one.py <harness_dir> <level> <reply> <logdir>
  os.environ["ANTICHEAT_GUARD"] = "1";  SDL_VIDEODRIVER=dummy
  sys.path += SpatialWorld/scripts, SpatialWorld/scripts/game
  import block3d_openai_evaluation as BE
  BE.create_openai_response = lambda *a, **k: Stub(json.dumps({"action": REPLY}))
      # Stub needs .output_text and .output = []
  asyncio.run(eval_runner.run_one(game="block3d", level=L, harness_dir=HD, log_dir=LOG, ...))
```

One episode per PROCESS — pygame is process-global — and five processes at a time puts 20 levels
in about four minutes. The result dict carries `won`, `vlm_calls`, `oracle_access` and `valid`, so
one sweep proves the mechanism runs, counts the model's participation, and checks anti-cheat
compliance at the same time.
