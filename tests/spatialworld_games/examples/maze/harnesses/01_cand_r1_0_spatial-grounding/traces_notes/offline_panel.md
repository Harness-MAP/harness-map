# Offline panel — how this candidate was developed and priced

No frame of this game exists on disk (the 20 anchor episode dirs hold only JSON; the engine and
renderer sources are off-limits to the proposer). So the band geometry could NOT be calibrated
against real pixels, and everything below is a **stand-in rig**, not a prediction of the live score.
It exists to answer mechanism questions — does the lattice stay in sync, does the frontier
terminate, does either half of the mechanism carry weight — which are answerable without the real
renderer.

## The rig

* `carve(w, h, seed)` — randomised-DFS perfect maze, `S` at (1,1), `E` the farthest open cell.
* The transition is the one `diagnosis.md` §7 fitted from the real tapes: initial heading = row−1
  (so the cell ahead at step 0 is a wall), `turn_right` clockwise, `move_forward`
  small/medium/large = 1/2/3 single-cell steps **stopping early at a wall**, entering `E` ends the
  episode, 100-action budget.
* The renderer is depth-faithful and nothing more: for each of the three bands it computes the ray
  depth to the first wall and paints a flat face whose apparent height shrinks with depth, with
  receding floor/ceiling banding outside it. It is a pure function of (cell, heading) — which is
  the only property of a renderer this harness relies on.
* The "VLM" is a scripted reflex. Five of them, including `anchor_orbit`, the literal period-6
  tape `diagnosis.md` §2.2 recorded as byte-identical on L08 and L16.

## Result — 16 mazes (7×7…13×13) × 5 reflexes = 80 episodes

| arm | wins | mean nulls | mean cells | mean steps |
|---|---:|---:|---:|---:|
| anchor (pass-through) | **2/80** | 66.3 | 2.4 | 97.9 |
| this candidate | **58/80** | 3.2 | 27.6 | 60.3 |
| − census blinded (frame replaced by pose-keyed noise) | **0/80** | 5.2 | 3.7 | 100.0 |
| − ledger blinded (every view digest unique, so nothing is ever a null) | **32/80** | 35.6 | 17.0 | 79.0 |

The anchor arm reproduces the diagnosed failure shape (66 of ~98 forwards null, 2.4 cells visited),
which is the rig's own sanity check. Both ablations lose, and they lose *differently*: without the
census the agent still refuses measured walls (5.2 nulls) but cannot choose a side and never gets
anywhere (3.7 cells); without the ledger it keeps re-pushing walls (35.6 nulls) yet still wins 32,
because the census alone is a usable wall test. The two halves are complementary, not redundant.

`vlm_calls/step` is 1.00 in every arm — the frozen model is asked exactly once per frame.

## Bugs this rig caught (each was a real defect, not a rig artefact)

1. **Spin.** The scan is pose-relative, so recomputing the goal after each rotation let the aim
   drift round with the agent; it rotated forever and never arrived. Cure: `_aim` holds the chosen
   heading until it is spent or the cell changes.
2. **Two-cell shuttle.** The census's 1.0 sort-key for the BACK side (which no band covers) was
   being read as a *veto*, condemning the one direction the agent came in through; the frontier
   then bounced between two mapped cells. Cure: `_said` speaks only for sides a band covers.
3. **Sealed cell.** A cell whose four sides were all recorded as walls left the frontier with
   nothing to pick and the agent spun. A cell you are standing in cannot have four walls, so the
   record must be wrong. Cure: `_amnesia`.
4. **Desync on long pushes.** A `medium`/`large` forward stops early at a wall, so its length is
   not creditable; honouring the model's long push across an unmapped edge drifted the cell.
   Cure: `_honours` refuses a run the ledger cannot vouch for; the override re-emits the same
   heading at a length it can. Win-neutral at n=80 (58/80 either way) but nulls 5.8 → 3.2 and
   cells 24.1 → 27.6, so it is kept as the cleaner invariant.
5. **Exhausted branch.** A purely local frontier has no way to leave a finished subtree. Cure: a
   `back` pointer per cell and a `_retreat` fallback — depth-first backtracking.

## What the rig cannot tell us

Whether the flatness statistic separates wall-at-1-cell from open corridor **on the real
renderer**, and where in the frame it does so. That is why `HORIZON` takes the whole column under
the HUD rather than a guessed band, why the threshold is learned online from the did-I-move label
instead of being written down, and why `_call` returns `None` — no opinion — until the two label
classes are separated by a measured margin. The first live run's override rate and null-forward
rate are the measurements that settle it.
