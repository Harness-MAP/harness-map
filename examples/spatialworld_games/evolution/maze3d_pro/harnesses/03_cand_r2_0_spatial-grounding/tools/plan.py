"""tools/plan.py — the ACTION-OUTCOME PLANNER: a closed-loop governor over the frozen model's
per-step proposal, now with a PIXEL-EXACT reading of the two things in this maze that matter and
cannot be inferred from motion alone: where the stair tile is, and where the gold exit tile is.

WHY THIS EXISTS. The anchor hands every decision straight to `evaluator.baseline_step` and keeps no
state, so nothing in the loop can tell a move that happened from a move that did not. The frozen
model is reactive: when its forward move is refused by a wall the screen comes back looking the same,
it proposes the same move again, and the episode spends its whole 100-step budget on one refused
action.

WHAT IT MEASURES — ALL OF IT FROM `frame_data.image`, the very frame the agent is shown. No engine
state, no layout, no coordinates, no distances.

(1) MOTION, by differencing two consecutive frames.
  * a translation that leaves the screen UNCHANGED did not happen -> the cell ahead is a wall. The
    engine stops a multi-cell stride early at a wall, so "zero cells moved" means the FIRST cell was
    refused, whatever granularity was asked for. That verdict is recorded against the frame
    FINGERPRINT, which is the agent's whole first-person view: whether something blocks the way
    ahead is a property of that view, so the ledger is exactly re-usable the next time the same view
    comes back (including after a four-turn circuit of the same cell).
  * a translation that CHANGES the screen did happen -> the agent moved.
  Equality is deliberately tolerant: identical fingerprints mean unchanged, and a changed
  fingerprint still counts as unchanged when under `_NULL_FRAC` of the sampled pixels differ. A real
  step or turn in a first-person 3D view repaints most of the frame, so this absorbs a one-pixel
  animation without ever mistaking a refused move for a successful one.

(2) THE TWO SPECIAL TILES, by an EXACT-COLOUR census of the whole frame. This renderer is
  flat-shaded: a sweep of 433 live frames over seven levels found exactly 30 distinct RGB values,
  and in that whole palette
      * every stair shade is the ONLY family with R == 0           -- (0,234,234) (0,214,214)
        (0,204,204) and the shaded faces (0,92,110) (0,84,100) (0,76,91) (0,68,81) (0,60,72)
        (0,52,62);
      * the gold exit is the single value (255,200,0), outlined in (255,255,255);
      * everything else is a wall brown with R > G > B (the brightest is (147,78,55)), the floor
        (50,60,65), or the ceiling (15,15,35) / (4,4,10).
  So `R == 0 and G >= 1` IS the stair mask and `R >= 200 and 150 <= G <= 230 and B <= 60` IS the
  exit mask -- exact masks, not colour tolerances, with nothing else in the palette able to satisfy
  either one.

(3) WHERE THAT TILE IS, from the AREA it paints. The painted area as a fraction of the frame is a
  distance ladder, and it is quantised: a stair straight down the corridor paints 1753 px five cells
  out, 15755 four cells out, 27401 three, 61505 two, and **243005 (50.6% of an 800x600 frame) when
  it is the adjacent cell dead ahead** -- then 0 the moment you step onto it, because the tile under
  your own feet is never drawn. The exit's ladder has the same shape (237009 px = 49.4% adjacent).
  The gap between "adjacent" (0.506) and "two cells away" (0.128) is enormous, so
  `_ADJACENT_FRAC = 0.30` separates them with no tuning at all.

(4) THEREFORE: STANDING ON A STAIR. The one fact the agent could never see. If the previous frame
  was saturated with stair paint, and the action executed from it was a translation, and the frame
  then CHANGED (so the step really happened) and the stair paint went to 0 -- the tile under the
  agent's feet is that stair. Measured end-to-end on all 25 levels: forcing `climb_up` at exactly
  that moment moves the agent to floor 2, which 25 of 25 episodes of the parent never once did.

WHAT IT DOES WITH THAT. The frozen model is asked EVERY step and is obeyed unless the frames say
its proposal cannot do anything, or unless the grounding has a strictly better-evidenced move:
  1. the gold exit IS the next cell -> step into it (this ends the episode),
  2. the floor is EXHAUSTED and a stair is under our feet -> take the floor transition,
  3. the floor is exhausted and the stair IS the next cell -> step onto it (never stride past it),
  4. a translation already proven refused from this exact view -> re-aimed,
  5. a climb with no stair measured under our feet -> obeyed `CLIMB_PROBES` times, then re-aimed,
  6. a turn proposed while the loop has stalled and the way ahead is untested -> re-aimed forward.
A re-aim is not a random nudge: it is one step of a right-hand wall follower (probe the way ahead;
if it is refused rotate to the next bearing in the order right, straight, left, back). Leaving the
came-from bearing for LAST is the whole of it -- a sweep that rotates always the same way tries the
bearing it just came from second and walks straight back out of every dead end.

WHY THE CLIMB IS GATED ON AN EXHAUSTED FLOOR, NOT TAKEN AT THE FIRST CHANCE. The exit is on the
START floor on a large minority of these levels, and a harness that climbs the instant it can walks
off the winning floor: measured over all 25 levels with a degenerate stub model, climbing at the
first opportunity won ten levels the parent lost and LOST six it had already won. A floor is called
exhausted when `STALE_LIMIT` consecutive decisions have gone by without the agent discovering a view
it had not already seen on this floor -- so the transition is taken only once walking around has
stopped paying, and a floor change resets the counter, which is also what stops an up/down 2-cycle
at one stair: the new floor has to go stale before the return is allowed.

A CLIMB IS NOT SELF-VERIFYING. A *successful* climb can leave the frame byte-identical (measured:
level 2's winning climb did), so an unchanged frame is never read as proof a climb failed. Grounded
climbs are presumed to have worked and are budgeted (`CLIMB_BUDGET`); ungrounded ones are presumed
not to have and are budgeted separately (`CLIMB_PROBES`). That budget, rather than a frame verdict,
is what makes the 82-consecutive-refused-climb absorption impossible.

AND IT TELLS THE MODEL. The governed verdict is useless if the model keeps proposing the same
refused action, so `render_memory()` renders the ledger into plain English and tools/vlm.py appends
it to the agent's system turn every step. The facts are the model's own observation history stated
back to it: what it last proposed, whether the screen moved, which actions this exact view has
already refused, what the grounding can see, and whether a stair is under its feet.
"""
from __future__ import annotations

import hashlib

try:                                    # the engine hands us a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

from . import action as _act

_SAMPLE = 2          # fingerprint/compare every 2nd pixel: ~4x cheaper, still a faithful view id
_DIFF_EPS = 8        # per-channel tolerance for "this pixel changed"
_NULL_FRAC = 0.002   # < 0.2% of sampled pixels changed == the screen did not change

# ── the exact palette masks (see the module docstring, §2) ─────────────────────────────────
_STAIR_G_MIN = 1          # a stair pixel: R == 0 and G >= 1. Nothing else in the palette has R == 0.
_EXIT_R_MIN = 200         # the gold exit (255,200,0): R >= 200 and 150 <= G <= 230 and B <= 60.
_EXIT_G_MIN = 150         # the brightest wall brown is (147,78,55), so R >= 200 alone excludes
_EXIT_G_MAX = 230         # every wall; the (255,255,255) outline is excluded by B <= 60.
_EXIT_B_MAX = 60

# ── the distance ladder, as a fraction of the frame (see §3) ───────────────────────────────
# measured on 800x600 = 480000 px, straight down a corridor:
#   5 cells 1753 (0.0037) | 4 cells 15755 (0.0328) | 3 cells 27401 (0.0571)
#   2 cells 61505 (0.1281) | ADJACENT, DEAD AHEAD 243005 (0.5063) | standing on it 0
_ADJACENT_FRAC = 0.30     # >= this: the painted tile IS the next cell dead ahead
_NEAR_FRAC = 0.045        # >= this: within about three cells (the 3-cell rung is 0.0571)
_SIGHT_FRAC = 0.002       # >= this: some of the tile is painted somewhere in this view

#: the built-in working-memory template. memory.md (when present next to the harness) overrides it,
#: which is what keeps the wording a prose lever; the built-in keeps the mechanism alive when the
#: harness is bundled without it.
DEFAULT_TEMPLATE = """## WORKING MEMORY (self-derived: measured off the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- what the pixels show: {sight_line}
- floor: {floor_line}
- mode: {mode_line}

How to use this: an action that left the screen byte-for-byte identical did not happen - something
solid is there. Never re-propose an action listed as PROVEN REFUSED while this screen is in front of
you; turn to a bearing you have not tried yet from this screen and then move. A climb only does
something while a stair is under your feet; the line above says when that is measured to be true.

Reply with ONLY the JSON object in the format specified above."""


class ActionPlanner:
    """Episode-local state + the governor. One instance per episode, built in orchestrator.py.

    The four tuning constants are CLASS attributes, read through `self`, so an offline rig can
    sweep them (`plan.ActionPlanner.STALE_LIMIT = n`) without touching the shipped defaults."""

    STALE_LIMIT = 4        # decisions with no NEW view on this floor before the floor is exhausted
    CLIMB_BUDGET = 4       # grounded floor transitions the governor will force in one episode
    CLIMB_PROBES = 2       # ungrounded climbs the model is allowed to spend per floor

    def __init__(self, max_steps: int = 100, stall_limit: int = 4,
                 climb_lock_span: int = 8, template: str = ""):
        self.max_steps = int(max_steps or 100)
        self.stall_limit = int(stall_limit)
        self.climb_lock_span = int(climb_lock_span)
        self.template = _clean_template(template) or DEFAULT_TEMPLATE
        # ledgers keyed on the frame fingerprint (the agent's own view)
        self.refused: dict[str, set] = {}       # view -> translations proven not to move us
        self.tried: dict[str, set] = {}         # view -> actions executed from it
        self.views: dict[str, int] = {}         # view -> how often it has been shown
        # rolling state
        self.view = ""
        self._thumb = None
        self.step = 0
        self.since_move = 0
        self.last = None                        # the previous decision we committed
        self.sweep = False                      # True while escaping a proven-refused pocket
        self.sweep_turns = 0                    # consecutive re-aimed turns since the last move
        self.turn_sense = 1                     # +1 = right-hand wall follower, -1 = left-hand
        self.overrides = 0
        self.moves = 0
        self.base_prompt = None                 # stash of the agent's first-call system turn
        # ── the pixel grounding (§2-§4) ────────────────────────────────────────────────────
        self.stair = 0.0                        # stair paint, as a fraction of THIS frame
        self.exit = 0.0                         # gold-exit paint, same units
        self._stair_was = 0.0                   # the previous frame's stair paint
        self.on_stair = False                   # a stair is MEASURED to be under our feet
        self.floor_epoch = 0                    # bumped by every grounded floor transition
        self.seen_on_floor: set = set()         # (floor_epoch, view) ever shown
        self.stale_run = 0                      # decisions since this floor last showed a new view
        self.spent: dict[tuple, set] = {}       # (floor_epoch, view) -> climbs executed there
        self.probes = 0                         # ungrounded climbs spent on this floor
        self.climbs = 0                         # grounded floor transitions taken

    # ── outcome measurement ────────────────────────────────────────────────────────────────
    def _array(self, image):
        """The frame as an int-safe ndarray, or None when numpy or the frame is unavailable."""
        if image is None or _np is None:
            return None
        arr = _np.asarray(image)
        return arr if arr.ndim == 3 and arr.shape[2] >= 3 else None

    def _fingerprint(self, image, arr=None):
        """(fingerprint, downsampled copy) for the frame the agent is being shown."""
        if image is None:
            return "", None
        if _np is None:                                     # pragma: no cover
            try:
                return hashlib.md5(bytes(memoryview(image))).hexdigest()[:16], None
            except Exception:
                return "", None
        a = arr if arr is not None else _np.asarray(image)
        if a.ndim == 3:
            a = a[::_SAMPLE, ::_SAMPLE, :3]
        elif a.ndim == 2:
            a = a[::_SAMPLE, ::_SAMPLE]
        else:
            return "", None
        a = _np.ascontiguousarray(a)
        return hashlib.md5(a.tobytes()).hexdigest()[:16], a.astype(_np.int16)

    def _census(self, arr):
        """(stair_fraction, exit_fraction) of the WHOLE frame, from the two exact palette masks.

        Full-resolution on purpose: the fractions ARE the distance ladder of §3 and a sampled
        count would move the rungs. ~2 ms on 800x600, against a ~2000 ms model call."""
        if arr is None:
            return 0.0, 0.0
        r = arr[:, :, 0].astype(_np.int16)
        g = arr[:, :, 1].astype(_np.int16)
        b = arr[:, :, 2].astype(_np.int16)
        n = float(r.size) or 1.0
        stair = ((r == 0) & (g >= _STAIR_G_MIN)).sum() / n
        gold = ((r >= _EXIT_R_MIN) & (g >= _EXIT_G_MIN)
                & (g <= _EXIT_G_MAX) & (b <= _EXIT_B_MAX)).sum() / n
        return float(stair), float(gold)

    def _screen_changed(self, prev_view, prev_thumb, view, thumb) -> bool:
        if prev_view and view and prev_view == view:
            return False                                     # identical frame: nothing happened
        if prev_thumb is None or thumb is None or prev_thumb.shape != thumb.shape:
            return prev_view != view
        delta = _np.abs(prev_thumb - thumb)
        if delta.ndim == 3:
            delta = delta.max(axis=-1)
        return float((delta > _DIFF_EPS).mean()) > _NULL_FRAC

    def observe(self, image) -> None:
        """Fold the frame now on screen into the ledgers: score the PREVIOUS decision's outcome,
        then make this frame the current view. Called once per step before the model is asked."""
        arr = self._array(image)
        view, thumb = self._fingerprint(image, arr)
        self._stair_was = self.stair
        self.stair, self.exit = self._census(arr)
        prev = self.last
        if prev is not None:
            moved = self._screen_changed(prev["view"], self._thumb, view, thumb)
            name = prev["name"]
            if name in _act.MOVEMENT_ACTIONS and not moved:
                self.refused.setdefault(prev["view"], set()).add(name)
                self.since_move += 1
            elif name in _act.MOVEMENT_ACTIONS:
                self.moves += 1
                self.since_move = 0
                self.sweep = False              # a real step: hand control back to the model
                self.sweep_turns = 0
                # §4: a saturated stair tile that vanished under a step we really took is now
                # under our feet. Any other real step takes us off whatever we were standing on.
                self.on_stair = (self._stair_was >= _ADJACENT_FRAC
                                 and self.stair < _SIGHT_FRAC)
            else:
                self.since_move += 1
        self.view, self._thumb = view, thumb
        self.views[view] = self.views.get(view, 0) + 1
        self.step += 1
        if prev is not None and prev.get("grounded"):
            # We have just arrived on the far side of a floor transition, and THIS view is the
            # arrival tile. Travelling the same direction again from the tile you just landed on is
            # the one climb that cannot pay, so spend it here: that leaves the RETURN as the only
            # unspent transition at this stair, which is what makes the up/down pair alternate
            # instead of repeating, with each leg still gated on the new floor going stale.
            self.spent.setdefault((self.floor_epoch, view), set()).add(prev["name"])
        # floor staleness: a view never before shown ON THIS FLOOR is progress, anything else is not
        key = (self.floor_epoch, view)
        if key in self.seen_on_floor:
            self.stale_run += 1
        else:
            self.seen_on_floor.add(key)
            self.stale_run = 0
        if self.since_move >= self.stall_limit:
            self.sweep = True

    # ── the grounded intent ────────────────────────────────────────────────────────────────
    def floor_done(self) -> bool:
        """Has walking around this floor stopped paying? `STALE_LIMIT` consecutive decisions
        without a single view this floor had not already shown us."""
        return self.stale_run >= self.STALE_LIMIT

    def _unspent_climb(self):
        """The floor transition not yet executed from this exact screen on this floor, or None.
        Keyed on (floor_epoch, view) rather than the view alone because a climb can land on a
        byte-identical screen one floor up -- a view-only key would call the arrival 'already
        climbed' and only ever offer the return."""
        done = self.spent.get((self.floor_epoch, self.view), ())
        for want in ("climb_up", "climb_down"):
            if want not in done:
                return want
        return None

    def _aim(self):
        """(action, granularity, why) when the grounding has a strictly better-evidenced move than
        whatever was proposed, else None. Ordered by how decisive the evidence is: closing the exit
        ends the episode, so it outranks everything."""
        refused = self.refused.get(self.view, set())
        if self.exit >= _ADJACENT_FRAC and "move_forward" not in refused:
            return ("move_forward", "small", "the gold EXIT tile is the very next cell dead ahead")
        if not self.floor_done():
            return None                         # this floor still has somewhere new to go
        if self.on_stair:
            want = self._unspent_climb()
            if want and self.climbs < self.CLIMB_BUDGET:
                return (want, None, "a stair is measured under your feet and this floor has "
                                    "stopped showing you anywhere new")
        if self.stair >= _ADJACENT_FRAC and "move_forward" not in refused:
            return ("move_forward", "small", "the STAIR tile is the very next cell dead ahead and "
                                             "this floor has stopped showing you anywhere new")
        return None

    # ── the governor ───────────────────────────────────────────────────────────────────────
    def govern(self, name: str):
        """(action_name, granularity_or_None, reason). reason == "" means the proposal is obeyed.

        Two kinds of displacement, in order. First the GROUNDING's own move, when the pixels
        evidence one (`_aim`). Then the refusal vetoes: a translation this exact view already
        refused, a climb with no stair under us once the probe allowance is gone, or a turn chosen
        while the loop has stalled. Either way the replacement is computed FIRST and the model is
        obeyed whenever it already agrees with it, so agreement costs nothing."""
        aim = self._aim()
        if aim is not None:
            if aim[0] == name:
                return (name, None, "")                 # the model already wants it
            return (aim[0], aim[1], aim[2])
        refused = self.refused.get(self.view, set())
        if name in _act.CLIMB_ACTIONS and not self.on_stair:
            if self.probes < self.CLIMB_PROBES:
                return (name, None, "")                 # unevidenced, but the model may try it
            why = ("`%s` has already been spent from this floor with no stair measured under you; "
                   "it changes nothing" % name)
        elif name in _act.MOVEMENT_ACTIONS and name in refused:
            why = "`%s` already left this exact screen unchanged" % name
        elif self.sweep and name in _act.TURN_ACTIONS:
            why = "%d decisions without moving" % self.since_move
        else:
            return (name, None, "")
        want, want_gran, how = self._reaim()
        if want == name:
            return (name, None, "")                 # the model already wants the escape step
        return (want, want_gran, why + "; " + how)

    def _reaim(self):
        """The next step of the escape sweep: a RIGHT-HAND WALL FOLLOWER, which is the part that has
        to be got right. The rule is "probe the way ahead; if it is refused rotate to the next
        bearing", and the ORDER of those bearings decides whether it escapes or orbits. Arriving on
        heading h, the cell behind is h+2, so the bearings are tried h+1 (right), h, h-1, h+2 —
        i.e. ONE right turn on arrival and LEFT turns from then on, which leaves the came-from
        bearing for last. Rotating always the same way instead tries h+2 SECOND and walks straight
        back out of every dead end, which is a 2-cell orbit: measured at 2 cells visited in 100
        decisions on the offline rig, against 25+ for this order.

        A bearing whose forward is already in the refused ledger is rotated past without spending a
        probe on it. `sweep_turns` is the rotation count since the last successful move, so it is
        also the "have I arrived" flag: 0 means a fresh cell."""
        self.sweep = True
        refused = self.refused.get(self.view, set())
        if "move_forward" not in refused:
            return ("move_forward", "small", "probing the way ahead")
        if self.sweep_turns >= 4 and "move_backward" not in refused:
            self.sweep_turns = 0                    # guard: every bearing refused (a sealed cell)
            return ("move_backward", "small", "every bearing here is refused, stepping back")
        first = self.sweep_turns == 0
        self.sweep_turns += 1
        turn = "turn_right" if first == (self.turn_sense > 0) else "turn_left"
        return (turn, None, "rotating %s to the next untried bearing"
                % ("right" if turn == "turn_right" else "left"))

    def apply(self, proposal):
        """Govern one proposal and return the engine Action to execute."""
        name = _act.name_for(getattr(proposal, "key", None))
        meta = getattr(proposal, "metadata", None) or {}
        gran = meta.get("granularity")
        if name is None:                                     # unmapped key: pass it through untouched
            self._commit(None, None, None)
            return proposal
        chosen, chosen_gran, why = self.govern(name)
        if not why:
            # obeyed — but never let a long stride carry us PAST a tile we have decided to land on
            if (name == "move_forward" and gran not in (None, "small")
                    and max(self.stair, self.exit) >= _ADJACENT_FRAC):
                self.overrides += 1
                out = _act.retarget(proposal, name, "small",
                                    "the tile you want is the very next cell: one cell at a time",
                                    name)
                self._commit(name, "small", None)
                return out
            self._commit(name, gran, None)
            return proposal
        self.overrides += 1
        out = _act.retarget(proposal, chosen, chosen_gran or "small", why, name)
        self._commit(chosen, chosen_gran, name)
        return out

    def _commit(self, name, gran, overridden) -> None:
        grounded = bool(name in _act.CLIMB_ACTIONS and self.on_stair)
        self.last = {"view": self.view, "name": name, "gran": gran, "overridden": overridden,
                     "step": self.step, "grounded": grounded}
        if not name:
            return
        self.tried.setdefault(self.view, set()).add(name)
        if name not in _act.CLIMB_ACTIONS:
            return
        self.spent.setdefault((self.floor_epoch, self.view), set()).add(name)
        if grounded:
            # A SUCCESSFUL CLIMB CAN LEAVE THE FRAME BYTE-IDENTICAL, so there is no frame verdict to
            # wait for: a climb the pixels justified is presumed to have happened. Bumping the epoch
            # re-opens every view as "new on this floor", which resets the staleness counter — and
            # that is the anti-ping-pong lock: the new floor has to go stale before the return is
            # allowed. The arrival tile of a legal climb is itself a stair, so `on_stair` stands.
            self.climbs += 1
            self.floor_epoch += 1
            self.probes = 0
            self.stale_run = 0
        else:
            self.probes += 1            # unevidenced: presumed a no-op, and budgeted as one

    # ── the per-turn memory block ──────────────────────────────────────────────────────────
    def _distance_word(self, frac: float) -> str:
        """The measured rungs of §3 in words. Only the adjacent rung is stated as a cell count:
        it is the one separated by a factor of four from its neighbour and the one verified
        end-to-end. The rest is reported as a band, because an off-axis tile shrinks the paint
        the same way distance does and the two were not separately calibrated."""
        if frac >= _ADJACENT_FRAC:
            return "is the VERY NEXT CELL, dead ahead"
        if frac >= _NEAR_FRAC:
            return "is close - a few cells away down this line of sight"
        if frac >= _SIGHT_FRAC:
            return "is far off, or off to one side"
        return ""

    def _sight_line(self) -> str:
        bits = []
        w = self._distance_word(self.exit)
        if w:
            bits.append("the GOLD EXIT tile %s" % w)
        w = self._distance_word(self.stair)
        if w:
            bits.append("a CYAN STAIR tile %s" % w)
        if not bits:
            return ("no stair and no exit tile is painted anywhere in this view (remember the tile "
                    "under your own feet is never drawn)")
        return "; ".join(bits)

    def _floor_line(self) -> str:
        where = ("a STAIR IS UNDER YOUR FEET (measured: a saturated stair tile vanished as you "
                 "stepped onto it), so a climb will work from here"
                 if self.on_stair else
                 "no stair is measured under your feet, so `climb_up`/`climb_down` would do nothing")
        state = ("this floor has gone %d decisions without showing you anywhere new, so changing "
                 "floor is now the best move available" % self.stale_run if self.floor_done() else
                 "this floor still has somewhere new to go (%d/%d decisions without a new view)"
                 % (self.stale_run, self.STALE_LIMIT))
        return "%s. %s. Floor changes taken so far: %d." % (where, state, self.climbs)

    def _last_line(self) -> str:
        prev = self.last
        if not prev or not prev.get("name"):
            return "none yet - this is your first decision."
        name = prev["name"]
        moved = self.since_move == 0 and name in _act.MOVEMENT_ACTIONS
        if prev.get("overridden"):
            ran = "you proposed `%s`, which the screens had already settled, so `%s` was executed " \
                  "instead" % (prev["overridden"], name)
        else:
            ran = "`%s` was executed as you asked" % name
        if name in _act.MOVEMENT_ACTIONS:
            ran += " and the screen %s" % ("changed, so you really moved" if moved else
                                           "did NOT change at all, so something solid blocked it")
        elif name in _act.CLIMB_ACTIONS:
            # NEVER report a climb as simply "executed": a climb that worked can leave the frame
            # byte-identical, and a climb with no stair under it does nothing at all.
            ran += (" from a tile measured to be a stair, so you have changed floor"
                    if prev.get("grounded") else
                    " but no stair was measured under your feet, so nothing happened")
        return ran + "."

    def _mode_line(self) -> str:
        if not self.sweep:
            return "FOLLOWING YOU - your proposal is executed unless the screens have settled it."
        return ("ESCAPING - the way out of this pocket is being searched by turning to the next "
                "bearing this screen has not refused and stepping forward whenever the way opens. "
                "Propose the turn or the step that agrees with that and it will be executed as "
                "asked.")

    def fields(self) -> dict:
        refused = sorted(self.refused.get(self.view, set()))
        tried = self.tried.get(self.view, set())
        # climbs are offered as "untried" only where a stair is actually measured underfoot;
        # listing them at every fresh view is what invited the refused-climb loops.
        legal = [a for a in _act.LEGAL_ACTIONS
                 if a not in _act.CLIMB_ACTIONS or self.on_stair]
        untried = [a for a in legal if a not in tried]
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "view_visits": self.views.get(self.view, 1),
            "last_line": self._last_line(),
            "since_move": self.since_move,
            "blocked_here": ", ".join("`%s`" % a for a in refused) or "nothing proven yet",
            "untried_here": ", ".join("`%s`" % a for a in untried) or "none - all of them tried",
            "sight_line": self._sight_line(),
            "floor_line": self._floor_line(),
            "mode_line": self._mode_line(),
            "moves": self.moves,
            "overrides": self.overrides,
            "climbs": self.climbs,
        }

    def render_memory(self) -> str:
        """The block tools/vlm.py appends to the agent's system turn. Never raises."""
        flds = self.fields()
        try:
            return self.template.format(**flds)
        except Exception:
            try:
                return DEFAULT_TEMPLATE.format(**flds)
            except Exception:
                return ""


def _clean_template(text: str) -> str:
    """The renderable part of memory.md: everything after the last HTML comment (the file's header
    documents the template for the next proposer and must not reach the agent). Returns "" when the
    file has no renderable body or its placeholders do not match `fields()`."""
    if not text:
        return ""
    body = text.split("-->")[-1].strip() if "-->" in text else text.strip()
    if "{step}" not in body:
        return ""
    return body
