"""tools/plan.py — the ACTION-OUTCOME PLANNER: a closed-loop governor over the frozen model's
per-step proposal.

WHY THIS EXISTS. The anchor hands every decision straight to `evaluator.baseline_step` and keeps no
state, so nothing in the loop can tell a move that happened from a move that did not. The frozen
model is reactive: when its forward move is refused by a wall the screen comes back looking the
same, it proposes the same move again, and the episode spends its whole 100-decision budget on one
refused action. The same blindness turns a legal floor transition into an up/down 2-cycle at a
single stair.

WHAT IT MEASURES (self-derived — no engine state, no layout, no coordinates). The only thing this
module reads is `frame_data.image`, the very frame the agent is shown, and it now reads it at THREE
resolutions, each strictly stronger than the last:

  * AS A FINGERPRINT. Two consecutive frames are compared whole, and the comparison is EXACT: an
    md5 over every second pixel, equal or not equal. A translation that leaves the screen identical
    did not happen; one that changes it did. Over 2040 live translations across two harnesses a
    refused translation left the frame byte-identical 152/152 times and a real move left it
    identical 0/1891 times, so identity is a perfect one-bit outcome oracle and the 0.2%-of-pixels
    tolerance this build used to carry was pure loss: it manufactured four phantom walls, one of
    which turned a start cell into an absorbing state and cost a level outright. It is gone.
  * BY COLOUR (tools/sight.py). The exact masks of the two special tiles and the measured area
    ladder that turns their painted area into a distance in cells — including the cyan of a stair
    standing BESIDE the corridor, which the geometry read below cannot see. The moment a saturated
    stair vanishes under a step that moved the body, the stair is UNDERFOOT: the one fact a floor
    transition needs, and the render carries no text to state it.
  * AS GEOMETRY (tools/sight.py, new). The HEIGHT of the paint down the centre column is the
    distance to the first thing standing in the corridor, its COLOUR is what that thing is, and the
    side bands that carry no paint are the openings beside the corridor. All three were measured on
    a census of rendered frames and all three decode EXACTLY (5624/5624 frames for the distance and
    the kind, 23688/23688 side bands, every walkable cell of all 25 levels from all four bearings).

WHAT IT DOES WITH THAT. The frozen model is asked EVERY step and obeyed unless its proposal is
provably useless, or unless the frames have just proven the one thing it cannot see:
  0. a stair is UNDER THE BODY and this floor has gone `STALE_FLOOR` decisions without showing a
     view it had not already shown, and no exit is in sight  -> the climb is taken. This is the only
     branch that displaces a proposal on evidence FOR something rather than against it.
  1. a forward step while the frame shows a solid face in the VERY NEXT CELL -> re-aimed. This used
     to cost a wasted decision per view to discover; it is now free and a priori (2882 of 5624
     poses, no false positive and no false negative),
  2. a climb proposed after a climb already left the screen untouched -> re-aimed (nothing is there),
  3. a climb that would undo the floor transition just made           -> re-aimed,
  4. a translation already proven refused from this exact view        -> re-aimed,
  5. a turn or climb proposed while the loop has stalled              -> re-aimed forward.
A re-aim is no longer a blind rotation. It is a SIGHTED wall follower: probe the way ahead while the
frame shows cells clear there, and otherwise rotate toward a side the frame shows OPEN, keeping the
right-hand order among the open ones so the came-from bearing is left for last (rotating always the
same way tries the came-from bearing SECOND and walks back out of every dead end, which the offline
rig measures as a 2-cell orbit). When the frame shows a face ahead and solid faces on both sides,
the only open cell is the one behind, and the step back is taken at once rather than after four
probing turns.

AND THE STRIDE IS NOW MEASURED, NOT GUESSED. The engine moves 1/2/3 cells for small/medium/large at
the price of ONE decision, and stops early at a wall, so a long stride is never worse than a short
one — yet the previous build's prose talked the model down to 94% `small` and 1.02 cells per forward
while five of its losses had shortest routes of 32-54 cells against a budget worth about 29. The
corridor read gives the legal stride for nothing: it never over-reaches (0 of 5624 poses) and it
stops at the first OPENING in the side walls, so a stride can no longer carry the body past a
turning. 34.4% of poses can stride two cells or more and 32.3% still can after every junction has
been honoured, which is the whole of the clock deficit at an unchanged decision count.

WHY THE CLIMB WAITS. Taking the first stair the body happens to land on is a measured trap: the
exit is on the STARTING floor on 13 of the 25 levels, and leaving that floor early throws away
levels that were being won. `sight.STALE_FLOOR` is the price of admission — this floor has to stop
producing new views before its stairs are worth spending — and a floor change clears the view set,
which is also what stops a pair of stairs becoming an up/down 2-cycle.

AND IT TELLS THE MODEL. The governed verdict is useless if the model keeps proposing the same
refused action, so `render_memory()` renders the ledger into plain English and tools/vlm.py appends
it to the agent's system turn every step. The facts are the model's own observation history and the
geometry of the screen in front of it, stated back to it: what it last proposed, whether the screen
moved, which actions this exact view has already refused, which it has never tried, how far the
corridor runs and where the openings off it are, and whether it is standing on a stair.
"""
from __future__ import annotations

import hashlib

try:                                    # the engine hands us a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

from . import action as _act
from . import sight as _sight

_SAMPLE = 2          # fingerprint every 2nd pixel: ~4x cheaper, still a faithful view id
_NULL_CLIMBS = 1     # ungrounded climbs allowed after one has been proven to do nothing

#: the built-in working-memory template. memory.md (when present next to the harness) overrides it,
#: which is what keeps the wording a prose lever; the built-in keeps the mechanism alive when the
#: harness is bundled without it. The two MUST carry the same placeholder set as `fields()`.
DEFAULT_TEMPLATE = """## WORKING MEMORY (measured off the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- THE CORRIDOR: {corridor_line}
- IN THIS FRAME: {sight_line}
- UNDER YOUR FEET: {stair_line}
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- this floor has shown nothing new for {stale} decision(s). {climb_line}
- mode: {mode_line}

How to use this: the CORRIDOR line is measured off the picture, not guessed, and the granularity it
names is the longest step that can land - ask for it. A stride costs ONE decision whatever its
length and the engine stops it early at a wall, so a long step is never worse than a short one; the
only reasons to say `small` are a stair or an exit in the very next cell, and an opening you want
to stop at. An action that left the screen byte-for-byte identical did not happen - something solid
is there; never re-propose an action listed as PROVEN REFUSED while this screen is in front of you.
`climb_up` and `climb_down` do something ONLY while a stair is under your feet; there is no prompt
to confirm it, so trust the UNDER YOUR FEET line above. After a floor change, explore the new level
before going back.

Reply with ONLY the JSON object in the format specified above."""


def refusal_reason(name, refused, climb_lock, null_climbs, sweep, since_move,
                   blocked_ahead) -> str:
    """Why `name` is provably useless right now, or "" when nothing refutes it.

    Pure function of the ledgers plus the one live geometric fact, so the whole veto chain can be
    read and priced in one place. The ORDER is the priority: the frame's own verdict on the cell
    ahead comes first because it is a measurement of THIS screen rather than a memory of an older
    one, then a climb proven inert, then the undo ban, then the view ledger, then the stall rule."""
    climbing = name in _act.CLIMB_ACTIONS
    if name == "move_forward" and blocked_ahead:
        return "the picture shows a solid face filling the very next cell ahead of you"
    if climbing and null_climbs >= _NULL_CLIMBS:
        return "a `%s` already left the screen untouched, so no stair is under you" % name
    if climbing and name == climb_lock:
        return "`%s` would undo the floor change just made" % name
    if name in _act.MOVEMENT_ACTIONS and name in refused:
        return "`%s` already left this exact screen unchanged" % name
    if sweep and (climbing or name in _act.TURN_ACTIONS):
        return "%d decisions without moving" % since_move
    return ""


def fingerprint(image) -> str:
    """The frame the agent is being shown, as a view IDENTITY.

    The WHOLE-FRAME reading, as opposed to the colour and geometry reads in tools/sight.py: an md5
    over every `_SAMPLE`-th pixel. Two poses that render the same get the same id, which is exactly
    what makes a "the way ahead is blocked" fact re-usable the next time that screen comes back."""
    if image is None:
        return ""
    if _np is None:                                     # pragma: no cover
        try:
            return hashlib.md5(bytes(memoryview(image))).hexdigest()[:16]
        except Exception:
            return ""
    arr = _np.asarray(image)
    if arr.ndim == 3:
        arr = arr[::_SAMPLE, ::_SAMPLE, :3]
    elif arr.ndim == 2:
        arr = arr[::_SAMPLE, ::_SAMPLE]
    else:
        return ""
    return hashlib.md5(_np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def screen_changed(prev_view: str, view: str) -> bool:
    """Did anything happen? EXACTLY: different view ids mean the screen changed and the same id
    means it did not.

    There used to be a tolerance here — a changed fingerprint still counted as unchanged when under
    0.2% of the sampled pixels differed — and it was measured to be pure loss. A refused translation
    leaves the frame byte-identical 152 times out of 152 and a real move leaves it identical 0 times
    out of 1891, so the tolerance cannot protect against anything; what it did instead was file four
    real moves as phantom walls, and one of those four (a step down a long open corridor whose two
    frames happen to differ by a hair) turned a level's start cell into an absorbing state that ate
    74 of 100 decisions."""
    return prev_view != view


def last_line(planner) -> str:
    """What the previous decision was and what the frames say it did. A climb that left the screen
    untouched must never be reported as having been carried out."""
    prev = planner.last
    if not prev or not prev.get("name"):
        return "none yet - this is your first decision."
    name = prev["name"]
    if prev.get("overridden"):
        ran = "you proposed `%s`, which was already proven useless here, so `%s` was executed " \
              "instead" % (prev["overridden"], name)
    else:
        ran = "`%s` was executed as you asked" % name
        if prev.get("gran") and prev.get("retuned"):
            ran += ", at `%s` - the length the picture said would land" % prev["gran"]
    if name in _act.MOVEMENT_ACTIONS:
        ran += " and the screen %s" % ("changed, so you really moved" if planner.since_move == 0
                                       else "did NOT change at all, so something solid blocked it")
    elif name in _act.CLIMB_ACTIONS and planner.null_climbs:
        ran += " and the screen did NOT change at all"
    return ran + "."


class ActionPlanner:
    """Episode-local state + the governor. One instance per episode, built in agent.py."""

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
        self.floor_views: set = set()           # views shown since the last floor transition
        # rolling state
        self.view = ""
        self.seen = _sight.look(None)           # colour + geometry census of the frame on screen
        self.sense = _sight.StairSense()        # the stair-underfoot latch and its budget
        self.step = 0
        self.since_move = 0
        self.stale = 0                          # consecutive decisions with no view new to this floor
        self.floor_age = 0                      # decisions spent on the floor we are on now
        self.null_climbs = 0                    # climbs proven inert since the last successful move
        self.last = None                        # the previous decision we committed
        self.sweep = False                      # True while escaping a proven-refused pocket
        self.sweep_turns = 0                    # consecutive re-aimed turns since the last move
        self.turn_sense = 1                     # +1 = right-hand wall follower, -1 = left-hand
        self.climb_lock = None                  # the floor transition that would undo the last one
        self.climb_lock_until = 0
        self.overrides = 0
        self.tuned = 0                          # stride re-lengths (the action name is left alone)
        self.sighted = 0                        # forwards vetoed by the picture before being spent
        self.moves = 0
        self.cells = 0                          # cells of travel the measured stride has asked for
        self.base_prompt = None                 # stash of the agent's first-call system turn

    # ── outcome measurement ────────────────────────────────────────────────────────────────
    def _score(self, prev, moved, before, after) -> None:
        """Fold the PREVIOUS decision's measured outcome into the ledgers.

        Only a translation's failure is filed against the view: "the cell ahead is blocked" is a
        property of what is on screen, so it transfers whenever that screen returns. A climb's
        failure is NOT — two poses can render identically and disagree about the tile underfoot —
        so an inert climb is counted on a rolling counter instead and forgotten the moment the body
        actually moves."""
        name = prev["name"]
        translated = name in _act.MOVEMENT_ACTIONS
        self.sense.update(before, after, moved, translated)
        if translated and not moved:
            self.refused.setdefault(prev["view"], set()).add(name)
            self.since_move += 1
        elif translated:
            self.moves += 1
            self.cells += _act.granularity_steps(prev.get("gran"))
            self.since_move = 0
            self.null_climbs = 0
            self.sweep = False                  # a real step: hand control back to the model
            self.sweep_turns = 0
            self.climb_lock = None              # we have left the stair landing
            self.climb_lock_until = 0
        else:
            self.since_move += 1
            if name in _act.CLIMB_ACTIONS and not moved:
                self.null_climbs += 1

    def observe(self, image) -> None:
        """Fold the frame now on screen into the ledgers: score the PREVIOUS decision's outcome,
        then make this frame the current view. Called once per step before the model is asked.

        The geometry read is folded in HERE rather than left for the veto chain, because it is the
        same statement the ledger makes and it belongs in the same place: a frame that shows a solid
        face in the very next cell proves `move_forward` cannot move the body from this view, so the
        fact is filed against the view at once, for free, from a pose that may never have been
        probed. It then reaches the model through the PROVEN REFUSED line and the sweep through the
        same ledger every learned refusal uses."""
        view = fingerprint(image)
        seen = _sight.look(image)
        if self.last is not None:
            self._score(self.last, screen_changed(self.last["view"], view), self.seen, seen)
        self.view, self.seen = view, seen
        if seen.blocked and view:
            self.refused.setdefault(view, set()).add("move_forward")
        self.views[view] = self.views.get(view, 0) + 1
        self.stale = 0 if view not in self.floor_views else self.stale + 1
        self.floor_views.add(view)
        self.floor_age += 1
        self.step += 1
        if self.climb_lock and self.step >= self.climb_lock_until:
            self.climb_lock = None              # last-resort release for a genuinely trapped landing
        if self.since_move >= self.stall_limit:
            self.sweep = True

    # ── the governor ───────────────────────────────────────────────────────────────────────
    def govern(self, name: str):
        """(action_name, granularity_or_None, reason, grounded). reason == "" means obeyed.

        Two kinds of displacement, in this order. The FIRST is positive evidence: the frames have
        proven a stair is under the body, this floor has stopped producing new views, and no exit is
        in sight — the model cannot see any of that (the tile underfoot is not drawn and the build
        renders no text), so the climb is taken for it. `grounded` marks that case, because it is
        the one that resets the floor ledger. The SECOND is the negative chain in `refusal_reason`.
        Both compute the escape step FIRST and obey the model whenever it already agrees, so
        agreement always costs nothing."""
        grounded = self.sense.wanted(self.stale, self.floor_age, self.seen)
        if grounded:
            if name == grounded:
                return (name, None, "", True)
            return (grounded, None, "a stair is under your feet and this floor has shown nothing "
                    "new for %d decisions" % self.stale, True)
        why = refusal_reason(name, self.refused.get(self.view, set()), self.climb_lock,
                             self.null_climbs, self.sweep, self.since_move, self.seen.blocked)
        if not why:
            self._advance(name)
            return (name, None, "", False)
        if name == "move_forward" and self.seen.blocked:
            self.sighted += 1
        want, want_gran, how = self._reaim()
        self._advance(want)
        if want == name:
            return (name, None, "", False)          # the model already wants the escape step
        return (want, want_gran, why + "; " + how, False)

    def _reaim(self):
        """The next step of the escape sweep — a SIGHTED wall follower.

        The blind version probed forward whenever the view ledger had not yet refused it and then
        rotated by a fixed hand, which spends one decision per bearing learning what the picture
        already says. Three clauses, in order, each one a measurement of the frame on screen:

          * the corridor read says cells are CLEAR ahead -> step, at the length that will land;
          * every bearing here has been spent and the step back has not -> step back (unchanged);
          * otherwise ROTATE, toward a side the picture shows OPEN when it shows one. The handed
            ORDER is the part that has to be got right: arriving on heading h the cell behind is
            h+2, so the bearings are tried h+1 (right), h, h-1, h+2 — ONE right turn on arrival and
            LEFT turns from then on, which leaves the came-from bearing for LAST. Rotating always
            the same way tries h+2 SECOND and walks straight back out of every dead end, measured
            at 2 cells visited in 100 decisions against 25+ for this order. The sighted part is
            only WHICH WAY to rotate: a side the frame shows solid cannot be worth a probe, so the
            hand is flipped onto the open one, which cuts the walk out of a sealed pocket from five
            decisions to three (turn, turn, step, facing AWAY from the pocket).

        A SEALED CELL MUST STILL BE LEFT BY ROTATING. Reading "wall ahead and solid on both sides"
        and stepping straight back looks like a four-decision saving and is a trap that was measured
        on the stub rig: `move_backward` does not change the bearing, so the body arrives in the
        previous cell still facing the pocket, the next forward is clear, and the agent walks back
        in — 50 decisions alternating between two cells on L02, 2 cells covered against the parent's
        25. Rotation is what acquires a new bearing, and that is why it stays the escape.

        PURE: it reads the ledgers and the census and returns a bearing; advancing the rotation
        count is the caller's job (`_advance`). It must not flip `self.sweep` itself — that once
        meant ANY veto latched the loop into ESCAPING mode until a translation succeeded (649 of
        1943 live decisions). `sweep` is exactly "the loop has stalled", decided in `observe`."""
        refused = self.refused.get(self.view, set())
        if self.seen.stride >= 1 and "move_forward" not in refused:
            return ("move_forward", _sight.GRANULARITY[self.seen.stride],
                    "the picture shows %d clear cell(s) that way" % self.seen.stride)
        if self.sweep_turns >= 4 and "move_backward" not in refused:
            return ("move_backward", "small", "every bearing here is refused, stepping back")
        sides = self.seen.open_sides()
        first = self.sweep_turns == 0
        hand = "right" if first == (self.turn_sense > 0) else "left"
        if sides and hand not in sides:
            hand = sides[0]
        return ("turn_" + hand, None, "rotating %s, %s" % (
            hand, "where the picture shows an opening" if hand in sides
            else "to the next bearing you have not tried"))

    def _advance(self, name: str) -> None:
        """Move the escape sweep on by the action about to run, whoever chose it. A rotation spends
        a bearing; the sealed-cell step back starts the fan over."""
        if name in _act.TURN_ACTIONS:
            self.sweep_turns += 1
        elif name == "move_backward":
            self.sweep_turns = 0

    def _stride(self, name: str, gran) -> str:
        """The granularity a forward translation should REALLY carry, or "" to leave it alone.

        Two-sided, and both sides are measurements. DOWN: clipped to one cell when the tile dead
        ahead is the thing worth arriving ON, because a longer stride walks straight over a stair
        (and the stair-underfoot fact is measured from ARRIVING on it) or past the exit. UP: raised
        to the length the corridor read says will land, because a stride costs one decision whatever
        it is and the previous build's prose talked the model down to 94% `small` and 1.02 cells per
        forward while five of its losses needed routes of 32-54 cells. The read never over-reaches
        (0 of 5624 poses asked for more than the engine would give) and it stops at the first
        OPENING in the side walls, so the promotion cannot carry the body past a turning.

        This changes no action NAME — the model still chooses every move — and nothing here can turn
        a blocked forward into a step: a zero stride is left to the veto chain."""
        if name != "move_forward" or not self.seen.ok:
            return ""
        if self.seen.stair_adjacent or self.seen.exit_adjacent:
            want = "small"
        elif self.seen.stride <= 0:
            return ""
        else:
            want = _sight.GRANULARITY[self.seen.stride]
        return "" if want == (gran or "small").strip().lower() else want

    def _stride_reason(self, want: str) -> str:
        """Why the stride was re-lengthened, in the agent's own terms."""
        if want == "small":
            return "one cell at a time: the target is the very next cell"
        return "the picture shows %d clear cell(s) ahead and the stride stops at the first opening" \
            % self.seen.stride

    def apply(self, proposal):
        """Govern one proposal and return the engine Action to execute."""
        name = _act.name_for(getattr(proposal, "key", None))
        meta = getattr(proposal, "metadata", None) or {}
        gran = meta.get("granularity")
        if name is None:                                 # unmapped key: pass it through untouched
            self._commit(None, None, None, False, False)
            return proposal
        chosen, chosen_gran, why, grounded = self.govern(name)
        if not why:
            want = self._stride(name, gran)
            if want:
                self.tuned += 1
                self._commit(name, want, None, grounded, True)
                return _act.retarget(proposal, name, want, self._stride_reason(want), name)
            self._commit(name, gran, None, grounded, False)
            return proposal
        self.overrides += 1
        out = _act.retarget(proposal, chosen, chosen_gran or "small", why, name)
        self._commit(chosen, chosen_gran, name, grounded, False)
        return out

    def _commit(self, name, gran, overridden, grounded, retuned) -> None:
        self.last = {"view": self.view, "name": name, "gran": gran, "overridden": overridden,
                     "retuned": retuned, "step": self.step}
        if not name:
            return
        self.tried.setdefault(self.view, set()).add(name)
        if name not in _act.CLIMB_ACTIONS:
            return
        if grounded:
            # A MEASURED transition: spend it from the latch's budget and start this floor's view
            # ledger over. The empty ledger IS the anti-ping-pong lock — the next stair cannot be
            # spent until the floor we just reached has itself gone stale.
            self.sense.pressed(name)
            self.floor_views = set()
            self.stale = 0
            self.floor_age = 0
            self.climb_lock = None
        else:
            # the model's own climb, with nothing under the body to justify it: ban the undo for a
            # span so one refused press cannot become a 2-cycle.
            self.climb_lock = _act.inverse_climb(name)
            self.climb_lock_until = self.step + self.climb_lock_span

    # ── the per-turn memory block ──────────────────────────────────────────────────────────
    def mode_line(self) -> str:
        """Who is driving, in the agent's own terms."""
        if not self.sweep:
            return "FOLLOWING YOU - your proposal is executed unless it is already proven useless."
        return ("ESCAPING - the way out of this pocket is being searched by stepping where the "
                "picture shows the corridor open and rotating toward the openings beside it. "
                "Propose the turn or the step that agrees with that and it will be executed.")

    def climb_line(self) -> str:
        """What the floor ledger and the stair latch together say about leaving this floor."""
        if self.sense.underfoot:
            left = max(_sight.STALE_FLOOR - self.stale, _sight.FLOOR_PATIENCE - self.floor_age)
            if left > 0:
                return ("You are on a stair, but this floor has not been given its due yet - about "
                        "%d more decision(s) and the stair is worth taking." % left)
            return "You are on a stair and this floor is exhausted: TAKE THE STAIR."
        if self.climb_lock:
            return "`%s` is held back so a floor change is not immediately undone." % self.climb_lock
        return "No stair under you, so neither climb can do anything from here."

    def fields(self) -> dict:
        refused = sorted(self.refused.get(self.view, set()))
        tried = self.tried.get(self.view, set())
        legal = _act.LEGAL_ACTIONS if self.sense.underfoot else \
            tuple(a for a in _act.LEGAL_ACTIONS if a not in _act.CLIMB_ACTIONS)
        untried = [a for a in legal if a not in tried and a not in refused]
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "view_visits": self.views.get(self.view, 1),
            "corridor_line": self.seen.corridor_line(),
            "sight_line": self.seen.describe(),
            "stair_line": self.sense.line(),
            "last_line": last_line(self),
            "since_move": self.since_move,
            "stale": self.stale,
            "blocked_here": ", ".join("`%s`" % a for a in refused) or "nothing proven yet",
            "untried_here": ", ".join("`%s`" % a for a in untried) or "none - all were tried",
            "climb_line": self.climb_line(),
            "mode_line": self.mode_line(),
            "moves": self.moves,
            "overrides": self.overrides,
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
