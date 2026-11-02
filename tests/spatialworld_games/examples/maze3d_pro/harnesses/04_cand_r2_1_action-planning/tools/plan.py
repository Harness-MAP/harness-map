"""tools/plan.py — the ACTION-OUTCOME PLANNER: a closed-loop governor over the frozen model's
per-step proposal.

WHY THIS EXISTS. The anchor hands every decision straight to `evaluator.baseline_step` and keeps no
state, so nothing in the loop can tell a move that happened from a move that did not. The frozen
model is reactive: when its forward move is refused by a wall the screen comes back looking the same,
it proposes the same move again, and the episode spends its whole 100-step budget on one refused
action. The same blindness turns a legal floor transition into an up/down 2-cycle at a single stair.

WHAT IT MEASURES (self-derived — no engine state, no layout, no coordinates). The only thing this
module reads is `frame_data.image`, the very frame the agent is shown, and it now reads it at TWO
resolutions:

  * AS A FINGERPRINT. Two consecutive frames are compared whole. A translation that leaves the
    screen UNCHANGED did not happen -> the cell ahead is a wall. The engine stops a multi-cell
    stride early at a wall, so "zero cells moved" means the FIRST cell was refused, whatever
    granularity was asked for. That verdict is recorded against the fingerprint, which is the
    agent's whole first-person view: whether something blocks the way ahead is a property of that
    view, so the ledger is exactly re-usable the next time the same view comes back (including after
    a four-turn circuit of the same cell). A translation that CHANGES the screen did happen.
    Equality is deliberately tolerant: identical fingerprints mean unchanged, and a changed
    fingerprint still counts as unchanged when under `_NULL_FRAC` of the sampled pixels differ.
  * BY COLOUR (tools/sight.py). The exact masks of the two special tiles, and the measured area
    ladder that turns their painted area into a distance in cells. This is what the fingerprint
    could never give: the frame says WHERE the stair and the exit are, and the moment a saturated
    stair vanishes under a step that moved the body, it says the stair is UNDERFOOT. That is the
    one fact a floor transition needs, and the render carries no text to state it.

WHAT IT DOES WITH THAT.  The frozen model is asked EVERY step and is obeyed unless its proposal is
provably useless, or unless the frames have just proven the one thing it cannot see:
  0. a stair is UNDER THE BODY and this floor has gone `STALE_FLOOR` decisions without showing a
     view it had not already shown, and no exit is in sight  -> the climb is taken. This is the only
     branch that displaces a proposal on evidence FOR something rather than against it.
  1. a climb proposed after a climb already left the screen untouched -> re-aimed (nothing is there),
  2. a climb that would undo the floor transition just made           -> re-aimed,
  3. a translation already proven refused from this exact view        -> re-aimed,
  4. a turn or climb proposed while the loop has stalled              -> re-aimed forward.
A re-aim is not a random nudge: it is one step of a right-hand wall follower (probe the way ahead;
if it is refused rotate to the next bearing in the order right, straight, left, back). Leaving the
came-from bearing for LAST is the whole of it — a sweep that rotates always the same way tries the
bearing it just came from second and walks straight back out of every dead end, which the offline
rig measures as a 2-cell orbit. Nothing the current view has already refused is ever re-probed, and
the sweep is released the moment a translation succeeds, so the model drives again.

WHY THE CLIMB WAITS. Taking the first stair the body happens to land on is a measured trap: the exit
is on the STARTING floor on a large minority of levels, and leaving that floor early throws away
levels that were being won. `sight.STALE_FLOOR` is the price of admission — this floor has to stop
producing new views before its stairs are worth spending — and a floor change clears the view set,
which is also what stops a pair of stairs becoming an up/down 2-cycle.

AND IT TELLS THE MODEL. The governed verdict is useless if the model keeps proposing the same
refused action, so `render_memory()` renders the ledger into plain English and tools/vlm.py appends
it to the agent's system turn every step. The facts are the model's own observation history, stated
back to it: what it last proposed, whether the screen moved, which actions this exact view has
already refused, which it has never tried, what the colours in this frame are, and whether it is
standing on a stair.
"""
from __future__ import annotations

import hashlib

try:                                    # the engine hands us a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

from . import action as _act
from . import sight as _sight

_SAMPLE = 2          # fingerprint/compare every 2nd pixel: ~4x cheaper, still a faithful view id
_DIFF_EPS = 8        # per-channel tolerance for "this pixel changed"
_NULL_FRAC = 0.002   # < 0.2% of sampled pixels changed == the screen did not change
_NULL_CLIMBS = 1     # ungrounded climbs allowed after one has been proven to do nothing

#: the built-in working-memory template. memory.md (when present next to the harness) overrides it,
#: which is what keeps the wording a prose lever; the built-in keeps the mechanism alive when the
#: harness is bundled without it. The two MUST carry the same placeholder set as `fields()`.
DEFAULT_TEMPLATE = """## WORKING MEMORY (measured off the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- IN THIS FRAME: {sight_line}
- UNDER YOUR FEET: {stair_line}
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- this floor has shown nothing new for {stale} decision(s). {climb_line}
- mode: {mode_line}

How to use this: an action that left the screen byte-for-byte identical did not happen - something
solid is there. Never re-propose an action listed as PROVEN REFUSED while this screen is in front of
you; turn to a bearing you have not tried yet from this screen and then move. A stair or an exit
that is the VERY NEXT CELL dead ahead is reached with one `move_forward` at `small`. `climb_up` and
`climb_down` do something ONLY while a stair is under your feet; there is no prompt to confirm it,
so trust the UNDER YOUR FEET line above. After a floor change, explore the new level before going
back.

Reply with ONLY the JSON object in the format specified above."""


def refusal_reason(name, refused, climb_lock, null_climbs, sweep, since_move) -> str:
    """Why `name` is provably useless right now, or "" when nothing refutes it.

    Pure function of the ledgers, so the whole veto chain can be read and priced in one place. The
    ORDER is the priority: a climb proven inert beats the undo ban, and both beat the stall rule."""
    climbing = name in _act.CLIMB_ACTIONS
    if climbing and null_climbs >= _NULL_CLIMBS:
        return "a `%s` already left the screen untouched, so no stair is under you" % name
    if climbing and name == climb_lock:
        return "`%s` would undo the floor change just made" % name
    if name in _act.MOVEMENT_ACTIONS and name in refused:
        return "`%s` already left this exact screen unchanged" % name
    if sweep and (climbing or name in _act.TURN_ACTIONS):
        return "%d decisions without moving" % since_move
    return ""


def fingerprint(image):
    """(fingerprint, downsampled copy) for the frame the agent is being shown.

    The WHOLE-FRAME reading, as opposed to the colour census in tools/sight.py: an md5 over every
    `_SAMPLE`-th pixel, which is a view identity — two poses that render the same get the same id,
    which is exactly what makes a "the way ahead is blocked" fact re-usable."""
    if image is None:
        return "", None
    if _np is None:                                     # pragma: no cover
        try:
            return hashlib.md5(bytes(memoryview(image))).hexdigest()[:16], None
        except Exception:
            return "", None
    arr = _np.asarray(image)
    if arr.ndim == 3:
        arr = arr[::_SAMPLE, ::_SAMPLE, :3]
    elif arr.ndim == 2:
        arr = arr[::_SAMPLE, ::_SAMPLE]
    else:
        return "", None
    arr = _np.ascontiguousarray(arr)
    return hashlib.md5(arr.tobytes()).hexdigest()[:16], arr.astype(_np.int16)


def screen_changed(prev_view, prev_thumb, view, thumb) -> bool:
    """Did anything happen? Identical ids mean no; otherwise the change has to clear `_NULL_FRAC` of
    the sampled pixels, so a one-pixel flicker cannot masquerade as a step while a real step — which
    repaints most of a first-person view — always registers."""
    if prev_view and view and prev_view == view:
        return False                                     # identical frame: nothing happened
    if prev_thumb is None or thumb is None or prev_thumb.shape != thumb.shape:
        return prev_view != view
    delta = _np.abs(prev_thumb - thumb)
    if delta.ndim == 3:
        delta = delta.max(axis=-1)
    return float((delta > _DIFF_EPS).mean()) > _NULL_FRAC


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
    if name in _act.MOVEMENT_ACTIONS:
        ran += " and the screen %s" % ("changed, so you really moved" if planner.since_move == 0
                                       else "did NOT change at all, so something solid blocked it")
    elif name in _act.CLIMB_ACTIONS and planner.null_climbs:
        ran += " and the screen did NOT change at all"
    return ran + "."


def mode_line(planner) -> str:
    """Who is driving, in the agent's own terms."""
    if not planner.sweep:
        return "FOLLOWING YOU - your proposal is executed unless it is already proven useless."
    return ("ESCAPING - the way out of this pocket is being searched by probing the way ahead "
            "and rotating to the next bearing it has not refused. Propose the turn or the step "
            "that agrees with that and it will be executed as asked.")


def climb_line(planner) -> str:
    """What the floor ledger and the stair latch together say about leaving this floor."""
    if planner.sense.underfoot:
        left = max(_sight.STALE_FLOOR - planner.stale,
                   _sight.FLOOR_PATIENCE - planner.floor_age)
        if left > 0:
            return ("You are on a stair, but this floor has not been given its due yet - about "
                    "%d more decision(s) and the stair is worth taking." % left)
        return "You are on a stair and this floor is exhausted: TAKE THE STAIR."
    if planner.climb_lock:
        return "`%s` is held back so a floor change is not immediately undone." % planner.climb_lock
    return "No stair under you, so neither climb can do anything from here."


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
        self._thumb = None
        self.seen = _sight.look(None)           # colour census of the frame now on screen
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
        self.tuned = 0                          # granularity trims (the action name is left alone)
        self.moves = 0
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
        then make this frame the current view. Called once per step before the model is asked."""
        view, thumb = fingerprint(image)
        seen = _sight.look(image)
        if self.last is not None:
            moved = screen_changed(self.last["view"], self._thumb, view, thumb)
            self._score(self.last, moved, self.seen, seen)
        self.view, self._thumb, self.seen = view, thumb, seen
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
        renders no text), so the climb is taken for it. `grounded` marks that case, because it is the
        one that resets the floor ledger. The SECOND is the old negative chain in `refusal_reason`.
        Both compute the escape step FIRST and obey the model whenever it already agrees, so
        agreement always costs nothing."""
        grounded = self.sense.wanted(self.stale, self.floor_age, self.seen)
        if grounded:
            if name == grounded:
                return (name, None, "", True)
            return (grounded, None, "a stair is under your feet and this floor has shown nothing "
                    "new for %d decisions" % self.stale, True)
        why = refusal_reason(name, self.refused.get(self.view, set()), self.climb_lock,
                             self.null_climbs, self.sweep, self.since_move)
        if not why:
            self._advance(name)
            return (name, None, "", False)
        want, want_gran, how = self._reaim()
        self._advance(want)
        if want == name:
            return (name, None, "", False)          # the model already wants the escape step
        return (want, want_gran, why + "; " + how, False)

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
        also the "have I arrived" flag: 0 means a fresh cell.

        PURE: it reads the ledgers and returns a bearing, and advancing the rotation count is the
        caller's job (`_advance`). It used to flip `self.sweep` itself, which meant ANY veto — a
        climb lock, one refused translation — latched the loop into ESCAPING mode until a
        translation succeeded; measured at 649 of 1943 decisions in the live run. `sweep` is now
        exactly "the loop has stalled", decided in `observe` and nowhere else."""
        refused = self.refused.get(self.view, set())
        if "move_forward" not in refused:
            return ("move_forward", "small", "probing the way ahead")
        if self.sweep_turns >= 4 and "move_backward" not in refused:
            return ("move_backward", "small", "every bearing here is refused, stepping back")
        first = self.sweep_turns == 0
        turn = "turn_right" if first == (self.turn_sense > 0) else "turn_left"
        return (turn, None, "rotating %s to the next untried bearing"
                % ("right" if turn == "turn_right" else "left"))

    def _advance(self, name: str) -> None:
        """Move the escape sweep on by the action about to run, whoever chose it. A rotation spends
        a bearing; the sealed-cell step back starts the fan over."""
        if name in _act.TURN_ACTIONS:
            self.sweep_turns += 1
        elif name == "move_backward":
            self.sweep_turns = 0

    def _trim(self, name: str, gran) -> str:
        """The granularity a translation should really carry, or "" to leave the proposal alone.

        A stride is clipped to one cell when the tile the frame shows dead ahead is the thing worth
        arriving ON: a longer stride would walk straight over a stair (and the stair-underfoot fact
        is measured from ARRIVING on it) or past the exit. This changes no action NAME, so the model
        still chooses every move; it only stops a `large` from overshooting a one-cell target."""
        if name != "move_forward" or (gran or "small") == "small":
            return ""
        return "small" if (self.seen.stair_adjacent or self.seen.exit_adjacent) else ""

    def apply(self, proposal):
        """Govern one proposal and return the engine Action to execute."""
        name = _act.name_for(getattr(proposal, "key", None))
        meta = getattr(proposal, "metadata", None) or {}
        gran = meta.get("granularity")
        if name is None:                                     # unmapped key: pass it through untouched
            self._commit(None, None, None, False)
            return proposal
        chosen, chosen_gran, why, grounded = self.govern(name)
        if not why:
            trim = self._trim(name, gran)
            if trim:
                self.tuned += 1
                self._commit(name, trim, None, grounded)
                return _act.retarget(proposal, name, trim,
                                     "one cell at a time: the target is the very next cell", name)
            self._commit(name, gran, None, grounded)
            return proposal
        self.overrides += 1
        out = _act.retarget(proposal, chosen, chosen_gran or "small", why, name)
        self._commit(chosen, chosen_gran, name, grounded)
        return out

    def _commit(self, name, gran, overridden, grounded) -> None:
        self.last = {"view": self.view, "name": name, "gran": gran,
                     "overridden": overridden, "step": self.step}
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
    def fields(self) -> dict:
        refused = sorted(self.refused.get(self.view, set()))
        tried = self.tried.get(self.view, set())
        legal = _act.LEGAL_ACTIONS if self.sense.underfoot else \
            tuple(a for a in _act.LEGAL_ACTIONS if a not in _act.CLIMB_ACTIONS)
        untried = [a for a in legal if a not in tried]
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "view_visits": self.views.get(self.view, 1),
            "sight_line": self.seen.describe(),
            "stair_line": self.sense.line(),
            "last_line": last_line(self),
            "since_move": self.since_move,
            "stale": self.stale,
            "blocked_here": ", ".join("`%s`" % a for a in refused) or "nothing proven yet",
            "untried_here": ", ".join("`%s`" % a for a in untried) or "none - all were tried",
            "climb_line": climb_line(self),
            "mode_line": mode_line(self),
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
