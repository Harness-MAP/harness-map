"""tools/plan.py — the ACTION-OUTCOME PLANNER: a closed-loop governor over the frozen model's
per-step proposal.

WHY THIS EXISTS. The anchor hands every decision straight to `evaluator.baseline_step` and keeps no
state, so nothing in the loop can tell a move that happened from a move that did not. The frozen
model is reactive: when its forward move is refused by a wall the screen comes back looking the
same, it proposes the same move again, and the episode spends its whole budget on one refused
action. The same blindness turns a legal floor transition into an up/down 2-cycle at a stair.

WHAT IT MEASURES (self-derived — no engine state, no layout, no coordinates). The only thing this
module reads is `frame_data.image`, the very frame the agent is shown, and it reads it at three
resolutions:

  * AS A FINGERPRINT. Two consecutive frames are compared whole. A translation that leaves the
    screen UNCHANGED did not happen -> the cell ahead is a wall. That verdict is filed against the
    fingerprint, which is the agent's whole first-person view, so the ledger is exactly re-usable
    the next time that view comes back. The comparison is now EXACT — identical md5 or not. The
    0.2%-of-sampled-pixels tolerance this used to carry was measured against both harnesses' live
    tapes: a refused translation leaves the frame byte-identical 83 times out of 83 and a real move
    leaves it identical 0 times out of 1077, so the tolerance could only ever manufacture a false
    "wall ahead" — which is exactly what it did four times, one of them turning L20's start cell
    into an absorbing state that cost the level outright.
  * BY COLOUR CENSUS (tools/sight.py). The exact masks of the two special tile families: whether a
    stair or the exit is anywhere in view at all.
  * BY GEOMETRY (tools/sight.py). The centre column is a RANGEFINDER and the flanks are a MAP
    STRIP: what stands dead ahead and how many cells away, how far the body can actually walk, and
    which of the two bearings a turn would take it to are open. All three are exact inversions of
    the renderer, scored at 1820/1820 against the engine's own layout.

WHAT IT DOES WITH THAT. The frozen model is asked EVERY step and obeyed unless the frames already
settle the matter:
  0. a stair is UNDER THE BODY, this floor has paid its price and shown nothing new, and no exit is
     in sight  -> the climb is taken. The only branch that displaces a proposal on evidence FOR
     something rather than against it.
  1. a climb while the frames prove no stair is underfoot, or before this floor has been given its
     PATIENCE -> re-aimed. The price used to gate only the harness's own climb, so a model-proposed
     climb was a free floor departure: 13 presses on the parent's tape, 12 of them inert and the one
     that worked cost L23 a level whose exit was on the floor it left. The two climbs are now priced
     DIFFERENTLY on purpose, and the asymmetry is the one the live runs forced: the harness's own
     climb needs the floor to have gone STALE as well, because it fires unasked; the model's climb
     needs only the patience, because it is the escape hatch for the case the staleness clause
     cannot see. A faster agent re-sees fewer views, so `stale` reaches 1 less often, and with the
     model's climb also gated on it an episode can stand on the only stair of a level whose exit is
     upstairs and never leave — measured live on L5, 0 floor changes in 3 of 3 paired episodes
     against the parent's 3 of 3 wins.
  2. a forward move the RANGEFINDER puts against a wall  -> re-aimed before it is spent, not after.
  3. a translation already proven refused from this exact view  -> re-aimed (this is what still
     covers `move_backward`, the one direction the frame cannot see).
  4. a turn proposed while the loop has stalled  -> re-aimed.
  5. a TRANSLATION proposed while the body stands on a stair this floor has already paid for
     -> re-aimed to a turn. This is the fault the live runs exposed and the one the supplied stride
     created: a one-cell-at-a-time agent loiters near a stair and comes back to it (the parent stood
     on L5's only stair on 14 of 73 decisions and climbed at the third opportunity), while an agent
     that walks three cells per decision arrives once and is gone — L5's latch was set on exactly
     ONE decision of 100, with the patience already paid and the floor not yet stale, and the stride
     carried the body off before the staleness clause could ever be satisfied. A turn does not move
     the body, and it is also the fastest way to finish exhausting the floor, since the four
     bearings of this cell are either new views or repeats that make `stale` tick.

AND IT SUPPLIES THE STRIDE. A decision buys one action whatever distance it covers, and the engine
stops a long stride early at a wall, so asking for one cell when three are clear throws away two
cells of route for nothing. On the parent's tape the model asked for `small` on 94% of its forward
moves and gained 1.02 cells per forward where its own parent gained 1.54 — a prose artefact worth
~500 cells of travel across a sweep whose losses need 32-54 cells of route they never get. The
granularity is therefore no longer the model's to guess: the rangefinder computes the legal stride
and the harness asks for it, capped to land ON the exit (which ends the episode), to stop ONE CELL
SHORT of a stair (the arrival latch is set by a stair that was one cell ahead and then vanished,
and the engine walks straight through a stair tile without stopping), and to stop AT a side branch
rather than past it. The action NAME is always still the model's.

WHY THE CLIMB WAITS. Taking the first stair the body lands on is a measured trap: the exit is on
the STARTING floor on 13 of 25 levels, and leaving early throws away levels that were being won.
`StairSense`'s price is the admission — this floor must stop producing new views AND have been
given its patience — and a floor change clears the view set, which is also what makes an up/down
2-cycle structurally impossible rather than merely discouraged.

AND IT TELLS THE MODEL. `render_memory()` renders the ledger into plain English and tools/vlm.py
appends it to the agent's system turn every step: what it last proposed, whether the screen moved,
what the rangefinder says is ahead, which bearings are open, which actions this view has refused,
and whether a stair is underfoot.
"""
from __future__ import annotations

import hashlib

try:                                    # the engine hands us a numpy frame; never hard-fail on it
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

from . import action as _act
from . import sight as _sight

#: the built-in working-memory template. memory.md (when present next to the harness) overrides it,
#: which is what keeps the wording a prose lever; the built-in keeps the mechanism alive when the
#: harness is bundled without it. The two MUST carry the same placeholder set as `fields()`.
DEFAULT_TEMPLATE = """## WORKING MEMORY (measured off the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- DEAD AHEAD: {ahead_line}
- OPEN BEARINGS: {open_line}
- IN THIS FRAME: {sight_line}
- UNDER YOUR FEET: {stair_line}
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- this floor has shown nothing new for {stale} decision(s). {climb_line}
- mode: {mode_line}

How to use this: the DEAD AHEAD line is measured off this very screen and is exact, so a
`move_forward` into a wall in the next cell is a wasted decision; turn to a bearing the OPEN
BEARINGS line says is open and then move. The distance of your stride is chosen for you from the
same reading - ask for `move_forward` and the longest safe stride is what runs - so never hold
yourself to one cell to be careful. `climb_up` and `climb_down` do something ONLY while a stair is
under your feet; there is no prompt to confirm it, so trust the UNDER YOUR FEET line. After a floor
change, explore the new level before going back.

Reply with ONLY the JSON object in the format specified above."""


class ActionPlanner:
    """Episode-local state + the governor. One instance per episode, built in agent.py.

    Everything that used to be a module-level helper here is a method: the ledgers, the frame
    comparison, the veto chain and the four lines of the working-memory block all read nothing but
    this object, and keeping them inside it is what makes the state machine readable in one place.
    """

    #: fingerprint every Nth pixel: ~4x cheaper than the whole frame and still a faithful view id,
    #: because a first-person step repaints most of the view.
    SAMPLE = 2
    #: cells per granularity label, and its inverse. Mirrors the engine's own clamp to [1, 3].
    LABELS = {1: "small", 2: "medium", 3: "large"}
    #: decisions without the body moving before the governor starts aiming the sweep itself.
    STALL_LIMIT = 4

    def __init__(self, max_steps: int = 100, stall_limit: int = 0, template: str = ""):
        self.max_steps = int(max_steps or 100)
        self.stall_limit = int(stall_limit or self.STALL_LIMIT)
        self.template = self.clean_template(template) or DEFAULT_TEMPLATE
        # ledgers keyed on the frame fingerprint (the agent's own view)
        self.refused: dict[str, set] = {}       # view -> translations proven not to move us
        self.tried: dict[str, set] = {}         # view -> actions executed from it
        self.views: dict[str, int] = {}         # view -> how often it has been shown
        self.floor_views: set = set()           # views shown since the last floor transition
        # rolling state
        self.view = ""
        self.seen = _sight.Scene.of(None)       # the decode of the frame now on screen
        self.sense = _sight.StairSense()        # the stair-underfoot latch and its price
        self.step = 0
        self.since_move = 0
        self.stale = 0                      # consecutive decisions with no view new to this floor
        self.floor_age = 0                  # decisions spent on the floor we are on now
        self.last = None                    # the previous decision we committed
        self.sweep = False                  # True while escaping a proven-refused pocket
        self.turn_sense = 1                 # +1 = right-hand sweep, -1 = left-hand
        self.overrides = 0
        self.tuned = 0                      # strides supplied (the action name is left alone)
        self.moves = 0
        self.cells = 0                      # cells of stride asked for, as a travel counter
        self.base_prompt = None             # stash of the agent's first-call system turn

    # ── the frame as a view identity ───────────────────────────────────────────────────────
    @classmethod
    def fingerprint(cls, image) -> str:
        """The WHOLE-FRAME reading: an md5 over every SAMPLEth pixel, which is a VIEW IDENTITY.
        Two poses that render the same get the same id, which is exactly what makes a "the way
        ahead is blocked" fact re-usable the next time that screen comes back."""
        if image is None:
            return ""
        if _np is None:                                     # pragma: no cover
            try:
                return hashlib.md5(bytes(memoryview(image))).hexdigest()[:16]
            except Exception:
                return ""
        arr = _np.asarray(image)
        if arr.ndim == 3:
            arr = arr[::cls.SAMPLE, ::cls.SAMPLE, :3]
        elif arr.ndim == 2:
            arr = arr[::cls.SAMPLE, ::cls.SAMPLE]
        else:
            return ""
        return hashlib.md5(_np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

    @staticmethod
    def screen_changed(prev_view: str, view: str) -> bool:
        """Did anything happen? EXACTLY: the same view id means the same frame means nothing moved.

        There is no tolerance here on purpose. Over 2040 translations across two live tapes a
        refused translation left the frame byte-identical 83/83 times and a real move left it
        identical 0/1077 times, so md5 identity is a perfect one-bit outcome oracle and any slack
        in it can only invent walls that are not there.
        """
        return prev_view != view

    # ── outcome measurement ────────────────────────────────────────────────────────────────
    def score(self, prev, moved, before, after) -> None:
        """Fold the PREVIOUS decision's measured outcome into the ledgers.

        Only a translation's failure is filed against the view: "the cell ahead is blocked" is a
        property of what is on screen, so it transfers whenever that screen returns. A climb's
        failure is NOT — two poses can render identically and disagree about the tile underfoot —
        and climbs are now gated on the latch rather than on an outcome, so nothing is filed for
        them at all."""
        name = prev["name"]
        translated = name in _act.MOVEMENT_ACTIONS
        self.sense.update(before, after, moved, translated)
        if translated and not moved:
            self.refused.setdefault(prev["view"], set()).add(name)
            self.since_move += 1
        elif translated:
            self.moves += 1
            self.since_move = 0
            self.sweep = False                  # a real step: hand control back to the model
        else:
            self.since_move += 1

    def observe(self, image) -> None:
        """Fold the frame now on screen into the ledgers: score the PREVIOUS decision's outcome,
        then make this frame the current view. Called once per step before the model is asked."""
        view = self.fingerprint(image)
        seen = _sight.Scene.of(image)
        if self.last is not None:
            moved = self.screen_changed(self.last["view"], view)
            self.score(self.last, moved, self.seen, seen)
        self.view, self.seen = view, seen
        self.views[view] = self.views.get(view, 0) + 1
        self.stale = 0 if view not in self.floor_views else self.stale + 1
        self.floor_views.add(view)
        self.floor_age += 1
        self.step += 1
        if self.since_move >= self.stall_limit:
            self.sweep = True

    # ── the governor ───────────────────────────────────────────────────────────────────────
    def blocked_ahead(self) -> bool:
        """Does the frame PROVE a forward move cannot move the body? `reach == 0` means a wall
        stands in the very next cell, and it agreed with the engine on 1820 of 1820 decisions."""
        return bool(self.seen) and self.seen.ok and self.seen.reach <= 0

    def holding_stair(self) -> bool:
        """Is the body standing on a stair this floor has already paid for in clock?

        While this holds, a translation is refused and the escape step is a TURN, because a turn is
        the one legal action that does not move the body off the tile. It is read by BOTH `refusal`
        and `reaim`: vetoing the model's step is useless if the re-aim then steps off anyway, which
        is exactly the hole the first version of this clause had.
        """
        if not self.sense.underfoot or self.sense.patience_owed(self.floor_age):
            return False
        return not (self.seen is not None and self.seen.exit > 0.0)

    def refusal(self, name: str) -> str:
        """Why `name` is provably useless right now, or "" when nothing refutes it.

        The ORDER is the priority. The climb clauses come first because they are the ones that can
        cost a level rather than a decision; the rangefinder's wall beats the view ledger because it
        is available on the FIRST encounter with a view rather than after one refused attempt; the
        ledger still owns `move_backward`, which no reading of the frame can see.
        """
        if name in _act.CLIMB_ACTIONS:
            if not self.sense.underfoot:
                return "no stair is under you, so `%s` cannot change anything" % name
            owed = self.sense.patience_owed(self.floor_age)
            if owed > 0:
                return ("this floor is owed about %d more decision(s) before its stair is worth "
                        "spending" % owed)
            if self.seen is not None and self.seen.exit > 0.0:
                return "the exit is in view on THIS floor, which no floor change can improve on"
            return ""              # the price is paid: the model may leave even on a brand-new view
        if name in _act.MOVEMENT_ACTIONS and self.holding_stair():
            return ("you are standing on a stair this floor has already paid for; stepping off it "
                    "throws the floor change away")
        if name == "move_forward" and self.blocked_ahead():
            return "this screen shows a wall in the very next cell, so `move_forward` cannot move you"
        if name in _act.MOVEMENT_ACTIONS and name in self.refused.get(self.view, set()):
            return "`%s` already left this exact screen unchanged" % name
        if self.sweep and name in _act.TURN_ACTIONS:
            return "%d decisions without moving" % self.since_move
        return ""

    def govern(self, name: str):
        """(action_name, granularity_or_None, reason, grounded). reason == "" means obeyed.

        Two kinds of displacement, in this order. The FIRST is positive evidence: the frames have
        proven a stair is under the body, this floor has paid its price, and no exit is in sight —
        the model cannot see any of that (the tile underfoot is not drawn and the build renders no
        text), so the climb is taken for it. `grounded` marks that case, because it is the one that
        resets the floor ledger. The SECOND is the negative chain in `refusal`. Both compute the
        escape step FIRST and obey the model whenever it already agrees, so agreement costs nothing.
        """
        grounded = self.sense.wanted(self.stale, self.floor_age, self.seen)
        if grounded:
            if name == grounded:
                return (name, None, "", True)
            return (grounded, None, "a stair is under your feet and this floor has shown nothing "
                    "new for %d decisions" % self.stale, True)
        why = self.refusal(name)
        if not why:
            return (name, None, "", False)
        want, want_gran, how = self.reaim()
        if want == name:
            return (name, want_gran, "", False)         # the model already wants the escape step
        return (want, want_gran, why + "; " + how, False)

    def reaim(self):
        """The next step of the escape sweep, AIMED off the frame rather than probed for.

        The rule used to be "probe the way ahead; if the probe is refused, rotate", and the order
        of those bearings was the whole of it, because a rotation that retries the came-from bearing
        second walks straight back out of every dead end. The flank strip makes the probe
        unnecessary: the frame already says whether each side of this cell is a wall, so the sweep
        turns straight to a side it can see is open and never spends a decision learning that one
        is not. The came-from bearing cannot be picked by accident, because the body arrived facing
        forward and so the way back is BEHIND it, not beside it — a dead end costs two rotations
        and is left for last by construction rather than by a hand-tuned order.
        """
        refused = self.refused.get(self.view, set())
        holding = self.holding_stair()
        if not holding and not self.blocked_ahead() and "move_forward" not in refused:
            return ("move_forward", self.granularity(), "this screen shows the way ahead open")
        left, right = self.seen.bearings() if self.seen is not None else (False, False)
        order = (("turn_right", right), ("turn_left", left)) if self.turn_sense > 0 else \
                (("turn_left", left), ("turn_right", right))
        if holding:
            # Stay on the tile and look around it: a bearing this view has not been turned to yet
            # finishes exhausting the floor fastest, and when none is left `stale` ticks and the
            # grounded climb in `govern` fires on the next decision.
            tried = self.tried.get(self.view, set())
            for turn, _open in order:
                if turn not in tried:
                    return (turn, None, "staying on this stair and looking round it")
            return (order[0][0], None, "staying on this stair until the floor is exhausted")
        for turn, is_open in order:
            if is_open:
                return (turn, None, "this screen shows that side open")
        return (order[0][0], None,
                "both sides of this cell are walled, turning toward the way you came")

    # ── the stride ─────────────────────────────────────────────────────────────────────────
    def granularity(self, limit: int = 3) -> str:
        """The granularity label for the longest stride the frame licenses right now. One cell when
        the frame could not be read at all, because an unreadable frame has measured nothing."""
        readable = self.seen is not None and self.seen.ok
        cells = self.seen.stride(limit) if readable else 1
        return self.LABELS.get(max(min(cells, limit), 1), "small")

    def supply(self, name: str, asked) -> str:
        """The granularity a forward move should really carry, or "" to leave the proposal alone.

        This replaces the old one-sided trim, which only ever clipped a long stride down to one
        cell when a stair or the exit was adjacent. The rangefinder knows the whole clear run, so
        the stride is now SUPPLIED in both directions: shortened when something worth arriving on
        stands close, and lengthened when the corridor is plainly empty. `move_backward` is left
        exactly as proposed — the cell behind the body is never drawn, so there is nothing to
        supply it from.
        """
        if name != "move_forward" or self.seen is None or not self.seen.ok:
            return ""                   # an unreadable frame licenses nothing: obey the proposal
        want = self.granularity()
        return "" if want == (asked or "small") else want

    def apply(self, proposal):
        """Govern one proposal and return the engine Action to execute."""
        name = _act.name_for(getattr(proposal, "key", None))
        meta = getattr(proposal, "metadata", None) or {}
        gran = meta.get("granularity")
        if name is None:                                # unmapped key: pass it through untouched
            self.commit(None, None, None, False)
            return proposal
        chosen, chosen_gran, why, grounded = self.govern(name)
        if not why:
            supplied = chosen_gran or self.supply(name, gran)
            if supplied and supplied != (gran or "small"):
                self.tuned += 1
                self.commit(name, supplied, None, grounded)
                return _act.retarget(proposal, name, supplied,
                                     "stride supplied from the rangefinder: %s cell(s) of the way "
                                     "ahead are clear" % _act.granularity_steps(supplied), name)
            self.commit(name, gran, None, grounded)
            return proposal
        self.overrides += 1
        out = _act.retarget(proposal, chosen, chosen_gran or "small", why, name)
        self.commit(chosen, chosen_gran or "small", name, grounded)
        return out

    def commit(self, name, gran, overridden, grounded) -> None:
        """Record what is about to run, and spend a grounded climb from the latch's budget."""
        self.last = {"view": self.view, "name": name, "gran": gran,
                     "overridden": overridden, "step": self.step}
        if not name:
            return
        self.tried.setdefault(self.view, set()).add(name)
        if name in _act.MOVEMENT_ACTIONS:
            self.cells += _act.granularity_steps(gran)
        if name not in _act.CLIMB_ACTIONS or not grounded:
            return
        # A MEASURED transition: spend it from the latch's budget and start this floor's view
        # ledger over. The empty ledger IS the anti-ping-pong lock — the next stair cannot be spent
        # until the floor just reached has itself paid the price in full.
        self.sense.pressed(name)
        self.floor_views = set()
        self.stale = 0
        self.floor_age = 0

    # ── the per-turn memory block ──────────────────────────────────────────────────────────
    def last_line(self) -> str:
        """What the previous decision was and what the frames say it did."""
        prev = self.last
        if not prev or not prev.get("name"):
            return "none yet - this is your first decision."
        name = prev["name"]
        if prev.get("overridden"):
            ran = "you proposed `%s`, which this screen had already settled, so `%s` was executed " \
                  "instead" % (prev["overridden"], name)
        else:
            ran = "`%s` was executed as you asked" % name
            if prev.get("gran") and name in _act.MOVEMENT_ACTIONS:
                ran += ", at a stride of %d cell(s)" % _act.granularity_steps(prev["gran"])
        if name in _act.MOVEMENT_ACTIONS:
            ran += " and the screen %s" % ("changed, so you really moved" if self.since_move == 0
                                           else "did NOT change at all, so something solid blocked it")
        return ran + "."

    def mode_line(self) -> str:
        """Who is driving, in the agent's own terms."""
        if not self.sweep:
            return "FOLLOWING YOU - your proposal is executed unless this screen already settles it."
        return ("ESCAPING - the way out of this pocket is being chosen off the screen itself, by "
                "turning to a side it shows open. Propose the turn or the step that agrees with "
                "that and it will be executed as asked.")

    def climb_line(self) -> str:
        """What the floor ledger and the stair latch together say about leaving this floor."""
        if self.sense.underfoot:
            # the price QUOTED is the model's own (clock only), because this line is addressed to it
            # and has to be true about what its next `climb_up` will do.
            owed = self.sense.patience_owed(self.floor_age)
            if owed > 0:
                return ("You are on a stair, but this floor has not been given its due yet - about "
                        "%d more decision(s) and the stair is worth taking." % owed)
            return "You are on a stair and this floor has had its due: the stair is yours to take."
        return "No stair under you, so neither climb can do anything from here."

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
            "ahead_line": self.seen.ahead_line(),
            "open_line": self.seen.open_line(),
            "sight_line": self.seen.describe(),
            "stair_line": self.sense.line(),
            "last_line": self.last_line(),
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
        try:
            flds = self.fields()
        except Exception:
            return ""
        try:
            return self.template.format(**flds)
        except Exception:
            try:
                return DEFAULT_TEMPLATE.format(**flds)
            except Exception:
                return ""

    @staticmethod
    def clean_template(text: str) -> str:
        """The renderable part of memory.md: everything after the last HTML comment (the file's
        header documents the template for the next proposer and must not reach the agent). Returns
        "" when the file has no renderable body."""
        if not text:
            return ""
        body = text.split("-->")[-1].strip() if "-->" in text else text.strip()
        return body if "{step}" in body else ""
