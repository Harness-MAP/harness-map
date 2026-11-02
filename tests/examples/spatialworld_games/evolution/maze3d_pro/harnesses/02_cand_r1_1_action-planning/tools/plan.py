"""tools/plan.py — the ACTION-OUTCOME PLANNER: a closed-loop governor over the frozen model's
per-step proposal.

WHY THIS EXISTS. The anchor hands every decision straight to `evaluator.baseline_step` and keeps no
state, so nothing in the loop can tell a move that happened from a move that did not. The frozen
model is reactive: when its forward move is refused by a wall the screen comes back looking the same,
it proposes the same move again, and the episode spends its whole 100-step budget on one refused
action. The same blindness turns a legal floor transition into an up/down 2-cycle at a single stair.

WHAT IT MEASURES (self-derived — no engine state, no layout, no coordinates). The only thing this
module reads is `frame_data.image`, the very frame the agent is shown. Two consecutive frames are
compared:

  * a translation that leaves the screen UNCHANGED did not happen -> the cell ahead is a wall. The
    engine stops a multi-cell stride early at a wall, so "zero cells moved" means the FIRST cell was
    refused, whatever granularity was asked for. That verdict is recorded against the frame
    FINGERPRINT, which is the agent's whole first-person view: whether something blocks the way
    ahead is a property of that view, so the ledger is exactly re-usable the next time the same view
    comes back (including after a four-turn circuit of the same cell).
  * a translation that CHANGES the screen did happen -> the agent moved, which also means it has
    left any stair landing.

Equality is deliberately tolerant: identical fingerprints mean unchanged, and a changed fingerprint
still counts as unchanged when under `_NULL_FRAC` of the sampled pixels differ. A real step or turn
in a first-person 3D view repaints most of the frame, so this absorbs a HUD counter or a one-pixel
animation without ever mistaking a refused move for a successful one.

WHAT IT DOES WITH THAT.  The frozen model is asked EVERY step and is obeyed unless its proposal is
provably useless:
  1. a translation already proven refused from this exact view          -> re-aimed,
  2. a climb that would undo the floor transition just made             -> re-aimed,
  3. a turn proposed while the loop has stalled and the way ahead is untested -> re-aimed forward.
A re-aim is not a random nudge: it is one step of a right-hand wall follower (probe the way ahead;
if it is refused rotate to the next bearing in the order right, straight, left, back). Leaving the
came-from bearing for LAST is the whole of it — a sweep that rotates always the same way tries the
bearing it just came from second and walks straight back out of every dead end, which the offline
rig measures as a 2-cell orbit. Nothing the current view has already refused is ever re-probed, and
the sweep is released the moment a translation succeeds, so the model drives again.

AND IT TELLS THE MODEL. The governed verdict is useless if the model keeps proposing the same
refused action, so `render_memory()` renders the ledger into plain English and tools/vlm.py appends
it to the agent's system turn every step. The facts are the model's own observation history, stated
back to it: what it last proposed, whether the screen moved, which actions this exact view has
already refused, which it has never tried.
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

#: the built-in working-memory template. memory.md (when present next to the harness) overrides it,
#: which is what keeps the wording a prose lever; the built-in keeps the mechanism alive when the
#: harness is bundled without it.
DEFAULT_TEMPLATE = """## WORKING MEMORY (self-derived: computed by comparing the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- stair lock: {climb_line}
- mode: {mode_line}

How to use this: an action that left the screen byte-for-byte identical did not happen - something
solid is there. Never re-propose an action listed as PROVEN REFUSED while this screen is in front of
you; turn to a bearing you have not tried yet from this screen and then move. After a floor change,
explore the new level before going back.

Reply with ONLY the JSON object in the format specified above."""


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
        # rolling state
        self.view = ""
        self._thumb = None
        self.step = 0
        self.since_move = 0
        self.last = None                        # the previous decision we committed
        self.sweep = False                      # True while escaping a proven-refused pocket
        self.sweep_turns = 0                    # consecutive re-aimed turns since the last move
        self.turn_sense = 1                     # +1 = right-hand wall follower, -1 = left-hand
        self.climb_lock = None                  # the floor transition that would undo the last one
        self.climb_lock_until = 0
        self.overrides = 0
        self.moves = 0
        self.base_prompt = None                 # stash of the agent's first-call system turn

    # ── outcome measurement ────────────────────────────────────────────────────────────────
    def _fingerprint(self, image):
        """(fingerprint, downsampled copy) for the frame the agent is being shown."""
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
        view, thumb = self._fingerprint(image)
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
                self.climb_lock = None          # we have left the stair landing
                self.climb_lock_until = 0
            else:
                self.since_move += 1
        self.view, self._thumb = view, thumb
        self.views[view] = self.views.get(view, 0) + 1
        self.step += 1
        if self.climb_lock and self.step >= self.climb_lock_until:
            self.climb_lock = None              # last-resort release for a genuinely trapped landing
        if self.since_move >= self.stall_limit:
            self.sweep = True

    # ── the governor ───────────────────────────────────────────────────────────────────────
    def govern(self, name: str):
        """(action_name, granularity_or_None, reason). reason == "" means the proposal is obeyed.

        A proposal is only ever displaced when it is provably useless: a translation this exact view
        already refused, the undo of the floor change just made, or a turn chosen while the loop has
        stalled (where the handedness of the rotation is what decides whether the sweep escapes or
        retraces). Even then the escape step is computed FIRST and the model is obeyed whenever it
        already agrees with it, so agreement costs nothing."""
        refused = self.refused.get(self.view, set())
        if name in _act.CLIMB_ACTIONS and name == self.climb_lock:
            why = "`%s` would undo the floor change just made" % name
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
        if self.sweep_turns > 4 and "move_backward" not in refused:
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
            self._commit(name, gran, None)
            return proposal
        self.overrides += 1
        out = _act.retarget(proposal, chosen, chosen_gran or "small", why, name)
        self._commit(chosen, chosen_gran, name)
        return out

    def _commit(self, name, gran, overridden) -> None:
        self.last = {"view": self.view, "name": name, "gran": gran,
                     "overridden": overridden, "step": self.step}
        if name:
            self.tried.setdefault(self.view, set()).add(name)
            if name in _act.CLIMB_ACTIONS:
                # lock the undo of ANY executed climb; released by a successful move (or the span)
                self.climb_lock = _act.inverse_climb(name)
                self.climb_lock_until = self.step + self.climb_lock_span

    # ── the per-turn memory block ──────────────────────────────────────────────────────────
    def _last_line(self) -> str:
        prev = self.last
        if not prev or not prev.get("name"):
            return "none yet - this is your first decision."
        moved = self.since_move == 0 and prev["name"] in _act.MOVEMENT_ACTIONS
        if prev.get("overridden"):
            ran = "you proposed `%s`, which was already proven useless here, so `%s` was executed " \
                  "instead" % (prev["overridden"], prev["name"])
        else:
            ran = "`%s` was executed as you asked" % prev["name"]
        if prev["name"] in _act.MOVEMENT_ACTIONS:
            ran += " and the screen %s" % ("changed, so you really moved" if moved else
                                           "did NOT change at all, so something solid blocked it")
        return ran + "."

    def _mode_line(self) -> str:
        if not self.sweep:
            return "FOLLOWING YOU - your proposal is executed unless it is already proven useless."
        return ("ESCAPING - the way out of this pocket is being searched by always rotating the "
                "same direction and stepping forward whenever the way opens. Propose the turn or "
                "the step that agrees with that and it will be executed as asked.")

    def fields(self) -> dict:
        refused = sorted(self.refused.get(self.view, set()))
        tried = self.tried.get(self.view, set())
        untried = [a for a in _act.LEGAL_ACTIONS if a not in tried]
        lock = self.climb_lock
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "view_visits": self.views.get(self.view, 1),
            "last_line": self._last_line(),
            "since_move": self.since_move,
            "blocked_here": ", ".join("`%s`" % a for a in refused) or "nothing proven yet",
            "untried_here": ", ".join("`%s`" % a for a in untried) or "none - all six were tried",
            "climb_line": ("`%s` is locked: you just changed floor, explore this level before "
                           "going back" % lock) if lock else "none",
            "mode_line": self._mode_line(),
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
