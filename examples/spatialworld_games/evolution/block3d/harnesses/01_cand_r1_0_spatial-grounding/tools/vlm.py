"""VLM — ask the frozen model every decision, and arbitrate its answer against the picture.

The split is decided by what the two parties are measurably good at. Routing a cursor through an
isometric projection is the half the frozen model fails at — it emits a step more often exactly
when that step is against a wall — so the harness owns routing outright. Deciding whether the
cell under the cursor belongs to the structure is the half the model gets right well above its
own base rate, so that vote is left with it wherever the picture does not already settle the
question.

Concretely, per decision:

  1. read the panels once per episode and the cursor every turn;
  2. re-derive the goal cell from the residual projection debt;
  3. hand the model a symbolic brief — cursor, goal, what is owed, which steps are dead, budget —
     by REBUILDING the system turn from the text the engine itself composed, never by splicing
     into it, so the response contract the engine appended survives every later call;
  4. take the model's one answer, map its key back to a name, and keep it unless the picture
     proves it wasted: a step that cannot change the cursor, a walk away from the goal, a toggle
     that would lift a block the panels require;
  5. on a cell the panels do not force, HONOUR the model's own commit-or-walk-on choice;
  6. book whatever is actually returned and stamp the proposal, the applied action and the
     override reason onto the saved response record.

The first decision is left pristine: nothing is injected into the system turn before the engine
has composed it once, because the composed turn is also where the response contract lives, and
a level whose structure is the spawn cell alone is won by that untouched first answer.

If the panel read ever fails its own consistency test the whole wrapper stands down and this
module behaves exactly like the pass-through it replaced.
"""

from tools.debt import forced_set, pick_cell, residual_debt
from tools.keys import TOGGLE, key_for, name_for
from tools.panels import decode_panels
from tools.reckon import Ledger, cursor_cell, route_names
from tools.frame import capture_frame, pixels_of

READ_TRIES = 3
BRIEF_HEAD = "GROUNDED STATE (read off the picture you are being shown, not advice):"
_fmt = lambda c: "(%d,%d,%d)" % c
_row = lambda label, body: "- " + label + ": " + body


def choose_key(shadows, ledger, goal, forced, proposal):
    """The action to actually return, and why it differs from the proposal when it does."""
    owed = goal is not None
    standing = owed and ledger.cursor == goal
    lifting = owed and goal in ledger.placed
    if proposal == TOGGLE:
        if standing:
            return TOGGLE, ""
        if ledger.cursor in forced and ledger.cursor not in ledger.placed:
            return TOGGLE, ""
        return (route_names(ledger.cursor, goal)[0] if owed else
                ledger.live_steps()[0]), "this cell is not one the panels ask for"
    if standing:
        return TOGGLE, ("the panels say this cell comes off" if lifting else
                        "the panels put the next block here")
    if not owed:
        return (proposal if proposal in ledger.live_steps() else ledger.live_steps()[0]), \
            ("" if proposal in ledger.live_steps() else "the proposed step is dead")
    if ledger.closer(proposal, goal):
        return proposal, ""
    return route_names(ledger.cursor, goal)[0], "the proposal does not shorten the walk"


class Arbiter:
    """One episode of grounded decision-making over the evaluator's own per-step model call."""

    def __init__(self, evaluator, base="", reader=None, grab=None):
        self.ev = evaluator
        self.base = base
        self.reader = reader if reader is not None else decode_panels
        self.grab = grab if grab is not None else capture_frame
        self.ledger = Ledger()
        self.shadows = None
        self.forced = frozenset()
        self.stash = None
        self.tries = 0
        self.overrides = 0

    def read_panels(self, tries=3):
        """Decode the strip once; a failed or inconsistent read retires the wrapper quietly.

        The strip is static for the whole episode, so one clean read is all that is needed; the
        retry budget exists only for the first frames, where the surface may not be painted yet.
        """
        if self.shadows is not None or self.tries >= tries:
            return self.shadows
        self.tries += 1
        seen = self.reader(self.ev)
        if seen is not None and seen.ok:
            self.shadows = seen
            self.forced = forced_set(seen)
        return self.shadows

    def look(self):
        """Refresh the cursor from the frame when the highlight resolves to exactly one cell."""
        grab = pixels_of(self.grab(self.ev))
        return self.ledger.sight(cursor_cell(*grab) if grab is not None else None)

    def brief(self, goal, wanted=12, filled=12):
        """The symbolic state block: every line of it is derived, none of it is engine state.

        Both cell lists are capped. The point of the block is to spare the model the arithmetic it
        gets wrong, not to hand it a wall of coordinates to re-read every turn; the cells nearest
        the front of the build order are the ones the next few decisions can act on.
        """
        owed = residual_debt(self.shadows, self.ledger.placed)
        unexplained = 0
        for panel in owed:
            unexplained += len(panel)
        todo = sorted(self.shadows.admitted - self.ledger.placed)
        lines = [BRIEF_HEAD,
                 _row("cursor", _fmt(self.ledger.cursor)),
                 _row("structure cells still to fill",
                      " ".join(_fmt(c) for c in todo[:wanted]) or "none"),
                 _row("already filled by you",
                      " ".join(_fmt(c) for c in sorted(self.ledger.placed)[:filled]) or "none"),
                 _row("panel squares still unexplained", "%d" % unexplained),
                 _row("next cell this harness is walking to", _fmt(goal) if goal else "none"),
                 _row("dead steps from here", " ".join(self.ledger.dead_steps()) or "none"),
                 _row("decisions left", "%d" % self.ledger.left()),
                 "Place a block only on a listed cell; a second toggle on a filled cell REMOVES it."]
        return "\n".join(lines)

    def speak(self, text):
        """Rebuild the system turn as composed-text + brief. Never splice, never truncate.

        The engine keeps message zero forever and re-sends it on every request, so rewriting it in
        place is a genuine per-turn channel — but it is also where the engine appended the JSON
        response contract. Stashing the composed text on first sight and re-appending the brief to
        THAT stash is what keeps the contract intact on every later call; truncating at a marker
        and appending deletes it from every call after the first.
        """
        hist = getattr(self.ev, "conversation_history", None)
        if not hist:
            return                                     # turn one: the engine has not composed yet
        try:
            part = hist[0]["content"][0]
        except (KeyError, IndexError, TypeError):
            return
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            return
        if self.stash is None:
            self.stash = part["text"]
        part["text"] = self.stash + "\n\n" + text

    def record(self, frame_number, entry):
        """Stamp the decision onto the evaluator's own saved response record, for the next pass."""
        rows = getattr(self.ev, "model_responses", None)
        if not rows:
            return
        try:
            fresh = rows[-1].get("frame_number") == frame_number
        except AttributeError:
            return
        if fresh:
            rows[-1]["harness"] = entry

    async def step(self, frame_data):
        """One decision: ground, ask, arbitrate, book, stamp."""
        ask = getattr(self.ev, "baseline_step", None) or self.ev.openai_ai_model
        if self.read_panels() is None:
            return await ask(frame_data)
        self.look()
        goal = pick_cell(self.shadows, self.ledger, self.forced)
        self.speak(self.brief(goal))
        action = await ask(frame_data)
        if action is None:
            return action
        meta = action.metadata if isinstance(action.metadata, dict) else {}
        proposal = name_for(action.key) or str(meta.get("function_called") or "")
        final, why = choose_key(self.shadows, self.ledger, goal, self.forced, proposal)
        if why:
            self.overrides += 1
            action.key = key_for(final) or action.key
            meta["function_called"] = final
        meta["model_proposal"] = proposal
        meta["harness_applied"] = final
        meta["harness_reason"] = why
        meta["cursor"] = list(self.ledger.cursor)
        meta["goal"] = list(goal) if goal else None
        meta["admitted"] = len(self.shadows.admitted)
        meta["forced"] = len(self.forced)
        meta["left"] = self.ledger.left()
        action.metadata = meta
        self.ledger.book(final)
        self.record(getattr(frame_data, "frame_number", -1),
                    {"proposal": proposal, "applied": final, "reason": why,
                     "cursor": list(self.ledger.cursor), "placed": len(self.ledger.placed),
                     "overrides": self.overrides, "nulls": self.ledger.nulls,
                     "cancels": self.ledger.cancels, "corrections": self.ledger.corrections})
        return action


async def decide(evaluator, frame_data, reader=decode_panels, grab=capture_frame):
    """Episode-scoped entry point: one Arbiter per evaluator, one model call per decision.

    The two perception sources are injected rather than imported inside the arbiter, so an offline
    rig can BLIND either one of them in a single line and price what that half of the grounding is
    actually worth: a blinded panel reader must collapse this harness onto the parent's answers, and
    a blinded frame grab must leave the booked cursor carrying the episode on its own.
    """
    seat = getattr(evaluator, "_debt_arbiter", None)
    if seat is None:
        seat = Arbiter(evaluator, getattr(evaluator, "current_prompt", "") or "",
                       reader=reader, grab=grab)
        evaluator._debt_arbiter = seat
    try:
        return await seat.step(frame_data)
    except Exception:
        seat.shadows = None
        seat.tries = READ_TRIES
        return await (getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model)(frame_data)
