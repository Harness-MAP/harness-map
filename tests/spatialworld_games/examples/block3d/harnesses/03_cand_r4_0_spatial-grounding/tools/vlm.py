"""THE DECISION LOOP — ask the frozen model every turn, measure what the last press actually did,
then price the model's answer against the lift.

Control flow, once per callback:

  1. FIRST TURN ONLY — fit the ruler off the frame, hand the fit back to the frame for
     confirmation, read the three target panels, back-project the envelope, compile the tape. A
     refused fit or a refused read means the harness never acts on a guess: it stands down to the
     evaluator's pristine per-step VLM fn for the whole episode.
  2. EVERY TURN — ask the frozen model, through that same pristine per-step fn. The call captures
     the frame, builds the image+text history, calls the VLM and parses the JSON, so the agent
     sees exactly what the baseline agent sees. It is also pure with respect to the game — the
     proposal costs no engine step — so asking is free and the model stays a real participant in
     every single decision.
  3. GRADE THE LAST PRESS, then MEASURE THE CURSOR. The identity fence hashes one fixed sub-frame:
     if a move left it unchanged the move clamped, and that pins one coordinate exactly
     (tools/book.py::grade). Independently, the calibrated projection reads the cursor straight
     off the highlight (tools/optics.py::locate) and the ledger adopts it. Dead reckoning is the
     prior; the picture is the measurement; a refused readout leaves the prior standing.
  4. ARBITRATE ON PRICE, never on agreement. The model's answer is applied whenever applying it
     cannot make the plan longer: a move that strictly closes the L1 gap to the goal is on some
     shortest route and is honoured; `place_block` is honoured only when the cursor stands on the
     plan's own current goal, because an out-of-order toggle breaks the prefix property the whole
     plan rests on. Anything else — a move away from the goal, a move into a grid face, an
     unparsed or unmapped name — is replaced by the plan's own press.
  5. APPLY, book the structure, and append one line of telemetry.

Every byte this module reads about the world comes from `frame_data.image` and from the actions it
chose itself. It never touches `evaluator.game`, never re-renders, and opens no file but its own
append-only trace.
"""
from __future__ import annotations

import hashlib
import json
import os

from . import frame as F
from . import optics as O
from . import route as R
from . import silhouette as S
from .action import LEGAL, TOGGLE, X, Y, Z, key_for
from .book import BuildState

TRACE = "gauge_trace.jsonl"
FENCE = (0, 540, 222)          # the sub-frame the identity fence hashes: the viewport, no clock

_DIGEST = lambda a: hashlib.md5(a[FENCE[0]:FENCE[1], FENCE[2]:].tobytes()).hexdigest()
_PRESS = lambda nm: key_for(nm)
_REPORT = lambda g: O.basis_report(g) if g else {}
_UNSEEN = lambda g: tuple(k for k in (X, Y, Z) if g is not None and k not in g.basis)
_LOOKABLE = lambda g, seen: seen and g.ready() and len(seen[0]) == g.whole


def _action_cls(proposal):
    """The engine `Action` class — imported when importable, else the live proposal's own class."""
    try:
        from core.data_classes import Action
        return Action
    except Exception:                                  # noqa: BLE001 — fall back to the live class
        return proposal.__class__


def _proposed_name(proposal):
    """The action NAME the frozen model asked for, or None when it is not a legal action.

    The evaluator maps an unknown name to `move_right` and leaves the bogus name in
    `metadata["function_called"]`, so reading that field is what separates a real proposal from a
    stray `right` the parser invented."""
    meta = getattr(proposal, "metadata", None) or {}
    name = meta.get("function_called")
    if isinstance(name, str) and name in LEGAL:
        return name
    return None


class Pilot:
    """Per-episode state and the closed perception loop: ruler, projection, scene, tape, ledger."""

    def __init__(self, evaluator):
        self.ev = evaluator
        self.st = BuildState()
        self.gauge = None
        self.panels = None
        self.hull = None
        self.need = frozenset()
        self.sure = False
        self.tape = []
        self.trim = None
        self.ready = False
        self.fence = None
        self.seen = None

    def measure(self, frame_data):
        """One shot on turn one: fit the ruler, audit it, read the scene, compile the tape."""
        pix = F.raster(frame_data)
        self.gauge = O.fit_gauge(pix)
        if not all((self.gauge, self.gauge and O.gauge_audit(pix, self.gauge))):
            return False
        scene = S.read_scene(pix, self.gauge)
        if scene is None:
            return False
        self.panels, self.hull = scene
        self.need = frozenset(O.forced(self.hull, self.panels))
        if not self.need <= self.hull:
            return False                               # a proof outside its own envelope
        self.sure = O.proved(self.hull, self.panels)
        self.tape = R.build_tape(self.hull, self.panels, self.st.cursor)
        self.ready = len(self.tape) > 0
        return self.ready

    def sight(self, frame_data):
        """CLOSE THE LOOP: calibrate on the last applied move, then read the cursor off THIS
        frame. Returns the measured cell, or None.

        None covers two different situations and both of them must leave the ledger's own
        dead-reckoned cursor standing: the projection is not calibrated yet, or THE CURSOR IS NOT
        IN THE PICTURE. The second one is real and it is not rare — the highlight is drawn in the
        scene, so a cube standing nearer the camera can hide it completely, and on one measured
        level that happens on the very press that would have won the episode. A readout is an
        opportunity to correct the prior, never a precondition for having one."""
        now = self.gauge.spot(F.raster(frame_data))
        self.gauge.learn(self.st.last, self.seen, now)
        if now and self.gauge.anchor is None:
            self.gauge.imprint(now)
        self.seen = now
        return self.gauge.locate(now) if _LOOKABLE(self.gauge, now) else None

    def goal(self):
        """The cell this turn is for: the first unplaced tape cell, else the first live trim."""
        for cell in self.tape:
            if cell not in self.st.placed:
                return cell, "build"
        if self.trim is None:
            self.trim = R.trim_tape(self.st.placed, self.panels, self.st.cursor)
        for cell in self.trim:
            if cell in self.st.placed and not self.st.booked(cell):
                return cell, "trim"
        return None, "dry"

    def log(self, row):
        """Append one line of telemetry beside the evaluator's own logs, or silently skip."""
        where = getattr(self.ev, "log_dir", None)
        if where:
            try:
                with open(os.path.join(where, TRACE), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(dict(row, ruler=_REPORT(self.gauge)), default=str) + "\n")
            except OSError:                            # telemetry never kills an episode
                pass


async def decide(evaluator, frame_data):
    """One decision -> engine Action. Asks the frozen model, measures the board, then prices."""
    pilot = getattr(evaluator, "_gauge_pilot", None)
    if pilot is None:
        pilot = Pilot(evaluator)
        evaluator._gauge_pilot = pilot
    st = pilot.st
    st.turn += 1
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model

    if st.turn == 1 and not pilot.ready:
        try:
            st.down = not pilot.measure(frame_data)
        except Exception as exc:                       # noqa: BLE001 — a bad read is a stand-down
            st.down = True
            st.notes.append((st.turn, repr(exc)))

    proposal = await step(frame_data)
    if st.down:
        pilot.log(dict(st.summary(), mode="stood-down"))
        return proposal

    now = _DIGEST(F.raster(frame_data))
    st.grade(now == pilot.fence)
    pilot.fence = now
    st.correct(pilot.sight(frame_data))

    goal, phase = pilot.goal()
    if goal is None:
        st.down = True
        pilot.log(dict(st.summary(), mode="dry"))
        return proposal

    asked = _proposed_name(proposal)
    want = TOGGLE if st.cursor == goal else R.step_toward(st.cursor, goal, _UNSEEN(pilot.gauge))
    if asked == want:
        applied, why = asked, "agrees"
    elif all((asked, want != TOGGLE, R.closes_gap(st.cursor, goal, asked))):
        applied, why = asked, "free-leg"
    else:
        applied, why = want, "planned"
    st.honoured += why != "planned"
    st.apply(applied)
    pilot.log(dict(st.summary(), phase=phase, goal=list(goal), asked=asked, applied=applied,
                   why=why, cursor=list(st.cursor), hull=len(pilot.hull), tape=len(pilot.tape),
                   sure=pilot.sure))
    if applied == asked:
        return proposal
    cls = _action_cls(proposal)
    meta = dict(getattr(proposal, "metadata", None) or {})
    meta.update({"frame_number": getattr(frame_data, "frame_number", st.turn),
                 "function_called": applied, "proposed": asked, "arbitration": why,
                 "goal": list(goal), "phase": phase})
    return cls(type="key_press", key=_PRESS(applied), metadata=meta)
