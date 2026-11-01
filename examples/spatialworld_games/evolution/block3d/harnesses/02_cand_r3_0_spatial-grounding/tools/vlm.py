"""The decision loop: the planned tape, the frozen model's vote, and SILENCE as the default.

The parent spent one model call on every one of its presses and let the answer decide everything. The
measurement that reshapes this loop: on the parent's sweep, committing was informative — a toggle was
two and a half times likelier when the cursor stood on a wanted cell — while MOVING was worse than not
looking, every one of the six directions being emitted more often exactly when it was blocked. So the
route belongs to the plan and the model keeps the choices the plan is indifferent about.

That gives an ask policy with one rule: ASK ONLY WHEN THE ANSWER CAN CHANGE THE KEY.

  * the first decision is always the pristine pass-through, uninjected — the opening frame is the one
    the parent is best at, it is the only decision made before the plan exists, and it is what keeps
    the episode's model-call count honest;
  * while TRAVELLING, every key that strictly closes the L1 gap costs the same and leaves the plan on
    the same chain, so when two or more axes are open the model picks which one to close, verbatim;
  * when the tape is SPENT and surplus blocks are standing, which one to peel is a free choice between
    equals, so the model makes it;
  * standing on the next tape cell, the key is the toggle and no answer could change it — so nothing is
    asked, and the press is spent in silence.

Everything else is routed, and two presses are refused outright: a move into a grid face (the ledger
knows it cannot move the cursor) and a toggle that would cancel a block the plan still wants.

If the strip will not decode, or the three marginals disagree, or the panels admit no structure, the
whole mechanism stands down and the parent's pass-through runs for the rest of the episode. A frame
this planner does not understand costs nothing and is never guessed at.
"""
import json
import os

from tools import frame, layers, panels, press, serpent
from tools.ledger import Ledger

_SESSIONS = {}
ACK = "OK."
BRIEF_CAP = 12
# Module-level one-liners: the presentation shaping, kept out of the way of the decision logic.
_RECORD = lambda cells: [list(cell) for cell in cells]
_DEPTHS = lambda census: {str(z): list(pair) for z, pair in census.items()}
_NAMES = lambda keys: [press.action_name(k) for k in keys]
_SEEN = lambda placed, cap: sorted(tuple(cell) for cell in placed)[:cap]
_LAYERS = lambda census: " ".join("z%d:%dx%d" % (z, w, t) for z, (w, t) in sorted(census.items()))


class Session:
    """Episode-scoped state: the ledger, the decoded panels, the tape, the trace sink."""

    def __init__(self):
        self.ledger = Ledger()
        self.tape = None
        self.wanted = set()
        self.shape = []
        self.sole = []
        self.census = {}
        self.spare = []
        self.thinnest = 0
        self.degraded = False
        self.calls = 0
        self.asks = 0
        self.honoured = 0
        self.routed = 0
        self.template = ""
        self.trace = None
        self.stood_down = ""
        self.published = False

    def stand_down(self, why):
        """Refuse the whole mechanism, by name. A frame this planner does not understand must cost the
        episode nothing, and the REASON has to reach the trace — a silent fallback is the one failure
        the next diagnosis cannot see."""
        self.degraded = True
        self.stood_down = why
        return False

    def lay_plan(self, image):
        """Decode the strip once and compile the tape, or stand down. Returns True while planning."""
        masks = panels.read_strip(image)
        if masks is None:
            return self.stand_down("strip-unreadable")
        if not panels.marginals_agree(masks):
            return self.stand_down("marginals-disagree")
        hull = panels.hull_of(masks)
        if len(hull) < 1:
            return self.stand_down("hull-empty")
        shape = layers.thin_shape(masks)
        self.wanted = panels.squares_wanted(masks)
        if not shape:
            return self.stand_down("no-admitted-structure")
        if not layers.settles(shape, self.wanted):
            return self.stand_down("plan-leaves-debt")
        self.census = panels.census(masks)
        self.shape = shape
        self.sole = layers.anchored(hull, self.wanted)
        self.tape = serpent.schedule(hull, self.sole, self.ledger.cursor)
        self.spare = layers.phantoms(hull, self.wanted)
        self.thinnest = max(layers.support(cell, hull) for cell in hull)
        return True

    def certificate(self):
        """What the plan has promised and what it has cost so far — self-derived, for the trace."""
        tape = self.tape or []
        return {"floor": len(self.shape), "forced": len(self.sole), "tape": len(tape),
                "asks": self.asks, "honoured": self.honoured, "routed": self.routed,
                "thinnest": self.thinnest, "stood_down": self.stood_down,
                "toll": press.toll(press.SPAWN, tape),
                "steps": press.engine_steps(press.toll(press.SPAWN, tape)),
                "per_cell": round(press.toll(press.SPAWN, tape) / max(len(tape), 1), 2)}

    def publish(self, evaluator):
        """Write the compiled plan ONCE, so the next diagnosis reads it instead of re-deriving it.

        Reconstructing what a block3d harness believed was the whole cost of the last diagnosis: no
        frames and no plan are archived, so every perception claim had to be re-measured from scratch.
        One file per episode fixes that, and it is written from self-derived state only.
        """
        if self.published or self.trace is None:
            return False
        self.published = True
        if not self.trace:
            return False
        try:
            with open(self.trace.replace("trace.jsonl", "plan.json"), "w", encoding="utf-8") as sink:
                json.dump({"census": _DEPTHS(self.census), "floor": _RECORD(self.shape),
                           "forced": _RECORD(self.sole), "spare": _RECORD(self.spare),
                           "tape": _RECORD(self.tape or []),
                           **self.certificate()}, sink, indent=2, default=str)
        except Exception:                            # noqa: BLE001 — publishing is never fatal
            return False
        return True

    def note(self, evaluator, record):
        """Append one decision to the episode journal. Tracing is never allowed to be fatal."""
        if self.trace is None:
            room = getattr(evaluator, "log_dir", None) or "."
            try:
                os.makedirs(room, exist_ok=True)
                self.trace = os.path.join(room, "sweep_trace.jsonl")
            except Exception:                        # noqa: BLE001
                self.trace = ""
        if self.trace:
            try:
                with open(self.trace, "a", encoding="utf-8") as sink:
                    sink.write(json.dumps(record, default=str) + "\n")
            except Exception:                        # noqa: BLE001
                pass


def session_for(evaluator):
    held = _SESSIONS.get(id(evaluator))
    if held is None:
        held = Session()
        _SESSIONS[id(evaluator)] = held
    return held


async def _frozen(evaluator, frame_data):
    """The pristine per-step VLM fn (never `openai_ai_model`, which is now this very agent)."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    return await step(frame_data)


def _inject(evaluator, text):
    """Put the live brief in front of the model WITHOUT touching the frozen system turn.

    `build_conversation_history` reuses the first history entry as the system message for every later
    call, so the response contract and the prompt prefix must stay byte-identical — a brief spliced into
    that entry deletes the JSON contract from every subsequent call. The brief therefore goes in the
    DIALOGUE tail as a user turn plus a one-word acknowledgement, a PAIR because a trailing odd message
    is dropped by the history truncation. Skipped while the history is empty, which is the first
    decision: that one stays pristine.
    """
    talk = getattr(evaluator, "conversation_history", None)
    if not talk:
        return False
    try:
        talk.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
        talk.append({"role": "assistant", "content": [{"type": "output_text", "text": ACK}]})
    except Exception:                                # noqa: BLE001 — never break the model's turn
        return False
    return True


def _brief(session, goal, peel):
    """Fill the memory.md scratchpad template with self-derived state only — never engine state."""
    led = session.ledger
    here = led.cursor
    if goal is not None:
        errand = "PUT A BLOCK at %s — you are %d move(s) away" % (tuple(goal), press.l1(here, goal))
    elif peel:
        errand = ("REMOVE one block that is not part of the shape; %d are equally allowed %s — walk "
                  "toward whichever you believe is surplus" % (len(peel), peel[:BRIEF_CAP]))
    else:
        errand = "HOLD the structure you have; the plan is settled"
    open_axes = _NAMES(press.closing_keys(here, goal))
    if goal is not None and tuple(here) == tuple(goal):
        open_axes = ["place_block"]
    cert = session.certificate()
    lines = [
        "HARNESS BRIEF — working state derived from your own presses, not read from the game:",
        "- cursor_now: %s" % (tuple(here),),
        "- blocks_you_have_placed: %s" % (_SEEN(led.placed, BRIEF_CAP),),
        "- presses: %d spent, %d left (every action you return costs one)" % (led.presses, led.left()),
        "- the panels need AT LEAST %d blocks and AT MOST %d; %d of them are load-bearing"
        % (cert["floor"], cert["tape"], cert["forced"]),
        "- per-height demand (columns x rows): %s" % _LAYERS(session.census),
        "- objective_now: %s" % errand,
        "- then: %s" % (serpent.ahead(session.tape or [], led.placed)[1:] or ["nothing"],),
        "- equally_short_choices: %s" % (open_axes or ["none"],),
        "- recent_presses: %s" % (led.recent(3),),
        "",
        (session.template or "").strip(),
        "",
        'Reply with ONLY the JSON object: {"action": "<one of the seven action names>"}.',
    ]
    return "\n".join(line for line in lines if line is not None)


def _arbitrate(session, goal, peel, prefer):
    """(key, reason). The ladder: a free rung is honoured verbatim, everything else is routed."""
    led = session.ledger
    here = led.cursor
    if goal is None:
        if not peel:
            for key in press.MOVE_KEYS:
                if press.moves_cursor(here, key):
                    return (key, "settled:hold")
            return (press.MOVE_KEYS[0], "settled:boxed")
        if tuple(here) in serpent._SEATS(peel):
            return (press.TOGGLE, "peel:toggle")
        if prefer is not None:
            steered = [cell for cell in peel if prefer in press.closing_keys(here, cell)]
            if steered:
                return (prefer, "peel:steered")
        seat = min(peel, key=lambda cell: (press.l1(here, cell), cell))
        return (press.closing_keys(here, seat)[0], "peel:routed")
    if tuple(here) == tuple(goal):
        return (press.TOGGLE, "place:toggle")
    open_axes = press.closing_keys(here, goal)
    if prefer in open_axes:
        session.honoured = session.honoured + 1
        return (prefer, "travel:honoured")
    if prefer in press.DELTA_BY_KEY and not press.moves_cursor(here, prefer):
        led.refused = led.refused + 1
    session.routed = session.routed + 1
    return (open_axes[0], "travel:routed")


async def decide(evaluator, frame_data):
    """One decision -> engine Action. The model is asked only when its answer can change the key."""
    s = session_for(evaluator)
    s.calls = s.calls + 1
    image = frame.capture_array(evaluator, frame_data)
    first = s.calls < 2
    if image is None:
        return await _frozen(evaluator, frame_data)  # no array, nothing to read: the parent's loop
    if s.tape is None and not s.degraded:
        s.lay_plan(image)
    if s.degraded:
        return await _frozen(evaluator, frame_data)
    if not s.tape:
        s.stand_down("tape-empty")
        return await _frozen(evaluator, frame_data)
    s.ledger.audit(frame.digest_of(image), frame.cursor_from_ink(image),
                   frame.highlight_mask(image) is not None)
    goal = serpent.outstanding(s.tape, s.ledger.placed)
    peel = serpent.surplus(s.tape, s.ledger.placed) if goal is None else []
    here = s.ledger.cursor
    if goal is not None and not press.inside(goal):
        s.stand_down("goal-off-grid")
        return await _frozen(evaluator, frame_data)
    open_axes = press.closing_keys(here, goal)
    free = len(open_axes) if goal is not None else len(peel)
    ajar = free > 1
    proposal = None
    prefer = None
    if first or ajar:  # ask only when the answer could change the key
        s.asks = s.asks + 1
        if not first:
            _inject(evaluator, _brief(s, goal, peel))
        proposal = await _frozen(evaluator, frame_data)
        prefer = getattr(proposal, "key", None)
    key, reason = _arbitrate(s, goal, peel, prefer)
    s.ledger.spend(key)
    s.publish(evaluator)
    record = {"call": s.calls, "key": key, "proposed": prefer, "reason": reason,
              "goal": None if goal is None else list(goal), "asked": bool(proposal),
              **s.ledger.snapshot(), **s.certificate()}
    s.note(evaluator, record)
    if proposal is not None:
        proposal.key = key
        meta = proposal.metadata if isinstance(getattr(proposal, "metadata", None), dict) else {}
        meta["function_called"] = press.action_name(key)
        meta["sweep"] = record
        meta.pop("repeat", None)
        proposal.metadata = meta
        return proposal
    from core.data_classes import Action
    return Action(type="key_press", key=key,
                  metadata={"function_called": press.action_name(key), "sweep": record})
