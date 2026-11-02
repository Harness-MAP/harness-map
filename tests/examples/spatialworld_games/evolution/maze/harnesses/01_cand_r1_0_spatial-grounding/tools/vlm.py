"""Perceive -> ground -> VLM -> arbitrate -> act for the maze harness.

SPATIAL GROUNDING. The frozen VLM is still asked exactly once per frame, but its proposal is now
checked against a metric model of the agent's own surroundings that this module builds out of the
rendered frames alone: a dead-reckoned cell/heading lattice, a per-side wall/open edge ledger, and
a palette-free flatness census of three vertical view bands whose wall threshold is CALIBRATED AT
RUN TIME against the one free label the engine hands back every step -- whether the view changed at
all after a forward.

Nothing here asserts anything about pixels. The bands are FRACTIONS of whatever frame arrives, the
statistic is a luminance gradient (no colour constants, no HUD crop, no glyph templates), and the
threshold that turns that statistic into "wall" or "open" is learned from paired samples: the front
flatness measured immediately before a one-cell forward, labelled on the next frame by whether the
view held. Until both label classes are populated AND separated by a measured margin the census has
no opinion at all (`_call` returns None), and the lattice runs on the edge ledger alone -- which is
the honest floor of this mechanism, and the reason a mis-sited band costs probes rather than
correctness.

WHERE THE TWO SIGNALS RANK. The ledger is the authority: an edge the agent has walked through is
open for good, and `_verdict` consults `open` before `walls` because a held digest is only an
absence of change while a crossing is a positive observation. The census is allowed to do two
things with the sides the ledger has no entry for: ORDER them (`_rank`, so the first probe goes to
the most open-looking side) and SKIP them (`_said`, so an obvious wall is not walked into). The
skip is what buys the step budget back, and its failure mode is bounded on purpose: a side the
census wrongly calls a wall is still reachable through `_unproven`, the last fallback before
`_amnesia`, so a bad band loses steps rather than losing the level.

No engine state, maze file, pose API or target information is read. Every number below is either a
fraction of the captured frame or a count of the agent's own past observations.
"""

import json

# ── the view geometry, as FRACTIONS of the arriving frame; never pixel counts ──────────────────
HORIZON = (0.085, 0.975)        # vertical slab: everything under the HUD strip. Deliberately WIDE
#                                 -- where a corridor's floor/ceiling convergence falls in the
#                                 frame is exactly what cannot be guessed without frames in hand,
#                                 so the slab takes the whole column instead of betting on a band.
DOWN, QUANT = 6, 18.0           # view-digest decimation + luminance quantum (glow-insensitive)
CAL_MIN, CAL_GAP = 4, 0.035     # samples needed per label class, and the separation to trust them
NAMES = {"w": "move_forward", "a": "turn_left", "d": "turn_right"}
TURN = {"a": 3, "d": 1}         # heading increment mod 4 for each rotation key
STEPS = {"small": 1, "medium": 2, "large": 3}
RUNS = ("small", "small", "medium", "large")     # proven-open reach -> granularity label
DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))       # heading 0..3 -> (drow, dcol), start-relative
RING = (1, 0, 3, 2)             # tie-break scan order: right, straight, left, back
BAND_OF = {0: 1, 1: 2, 3: 0}    # relative heading -> census index (the back side has no band)
ORIGIN = (0, 0)

# ── module-level numeric kernels. Pure, tiny, and deliberately kept OUT of the four components
#    below so each component's own control flow stays readable at a glance. ────────────────────
_lum = lambda px: px[..., :3].mean(axis=2)
_rows = lambda img, lo, hi: img[int(img.shape[0] * lo):int(img.shape[0] * hi)]
_slab = lambda img, lo, hi: _rows(img, HORIZON[0], HORIZON[1])[:, int(img.shape[1] * lo):
                                                               int(img.shape[1] * hi)]
_grad = lambda g: float(abs(g[1:] - g[:-1]).mean())
_flat = lambda img, lo, hi: 1.0 / (1.0 + _grad(_lum(_slab(img, lo, hi))))
_view = lambda img: hash((_lum(img)[::DOWN, ::DOWN] // QUANT).tobytes())
_meta = lambda a: a.metadata or {}

# the run-time calibration: two running (count, sum) accumulators and the cut between their means
_acc = lambda: [0, 0.0]
_tally = lambda acc, x: acc.__setitem__(0, acc[0] + 1) or acc.__setitem__(1, acc[1] + x)
_mean = lambda acc: acc[1] / max(acc[0], 1)
_ready = lambda lo, hi: min(lo[0], hi[0]) >= CAL_MIN and _mean(hi) - _mean(lo) >= CAL_GAP
_cut = lambda lo, hi: (_mean(lo) + _mean(hi)) / 2.0
_call = lambda lo, hi, x: (x > _cut(lo, hi)) if _ready(lo, hi) else None

# the lattice algebra
_edge = lambda cell, h: (cell[0], cell[1], h)
_step = lambda cell, h, n: (cell[0] + DELTA[h][0] * n, cell[1] + DELTA[h][1] * n)
_rev = lambda h: (h + len(DELTA) // 2) % len(DELTA)
_look = lambda census, r: census[BAND_OF[r]] if r in BAND_OF else 1.0
# `_look`'s 1.0 default for the BACK side is a sort key, not a reading -- there is no band behind
# the agent. A veto may only speak for a side some band actually covers, or the census silently
# condemns the one direction the agent came in through.
_said = lambda st, census, g: (g - st.heading) % len(DELTA) in BAND_OF and _call(
    st.lo, st.hi, _look(census, (g - st.heading) % len(DELTA))) is True
_rank = lambda st, census, g: (_step(st.cell, g, 1) in st.seen,
                               _look(census, (g - st.heading) % len(DELTA)),
                               RING.index((g - st.heading) % len(DELTA)))
_scan = lambda st, census: tuple(sorted(((st.heading + r) % len(DELTA) for r in RING),
                                        key=lambda g: _rank(st, census, g)))
_spent = lambda st, census, g: (g in st.tried or _edge(st.cell, g) in st.walls
                                or _said(st, census, g)) and _edge(st.cell, g) not in st.open

# THE EXPLORATION ORDER, as a chain of fallbacks over measurements only:
#   _novel     a side neither spent nor census-vetoed whose cell is somewhere NEW, best-looking
#              band first -- this is where the census earns its step budget;
#   _retreat   else the side pointing back the way this cell was first entered. Depth-first
#              backtracking; without it an exhausted branch ping-pongs between two mapped cells;
#   _openside  else any side already proven open (reachable at the start cell, which has no way
#              back, and after an `_amnesia` repair);
#   _unproven  else any side the LEDGER has not ruled out -- the census's veto is dropped here, so
#              a band that wrongly called an opening a wall costs steps and not the level;
#   _amnesia   else the record for this cell must be wrong: a cell you are STANDING IN cannot have
#              four walls, you came in through one of them. Forget it and measure again.
_novel = lambda st, census: tuple(g for g in _scan(st, census)
                                  if not _spent(st, census, g)
                                  and _step(st.cell, g, 1) not in st.seen)
_retreat = lambda st: tuple(g for g in (st.back.get(st.cell),) if g is not None)
_openside = lambda st, census: tuple(g for g in _scan(st, census)
                                     if _edge(st.cell, g) in st.open)
_unproven = lambda st, census: tuple(g for g in _scan(st, census)
                                     if g not in st.tried and _edge(st.cell, g) not in st.walls)
_amnesia = lambda st: (st.tried.clear(), tuple(st.walls.discard(_edge(st.cell, g))
                                               for g in range(len(DELTA))), ())[-1]
_frontier = lambda st, census: (_novel(st, census) or _retreat(st) or _openside(st, census)
                                or _unproven(st, census) or _amnesia(st)
                                or _unproven(st, census) or (st.heading,))[0]

# COMMITMENT. The scan is pose-relative, so recomputing it after every rotation lets the aim drift
# round with the agent and it never arrives -- the spin that burns whole episodes. Hold the chosen
# heading until it is spent (probed, proven a wall, or vetoed by the census) or the cell changes.
_aim = lambda st, census: st.goal if st.goal is not None and not _spent(st, census, st.goal) \
    else _frontier(st, census)

# `open` is read BEFORE `walls`: a side the agent walked through is a positive observation, while a
# held digest is only an absence of change, and a false wall is the costly direction of the error.
_verdict = lambda st, front: False if _edge(st.cell, st.heading) in st.open else (
    True if _edge(st.cell, st.heading) in st.walls else _call(st.lo, st.hi, front))

# When is the model's forward worth taking? Not into a measured wall, AND either the lattice wanted
# this heading anyway or the cell ahead is somewhere new. Without the second clause an "always
# forward" reflex re-walks a corridor it has already mapped instead of resuming the frontier.
# A `medium`/`large` proposal across an UNMAPPED edge is refused even when the edge is not a known
# wall: such a push stops early at the first wall it meets, so its length is not a number the
# lattice can credit, and honouring it would desynchronise the dead-reckoned cell. The override
# below then re-emits the same heading at the run length the ledger can actually vouch for.
_runof = lambda a: STEPS.get(_meta(a).get("granularity", "small"), 1)
_honours = lambda st, proposal, goal, blocked: blocked is not True \
    and _runof(proposal) <= _reach(st, 1) and (
        goal == st.heading or _step(st.cell, st.heading, 1) not in st.seen)
# And the model's TURN? Only while the lattice still wants another heading, and only onto a heading
# not already faced since the pose last changed -- a left after a right with nothing moving in
# between cancels, which is how two episodes burned thirty-two turns each.
_turnable = lambda st, goal, turned: goal != st.heading and turned not in st.faced

# a long push STOPS EARLY at a wall, so crediting its full length would desynchronise the cell:
# credit `run` only when every edge it crossed was already proven open, else credit one cell
_credit = lambda st, run: run if run <= _reach(st, 1) else 1
_reach = lambda st, n: n if n >= len(RUNS) - 1 or _edge(_step(st.cell, st.heading, n),
                                                        st.heading) not in st.open \
    else _reach(st, n + 1)
_advance = lambda st, run: (setattr(st, "cell", _step(st.cell, st.heading, _credit(st, run))),
                            st.back.setdefault(st.cell, _rev(st.heading)),
                            st.seen.add(st.cell), st.tried.clear(),
                            setattr(st, "goal", None), setattr(st, "faced", {st.heading}))
_record = lambda key, census, run: (key, census[1], census[3], run)
_mark = lambda st, held, run, flat, grade: (
    _tally(st.hi if held else st.lo, flat) if grade else None,
    (st.walls if held else st.open).add(_edge(st.cell, st.heading)),
    st.tried.add(st.heading),
    None if held else _advance(st, run))


def _band_statistics(frame_data):
    """The whole per-frame measurement: left / front / right flatness, plus a view digest.

    Flatness is 1/(1 + mean |vertical luminance difference|) over a band's horizon slab. A wall
    face one cell away is a near-constant vertical wash, so its flatness is high; a corridor
    receding to a vanishing point is not, so its flatness is low. The statistic is monotone in
    "how blocked is that side" without naming one colour, which is what lets the same reader work
    whatever palette the renderer actually uses and whatever size the capture arrives at. The
    digest is a decimated, coarsely quantised luminance grid, so a pulsing exit panel cannot
    masquerade as a pose change."""
    image = frame_data.image
    return (_flat(image, 0.055, 0.285), _flat(image, 0.395, 0.605),
            _flat(image, 0.715, 0.945), _view(image))


def _probe_action(proposal, key, frame_number, granularity):
    """A fresh legal engine Action. Never inherits the vetoed proposal's granularity or repeat."""
    name = NAMES[key] if key in NAMES else NAMES["d"]
    meta = {"frame_number": frame_number, "ai_model": "openai_ai+grounded_lattice",
            "function_called": name, "grounding_override": True}
    if key == "w":
        meta["granularity"] = granularity
        meta["repeat"] = STEPS[granularity]
    return type(proposal)(type="key_press", key=key, metadata=meta)


def _overwrite_last_action(evaluator, action):
    """Rewrite the newest assistant turn to the action actually executed.

    The evaluator feeds a rolling image+text window back to the model every step. Leaving a vetoed
    wall-push inside that window teaches the policy that the push succeeded, which is exactly the
    self-confirming loop that produced byte-identical hundred-step tapes on different mazes."""
    payload = {"action": NAMES[action.key]}
    if action.key == "w":
        payload["granularity"] = _meta(action).get("granularity", "small")
    turns = getattr(evaluator, "conversation_history", None)
    if not isinstance(turns, list):
        return
    text = json.dumps(payload, separators=(",", ":"))
    for turn in reversed(turns):
        if turn.get("role") == "assistant":
            turn["content"] = [{"type": "output_text", "text": text}]
            return


class _GroundedLatticeRecovery:
    """The agent's own metric map of where it has been, and the frontier policy that reads it.

    Every field is derived from observations the model was shown. A one-cell forward that holds the
    view digest proves the side ahead is a wall; one that changes it proves the side is open and
    advances the dead-reckoned cell. The same pair of events labels the flatness samples, so the
    pixel census is AUDITED by the ledger instead of being trusted ahead of it, and `medium` or
    `large` runs are emitted only along edges the ledger has already proven open -- which is what
    keeps dead reckoning exact, since a long push that stops early would desynchronise the cell."""

    def __init__(self):
        self.cell = ORIGIN
        self.heading = 0
        self.walls = set()
        self.open = set()
        self.seen = {ORIGIN}
        self.back = {}
        self.goal = None
        self.tried = set()
        self.faced = {self.heading}
        self.last = None
        self.graded = False
        self.lo = _acc()
        self.hi = _acc()

    def settle(self, census):
        """Attribute the PREVIOUS action's outcome. One comparison, and it is a measurement."""
        prior = self.last
        if prior is None:
            return
        key, flat, digest, run = prior
        self.heading = (self.heading + TURN.get(key, 0)) % len(DELTA)
        self.faced.add(self.heading)
        if key == "w":
            _mark(self, digest == census[3], run, flat, self.graded)

    def arbitrate(self, proposal, census, frame_number):
        """Let the model drive; override only what the lattice has already measured impossible."""
        blocked = _verdict(self, census[1])
        goal = _aim(self, census)
        self.goal = goal
        turned = (self.heading + TURN.get(proposal.key, 0)) % len(DELTA)
        if proposal.key == "w" and _honours(self, proposal, goal, blocked):
            return proposal, False
        if proposal.key != "w" and _turnable(self, goal, turned):
            return proposal, False
        if goal != self.heading or blocked is True:
            return _probe_action(proposal, "d", frame_number, "small"), True
        return _probe_action(proposal, "w", frame_number,
                             RUNS[_reach(self, 1)] if blocked is False else "small"), True

    def commit(self, action, census):
        """Remember what we returned, and the flatness that will label its outcome next frame."""
        run = STEPS.get(_meta(action).get("granularity", "small"), 1)
        self.last = _record(action.key, census, run)
        self.graded = action.key == "w" and run == 1


async def decide(evaluator, frame_data):
    """One grounded decision per frame: measure -> settle -> ask the VLM once -> arbitrate."""
    lattice = getattr(evaluator, "_maze_grounded_lattice", None)
    if lattice is None:
        lattice = _GroundedLatticeRecovery()
        evaluator._maze_grounded_lattice = lattice
    census = _band_statistics(frame_data)
    lattice.settle(census)
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    proposal = await step(frame_data)
    action, overridden = lattice.arbitrate(proposal, census, frame_data.frame_number)
    if overridden:
        _overwrite_last_action(evaluator, action)
    lattice.commit(action, census)
    return action
