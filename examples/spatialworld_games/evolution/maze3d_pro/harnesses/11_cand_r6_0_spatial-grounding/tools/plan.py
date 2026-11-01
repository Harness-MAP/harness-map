"""tools/plan.py — the CHART the flank ladder fills, and the pilot that routes on it.

THE CONSTRAINT THIS ANSWERS. The ancestors of this harness that could already read the corridor
paid ONE DECISION PER CELL for completeness, because the only neighbours they could classify were
the two beside the body: a corridor cell crossed by a stride kept two unknown neighbours, every
unknown neighbour made its cell a frontier, and the route turned back to look at ground it had
already walked. tools/sight.py now classifies the neighbours of every corridor cell the ray proves
open (4.43 cells per look against 2, 24938/24938 exact), so a cell is surrounded by settled map
before the body ever reaches it, and the frontier collapses onto the openings that are genuinely
unexplored instead of onto the corridor the agent is standing in.

WHAT IS CHARTED, AND WITH WHAT PROVENANCE. `MazeChart` holds one occupancy map per EPOCH (a map
namespace, one per floor entered) in a single reckoned coordinate frame, and it keeps two kinds of
"open" strictly apart:

  * PLAIN — a cell a forward ray passed THROUGH, or one the body has stood on. The ray stops at the
    first cell that is not plain floor (a wall, a stair and the exit all terminate it), so every
    cell it passes through is proven to be ordinary floor, i.e. proven NOT to be the goal.
  * PASSABLE-ONLY — a cell a FLANK band showed empty. The renderer paints the exit only as a front
    face, so an exit cell beside the corridor paints nothing and reads exactly like open floor.
    Such a cell is enterable and nothing more.

A cell is SETTLED when it is a wall, a stair, the exit, or proven plain. "No reachable frontier left
and no exit charted" therefore PROVES the exit is not on this floor, and that proof is what makes a
floor change free. Leaving a floor on the first stair instead costs every level whose exit is on the
starting floor, which is about half of them.

HOW THE POSE IS KEPT. A turn is never refused by the engine and `sight.reach` states before the move
how far a translation will travel (0 over-reaches in 5624 poses), so the pose is reckoned exactly and
the single bit "the frame did not change at all" settles a refusal. The one event that is NOT
observable is a floor change — a real climb can leave the frame byte-identical — so a climb is
committed optimistically into a fresh epoch and settled afterwards by comparing per-pose frame
digests across epochs: one mismatch confirms the transition, two matches retire the new epoch as a
refused climb and restore the map underneath it.

WHAT A DECISION BUYS (new this iteration). The episode is billed one decision per `ai_model_func`
call and nothing else, and a translation already buys up to three cells of it through `granularity`.
A ROTATION has the same knob and this lineage never used it, so it paid a whole decision per 90
degrees while turning took 48% of the budget (396 of 823 decisions on the panel, 309 of them spent
purely to acquire a view). `pivot_step` plans the rotation instead of the quarter: `look_bearing`
picks the direction that sweeps every bearing this cell still has to show in the shortest arc, and
`blind_arc` spends the whole arc on one decision only when `ray_charted` proves that every heading
it passes THROUGH has nothing left to reveal — an uncharted cell, or one known only from a flank
band and so possibly the exit, cuts the arc short so the next frame arrives from there. Nothing is
seen that the ancestor would have seen; the same looks simply cost fewer decisions.

THE MODEL STILL DECIDES. `evaluator.baseline_step` is called every step and its proposal executes
unless the chart proves it cannot help: a translation the frame shows blocked, a climb with no stair
charted underfoot, or a proposal that disagrees with a route the chart can prove. The route step is
computed first and the proposal is kept whenever it agrees, so agreeing costs the model nothing, and
every measured fact is stated back to it in plain words by `render_memory`.
"""
from __future__ import annotations

from collections import deque

from . import action as _act
from .sight import Sight, WALL, STAIR, EXIT, OPEN

PASSABLE = frozenset((OPEN, STAIR, EXIT))

#: how a floor change is ranked at a charted stair, cheapest first. New ground outranks a floor
#: already charted (GROUND_RANK); and a floor the harness has never stood on is outranked by one it
#: has (UNSEEN_RANK), which is the whole prior — it arrived on this floor by climbing, so the floor
#: it came FROM is known to exist while the one beyond this floor is pure hope. REFUTED, measured,
#: do not rebuild: also remembering per-FLOOR which direction the retirement test has disproved
#: ("the ceiling") makes one of that test's false positives permanent and floor-wide, and a level
#: whose every charted stair is then struck out becomes unwinnable at ANY budget (L23 did). The
#: refusal stays contained to one (cell, direction) in `chart.tried`, exactly as the ancestor had it.
GROUND_RANK, UNSEEN_RANK = 1, 2

#: which way each floor change counts.
STOREY_STEP = {"climb_up": 1, "climb_down": -1}


#: the working-memory block rendered into the agent's system turn every step. memory.md overrides
#: this whenever its placeholders match `Pilot.fields()`, which is what keeps the wording a lever.
DEFAULT_TEMPLATE = """## MEASURED STATE (read off the screen in front of you — not a guess)
- decision {step} of {max_steps}.
- straight ahead: {ahead_line}
- the cells beside the corridor: {flank_line}
- under your feet: {foot_line}
- your last decision: {last_line}
- this floor so far: {chart_line}
- the route: {plan_line}

Distances are counted in grid cells and are exact. The tile you are standing on is never drawn, and
the screen carries no HUD, no text and no coordinates. A move into a wall changes nothing at all, so
it is never worth a decision.

Reply with ONLY the JSON object in the format specified above."""


class MazeChart:
    """Occupancy, per epoch, in one reckoned coordinate frame. Pure bookkeeping: no pixels."""

    __slots__ = ("cells", "plain", "visits", "digests", "links", "tried", "epochs", "conflicts")

    #: an upgrade order, so a later look may sharpen an earlier one but never blunt it. A flank band
    #: reports the exit as merely open, so `open -> exit` has to be allowed; nothing is ever
    #: downgraded to `open`, and two DIFFERENT solid readings for one cell are counted as a conflict
    #: because the only way that happens is a broken pose.
    RANK = {OPEN: 0, WALL: 1, STAIR: 2, EXIT: 3}

    def __init__(self):
        self.cells = {}          # (epoch, x, y) -> kind
        self.plain = set()       # (epoch, x, y) proven ordinary floor: a ray went through it
        self.visits = {}         # (epoch, x, y) -> times stood on
        self.digests = {}        # (epoch, x, y, heading) -> frame digest
        self.links = {}          # (epoch, x, y, climb) -> the epoch it leads to
        self.tried = {}          # (epoch, x, y) -> climbs already spent from there
        self.epochs = {0}
        self.conflicts = 0

    def tag(self, epoch, cell):
        return self.cells.get((epoch,) + tuple(cell))

    def mark(self, epoch, cell, kind) -> None:
        """File one reading, keeping the sharper of the two when the cell is already charted."""
        key = (epoch,) + tuple(cell)
        old = self.cells.get(key)
        if old is None or old == kind:
            self.cells[key] = kind
            return
        if self.RANK[kind] > self.RANK[old]:
            if self.RANK[old] >= self.RANK[WALL]:
                self.conflicts += 1
            self.cells[key] = kind
        elif self.RANK[kind] >= self.RANK[WALL]:
            self.conflicts += 1

    def prove_plain(self, epoch, cell) -> None:
        """Record that this cell is ordinary floor — a ray went through it, or we stand on it."""
        self.mark(epoch, cell, OPEN)
        self.plain.add((epoch,) + tuple(cell))

    def settled(self, epoch, cell) -> bool:
        """Is this cell's identity decided? An unseen cell is not, and neither is a cell known only
        from a flank band, because that reading cannot tell the exit from empty floor."""
        key = (epoch,) + tuple(cell)
        kind = self.cells.get(key)
        return kind is not None and (kind != OPEN or key in self.plain)

    def unsettled_bearings(self, epoch, cell):
        """The bearings out of `cell` whose neighbour still has to be settled."""
        return [b for b in range(4) if not self.settled(epoch, _act.step_from(cell, b))]

    def frontier(self, epoch, cell) -> bool:
        """A reachable cell from which something unsettled can still be looked at."""
        if self.cells.get((epoch,) + tuple(cell)) not in PASSABLE:
            return False
        return bool(self.unsettled_bearings(epoch, cell))

    def alive(self, epoch) -> bool:
        """Does this epoch still hold a frontier — i.e. is that floor unfinished?"""
        return any(self.frontier(epoch, (x, y)) for (e, x, y) in list(self.cells) if e == epoch)

    def reach(self, epoch, start):
        """BFS over the charted-passable cells of one epoch: (parents, depth, cells in depth order).

        A route therefore never walks into a cell the frame has not already shown to be enterable,
        and `order` is non-decreasing in depth, so "the nearest such cell" is the first match in it.
        """
        parents, depth, order = {tuple(start): None}, {tuple(start): 0}, [tuple(start)]
        queue = deque(order)
        while queue:
            cur = queue.popleft()
            for b in range(_act.QUARTERS):
                nxt = _act.step_from(cur, b)
                fresh = nxt not in parents and self.cells.get((epoch,) + nxt) in PASSABLE
                if fresh:
                    parents[nxt], depth[nxt] = cur, depth[cur] + 1
                    order.append(nxt)
                    queue.append(nxt)
        return parents, depth, order

    @staticmethod
    def route(parents, target):
        path = [tuple(target)]
        while parents.get(path[-1]) is not None:
            path.append(parents[path[-1]])
        path.reverse()
        return path

    def retire(self, epoch) -> None:
        """Erase an epoch that turned out never to have been entered (a refused climb)."""
        for key in [k for k in self.cells if k[0] == epoch]:
            del self.cells[key]
        for key in [k for k in self.visits if k[0] == epoch]:
            del self.visits[key]
        self.plain -= {k for k in self.plain if k[0] == epoch}
        self.epochs.discard(epoch)


class Pilot:
    """Episode-local perception, reckoning, charting and governing. One instance per episode."""

    # ── levers (CLASS attributes: harness.py purges sys.modules["tools*"], so a rig must set
    #    them AFTER build_agent or it silently measures the shipped config) ──
    OBEY_MODE = "route"        # "route": a proposal that disagrees with the charted route is
                               #   displaced (the model is still obeyed whenever it agrees, when
                               #   there is no route at all, and it still votes on frontier ties);
                               #   "coverage": insist only on a PROVEN route (a charted exit or an
                               #   exhausted floor); "off": only measured-inert proposals move.
    FLOOR_GATE = "exhaust"     # "exhaust": leave a floor only once its exit is disproved by the
                               #   chart; "stale": also leave it after STALE_LIMIT decisions that
                               #   charted nothing new. The proof is the one with evidence behind
                               #   it; staleness is kept as a priced alternative, not a belief.
    STALE_LIMIT = 8            # decisions charting nothing new, for FLOOR_GATE == "stale".
    STRIDE_MAX = 3             # the engine's `large`; a stride costs one decision at any length.
    MAX_CLIMBS = 8             # a bound on floor transitions per episode, not a knob.
    TWIN_MATCHES = 2           # matching per-pose digests that retire an epoch as a refused climb.
    TWIN_WINDOW = 6            # decisions the retirement test stays open for a comparison.
    STALL_LIMIT = 4            # decisions without motion before the blind escape engages.
    LADDER = Sight.LADDER      # neighbour pairs filed per look; 0 reproduces the one-cell ancestor.
    PIVOT_MAX = 2              # quarter turns one decision may spend. 1 reproduces the ancestor,
                               #   which paid a decision per 90 degrees. SATURATED, not tuned: no
                               #   rotation onto a bearing is ever more than a half turn (`pivot_to`
                               #   takes the short way round), so 3 is byte-identical to 2 on the
                               #   whole panel. An arc is only ever spent whole when every heading
                               #   it passes THROUGH is already charted out.
    SWEEP = True               # rotate ONE way through all the bearings this cell still has to
                               #   look down; False reproduces the ancestor's nearest-pending pick.
    CEILING_LEARNS = False     # REFUTED, measured, do not turn on: ranking a floor change by
                               #   whether the harness has stood on the floor it leads to really
                               #   does kill every refused climb (5 of 25 levels, 17 decisions) —
                               #   and ON ITS OWN it costs SIX WINS, because reordering the climbs
                               #   walks the per-visit epoch model into a state where its epoch and
                               #   the engine's floor disagree and all four climbs of a two-stair
                               #   floor are spent, after which the chart offers nothing at all.
                               #   Fix the epoch model (one map per FLOOR, re-joined on return)
                               #   before reaching for this again.

    def __init__(self, max_steps: int = 100, template: str = ""):
        self.max_steps = int(max_steps or 100)
        self.template = _clean_template(template) or DEFAULT_TEMPLATE
        self.chart = MazeChart()
        self.epoch, self.x, self.y, self.heading = 0, 0, 0, 0
        self.sight = Sight(None)
        self.step = 0
        self.last = None               # the decision committed on the previous step
        self.prev_digest = None
        self.twin = None               # the pending "did that climb happen?" comparison
        self.climbs = 0
        self.charted_mark, self.since_new, self.since_move = 0, 0, 0
        self.moves, self.overrides, self.filed = 0, 0, 0
        self.pivots, self.quarters = 0, 0      # rotations spent, and quarter turns they carried
        self.lost = False              # reckoning contradicted: fall back to the blind ledger
        self.refused = {}              # frame digest -> translations that provably did nothing
        self.storey = {0: 0}           # epoch -> reckoned floor, counted from the starting floor
        self.plan_note = "nothing measured yet"
        self.base_prompt = None        # stash of the agent's first-call system turn

    # ── the pose ────────────────────────────────────────────────────────────────────────────
    def cell(self):
        return (self.x, self.y)

    def neighbour(self, offset, cell=None):
        return _act.step_from(self.cell() if cell is None else cell,
                              _act.side_of(self.heading, offset))

    def on_stair(self) -> bool:
        return (not self.lost) and self.chart.tag(self.epoch, self.cell()) == STAIR

    # ── observation ─────────────────────────────────────────────────────────────────────────
    def observe(self, image) -> None:
        """Fold the frame now on screen into the chart, score the previous decision, and advance the
        reckoned pose. Called once per step, before the frozen model is asked."""
        sight = Sight(image)
        changed = self.prev_digest is None or sight.digest != self.prev_digest
        if self.last is not None:
            settle_last(self, self.last, changed, sight)
        self.sight = sight
        self.prev_digest = sight.digest
        self.step += 1
        if sight.ok and not self.lost:
            file_look(self, sight)
            twin_verdict(self, sight)
            key = (self.epoch, self.x, self.y)
            self.chart.visits[key] = self.chart.visits.get(key, 0) + 1
        if len(self.chart.cells) > self.charted_mark:
            self.charted_mark, self.since_new = len(self.chart.cells), 0
        else:
            self.since_new += 1
        if self.chart.conflicts >= 2:
            self.lost = True

    # ── the route ───────────────────────────────────────────────────────────────────────────
    def _climb_choice(self, cell):
        """The floor transition worth spending at a charted stair cell, or None.

        Preference: a transition never tried from here, into an epoch that does not exist yet or
        still holds a frontier. `climb_up` is tried first so the first transition of an episode goes
        away from the starting floor rather than into the ground — but only until the frame has
        DISPROVED it. A refused climb is not free: it costs the decision that spends it plus the
        TWIN_MATCHES frames the retirement test needs, and the ancestor paid that at every stair it
        ever reached on the top floor, because nothing remembered the result. The twin test's own
        verdict is the evidence (`barred`), it is self-derived, and it is per-floor rather than
        per-tile, so a direction this floor has already proven inert is ranked LAST instead of being
        struck out — a stair that really does lead that way is still tried, just not first."""
        spent = self.chart.tried.get((self.epoch,) + tuple(cell), ())
        standing = self.storeys_stood_on()
        best = None
        for name in filter(lambda n: n not in spent, _act.CLIMB_ORDER):
            rank = climb_worth(self, cell, name, standing)
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, name)
        return best[1] if best else None

    def _climb_rank(self, cell):
        """How good the floor change at this stair is, cheapest first, or None when there is none.

        `plan` used to route to the NEAREST stair with any climb left, which on a floor reached by
        climbing is often the very stair it came up: the way back down leads to the floor it just
        charted out, so nothing is left there worth the walk. Ranking the stairs before routing to
        one costs nothing — they are all already charted — and sends the harness to the stair that
        actually opens ground. Measured INERT on the six-tape panel (the nearest stair was already
        the best one on all 25 levels); kept because the pathology it removes is real."""
        name = self._climb_choice(cell)
        if name is None:
            return None
        standing = self.storeys_stood_on()
        return climb_worth(self, cell, name, standing)

    def storeys_stood_on(self):
        """The reckoned floors this episode has actually been on, or () when the prior is off."""
        if not self.CEILING_LEARNS:
            return ()
        return frozenset(self.storey.get(e, 0) for e in self.chart.epochs)

    def _impatient(self) -> bool:
        """May a floor change pre-empt an unfinished floor? Under the default gate it never may:
        a stair is spent only once every reachable cell of this floor is settled, which is the
        point at which "the exit is not on this floor" is proven rather than guessed."""
        return self.FLOOR_GATE == "stale" and self.since_new >= self.STALE_LIMIT

    def plan(self, prefer: str = ""):
        """(action, cells, note, mode) — the harness's own next step, or None when it has no plan.

        `mode` is what the step is FOR, and only the two modes resting on a proof are insisted upon:
        "exit" (a charted exit cell, reachable through cells known to be floor) and "stair" (this
        floor is exhausted, so the exit provably is not on it). While exploring, the harness has no
        proof its guess beats the model's, so the model keeps the wheel for as long as it agrees."""
        if self.lost or not self.sight.ok:
            return None
        parents, depth, order = self.chart.reach(self.epoch, self.cell())
        exits = [c for c in order if self.chart.tag(self.epoch, c) == EXIT]
        if exits:
            return goal_step(self, parents, depth, order, exits, "exit", prefer)
        frontier = [c for c in order if self.chart.frontier(self.epoch, c)]
        stairs = best_stairs(self, order)
        if stairs and self.climbs < self.MAX_CLIMBS and (not frontier or self._impatient()):
            if self.cell() in stairs:
                name = self._climb_choice(self.cell())
                if name:
                    return (name, 1, "this floor is charted out and holds no exit; changing floor "
                            "here", "stair")
            return goal_step(self, parents, depth, order, stairs, "stair", prefer)
        if frontier:
            return goal_step(self, parents, depth, order, frontier, "explore", prefer)
        return last_resort(self, parents, depth, order, prefer)

    # ── the governor ────────────────────────────────────────────────────────────────────────
    def govern(self, name: str, cells: int):
        """(action, cells, reason). An empty reason means the model's own proposal is executed."""
        route = self.plan(prefer=name)
        mode = route[3] if route else ""
        self.plan_note = route[2] if route else "no route of its own: following you"
        why = inert_reason(self, name)
        if not why and route and self.OBEY_MODE != "off":
            if self.OBEY_MODE == "route" or mode in ("exit", "stair"):
                why = "the charted route leads to %s" % {
                    "exit": "the exit", "stair": "a floor change"}.get(mode, "unexplored ground")
        if not why:
            return clamp_translation(self, name, cells) or (name, cells, "")
        pick = route or (escape_step(self) if self.lost
                         or self.since_move >= self.STALL_LIMIT else None)
        if pick is None:
            return (name, cells, "")
        if pick[0] == name and pick[1] == cells:
            return (name, cells, "")                   # the model already wants the route step
        return (pick[0], pick[1], why + "; " + pick[2])

    def apply(self, proposal):
        """Govern one proposal and return the engine Action the evaluator will execute."""
        name = _act.name_for(getattr(proposal, "key", None))
        meta = getattr(proposal, "metadata", None) or {}
        cells = _act.granularity_steps(meta.get("granularity"))
        if name is None:
            self._commit(None, 0, None)
            return proposal
        chosen, chosen_cells, why = self.govern(name, cells)
        if not why:
            self._commit(name, cells, None)
            return proposal
        self.overrides += 1
        out = _act.retarget(proposal, chosen, chosen_cells, why, name)
        self._commit(chosen, chosen_cells, name)
        return out

    def _commit(self, name, cells, displaced) -> None:
        expect = 0
        if name == "move_forward":
            expect = self.sight.stride(cells) if self.sight.ok else cells
        elif name == "move_backward":
            expect = max(1, cells)                     # the backward ray is never rendered
        quarters = max(1, min(_act.PIVOT_CAP, int(cells))) if name in _act.TURN_ACTIONS else 0
        self.last = {"name": name, "cells": cells, "expect": expect, "displaced": displaced,
                     "quarters": quarters, "digest": self.sight.digest, "step": self.step}
        if name in _act.CLIMB_ACTIONS and not self.lost:
            commit_climb(self, name)

    # ── the per-turn block ──────────────────────────────────────────────────────────────────
    def fields(self) -> dict:
        return {"step": self.step, "max_steps": self.max_steps,
                "ahead_line": self.sight.ahead_line(), "flank_line": self.sight.flank_line(),
                "foot_line": foot_line(self), "last_line": last_line(self),
                "chart_line": chart_line(self), "plan_line": self.plan_note,
                "moves": self.moves, "overrides": self.overrides, "filed": self.filed,
                "pivots": self.pivots, "quarters": self.quarters}

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


# ── the route, the vetoes and the escape: decisions over a Pilot, kept out of its body ──────
def route_step(pilot, parents, target, note, stop_on: bool):
    """One step of the route to `target`: a rotation onto its bearing, or a stride along it.

    The stride is the measured clear run clipped to the straight part of the route, and it never
    strides PAST a cell the route wants to stop on. Crossing a cell costs nothing now that its
    neighbours are charted from a distance — which is exactly what the flank ladder bought. The
    rotation is priced the same way: `pivot_step` spends one decision on the whole arc whenever
    every heading it passes through is already charted out."""
    if target == pilot.cell():
        pending = pilot.chart.unsettled_bearings(pilot.epoch, pilot.cell())
        if pending:
            step = pivot_step(pilot, look_bearing(pilot, pending),
                              "looking down the one bearing here still unsettled", note)
            if step:
                return step
        return None
    path = pilot.chart.route(parents, target)
    if len(path) < 2:
        return None
    bearing = _act.bearing_to(path[0], path[1])
    if bearing is None:
        return None
    turn = pivot_step(pilot, bearing, "turning onto the route (%s)" % note, note)
    if turn:
        return turn
    run = 1
    while run + 1 < len(path) and _act.bearing_to(path[run], path[run + 1]) == bearing:
        run += 1
    cells = min(run, pilot.STRIDE_MAX, max(1, pilot.sight.reach))
    if stop_on:
        cells = min(cells, len(path) - 1)
    return ("move_forward", max(1, cells), "stepping along the route (%s)" % note, note)


def goal_step(pilot, parents, depth, order, goals, note, prefer):
    """One decision toward the nearest of `goals`, or None when none of them is reachable.

    Standing on a goal is not a route: for `explore` it is the cue to LOOK down whatever bearing
    this cell still has to show, which is the only way a passable-only neighbour — the one kind of
    cell that could still be the exit — ever gets settled."""
    wanted = frozenset(tuple(c) for c in goals)
    if pilot.cell() in wanted and note == "explore":
        look = look_step(pilot, note)
        if look:
            return look
        wanted = wanted - {pilot.cell()}
    reachable = list(filter(wanted.__contains__, order))
    if not reachable:
        return None
    target = reachable[0]
    if note == "explore":
        target = choose_target(pilot, parents, depth, reachable, prefer)
    return route_step(pilot, parents, target, note, True)


def look_step(pilot, note):
    """The LOOK: rotate onto a bearing out of this cell that still has something to show."""
    pending = pilot.chart.unsettled_bearings(pilot.epoch, pilot.cell())
    if not pending:
        return None
    return pivot_step(pilot, look_bearing(pilot, pending),
                      "looking down the one bearing here still unsettled", note)


def last_resort(pilot, parents, depth, order, prefer):
    """A door back through a floor that LOOKED finished, when the chart has nothing else on offer.

    Reaching here means the chart says: no exit charted, nothing unseen left to look at, and every
    stair's remaining climb leads to a floor already charted out. On a solvable level that judgement
    has to be wrong somewhere — a floor is retired per VISIT, so a second stair down to the same
    floor opens a component the first visit could never reach, and "charted out" is a statement
    about the component, not the floor. The ancestor returned None here and spent the rest of the
    episode executing whatever the frozen model happened to say: measured at 12 decisions of the
    arm's one losing level and, on one lever configuration, 190 decisions of pure wandering. So drop
    the liveness test, which is the only guess in the stair rung, and walk back through the nearest
    door with any untried climb left. It can do no worse than the wandering it replaces, and it
    cannot fire while any proof-carrying step remains."""
    reopened = []
    for cell in filter(lambda c: pilot.chart.tag(pilot.epoch, c) == STAIR, order):
        if untried_climbs(pilot, cell):
            reopened.append(cell)
    if not reopened:
        return None
    if pilot.cell() in reopened:
        return (untried_climbs(pilot, pilot.cell())[0], 1,
                "nothing unseen is left on this floor and every door looked closed; trying this "
                "one anyway", "stair")
    return goal_step(pilot, parents, depth, order, reopened, "stair", prefer)


def best_stairs(pilot, order):
    """The charted stairs of this epoch whose floor change is the best on offer, nearest first.

    `order` is non-decreasing in distance, so the list keeps that property and its head is the
    nearest stair worth walking to."""
    ranked = []
    for cell in filter(lambda c: pilot.chart.tag(pilot.epoch, c) == STAIR, order):
        rank = pilot._climb_rank(cell)
        if rank is not None:
            ranked.append((rank, cell))
    if not ranked:
        return []
    keenest = min(rank for rank, _cell in ranked)
    return [cell for rank, cell in ranked if rank == keenest]


def untried_climbs(pilot, cell):
    """The floor changes never yet spent at this charted stair, in a stable order."""
    spent = pilot.chart.tried.get((pilot.epoch,) + tuple(cell), ())
    return [n for n in _act.CLIMB_ORDER if n not in spent]


def climb_worth(pilot, cell, name, standing):
    """How much a floor change at `cell` in direction `name` is worth, cheapest first, or None.

    None means it is not worth spending at all: it leads back to a floor this episode has already
    charted out, so there is nothing to go back for. Otherwise new ground outranks a charted floor,
    and a floor the harness has stood on outranks one it has no evidence for."""
    dest = pilot.chart.links.get((pilot.epoch,) + tuple(cell) + (name,))
    if dest is None:
        rank = 0
    elif pilot.chart.alive(dest):
        rank = GROUND_RANK
    else:
        return None
    if pilot.storey.get(pilot.epoch, 0) + storey_delta(name) not in standing:
        rank += UNSEEN_RANK
    return rank


def storey_delta(action_name) -> int:
    """Floors a climb moves, counted upward; 0 when the action is not a floor change."""
    return STOREY_STEP.get(action_name, 0)


def file_look(pilot, sight) -> None:
    """Write ONE look into the chart: the forward ray, then the whole flank ladder beside it.

    The ray's cells are proven PLAIN (it passed through them); the ladder's open cells are only
    proven PASSABLE, because a side exit paints nothing. Filing the ladder is this harness's
    whole axis — it is what settles a corridor's walls without standing in the corridor."""
    here = pilot.cell()
    pilot.chart.prove_plain(pilot.epoch, here)
    for i in range(1, sight.depth):
        pilot.chart.prove_plain(pilot.epoch, _act.step_from(here, pilot.heading, i))
    if sight.face != OPEN:
        pilot.chart.mark(pilot.epoch, _act.step_from(here, pilot.heading, sight.depth), sight.face)
    for k, pair in enumerate(sight.flanks[:max(0, pilot.LADDER + 1)]):
        base = _act.step_from(here, pilot.heading, k)
        for offset, kind in ((_act.LEFT, pair[0]), (_act.RIGHT, pair[1])):
            pilot.chart.mark(pilot.epoch, pilot.neighbour(offset, base), kind)
            pilot.filed += 1
    pilot.chart.digests[(pilot.epoch, pilot.x, pilot.y, pilot.heading)] = sight.digest


def settle_last(pilot, prev, changed, sight) -> None:
    """Apply the previous decision's measured outcome to the pose, the chart and the ledgers."""
    name, cells = prev["name"], prev["cells"]
    if name in _act.TURN_ACTIONS:
        quarters = max(1, int(prev.get("quarters") or 1))
        pilot.heading = _act.heading_after(pilot.heading, name, quarters)
        pilot.quarters += quarters
        pilot.pivots += 1
        pilot.since_move += 1
        return
    if name not in _act.MOVEMENT_ACTIONS:
        pilot.since_move += 1           # a climb moved the epoch at commit time; see _commit
        return
    bearing = pilot.heading if name == "move_forward" \
        else _act.side_of(pilot.heading, _act.BEHIND)
    if not changed:
        pilot.since_move += 1
        pilot.refused.setdefault(prev["digest"], set()).add(name)
        pilot.chart.mark(pilot.epoch, _act.step_from(pilot.cell(), bearing), WALL)
        if prev["expect"] > 0:
            pilot.chart.conflicts += 1  # the frame said this would move and it did not
        return
    pilot.moves += 1
    pilot.since_move = 0
    travelled = prev["expect"] if prev["expect"] > 0 else cells
    if prev["expect"] <= 0:
        pilot.chart.conflicts += 1      # it moved where the frame said it could not
    pilot.x, pilot.y = _act.step_from(pilot.cell(), bearing, travelled)


def twin_verdict(pilot, sight) -> None:
    """Decide for free whether the last climb actually happened.

    A floor change is not observable — a real one can leave the frame byte-identical — so it is
    committed optimistically and compared against the previous epoch's per-pose digests as they
    arrive: one mismatch proves a different floor, TWIN_MATCHES matches retire the new epoch."""
    twin = pilot.twin
    if not twin or pilot.epoch != twin["new"]:
        return
    twin["age"] += 1
    was = pilot.chart.digests.get((twin["old"], pilot.x, pilot.y, pilot.heading))
    if was is not None:
        if was != sight.digest:
            pilot.twin = None
            return
        twin["same"] += 1
        if twin["same"] >= pilot.TWIN_MATCHES:
            pilot.chart.retire(twin["new"])
            pilot.epoch = twin["old"]
            pilot.chart.tried.setdefault((twin["old"],) + twin["cell"], set()).add(twin["name"])
            pilot.chart.links.pop((twin["old"],) + twin["cell"] + (twin["name"],), None)
            pilot.climbs = max(0, pilot.climbs - 1)
            pilot.twin = None
            file_look(pilot, sight)
            return
    if twin["age"] > pilot.TWIN_WINDOW:
        pilot.twin = None                       # nothing comparable: assume it happened


def commit_climb(pilot, name) -> None:
    """Move into the epoch this climb leads to, creating it when it is new ground."""
    cell = pilot.cell()
    key = (pilot.epoch,) + cell
    pilot.chart.tried.setdefault(key, set()).add(name)
    pilot.climbs += 1
    dest = pilot.chart.links.get(key + (name,))
    fresh = dest is None
    if fresh:
        dest = max(pilot.chart.epochs) + 1
        pilot.chart.links[key + (name,)] = dest
        pilot.chart.links[(dest,) + cell + (_act.inverse_climb(name),)] = pilot.epoch
    previous = pilot.epoch
    pilot.chart.epochs.add(dest)
    if fresh:                                      # re-entering a known epoch must NOT renumber
        pilot.storey[dest] = pilot.storey.get(previous, 0) + storey_delta(name)
    pilot.epoch = dest
    pilot.chart.mark(dest, cell, STAIR)             # a link is a stair on both of its sides
    if fresh:
        pilot.twin = {"new": dest, "old": previous, "cell": cell, "name": name,
                      "same": 0, "age": 0}


def ray_charted(pilot, cell, bearing) -> bool:
    """Would a look down `bearing` from `cell` show the harness anything it does not already hold?

    False the moment the ray would cross ground whose identity is open — an UNCHARTED cell, or one
    known only from a flank band (passable, and so possibly the exit, since the exit paints no side
    face). True only when every cell the ray can reach is PROVEN PLAIN up to a charted blocker, in
    which case the look repeats a reading the chart already carries: the ray re-proves plain cells
    and the flank ladder re-reads the same lateral pair from the other end of the corridor."""
    probe = cell
    for _ in range(Sight.MAX_DEPTH + 1):
        probe = _act.step_from(probe, bearing)
        kind = pilot.chart.tag(pilot.epoch, probe)
        if kind is None:
            return False
        if kind != OPEN:
            return True
        if (pilot.epoch,) + probe not in pilot.chart.plain:
            return False
    return True


def blind_arc(pilot, name, quarters) -> bool:
    """Is every heading this rotation passes THROUGH already charted out?

    The harness is handed a frame only at the heading it STOPS on, so spending an arc in one
    decision forfeits exactly the looks from its interior headings. Checking them against the chart
    is what makes the compression free rather than a gamble: when any of them still has ground to
    show, the arc is cut short so the next frame arrives from there."""
    for heading in _act.pivot_path(pilot.heading, name, quarters):
        if not ray_charted(pilot, pilot.cell(), heading):
            return False
    return True


def pivot_step(pilot, bearing, note, mode):
    """(action, quarters, note, mode) — the rotation onto `bearing`, or None when it already faces.

    The arc is cut to the longest prefix whose interior headings are charted out, and to PIVOT_MAX.
    With PIVOT_MAX == 1 this is the ancestor's one-quarter-per-decision rung exactly."""
    if bearing is None:
        return None
    name, quarters = _act.pivot_to(pilot.heading, bearing)
    if name is None:
        return None
    arc = min(int(quarters), max(1, int(pilot.PIVOT_MAX)))
    while arc > 1 and not blind_arc(pilot, name, arc):
        arc -= 1
    return (name, arc, note, mode)


def look_bearing(pilot, pending):
    """Which unsettled bearing to look down next — ONE consistent rotation, not the nearest one.

    Every bearing in `pending` has to be looked down before this cell stops being a frontier, and
    the harness re-decides after each look, so the pick is really "which way do I rotate". Rotating
    one way sweeps all four headings in three quarters; taking the nearest pending bearing each
    time can double back and spend four. The direction that reaches the whole pending set in the
    shortest arc wins, and the bearing returned is its FIRST pending heading, so a look is never
    skipped over. `SWEEP = False` restores the ancestor's nearest-pending pick."""
    if not pending:
        return None
    if not pilot.SWEEP:
        return min(pending, key=lambda b: _act.turns_to(pilot.heading, b))
    right = _act.sweep_arc(pilot.heading, "turn_right", pending)
    left = _act.sweep_arc(pilot.heading, "turn_left", pending)
    name = "turn_right" if right <= left else "turn_left"
    for quarter in range(1, _act.QUARTERS):
        heading = _act.heading_after(pilot.heading, name, quarter)
        if heading in pending:
            return heading
    return pending[0]


def choose_target(pilot, parents, depth, targets, prefer: str):
    """Which of several equally near targets to head for — THE MODEL'S FREE VOTE.

    Every candidate here sits at the same BFS distance, so choosing between them cannot cost a
    decision, and the model's own proposal decides: if its action is the first step toward one of
    them, that one wins. The vote is taken strictly INSIDE the turn-minimal subset, never across
    it — a vote that could add a turn is not free, and a degenerate proposer would otherwise
    steer the route with its own repetition and spin in place.

    REFUTED this iteration, measured, do not rebuild: ranking this tie by distance from the stair
    the sweep will depart through (so the last frontier retired is the one nearest the door) is
    INERT — ties at equal cell distance are too rare to carry it — and planning the whole route in
    decisions instead of cells COSTS 0.16-0.51 decisions an episode on the six-tape panel. A 1-wide
    maze has one path per pair of cells, so there is no turn-for-cell trade left to find."""
    best = min(depth[c] for c in targets)
    tied = list(filter(lambda c: depth[c] == best, targets))
    if len(tied) == 1:
        return tied[0]

    def opening(cell):
        path = pilot.chart.route(parents, cell)
        return _act.bearing_to(path[0], path[1]) if len(path) > 1 else pilot.heading

    def turns(cell):
        return _act.turns_to(pilot.heading, opening(cell))

    fewest = min(map(turns, tied))
    tied = list(filter(lambda c: turns(c) == fewest, tied))
    if prefer and len(tied) > 1:
        for cell in tied:
            bearing = opening(cell)
            move = _act.turn_toward(pilot.heading, bearing) or (
                "move_forward" if bearing == pilot.heading else "")
            if move == prefer:
                return cell
    return tied[0]


def clamp_translation(pilot, name, cells):
    """Shorten an OBEYED translation to the part the frame can account for, or None to leave it.

    This is what keeps the reckoned pose exact rather than nearly exact. `sight.reach` is the run the
    frame proves walkable and it is never longer than the engine's own (0 over-reaches in 5624
    poses), so executing at most `reach` cells means the travel is known before the move and settled
    exactly after it. It can only ever shorten a stride that would carry the body PAST the stair or
    the exit terminating the ray — which is the one place a longer stride is unwanted anyway, since
    a climb only works from on top of the tile. Backward is clamped to a single cell because nothing
    at all is drawn behind the body: one step back either happened or did not, and that the frame can
    still tell, while a three-cell reversal cannot be attributed."""
    if name == "move_forward" and pilot.sight.ok:
        allowed = pilot.sight.stride(cells)
        if 0 < allowed < cells:
            return ("move_forward", allowed,
                    "the screen accounts for %d cell(s) of that stride and stops at the %s ahead"
                    % (allowed, pilot.sight.face))
    if name == "move_backward" and cells > 1:
        return ("move_backward", 1,
                "nothing is drawn behind you, so only a single step back can be checked")
    return None


def inert_reason(pilot, name: str) -> str:
    """Why the model's proposal provably cannot change anything, or ""."""
    sight = pilot.sight
    if name == "move_forward" and sight.ok and sight.blocked():
        return "a wall stands in the very next cell, so `move_forward` cannot move at all"
    if name == "move_backward" and not pilot.lost:
        if pilot.chart.tag(pilot.epoch, pilot.neighbour(_act.BEHIND)) == WALL:
            return "the cell behind you is a wall, so `move_backward` cannot move at all"
    if name in _act.CLIMB_ACTIONS and not pilot.lost and not pilot.on_stair():
        return "there is no stair charted under your feet, so `%s` does nothing" % name
    if name in _act.MOVEMENT_ACTIONS and name in pilot.refused.get(sight.digest, ()):
        return "`%s` already left this exact screen unchanged" % name
    return ""


def escape_step(pilot):
    """The blind fallback: probe ahead if the frame has not refused it, else ROTATE.

    Rotation, never `move_backward`: stepping back does not change the bearing, so the body
    lands in the previous cell still facing the pocket and walks straight back into it — a
    two-cycle, measured. The flank ladder picks WHICH WAY to rotate, onto a side the frame shows
    open, which walks the body out of a sealed pocket in three decisions instead of five."""
    refused = pilot.refused.get(pilot.sight.digest, ())
    if "move_forward" not in refused and not pilot.sight.blocked():
        return ("move_forward", 1, "probing the way ahead", "escape")
    if pilot.sight.flanks:
        left, right = pilot.sight.flanks[0]
        if left != WALL and right == WALL:
            return ("turn_left", 1, "turning toward the open side beside you", "escape")
        if right != WALL and left == WALL:
            return ("turn_right", 1, "turning toward the open side beside you", "escape")
    return ("turn_right", 1, "rotating onto a bearing this screen has not refused", "escape")


# ── the lines of the per-turn block ─────────────────────────────────────────────────────────
def foot_line(pilot) -> str:
    if pilot.lost:
        return "unknown (the position estimate was dropped)."
    if pilot.on_stair():
        return "a STAIR tile — a floor change works from here."
    return "ordinary floor, not a stair."


def last_line(pilot) -> str:
    prev = pilot.last
    if not prev or not prev.get("name"):
        return "none yet — this is your first decision."
    if prev.get("displaced"):
        head = "you proposed `%s`; `%s` ran instead" % (prev["displaced"], prev["name"])
    else:
        head = "`%s` ran exactly as you asked" % prev["name"]
    if prev["name"] in _act.TURN_ACTIONS and prev.get("quarters", 1) > 1:
        head += " %d times in that one decision, so you are now facing %d degrees around" % (
            prev["quarters"], 90 * prev["quarters"])
    if prev["name"] in _act.MOVEMENT_ACTIONS:
        head += (" and you moved" if pilot.since_move == 0 else
                 " and the screen did not change at all, so something solid blocked it")
    return head + "."


def chart_line(pilot) -> str:
    mine = [(x, y) for (e, x, y) in pilot.chart.cells if e == pilot.epoch]
    settled = sum(1 for c in mine if pilot.chart.settled(pilot.epoch, c))
    frontier = sum(1 for c in mine if pilot.chart.frontier(pilot.epoch, c))
    stairs = sum(1 for c in mine if pilot.chart.tag(pilot.epoch, c) == STAIR)
    return ("%d cells identified, %d of them settled; %d still have something unseen beside them; "
            "%d stair(s) found." % (len(mine), settled, frontier, stairs))


def _clean_template(text: str) -> str:
    """The renderable part of memory.md: everything after its last HTML comment, since the file's
    header documents the template for the next proposer and must never reach the agent. Returns ""
    when there is no body or its placeholders do not match `Pilot.fields()`."""
    if not text:
        return ""
    body = text.split("-->")[-1].strip() if "-->" in text else text.strip()
    return body if "{ahead_line}" in body and "{flank_line}" in body and "{step}" in body else ""
