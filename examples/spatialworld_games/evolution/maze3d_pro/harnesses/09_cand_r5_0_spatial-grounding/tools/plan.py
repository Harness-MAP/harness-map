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
            for b in range(4):
                nxt = _act.step_from(cur, b)
                if nxt in parents or self.cells.get((epoch,) + nxt) not in PASSABLE:
                    continue
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
        self.lost = False              # reckoning contradicted: fall back to the blind ledger
        self.refused = {}              # frame digest -> translations that provably did nothing
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
            self._settle(self.last, changed, sight)
        self.sight = sight
        self.prev_digest = sight.digest
        self.step += 1
        if sight.ok and not self.lost:
            self._file(sight)
            self._resolve_twin(sight)
            key = (self.epoch, self.x, self.y)
            self.chart.visits[key] = self.chart.visits.get(key, 0) + 1
        if len(self.chart.cells) > self.charted_mark:
            self.charted_mark, self.since_new = len(self.chart.cells), 0
        else:
            self.since_new += 1
        if self.chart.conflicts >= 2:
            self.lost = True

    def _file(self, sight) -> None:
        """Write ONE look into the chart: the forward ray, then the whole flank ladder beside it.

        The ray's cells are proven PLAIN (it passed through them); the ladder's open cells are only
        proven PASSABLE, because a side exit paints nothing. Filing the ladder is this harness's
        whole axis — it is what settles a corridor's walls without standing in the corridor."""
        here = self.cell()
        self.chart.prove_plain(self.epoch, here)
        for i in range(1, sight.depth):
            self.chart.prove_plain(self.epoch, _act.step_from(here, self.heading, i))
        if sight.face != OPEN:
            self.chart.mark(self.epoch, _act.step_from(here, self.heading, sight.depth), sight.face)
        for k, pair in enumerate(sight.flanks[:max(0, self.LADDER + 1)]):
            base = _act.step_from(here, self.heading, k)
            for offset, kind in ((_act.LEFT, pair[0]), (_act.RIGHT, pair[1])):
                self.chart.mark(self.epoch, self.neighbour(offset, base), kind)
                self.filed += 1
        self.chart.digests[(self.epoch, self.x, self.y, self.heading)] = sight.digest

    def _settle(self, prev, changed: bool, sight) -> None:
        """Apply the previous decision's measured outcome to the pose, the chart and the ledgers."""
        name, cells = prev["name"], prev["cells"]
        if name in _act.TURN_ACTIONS:
            self.heading = _act.side_of(self.heading, _act.RIGHT if name == "turn_right"
                                       else _act.LEFT)
            self.since_move += 1
            return
        if name not in _act.MOVEMENT_ACTIONS:
            self.since_move += 1           # a climb moved the epoch at commit time; see _commit
            return
        bearing = self.heading if name == "move_forward" else _act.side_of(self.heading,
                                                                           _act.BEHIND)
        if not changed:
            self.since_move += 1
            self.refused.setdefault(prev["digest"], set()).add(name)
            self.chart.mark(self.epoch, _act.step_from(self.cell(), bearing), WALL)
            if prev["expect"] > 0:
                self.chart.conflicts += 1  # the frame said this would move and it did not
            return
        self.moves += 1
        self.since_move = 0
        travelled = prev["expect"] if prev["expect"] > 0 else cells
        if prev["expect"] <= 0:
            self.chart.conflicts += 1      # it moved where the frame said it could not
        self.x, self.y = _act.step_from(self.cell(), bearing, travelled)

    def _resolve_twin(self, sight) -> None:
        """Decide for free whether the last climb actually happened.

        A floor change is not observable — a real one can leave the frame byte-identical — so it is
        committed optimistically and compared against the previous epoch's per-pose digests as they
        arrive: one mismatch proves a different floor, TWIN_MATCHES matches retire the new epoch."""
        twin = self.twin
        if not twin or self.epoch != twin["new"]:
            return
        twin["age"] += 1
        was = self.chart.digests.get((twin["old"], self.x, self.y, self.heading))
        if was is not None:
            if was != sight.digest:
                self.twin = None
                return
            twin["same"] += 1
            if twin["same"] >= self.TWIN_MATCHES:
                self.chart.retire(twin["new"])
                self.epoch = twin["old"]
                self.chart.tried.setdefault((twin["old"],) + twin["cell"], set()).add(twin["name"])
                self.chart.links.pop((twin["old"],) + twin["cell"] + (twin["name"],), None)
                self.climbs = max(0, self.climbs - 1)
                self.twin = None
                self._file(sight)
                return
        if twin["age"] > self.TWIN_WINDOW:
            self.twin = None                       # nothing comparable: assume it happened

    # ── the route ───────────────────────────────────────────────────────────────────────────
    def _climb_choice(self, cell):
        """The floor transition worth spending at a charted stair cell, or None.

        Preference: a transition never tried from here, into an epoch that does not exist yet or
        still holds a frontier. `climb_up` is tried first so the first transition of an episode goes
        away from the starting floor rather than into the ground."""
        spent = self.chart.tried.get((self.epoch,) + tuple(cell), ())
        best = None
        for name in ("climb_up", "climb_down"):
            if name in spent:
                continue
            dest = self.chart.links.get((self.epoch,) + tuple(cell) + (name,))
            if dest is None:
                rank = 0
            elif self.chart.alive(dest):
                rank = 1
            else:
                continue                           # a finished floor: nothing to go back for
            if best is None or rank < best[0]:
                best = (rank, name)
        return best[1] if best else None

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
            return route_step(self, parents, exits[0], "exit", True)
        frontier = [c for c in order if self.chart.frontier(self.epoch, c)]
        stairs = [c for c in order if self.chart.tag(self.epoch, c) == STAIR
                  and self._climb_choice(c) is not None]
        if stairs and self.climbs < self.MAX_CLIMBS and (not frontier or self._impatient()):
            if stairs[0] == self.cell():
                name = self._climb_choice(stairs[0])
                if name:
                    return (name, 1, "this floor is charted out and holds no exit; changing floor "
                            "here", "stair")
            else:
                return route_step(self, parents, stairs[0], "stair", True)
        if frontier:
            return route_step(self, parents,
                              choose_target(self, parents, depth, frontier, prefer),
                              "explore", False)
        return None

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
        if pick[0] == name and (pick[0] not in _act.MOVEMENT_ACTIONS or pick[1] == cells):
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
        self.last = {"name": name, "cells": cells, "expect": expect, "displaced": displaced,
                     "digest": self.sight.digest, "step": self.step}
        if name in _act.CLIMB_ACTIONS and not self.lost:
            self._commit_climb(name)

    def _commit_climb(self, name) -> None:
        """Move into the epoch this climb leads to, creating it when it is new ground."""
        cell = self.cell()
        key = (self.epoch,) + cell
        self.chart.tried.setdefault(key, set()).add(name)
        self.climbs += 1
        dest = self.chart.links.get(key + (name,))
        fresh = dest is None
        if fresh:
            dest = max(self.chart.epochs) + 1
            self.chart.links[key + (name,)] = dest
            self.chart.links[(dest,) + cell + (_act.inverse_climb(name),)] = self.epoch
        previous = self.epoch
        self.chart.epochs.add(dest)
        self.epoch = dest
        self.chart.mark(dest, cell, STAIR)             # a link is a stair on both of its sides
        if fresh:
            self.twin = {"new": dest, "old": previous, "cell": cell, "name": name,
                         "same": 0, "age": 0}

    # ── the per-turn block ──────────────────────────────────────────────────────────────────
    def fields(self) -> dict:
        return {"step": self.step, "max_steps": self.max_steps,
                "ahead_line": self.sight.ahead_line(), "flank_line": self.sight.flank_line(),
                "foot_line": foot_line(self), "last_line": last_line(self),
                "chart_line": chart_line(self), "plan_line": self.plan_note,
                "moves": self.moves, "overrides": self.overrides, "filed": self.filed}

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
    """One step of the route to `target`: a turn onto its bearing, or a stride along it.

    The stride is the measured clear run clipped to the straight part of the route, and it never
    strides PAST a cell the route wants to stop on. Crossing a cell costs nothing now that its
    neighbours are charted from a distance — which is exactly what the flank ladder bought."""
    if target == pilot.cell():
        pending = pilot.chart.unsettled_bearings(pilot.epoch, pilot.cell())
        if pending:
            nearest = min(pending, key=lambda b: _act.turns_to(pilot.heading, b))
            turn = _act.turn_toward(pilot.heading, nearest)
            if turn:
                return (turn, 1, "looking down the one bearing here still unsettled", note)
        return None
    path = pilot.chart.route(parents, target)
    if len(path) < 2:
        return None
    bearing = _act.bearing_to(path[0], path[1])
    if bearing is None:
        return None
    turn = _act.turn_toward(pilot.heading, bearing)
    if turn:
        return (turn, 1, "turning onto the route (%s)" % note, note)
    run = 1
    while run + 1 < len(path) and _act.bearing_to(path[run], path[run + 1]) == bearing:
        run += 1
    cells = min(run, pilot.STRIDE_MAX, max(1, pilot.sight.reach))
    if stop_on:
        cells = min(cells, len(path) - 1)
    return ("move_forward", max(1, cells), "stepping along the route (%s)" % note, note)


def choose_target(pilot, parents, depth, targets, prefer: str):
    """Which of several equally near targets to head for — THE MODEL'S FREE VOTE.

    Every candidate here sits at the same BFS distance, so choosing between them cannot cost a
    decision, and the model's own proposal decides: if its action is the first step toward one of
    them, that one wins. The vote is taken strictly INSIDE the turn-minimal subset, never across
    it — a vote that could add a turn is not free, and a degenerate proposer would otherwise
    steer the route with its own repetition and spin in place."""
    best = min(depth[c] for c in targets)
    tied = [c for c in targets if depth[c] == best]
    if len(tied) == 1:
        return tied[0]

    def opening(cell):
        path = pilot.chart.route(parents, cell)
        return _act.bearing_to(path[0], path[1]) if len(path) > 1 else pilot.heading

    fewest = min(_act.turns_to(pilot.heading, opening(c)) for c in tied)
    tied = [c for c in tied if _act.turns_to(pilot.heading, opening(c)) == fewest]
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
