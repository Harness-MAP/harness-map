"""tools/action.py — THE DECISION ALGEBRA: the engine's key table, the dead-reckoned chart, and a
route planner whose currency is DECISIONS rather than cells.

WHAT THIS MODULE IS FOR. An episode of maze3d_pro is budgeted in exactly one resource: 100 calls to
the frozen model. Everything else is free. A translation costs ONE decision whether it travels one
cell or three and it stops early at a wall, so distance is nearly free; a 90-degree turn costs ONE
decision and travels nothing, so TURNING is the expensive thing. This lineage has planned its routes
by counting CELLS and then rendered each route step with an if-ladder that emits a turn onto the
bearing and then a forward stride along it. That renderer can only ever spend turn + turn + forward
— three decisions — to step one cell BACKWARD, and `move_backward` is a legal action it never once
emits: zero of the anchor's 2340 decisions, and zero in every descendant's route.

So this module plans in the currency the episode is actually billed in. `search` runs a
breadth-first search over POSE STATES `(x, y, heading)` whose edges are the legal decisions —
turn_left, turn_right, and forward/backward strides of 1, 2 or 3 map-proven cells — each costing
exactly one. Every edge costs the same, so plain BFS is already optimal, and what it returns is the
minimum NUMBER OF DECISIONS to stand somewhere facing something. Four consequences fall out of that
and none of them is special-cased:

  * a reversal costs 1 decision instead of 3, so `move_backward` becomes a first-class route move —
    and it is sound exactly where it is useful, because a retreat runs back over cells the agent has
    already stood on and the chart has already proven passable;
  * a stride packs itself to the longest proven run, and clips itself to land exactly ON a target,
    because the search only ever stops on states it enumerated;
  * a frontier's target state is `(cell, bearing)` for each bearing out of it still unsettled, so
    the price of a look includes the turn that performs it — there is no separate "which way do I
    face when I arrive" rule, it is the same search;
  * and the target that wins is the one reachable in the fewest DECISIONS, not the one nearest in
    cells — a frontier straight ahead beats a closer one that needs two turns to look at.

EVERY LEVER BELOW WAS PRICED, NOT ASSUMED. A stand-in engine (the documented move/turn/climb
semantics plus the painter formulas above) was driven by six degenerate stand-in proposers over 60
generated two-floor levels at two sizes — 720 PAIRED episodes per arm, identical levels and
identical proposals, one class attribute changed. The shipped configuration wins 385/720:

  | arm                                   | wins    | paired vs shipped |
  |---------------------------------------|---------|-------------------|
  | shipped                               | 385/720 | --                |
  | OBEY = "off" (model ungoverned)       | 138/720 | +9  / -256        |
  | PROBE_FIRST = False                   | 341/720 | +15 / -59         |
  | TRANSIT_ONLY = False                  | 351/720 | +13 / -47         |
  | REVERSE = False, STRIDE_MAX = 1       | 361/720 | +19 / -43         |
  | REVERSE = False (the lineage's routes)| 366/720 | +20 / -39         |
  | RANK = "value"                        | 381/720 | +9  / -13         |
  | COMMIT = False                        | 384/720 | +1  / -2          |
  | STRIDE_MAX = 1                        | 386/720 | +9  / -8          |

Read honestly, that is three findings, one neutral-and-kept, and one retraction:

  * ADMITTING `move_backward` IS THE RESULT: +19 of 720 against the lineage's turn-turn-forward
    renderer, and +25 with strides held to one cell, so the gain is not an artefact of stride length.
  * A LONG STRIDE IS ONLY SAFE OVER GROUND ALREADY WALKED (TRANSIT_ONLY, +34). The flank readout
    covers the cell the agent is IN, so striding past an unvisited cell buys position and leaves that
    cell's two side neighbours unread — a debt the exhaustion proof pays back later at a worse rate.
    Note STRIDE_MAX = 1 is NEUTRAL (+1): the value is not in the stride length at all, only in
    refusing to spend one where it costs information, which is exactly the ground a retreat covers.
  * REVERSALS MUST BE RATIONED TO VERIFIABLE GROUND (PROBE_FIRST, +44, the largest single lever).
    Lifting the gate nearly doubles the reversals (8049 against 4355) and loses 44 episodes, so the
    premise test is also the throttle that keeps retreats on ground the chart has stood on.
  * HOLDING THE CHOSEN LOOK (COMMIT) measured NEUTRAL at -1 and is kept anyway, because it costs
    nothing and removes a known hazard: pricing decisions instead of cells compresses the cost range,
    so candidate looks tie far more often than they did for the ancestors, and re-voting on every tie
    is the unbounded model vote this chain already measured as fatal. Stated as neutral, not as a win.
  * RETRACTED: ranking looks by `worth` (decisions per unseen cell) is INDISTINGUISHABLE from raw
    minimum decisions (381 against 385, paired 9 against 13). A 150-episode sample had said `worth`
    was what made reversals pay; the 720-episode sample refutes it. The default is the simpler rule
    and `worth` stays a documented lever rather than a claim.

WHAT IS REPRODUCED, NOT RE-DERIVED. The chart underneath the planner is this chain's measured result
and is kept deliberately: an exact frame-identity outcome bit (byte-identical frame <=> the last
translation was refused; 83/83 refusals identical and 0/1077 real moves identical, so equality is
exact and a tolerance is refuted), a per-floor occupancy map keyed on a dead-reckoned CELL and never
on the view (the frame digest aliases badly: 257 distinct digests over 5624 distinct poses),
provenance kept between "open because a ray passed through it" and "open because a flank said so"
(the painter draws the exit only as a front face, so an exit cell beside the agent reads exactly like
floor), a floor change gated on the FLOOR BEING EXHAUSTED rather than on the first stair (climbing
early costs the levels whose exit is on the start floor), and an optimistic climb into a fresh map
epoch settled afterwards by comparing per-pose digests (19.2% of this benchmark's floor changes leave
the frame byte-identical, so frame identity cannot decide a climb).

THE ONE UNTESTED PREMISE, PRICED. `move_backward` has never executed in this lineage, so its
semantics — translate opposite the facing, 1..3 cells, stopping early at a wall, heading unchanged —
are taken from the action contract in rules.md and confirmed by no measurement. `Chart.verify`
therefore prices the premise on its first use: a retreat lands on a pose the chart already holds a
frame digest for, and if the new frame's digest differs from the one recorded there the premise is
WRONG, reversals switch off for the rest of the episode and the contradiction is counted. Digests
alias, so agreement is weak evidence and disagreement is a proof; the guard is built around the
direction that proves something.
"""
from __future__ import annotations

from collections import deque

from . import frame as _f

# ── the engine contract ──────────────────────────────────────────────────────────────────────
# action_name -> engine key, exactly the adapter's execute_mapped_action dispatch table.
KEY_MAPPING = {
    "move_forward": "w",
    "move_backward": "s",
    "turn_left": "a",
    "turn_right": "d",
    "climb_up": "q",
    "climb_down": "e",
}
NAME_OF_KEY = {key: name for name, key in KEY_MAPPING.items()}
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}
STEPS_GRANULARITY = {steps: name for name, steps in GRANULARITY_STEPS.items()}
MOVEMENT_ACTIONS = frozenset(("move_forward", "move_backward"))
TURN_ACTIONS = frozenset(("turn_left", "turn_right"))
CLIMB_ACTIONS = frozenset(("climb_up", "climb_down"))
# The input source reads granularity from metadata (falling back to json_data) and `repeat` as the
# cell count, so all three must be dropped before a turn is rebuilt: a stale `repeat` inherited from
# a movement proposal would fire that many 90-degree turns instead of one.
STRIPPED_KEYS = ("granularity", "repeat", "json_data")

# bearings 0,1,2,3 = N,E,S,W, which makes turn_right exactly +1 and turn_left exactly -1 (mod 4).
STEP = ((0, -1), (1, 0), (0, 1), (-1, 0))
TURN_DELTA = {"turn_left": -1, "turn_right": 1}
CLIMB_INVERSE = {"climb_up": "climb_down", "climb_down": "climb_up"}
PASSABLE = frozenset((_f.OPEN, _f.STAIR, _f.EXIT))
# a later look may UPGRADE a cell's identity but never downgrade it to plain floor: a flank reports
# an exit cell as merely open, so "open" has to be allowed to become "exit".
TAG_RANK = {_f.OPEN: 0, _f.WALL: 1, _f.STAIR: 2, _f.EXIT: 3}


def key_for(action_name: str):
    """The engine key a legal action name executes, or None when the name is not legal."""
    return KEY_MAPPING.get(action_name)


def name_for(key):
    """The legal action name an engine key will actually execute, or None.

    Read the KEY, never `metadata["function_called"]`: the evaluator also writes non-action labels
    there (its parse fallback stamps `default_move_forward`), while the key is what runs."""
    if not isinstance(key, str):
        return None
    return NAME_OF_KEY.get(key.strip().lower())


def granularity_steps(name) -> int:
    """Cells a granularity label asks for, mirroring the adapter's clamp into 1..3."""
    return GRANULARITY_STEPS.get(str(name or "small").strip().lower(), 1)


def steps_granularity(cells: int) -> str:
    """The granularity label that asks for `cells` cells, clamped into the engine's 1..3 range."""
    return STEPS_GRANULARITY[max(1, min(3, int(cells)))]


def shift(cell, bearing: int, run: int = 1):
    """The cell `run` steps from `cell` along `bearing`, using the engine's own vectors."""
    dx, dy = STEP[bearing % 4]
    return (cell[0] + dx * run, cell[1] + dy * run)


def turn_toward(heading: int, bearing: int):
    """The single turn that rotates `heading` toward `bearing`, or None when already facing it.

    Offsets are mod 4, so +1 is one turn_right and +3 is one turn_left; the 180-degree case needs
    two turns and is reported as turn_right, which keeps the handedness consistent across a reversal
    instead of alternating and re-treading the same two bearings."""
    delta = (int(bearing) - int(heading)) % 4
    if delta == 0:
        return None
    return "turn_left" if delta == 3 else "turn_right"


def turns_to(heading: int, bearing: int) -> int:
    """How many 90-degree turns separate a heading from a bearing (0, 1 or 2)."""
    delta = (int(bearing) - int(heading)) % 4
    return 2 if delta == 2 else (0 if delta == 0 else 1)


class Chart:
    """The dead-reckoned pose and the per-floor occupancy map, both derived from nothing but
    `frame_data.image` and the agent's own committed decisions.

    COORDINATES ARE LOCAL AND THAT IS ENOUGH. The origin is wherever the episode started and the
    bearing labels are arbitrary; every observation is folded in through the same transform, so the
    chart is self-consistent and a route computed on it is correct even though it carries no idea
    where it is in the level. Nothing privileged is read, and nothing could be: the only input is the
    rendered frame.
    """

    DESYNC_LIMIT = 2       # contradictions between prediction and outcome before the pose is dropped
    TWIN_MATCHES = 2       # per-pose digest matches that retire an epoch as a refused climb
    TWIN_WINDOW = 6        # decisions the retirement test stays open for comparisons

    def __init__(self):
        self.epoch, self.x, self.y, self.heading = 0, 0, 0, 0
        self.cells = {}              # (epoch, x, y) -> tag
        self.clear = set()           # (epoch, x, y) proven PLAIN floor: a ray passed through it
        self.visits = {}             # (epoch, x, y) -> times stood on
        self.digests = {}            # (epoch, x, y, heading) -> frame digest
        self.tried = {}              # (epoch, x, y) -> set of climbs already spent there
        self.links = {}              # (epoch, x, y, climb) -> the epoch it leads to
        self.epochs = {0}
        self.climbs = 0
        self.twin = None
        self.scene = _f.Scene()
        self.step = 0
        self.last = None
        self.prev_digest = None
        self.known, self.since_new, self.since_move = 0, 0, 0
        self.moves, self.overrides, self.desync = 0, 0, 0
        self.lost = False
        self.reverse_ok = True       # the move_backward premise, until a probe refutes it
        self.reverse_used, self.reverse_agreed = 0, 0
        self.probe = None
        self.refused = {}            # digest -> translations that provably did nothing there

    def here(self):
        return (self.x, self.y)

    def tag(self, cell, epoch=None):
        return self.cells.get(((self.epoch if epoch is None else epoch),) + tuple(cell))

    def passable(self, cell, epoch=None) -> bool:
        return self.tag(cell, epoch) in PASSABLE

    def visited(self, cell, epoch=None) -> bool:
        return bool(self.visits.get(((self.epoch if epoch is None else epoch),) + tuple(cell)))

    def mark(self, cell, tag) -> None:
        """Record a cell's identity, upgrading only, and count a contradiction when two different
        SOLID readings land on one cell — the signature of a lost pose, not of a repaint."""
        key = (self.epoch,) + tuple(cell)
        old = self.cells.get(key)
        if old is None or old == tag:
            self.cells[key] = tag
            return
        if TAG_RANK[tag] > TAG_RANK[old]:
            if TAG_RANK[old] >= TAG_RANK[_f.WALL]:
                self.desync += 1
            self.cells[key] = tag
        elif TAG_RANK[tag] >= TAG_RANK[_f.WALL]:
            self.desync += 1

    def settled(self, cell, epoch=None) -> bool:
        """Is this cell's identity decided — a wall, a stair, the exit, or PROVEN plain floor?

        A cell known only from a flank is not settled, because that reading cannot tell the exit from
        empty floor. Every unsettled cell is cheap to settle: one ray through it terminates on it if
        it is anything other than floor."""
        ep = self.epoch if epoch is None else epoch
        key = (ep,) + tuple(cell)
        tag = self.cells.get(key)
        if tag is None:
            return False
        return tag != _f.OPEN or key in self.clear

    def unsettled_bearings(self, cell, epoch=None):
        """The bearings out of `cell` whose neighbour still has to be looked at."""
        ep = self.epoch if epoch is None else epoch
        return [b for b in range(4) if not self.settled(shift(cell, b), ep)]

    def frontier(self, cell, epoch=None) -> bool:
        """A reachable cell from which something unseen can still be looked at.

        A floor is finished exactly when no such cell is reachable, and at that moment every
        reachable cell is a wall, a stair, the exit or proven plain floor — so "no exit found and no
        frontier left" PROVES the exit is not on this floor. That proof is what makes a floor change
        free; taking a stair before it costs the levels whose exit is on the start floor."""
        ep = self.epoch if epoch is None else epoch
        if self.cells.get((ep,) + tuple(cell)) not in PASSABLE:
            return False
        return bool(self.unsettled_bearings(cell, ep))

    def epoch_cells(self, epoch=None):
        ep = self.epoch if epoch is None else epoch
        return [(x, y) for (e, x, y) in self.cells if e == ep]

    def epoch_alive(self, epoch) -> bool:
        return any(self.frontier(cell, epoch) for cell in self.epoch_cells(epoch))

    def fold(self, scene) -> None:
        """Write one frame into the chart: the whole forward ray, plus both lateral neighbours.

        PROVENANCE IS THE POINT. The ray terminates on the first cell that is not plain floor — a
        wall, a stair and the exit all stop it — so every cell it passes THROUGH is proven ordinary
        empty floor, which is to say proven not to be the goal. A flank proves no such thing, so the
        two kinds of "open" live in different containers and only the ray can retire a cell."""
        start = self.here()
        if self.tag(start) is None:
            self.mark(start, _f.OPEN)                 # standing on it proves it enterable
        self.clear.add((self.epoch,) + start)         # and proves it is not the exit: still playing
        for i in range(1, scene.depth):
            cell = shift(start, self.heading, i)
            self.mark(cell, _f.OPEN)
            self.clear.add((self.epoch,) + cell)
        if scene.kind != _f.OPEN:
            self.mark(shift(start, self.heading, scene.depth), scene.kind)
        for offset, reading in ((3, scene.left), (1, scene.right)):
            self.mark(shift(start, self.heading + offset), reading)
        self.digests[(self.epoch, self.x, self.y, self.heading)] = scene.digest

    def observe(self, image) -> None:
        """Score the previous decision against the frame it produced, advance the pose, fold the new
        observation in. Called once per step, before the frozen model is asked."""
        scene = _f.Scene(image)
        changed = self.prev_digest is None or scene.digest != self.prev_digest
        if self.last is not None:
            self.settle(self.last, changed)
        self.scene = scene
        self.prev_digest = scene.digest
        self.step += 1
        if scene.ok and not self.lost:
            self.verify(scene)
            self.retire(scene)
            self.fold(scene)
            key = (self.epoch,) + self.here()
            self.visits[key] = self.visits.get(key, 0) + 1
        if len(self.cells) > self.known:
            self.known, self.since_new = len(self.cells), 0
        else:
            self.since_new += 1
        if self.desync >= self.DESYNC_LIMIT:
            self.lost = True

    def settle(self, prev, changed: bool) -> None:
        """Apply the measured outcome of the previous decision to the pose, the map and the ledgers.

        A turn is never refused by the engine, so the heading is exact. A translation is scored by
        the one exact bit the frame carries: byte-identical means the engine refused it, which both
        proves a wall in the cell it aimed at and costs nothing to learn. A climb is not observable
        at all and was already committed at decision time."""
        name = prev["name"]
        if name in TURN_ACTIONS:
            self.heading = (self.heading + TURN_DELTA[name]) % 4
            self.since_move += 1
            return
        if name not in MOVEMENT_ACTIONS:
            self.since_move += 1
            return
        bearing = self.heading if name == "move_forward" else (self.heading + 2) % 4
        if not changed:
            self.since_move += 1
            self.refused.setdefault(prev["digest"], set()).add(name)
            self.mark(shift(self.here(), bearing), _f.WALL)
            if prev["expect"] > 0:
                self.desync += 1              # the measurement said it would move and it did not
            self.probe = None
            return
        self.moves += 1
        self.since_move = 0
        travelled = prev["expect"] if prev["expect"] > 0 else prev["cells"]
        if prev["expect"] <= 0:
            self.desync += 1                  # it moved where the measurement said it could not
        self.x, self.y = shift(self.here(), bearing, travelled)

    def verify(self, scene) -> None:
        """Price the `move_backward` premise against a pose whose frame this chart already holds.

        A retreat lands on a pose occupied earlier with the same heading, so a recorded digest is
        usually waiting there. Digests alias across poses, so a match is weak evidence and is only
        counted; a MISMATCH at a recorded pose proves the retreat did not land where the contract
        says it would, and it switches reversals off rather than letting a wrong pose poison the map.
        """
        pending, self.probe = self.probe, None
        if not pending:
            return
        was = self.digests.get(pending)
        if was is None:
            return
        if was == scene.digest:
            self.reverse_agreed += 1
            return
        self.reverse_ok = False
        self.desync += 1

    def climb_choice(self, cell):
        """The floor transition worth spending at a mapped stair, or None.

        Untried transitions into ground that does not exist yet come first, then ones into a floor
        that still has a frontier; a floor already finished is nothing to go back for. `climb_up` is
        offered before `climb_down` so the first transition leaves the start floor rather than
        burrowing under it."""
        spent = self.tried.get((self.epoch,) + tuple(cell), ())
        ranked = []
        for name in ("climb_up", "climb_down"):
            if name in spent:
                continue
            dest = self.links.get((self.epoch,) + tuple(cell) + (name,))
            if dest is None:
                ranked.append((0, name))
            elif self.epoch_alive(dest):
                ranked.append((1, name))
        return min(ranked)[1] if ranked else None

    def commit_climb(self, name) -> None:
        """Enter the epoch this climb leads to, creating it when it is new ground.

        The transition is committed OPTIMISTICALLY because it cannot be observed: 19.2% of this
        benchmark's floor changes render byte-identically. A fresh epoch is a clean namespace either
        way and `retire` settles the ambiguity afterwards for free."""
        cell = self.here()
        key = (self.epoch,) + cell
        self.tried.setdefault(key, set()).add(name)
        self.climbs += 1
        dest = self.links.get(key + (name,))
        fresh = dest is None
        if fresh:
            dest = max(self.epochs) + 1
            self.links[key + (name,)] = dest
            self.links[(dest,) + cell + (CLIMB_INVERSE[name],)] = self.epoch
        previous = self.epoch
        self.epochs.add(dest)
        self.epoch = dest
        self.cells.setdefault((dest,) + cell, _f.STAIR)   # a link is a stair on both sides
        if fresh:
            self.twin = {"new": dest, "old": previous, "cell": cell, "name": name,
                         "same": 0, "age": 0}

    def retire(self, scene) -> None:
        """Decide for free whether the last climb actually happened, by comparing the new epoch's
        frames against the old epoch's at the same pose. One mismatch confirms the floor changed;
        TWIN_MATCHES matches retire the new epoch and restore the mapped floor underneath it."""
        twin = self.twin
        if not twin or self.epoch != twin["new"]:
            return
        twin["age"] += 1
        was = self.digests.get((twin["old"], self.x, self.y, self.heading))
        if was is not None:
            if was != scene.digest:
                self.twin = None
                return
            twin["same"] += 1
            if twin["same"] >= self.TWIN_MATCHES:
                for store in (self.cells, self.visits):
                    for key in [k for k in store if k[0] == twin["new"]]:
                        del store[key]
                self.clear -= {k for k in self.clear if k[0] == twin["new"]}
                self.epochs.discard(twin["new"])
                self.epoch = twin["old"]
                self.tried.setdefault((twin["old"],) + twin["cell"], set()).add(twin["name"])
                self.links.pop((twin["old"],) + twin["cell"] + (twin["name"],), None)
                self.climbs = max(0, self.climbs - 1)
                self.twin = None
                return
        if twin["age"] > self.TWIN_WINDOW:
            self.twin = None                    # nothing comparable arrived: assume it happened

    def foot_line(self) -> str:
        if self.lost:
            return "unknown — the position estimate was dropped, so trust the screen alone."
        if self.tag(self.here()) == _f.STAIR:
            return "a STAIR tile, so a floor change from here works."
        return "ordinary floor, not a stair."

    def last_line(self) -> str:
        prev = self.last
        if not prev or not prev.get("name"):
            return "none yet — this is your first decision."
        if prev.get("displaced"):
            head = "you asked for `%s`; `%s` was executed instead" % (prev["displaced"], prev["name"])
        else:
            head = "`%s` was executed exactly as you asked" % prev["name"]
        if prev["name"] in MOVEMENT_ACTIONS:
            head += (" and you moved" if self.since_move == 0 else
                     " and the screen did not change at all, so something solid blocked it")
        return head + "."

    def map_line(self) -> str:
        mine = self.epoch_cells()
        return ("%d cells identified, %d of them settled, %d still have an unseen neighbour, "
                "%d stair(s) found." %
                (len(mine), sum(1 for c in mine if self.settled(c)),
                 sum(1 for c in mine if self.frontier(c)),
                 sum(1 for c in mine if self.tag(c) == _f.STAIR)))


# ── the search, and the economics on top of it ────────────────────────────────────────────────
def search(chart, stride: int, reverse: bool, transit_only: bool, probe_first: bool = True):
    """BFS over pose states `(x, y, heading)`; returns (cost, first) keyed by state.

    The edges ARE the legal decisions and every one costs exactly one, which is what makes an
    unweighted BFS already optimal in the episode's own currency. A stride edge of `k` cells needs
    all `k` of them map-proven passable, because the engine stops early at a wall and a stride that
    stopped short would break the dead reckoning; routing only through proven cells is therefore
    both the safe rule and the exact one.

    `transit_only` additionally forbids a stride from CROSSING a cell never stood on, which is the
    information price of a long stride. It is applied while exploring, where coverage is the point,
    and lifted when the destination is already proven — a mapped exit, or a stair off a finished
    floor — where there is nothing left to learn and only arriving cheaply matters.
    """
    start = (chart.x, chart.y, chart.heading)
    cost, first = {start: 0}, {start: None}
    queue = deque([start])
    # UNTIL THE PREMISE IS PROVEN, ONLY PLAN A VERIFIABLE REVERSAL. `move_backward` has never
    # executed in this lineage, so the first one planned must land on a pose whose frame the chart
    # already holds — which is exactly what makes `Chart.verify` able to refute it. Once one retreat
    # has agreed with its recorded frame, the restriction lifts.
    unproven = reverse and probe_first and not chart.reverse_agreed
    while queue:
        state = queue.popleft()
        x, y, heading = state
        edges = [(name, 1, (x, y, (heading + delta) % 4)) for name, delta in TURN_DELTA.items()]
        for name, sign in (("move_forward", 1), ("move_backward", -1)):
            if sign < 0 and not reverse:
                continue
            bearing = heading if sign > 0 else (heading + 2) % 4
            for run in range(1, stride + 1):
                cell = shift((x, y), bearing, run)
                if not chart.passable(cell):
                    break
                if sign < 0 and unproven and \
                        (chart.epoch, cell[0], cell[1], heading) not in chart.digests:
                    break                       # not a retreat this chart could check afterwards
                edges.append((name, run, (cell[0], cell[1], heading)))
                if transit_only and not chart.visited(cell):
                    break                       # crossing it would leave its flanks unread
        for name, run, nxt in edges:
            if nxt in cost:
                continue
            cost[nxt] = cost[state] + 1
            first[nxt] = first[state] or (name, run)
            queue.append(nxt)
    return cost, first


def look_states(chart, cells):
    """`(cell, bearing)` for every bearing out of a frontier cell that is still unsettled.

    Making the BEARING part of the target is what prices the look itself: arriving already facing the
    unseen neighbour costs nothing extra, while arriving facing away costs the turn that fixes it,
    and the search weighs both against every other candidate."""
    return [(cell[0], cell[1], bearing)
            for cell in cells for bearing in chart.unsettled_bearings(cell)]


def reveal(chart, cell, bearing: int) -> int:
    """How many unsettled cells one look from `cell` along `bearing` can decide.

    The ray is counted optimistically through cells the chart has not seen — an unknown cell may be
    floor and the ray may run on past it — and stops at the first cell already known to be solid or
    to be a stair or the exit, because that is where the painter stops it. Arriving at a cell never
    stood on also reads its two flanks for free, so those count too. This is the BENEFIT half of a
    look's price; `worth` is the ratio."""
    gain = 0
    for i in range(1, _f.DEPTH_MAX + 1):
        ahead = shift(cell, bearing, i)
        if not chart.settled(ahead):
            gain += 1
        if chart.tag(ahead) in (_f.WALL, _f.STAIR, _f.EXIT):
            break
    if not chart.visited(cell):
        gain += sum(1 for side in (bearing + 1, bearing + 3)
                    if not chart.settled(shift(cell, side)))
    return max(1, gain)


def worth(chart, state, cost) -> float:
    """Decisions per unseen cell for one candidate look — lower is better. `Tape.RANK = "value"`.

    The argument for it: minimum decisions alone ranks a one-decision peek at a nearby straggler
    above a two-decision walk to the mouth of an unexplored corridor, and the exit is only ever
    found by a ray. The measurement does not support the argument — 381 of 720 against 385 for raw
    cost, paired 9 against 13, which is a coin flip — so this is offered as a lever and not as a
    finding. It is worth re-pricing against the real engine, where corridors are longer than the
    stand-in's and the ray is worth more.
    """
    return (cost[state] + 1) / float(reveal(chart, (state[0], state[1]), state[2]))


def aim(cost, first, states, prefer: str, rank=None):
    """(first_action, cells, target_state) for the best of `states` — THE MODEL'S FREE VOTE.

    `rank` is the ordering and defaults to the plain decision cost; the explore branch passes
    `worth`. Among states that TIE on it, choosing between them cannot by definition cost anything,
    so the frozen model's own proposal decides: if its action is the first move toward one of them,
    that one wins. The vote is bounded to the best-ranked set and never across it, because an
    unbounded vote lets a degenerate proposer swing the target every step and spin in place."""
    reachable = [s for s in states if s in cost]
    if not reachable:
        return None
    rank = rank or (lambda state: cost[state])
    best = min(rank(s) for s in reachable)
    tied = [s for s in reachable if rank(s) == best]
    pick = min(tied)
    if prefer and len(tied) > 1:
        for state in sorted(tied):
            head = first.get(state)
            if head and head[0] == prefer:
                pick = state
                break
    head = first.get(pick)
    return (head[0], head[1], pick) if head else None


def inert_reason(chart, name: str) -> str:
    """Why the proposal provably cannot change anything, or "". Every clause is a measurement."""
    scene = chart.scene
    if name == "move_forward" and scene.ok and scene.walkable == 0:
        return "a wall fills the very next cell, so `move_forward` cannot move at all"
    if name == "move_backward" and not chart.lost:
        if chart.tag(shift(chart.here(), chart.heading + 2)) == _f.WALL:
            return "the cell behind you is a wall, so `move_backward` cannot move at all"
    if name in CLIMB_ACTIONS and not chart.lost:
        if chart.tag(chart.here()) != _f.STAIR:
            return "there is no stair under your feet, so `%s` does nothing" % name
    if name in MOVEMENT_ACTIONS and name in chart.refused.get(scene.digest, ()):
        return "`%s` already left this exact screen unchanged once" % name
    return ""


def gains(chart, name: str, cells: int) -> bool:
    """Does the proposal reach a cell not yet stood on, or look at one not yet settled?"""
    if chart.lost or not chart.scene.ok:
        return True
    if name in MOVEMENT_ACTIONS:
        bearing = chart.heading if name == "move_forward" else (chart.heading + 2) % 4
        reach = max(1, min(cells, max(1, chart.scene.walkable)))
        return any(not chart.visited(shift(chart.here(), bearing, i)) for i in range(1, reach + 1))
    if name in TURN_ACTIONS:
        return not chart.settled(shift(chart.here(), chart.heading + TURN_DELTA[name]))
    return True


def escape_step(chart):
    """The blind fallback for a dropped pose: probe ahead, else rotate consistently."""
    scene = chart.scene
    if "move_forward" not in chart.refused.get(scene.digest, ()) and scene.walkable != 0:
        return ("move_forward", 1, "probing the way ahead", "escape")
    return ("turn_right", 1, "rotating to a bearing this screen has not refused", "escape")


def ration(chart, name: str, cells: int) -> int:
    """The cells a translation the harness did NOT plan may ask for.

    A forward stride is self-limiting: the engine stops it at the wall the ray already measured, so
    the travel is known either way. A backward stride is not measurable at all — nothing renders
    behind the agent — so it is rationed to the run the chart has already proven, which keeps the
    dead reckoning exact instead of guessing at a displacement."""
    if name == "move_forward":
        return max(1, min(cells, max(1, chart.scene.walkable)) if chart.scene.ok else cells)
    if name != "move_backward" or chart.lost:
        return max(1, cells)
    proven = 0
    while proven < cells and chart.passable(shift(chart.here(), chart.heading + 2, proven + 1)):
        proven += 1
    return max(1, proven)


def rebuild(proposal, name: str, cells: int, reason: str, proposed: str):
    """A NEW engine Action for `name`, rebuilt from `proposal`'s metadata envelope.

    `type(proposal)` is reused so this module never imports the engine. The three distance keys are
    dropped first and re-added only for the two translations, where `repeat` carries the
    granularity's own cell count exactly as the evaluator writes it."""
    key = KEY_MAPPING.get(name)
    if key is None:
        return proposal
    meta = {k: v for k, v in (getattr(proposal, "metadata", None) or {}).items()
            if k not in STRIPPED_KEYS}
    meta["harness_override"] = {"proposed": proposed, "executed": name, "reason": reason}
    meta["function_called"] = name
    if name in MOVEMENT_ACTIONS:
        label = steps_granularity(cells)
        meta["granularity"] = label
        meta["repeat"] = GRANULARITY_STEPS[label]
    try:
        return type(proposal)(type="key_press", key=key, metadata=meta)
    except Exception:
        return proposal                 # an unexpected Action shape must never cost the step


class Tape:
    """The decision-cost route planner and the governor that spends the model's proposals against it.

    `plan` answers one question — what is the cheapest-per-unseen-cell NEXT DECISION — and `apply`
    turns the model's proposal plus that answer into the engine Action the evaluator runs. The model
    is asked on every single step and obeyed wherever the measurement cannot prove it wrong, so
    agreeing with the harness never costs it the wheel.
    """

    # ── swept levers. Class attributes, because build_agent purges sys.modules["tools*"] and would
    #    throw away any instance override a rig set before the call. ───────────────────────────
    STRIDE_MAX = 3             # the engine's `large`; a stride costs one decision at any length
    REVERSE = True             # admit move_backward as a route move (False => the lineage's routes)
    PROBE_FIRST = True         # the first reversal planned must land on a pose the chart already
                               # holds a frame for, so `Chart.verify` can refute the premise before
                               # anything relies on it. It costs reach early in an episode, which is
                               # why it is a lever and not a belief.
    TRANSIT_ONLY = True        # a multi-cell stride may only CROSS cells already stood on
    RANK = "cost"              # "cost": rank looks on raw minimum decisions | "value": rank them on
                               # decisions per unseen cell (`worth`). Measured indistinguishable over
                               # 720 paired episodes (385 against 381), so the simpler rule is the
                               # default and `worth` is the documented alternative.
    COMMIT = True              # hold the chosen look until it is settled or goes out of reach
    OBEY = "route"             # "route": disagreement with any plan is displaced | "proven": only a
                               # mapped exit or a stair off an exhausted floor displaces | "off":
                               # only measured-inert proposals are displaced
    FLOOR_GATE = "exhaust"     # "exhaust": spend a stair only once the floor is finished | "stale":
                               # also after STALE_LIMIT decisions with nothing new mapped
    STALE_LIMIT = 8
    MAX_CLIMBS = 8
    RESWEEP = True             # with no exit, no frontier and no worthwhile stair, re-look from the
                               # thinnest-walked ground instead of handing a frozen model the wheel
    STALL_LIMIT = 4            # decisions with no motion before the blind escape engages

    TEMPLATE = """## MEASURED STATE (read off the screen in front of you — not a guess)
- decision {step} of {max_steps}.
- straight ahead: {ahead_line}
- immediately to your left: {left_line}   |   immediately to your right: {right_line}
- under your feet: {foot_line}
- your last decision: {last_line}
- this floor so far: {map_line}
- cheapest plan from here: {plan_line}

There is no HUD, no text and no coordinates anywhere on screen. Distances above are counted in grid
cells and are exact. A translation into a wall changes nothing whatsoever, so never spend a decision
on one. A move costs one decision whether it covers one cell or three.

Reply with ONLY the JSON object in the format specified above."""

    def __init__(self, chart, max_steps: int = 100, template: str = ""):
        self.chart = chart
        self.max_steps = int(max_steps or 100)
        self.template = template or self.TEMPLATE
        self.note = "nothing measured yet"
        self.base_prompt = None
        self.goal = None             # the committed look: (epoch, x, y, bearing)
        self.switches = 0

    def hold(self, cost, first, states, prefer: str, rank=None):
        """`aim`, but holding the committed look until it is settled or drops out of reach.

        The tie-break vote is therefore spent only when a NEW look is chosen, never every step: a
        decision-priced cost ties often, and re-voting on every tie is the unbounded vote this chain
        already measured as fatal. Commitment costs nothing — the held look is still the one the
        search ranked best when it was picked — and it is abandoned the moment the cell stops being
        a frontier, which is exactly when it has been looked at."""
        if not self.COMMIT:
            return aim(cost, first, states, prefer, rank)
        goal = self.goal
        if goal is not None and goal[0] == self.chart.epoch:
            state = goal[1:]
            if state in cost and state in set(states):
                head = first.get(state)
                if head:
                    return (head[0], head[1], state)
            self.goal = None
        shot = aim(cost, first, states, prefer, rank)
        if shot:
            if self.goal is not None:
                self.switches += 1
            self.goal = (self.chart.epoch,) + shot[2]
        return shot

    def plan(self, prefer: str = ""):
        """(action_name, cells, note, mode) — the next decision the economics pick, or None.

        `mode` says what the decision is FOR, and only the two modes resting on a proof are insisted
        upon: "exit" (a mapped exit cell, reached through cells already shown to be floor) and
        "stair" (this floor is exhausted, so the exit provably is not on it). While exploring, the
        harness has no proof its guess beats the model's, so it keeps the wheel on agreement and on
        coverage instead."""
        chart = self.chart
        if chart.lost or not chart.scene.ok:
            return None
        reverse = self.REVERSE and chart.reverse_ok
        cost, first = search(chart, self.STRIDE_MAX, reverse, False, self.PROBE_FIRST)
        mine = chart.epoch_cells()
        exits = [c for c in mine if chart.tag(c) == _f.EXIT]
        if exits:
            shot = aim(cost, first, [(c[0], c[1], h) for c in exits for h in range(4)], prefer)
            if shot:
                return (shot[0], shot[1], "stepping onto the mapped EXIT", "exit")
        frontier = [c for c in mine if chart.frontier(c)]
        stairs = [c for c in mine if chart.tag(c) == _f.STAIR and chart.climb_choice(c) is not None]
        impatient = self.FLOOR_GATE == "stale" and chart.since_new >= self.STALE_LIMIT
        if stairs and chart.climbs < self.MAX_CLIMBS and (not frontier or impatient):
            if chart.here() in stairs:
                name = chart.climb_choice(chart.here())
                if name:
                    return (name, 1, "this floor is mapped out, so changing floor here", "stair")
            shot = aim(cost, first, [(c[0], c[1], h) for c in stairs for h in range(4)], prefer)
            if shot:
                return (shot[0], shot[1], "walking to the stair that leaves this floor", "stair")
        if frontier:
            # the exploring search is the restricted one; with the transit rule off it IS the search
            # already run above, so it is reused rather than recomputed.
            walk, lead = (search(chart, self.STRIDE_MAX, reverse, True, self.PROBE_FIRST)
                          if self.TRANSIT_ONLY else (cost, first))
            rank = None if self.RANK == "cost" else (lambda state: worth(chart, state, walk))
            shot = self.hold(walk, lead, look_states(chart, frontier), prefer, rank)
            if shot:
                return (shot[0], shot[1], "best look at unseen ground (%d decisions away, %d cell(s)"
                        " it can settle)" % (walk[shot[2]], reveal(chart, shot[2][:2], shot[2][2])),
                        "explore")
        if self.RESWEEP:
            spare = [c for c in mine if chart.passable(c) and c != chart.here()]
            if spare:
                thin = min(chart.visits.get((chart.epoch,) + c, 0) for c in spare)
                thinnest = [c for c in spare if chart.visits.get((chart.epoch,) + c, 0) == thin]
                shot = aim(cost, first, [(c[0], c[1], h) for c in thinnest for h in range(4)], prefer)
                if shot:
                    return (shot[0], shot[1],
                            "nothing unseen left: re-walking the thinnest ground", "sweep")
        return None

    def govern(self, name: str, cells: int):
        """(action, cells, reason). An empty reason means the model's own proposal is executed."""
        chart = self.chart
        route = self.plan(prefer=name)
        self.note = route[2] if route else "no plan of its own: following you"
        why = inert_reason(chart, name)
        if not why and route and self.OBEY != "off":
            if self.OBEY == "route" or route[3] in ("exit", "stair"):
                why = "the cheapest mapped plan is %s" % route[2]
            elif not gains(chart, name, cells):
                why = "`%s` only re-treads ground already covered" % name
        if not why:
            return (name, ration(chart, name, cells), "")
        stalled = chart.since_move >= self.STALL_LIMIT or chart.lost
        pick = route or (escape_step(chart) if stalled else None)
        if pick is None:
            return (name, ration(chart, name, cells), "")
        if pick[0] == name and (pick[0] not in MOVEMENT_ACTIONS or pick[1] == cells):
            return (name, cells, "")                     # the model already wants the planned step
        return (pick[0], pick[1], why)

    def apply(self, proposal):
        """Govern one proposal and return the engine Action the evaluator will execute."""
        name = name_for(getattr(proposal, "key", None))
        cells = granularity_steps((getattr(proposal, "metadata", None) or {}).get("granularity"))
        if name is None:
            self.commit(None, 0, None)
            return proposal
        chosen, chosen_cells, why = self.govern(name, cells)
        if not why and chosen_cells == cells:
            self.commit(name, cells, None)
            return proposal
        self.chart.overrides += 1
        out = rebuild(proposal, chosen, chosen_cells, why or "rationed to the proven run", name)
        self.commit(chosen, chosen_cells, name if why else None)
        return out

    def commit(self, name, cells, displaced) -> None:
        """Record what is about to run, with the displacement the chart will score it against."""
        chart = self.chart
        expect = 0
        if name == "move_forward":
            expect = min(max(1, cells), chart.scene.walkable) if chart.scene.ok else cells
        elif name == "move_backward":
            expect = max(1, cells)
            if not chart.lost:
                landing = shift(chart.here(), chart.heading + 2, expect)
                chart.probe = (chart.epoch, landing[0], landing[1], chart.heading)
                chart.reverse_used += 1
        chart.last = {"name": name, "cells": cells, "expect": expect, "displaced": displaced,
                      "digest": chart.scene.digest, "step": chart.step}
        if name in CLIMB_ACTIONS and not chart.lost:
            chart.commit_climb(name)

    def fields(self) -> dict:
        chart = self.chart
        return {"step": chart.step, "max_steps": self.max_steps,
                "ahead_line": chart.scene.ahead_line(),
                "left_line": chart.scene.flank_line("left"),
                "right_line": chart.scene.flank_line("right"),
                "foot_line": chart.foot_line(), "last_line": chart.last_line(),
                "map_line": chart.map_line(), "plan_line": self.note,
                "moves": chart.moves, "overrides": chart.overrides}

    def render(self) -> str:
        """The block tools/vlm.py appends to the agent's system turn. Never raises."""
        flds = self.fields()
        for text in (self.template, self.TEMPLATE):
            try:
                return text.format(**flds)
            except Exception:
                continue
        return ""
