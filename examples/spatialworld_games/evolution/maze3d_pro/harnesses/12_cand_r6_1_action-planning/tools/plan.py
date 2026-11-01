"""THE ROUTE, PRICED IN DECISIONS — every goal elected on price, and a governor that honours the
frozen model wherever its answer is free.

An episode is budgeted in exactly one resource: 100 calls to the frozen model. A translation costs
one decision whether it travels one cell or three and it stops itself at a wall; a 90-degree turn
costs one decision and travels nothing. Distance is therefore nearly free and TURNING is the
expensive thing, so every edge of the search below costs ONE DECISION and the route it returns is
the one a 100-decision budget can actually afford.

WHAT IS NEW HERE, against the pass-through this iteration started from and against every ancestor of
it: the harness aims at a GOAL that is chosen by PRICE, not at a BEARING chosen by the current view.
Those ancestors re-derived a bearing every single step from whatever the frame happened to show,
which is exactly how a two-cycle is born — the view at cell A recommends east, the view at cell B
recommends west, and the body shuttles between them until the clock runs out (one ancestor spent 74
moves on a period-16 shuttle without a single refused press). Price does not do that: moving toward
the cheapest goal makes that goal cheaper, so re-electing every step is self-stabilising. A version
that HELD the elected goal across decisions was built and measured first, and it came out one level
worse on every one of eight proposer arms, because it could not abandon a staircase that a look from
halfway there had already made pointless. The price is the cure; a commitment on top of it is a cost.

THE MODEL IS STILL ASKED EVERY STEP, AND IT IS OBEYED WHENEVER ITS ANSWER IS FREE. The governor
prices the proposal the way it prices its own route: re-plan from the pose the proposal would leave
the body in, and honour it when the remaining route is no longer than the one the harness's own
first decision would have left (`decisions_after <= decisions_now - 1`). That bound is the whole
discipline. An unbounded vote is refuted — a proposer that always answers `turn_right` swings the
target every step and spins in place — and a harness that never defers is a hand-coded solver with
a model-shaped ornament. Free-or-cheaper is the rule that keeps both failure modes out: on a long
corridor or at a fork with two equally short ways round, the model's answer is free and it rules;
on a wall it has already been refused by, it is not free and it is displaced.

THE FLOOR CHANGE IS COMMITTED, NOT GATED. A departure heuristic over "how much of this floor have I
explored" is refuted at both ends of its range: a tight cap wins two levels and loses twenty-six,
a loose one is inert, and no interior value exists, because one level must leave after five cells
stood and another must not leave after twenty-three. So there is no gate. A stair carries a TOLL in
decisions (`STAIR_TOLL`) and competes with the frontier on price: while cheap questions remain on
this floor they win, and when they are used up the stair wins at any distance. Each directed link
(this cell, upward) and (this cell, downward) may be elected AT MOST ONCE, which bounds the whole
mechanism to two traversals per stair cell and makes the way back a plannable goal rather than an
oscillation.
"""
from __future__ import annotations

import re
from collections import deque

from tools import action as A
from tools import chart as C
from tools.sight import EXIT, STAIR, look

#: decisions a FRESH floor change is priced at, on top of the route to it — and the measured answer
#: is ZERO. A staircase is not a destination that has to earn a premium: it IS a question, the
#: question "what is on the other floor", and it is priced exactly like every other question on the
#: board. Swept on the offline rig at 0 / 1 / 2 / 3 / 8 / 20 against five degenerate proposer arms:
#: 125 / 125 / 125 / 125 / 120 / 120 of 125. Every premium is a cost and none of them buys anything,
#: which is the departure-gate refutation one step further than it was taken upstream: not only is a
#: "have I explored enough of this floor" GATE wrong, so is a price.
STAIR_TOLL = 0
#: decisions a RETURN down the staircase the body just came up is priced at. THIS premium is the
#: load-bearing one, and it is the asymmetry that makes the whole tier work: at 0 the agent rides the
#: same staircase up and down forever (45 of 125), at 8, 50 and 100 it scores 125 of 125. Half the
#: budget keeps the way back permanently ON the board as a route — never behind a gate — while making
#: it the last thing elected, which is exactly what "commit on the way out, keep the way back" means
#: once it is priced instead of latched.
RETURN_TOLL = 50
#: the search's own ceiling. Four bearings over a floor of a few hundred cells settles far inside
#: this; it exists so a corrupt chart cannot turn one decision into an unbounded walk.
SEARCH_CAP = 4000
#: decisions the body spends standing ON a staircase before the climb is pressed — and the number is
#: not a tuning, it is THE REST OF THE COMPASS. A body that has just walked onto a stair has faced
#: exactly one of four bearings from that cell; three more decisions of rotation face the other
#: three, and every one of them files a corridor the chart did not have. Only then is a floor change
#: priced against a frontier that is actually complete. Swept on the offline rig at 1 / 2 / 3 / 4:
#: 21 / 23 / 24 / 20 levels of 25. It peaks at the compass and falls off the moment a fourth
#: decision buys a bearing already seen.
LOOK_AROUND = 3

DEFAULT_TEMPLATE = """## working memory (self-derived only)
- ahead: {ahead}
- beside: {beside}
- underfoot: {underfoot}
- mapped: {mapped}
- plan: {plan}
"""


def clean_template(text):
    """The working-memory template with its authoring comments removed.

    `memory.md` documents its own contract in an HTML comment at the top; that paragraph is for
    whoever edits the file next, not for the agent, so it is stripped before the block is rendered.
    A file that has lost its placeholders falls back to the built-in template rather than injecting
    a block the agent cannot use."""
    body = re.sub(r"<!--.*?-->", "", text or "", flags=re.S).strip()
    return body + "\n" if "{ahead}" in body else DEFAULT_TEMPLATE


class Commitment:
    """One elected goal: what the harness is aiming at on THIS decision.

    Re-elected every step from the current pose (see `Pilot.aim`), so the name records what the
    mechanism is for — a single target the whole step agrees on, including the governor — rather
    than a claim that it survives the step.

    `poses` is the set of `(cell, bearing)` the goal is satisfied at, with a bearing of None
    meaning any. `kind` is what raised it, and for a stair it carries the directed link that
    election spent."""

    __slots__ = ("kind", "poses", "cell", "link")

    def __init__(self, kind, poses, cell, link=None):
        self.kind = kind
        self.poses = set(poses)
        self.cell = cell
        self.link = link

    def satisfied(self, cell, heading):
        """Is the body standing where this goal wanted it?"""
        return (cell, None) in self.poses or (cell, heading) in self.poses

    def line(self):
        """The goal, in the plain words the working-memory block states back to the agent."""
        words = {EXIT: "walk onto the GOLD EXIT",
                 STAIR: "walk onto the CYAN STAIR and change floor",
                 "frontier": "reach unexplored ground"}
        return words.get(self.kind, self.kind)


def edges(chart, cell, bearing):
    """Every single-decision move out of one pose, as `((action, granularity), pose)`.

    FOUR FAMILIES, EACH EXACTLY ONE PRESS, and the order they are yielded in is the tie-break the
    search inherits: translations before turns and the LONGEST stride first, so a route that can
    travel does. A stride costs one decision whether it carries one cell or three, and spending it
    on three is free.

      * `move_forward` of 3, 2 or 1 cells, out of a pose no translation has already been refused in.
        A stride may LAND on ground a flank band merely glimpsed, but it may not continue THROUGH
        it: striding over cells nothing has looked along is the most expensive habit this search can
        have, worth more than the reversal it rations.
      * `move_backward`, same lengths, over RAY-PROVEN ground only. Admitting the reversal makes the
        cell behind cost one decision where turning around and walking costs three; restricting it
        to proven ground is what stops the planner backing blind into a cell it has only glimpsed.
      * the two turns, always available, one quarter-turn each.
    """
    for back in (False, True):
        if chart.refuted(cell, (bearing + 2) % 4 if back else bearing):
            continue
        name = "move_backward" if back else "move_forward"
        run = []
        for cells in (1, 2, 3):
            step = A.advance(cell, bearing, cells, back)
            if not (chart.verified(step) if back else chart.enterable(step)):
                break
            run.append((cells, step))
            if not chart.verified(step):
                break
        for cells, step in reversed(run):
            yield (name, A.granularity_for(cells)), (step, bearing)
    for turn in ("turn_right", "turn_left"):
        yield (turn, "small"), (cell, A.heading_after(bearing, turn))


def search(chart, cell, heading):
    """`{pose: (decisions, opening)}` for every pose reachable on this floor from one pose.

    Breadth-first with ONE DECISION per edge, which is what makes the result a BUDGET and not a
    length. `opening` is the single action that starts the cheapest route to that pose — carried
    down from the first edge rather than recovered by a second pass, so one search answers both
    "how much does this goal cost" and "what do I press now".

    The cap exists so a corrupt chart can never turn one decision into an unbounded walk; four
    bearings over a floor of a few hundred cells settle far inside it."""
    start = (cell, heading)
    table = {start: (0, None)}
    queue = deque([start])
    while queue and len(table) < SEARCH_CAP:
        here = queue.popleft()
        cost, opening = table[here]
        for move, nxt in edges(chart, here[0], here[1]):
            if nxt not in table:
                table[nxt] = (cost + 1, opening or move)
                queue.append(nxt)
    return table


def cheapest(poses, table):
    """`(decisions, opening)` for the cheapest of a goal's poses, or None if none is reachable.

    A pose with a bearing of None is satisfied facing any way, so all four bearings count; a pose
    that names a bearing counts only at that bearing, because a question is answered by LOOKING at
    it and the look only happens down the bearing that faces it."""
    found = [table[(c, hh)] for c, h in poses for hh in range(4)
             if h in (None, hh) and (c, hh) in table]
    return min(found) if found else None


def frontier_poses(chart):
    """The poses that would ANSWER an open question — the goals the frontier raises.

    A cell a flank band showed empty is answered by standing on it; a cell already walked through
    that still has an unknown neighbour is answered by FACING that neighbour, because the forward
    ray then measures it without spending a step on it. Seeding the route from the questions rather
    than from a compass bearing is what makes a look worth a move."""
    out = set()
    for q in chart.questions():
        if chart.kind(q) == C.PASSABLE:
            out.add((q, None))
            continue
        for h in range(4):
            if chart.kind(A.advance(q, h, 1)) == C.UNKNOWN:
                out.add((q, h))
    return out


def candidates(chart, used):
    """`[(kind, poses, cell, link, toll), ...]` — everything worth routing to, with its price.

    Three tiers and two tolls. The exit is free and outranks everything, because an exit anywhere on
    this floor ends the episode and no floor change can improve on that. The frontier is free. A
    stair is priced — `STAIR_TOLL` for a fresh link, `RETURN_TOLL` for the undo of the press that
    brought the body to this floor — and only in a direction this episode has not already spent.

    A LINK IS KEYED ON THE CELL AND THE DIRECTION, never on a floor number. There is no floor
    readout to key on, so a believed floor index can be wrong; a stair cell cannot. The key bounds
    the whole mechanism to at most one climb up and one climb down per staircase per episode, which
    is what makes a refused press cost one decision instead of an unbounded loop of them."""
    out = []
    for cell in chart.tiles(EXIT):
        out.append((EXIT, {(cell, None)}, cell, None, 0))
    for cell, h in frontier_poses(chart):
        out.append(("frontier", {(cell, h)}, cell, None, 0))
    for cell in chart.tiles(STAIR):
        for name in ("climb_up", "climb_down"):
            if (cell, name) in used:
                continue
            toll = RETURN_TOLL if chart.returning(cell, name) else STAIR_TOLL
            out.append((STAIR, {(cell, None)}, cell, (cell, name), toll))
    return out


def elect(chart, used):
    """The cheapest goal on offer, as a `Commitment`, or None when nothing is reachable.

    Price is `decisions to get there + the tier's toll`, and ties break on the goal's own cell so
    the choice is path-independent: two runs that reach the same pose with the same chart elect the
    same goal, which is what makes a win reproducible rather than lucky."""
    table = search(chart, chart.cell, chart.heading)
    best = None
    for kind, poses, cell, link, toll in candidates(chart, used):
        found = cheapest(poses, table)
        if found is None:
            continue
        if best is None or (found[0] + toll, cell) < best[0]:
            best = ((found[0] + toll, cell), Commitment(kind, poses, cell, link))
    return best[1] if best else None


def sweep(chart, sight, last=""):
    """The forced escape, for the step where no goal is reachable at all.

    Ranked NOVELTY FIRST — a bearing whose next cell the chart has never settled — then openness,
    then a constant handedness, and never the UNDO of the last turn — a left straight after a right
    spends two decisions to arrive back at the pose that was already refused, and it is how most
    lost episodes in this lineage end. Ranking openness first is refuted: it rotates the body through
    bearings this very frame has already refuted and manufactures a fresh two-cycle. The escape is
    always a TURN; a reversal as a dead-end fallback is refuted outright, because the pair
    `move_backward` then `move_forward` is itself a two-cycle and it was worth a quarter of the
    episodes it ran in."""
    ranked = []
    for turn in ("turn_right", "turn_left"):
        if A.is_undo(last, turn):
            continue
        bearing = A.heading_after(chart.heading, turn)
        ahead = A.advance(chart.cell, bearing, 1)
        fresh = 0 if chart.kind(ahead) == C.UNKNOWN else 1
        shut = 0 if chart.enterable(ahead) else 1
        ranked.append(((fresh, shut, 0 if turn == "turn_right" else 1), turn))
    if sight is not None and sight.ok and not sight.blocked():
        if chart.kind(A.advance(chart.cell, chart.heading, 1)) == C.UNKNOWN:
            return "move_forward", A.granularity_for(sight.stride(3))
    return (min(ranked)[1] if ranked else "turn_right"), "small"


class Pilot:
    """The episode: one chart, one commitment, one governor. Built once per level."""

    __slots__ = ("chart", "template", "budget", "base_prompt", "goal", "used", "sight",
                 "pending", "last_name", "stair_age", "decisions", "overrides", "climbs")

    def __init__(self, budget=100, template=""):
        self.chart = C.Chart()
        self.template = clean_template(template)
        self.budget = int(budget)
        self.base_prompt = None
        self.goal = None
        self.used = set()
        self.sight = None
        self.pending = None           # (action_name, cells it would travel if it was not refused)
        self.last_name = ""
        self.stair_age = 0
        self.decisions = 0
        self.overrides = 0
        self.climbs = 0

    # ── perceive ────────────────────────────────────────────────────────────────────────────
    def observe(self, image):
        """Settle the previous decision against this frame, then file what this frame proves."""
        seen = look(image)
        if self.pending is not None:
            name, cells = self.pending
            readable = seen.ok and self.sight is not None and self.sight.ok
            moved = readable and self.sight.digest != seen.digest
            if readable or name not in A.MOVEMENT_ACTIONS:
                self.chart.fold_outcome(name, cells, moved or name not in A.MOVEMENT_ACTIONS)
            self.pending = None
        self.chart.reconcile(seen)
        self.sight = seen
        self.stair_age = self.stair_age + 1 if self.chart.on_stair() else 0

    # ── plan ────────────────────────────────────────────────────────────────────────────────
    def aim(self):
        """Elect the cheapest goal FROM THIS POSE, every step. This is the axis, in three lines.

        There is deliberately no commitment held across decisions. A held goal was built and measured
        first, on the theory that re-deriving a target every step is what breeds a two-cycle — and it
        is, for a harness that re-derives the target from a BEARING. It is not, for one that
        re-derives it from a PRICE: moving toward the cheapest goal makes that goal cheaper, so plain
        re-election is self-stabilising, and the held version measured **one level worse on every one
        of eight proposer arms** (it could not abandon a staircase that a look from halfway there had
        just made pointless). The price is the cure; the commitment on top of it is a cost."""
        self.goal = elect(self.chart, self.used)
        return self.goal, search(self.chart, self.chart.cell, self.chart.heading)

    def want(self):
        """`(action_name, granularity)` — the harness's own next decision, in rung order.

        The exit comes first and unconditionally: gold in sight down the corridor outranks every
        other rung, including a floor change, because stepping onto it ends the episode."""
        seen = self.sight
        if seen is not None and seen.ok and seen.face == EXIT and seen.reach >= seen.depth:
            return "move_forward", A.granularity_for(min(3, seen.depth))
        goal, table = self.aim()
        climb = self.wanted_climb(goal)
        if climb:
            return climb, "small"
        if goal is not None:
            found = cheapest(goal.poses, table)
            if found is not None and found[1] is not None:
                return found[1]
        return sweep(self.chart, seen, self.last_name)

    def wanted_climb(self, goal):
        """The climb to press right now, or "".

        Four conditions, and the HUD is not one of them: this build draws no `[Q] UP` / `[E] DOWN`
        text at all, so the tile underfoot is the chart's answer. The body must be standing on a
        charted stair, the elected goal must be that stair's own link, the body must have stood there
        long enough to have faced the rest of the compass (`LOOK_AROUND`), and the link must be
        unspent."""
        if goal is None or goal.kind != STAIR or goal.link is None:
            return ""
        if not self.chart.on_stair() or self.chart.cell != goal.cell:
            return ""
        if self.stair_age < LOOK_AROUND or goal.link in self.used:
            return ""
        return goal.link[1]

    # ── govern ──────────────────────────────────────────────────────────────────────────────
    def travel(self, name, label):
        """Cells an action would really travel from this pose — the number the fold needs.

        A forward is measured by the pre-step ray, which is exact and free. A backward has no ray,
        so it is measured against the chart's VERIFIED ground and is pessimistic by construction."""
        asked = A.granularity_steps(label)
        if name == "move_forward":
            return self.sight.stride(asked) if self.sight is not None else asked
        if name == "move_backward":
            run = 0
            while run < asked and self.chart.verified(
                    A.advance(self.chart.cell, self.chart.heading, run + 1, True)):
                run += 1
            return run
        return 0

    def honours(self, name, label, goal, table):
        """Is the model's proposal FREE-OR-CHEAPER against the committed route?

        One comparison, in the currency the whole module is denominated in. Re-plan from the pose
        the proposal would leave the body in and keep it when the remaining route is no longer than
        the harness's own first decision would have left. A climb is never free — this build gives
        the model nothing to read a floor transition from — and the undo of the last turn is never
        free either.

        AN UNREADABLE FRAME HONOURS EVERYTHING. The whole mechanism rests on one decode; if that
        decode ever fails, every fact the chart holds is stale and every route over it is a guess, so
        the harness stands down to the pass-through it started life as rather than imposing a plan it
        can no longer justify. That is the floor of this candidate's behaviour, and it is the parent's
        score, not worse than it."""
        if self.sight is None or not self.sight.ok:
            return name is not None     # unreadable frame: degrade to the pass-through, not below it
        if name is None or name in A.CLIMB_ACTIONS:
            return False
        if A.is_undo(self.last_name, name):
            return False
        if name in A.MOVEMENT_ACTIONS and self.travel(name, label) <= 0:
            return False
        if goal is None:
            return True
        here = table.get((self.chart.cell, self.chart.heading))
        if here is None:
            return True
        after = self.after_pose(name, label)
        beyond = cheapest(goal.poses, search(self.chart, after[0], after[1]))
        return beyond is not None and beyond[0] <= here[0] - 1

    def after_pose(self, name, label):
        """The pose the proposal would leave the body in, folded the same way an outcome is."""
        if name in A.TURN_ACTIONS:
            return self.chart.cell, A.heading_after(self.chart.heading, name)
        cells = self.travel(name, label)
        back = name == "move_backward"
        return A.advance(self.chart.cell, self.chart.heading, cells, back), self.chart.heading

    def apply(self, proposal):
        """Govern one proposal and return the engine Action that will actually run."""
        self.decisions += 1
        name, label = A.read_proposal(proposal)
        goal, table = self.aim()
        if self.honours(name, label, goal, table):
            chosen, chosen_label, out = name, label, proposal
        else:
            chosen, chosen_label = self.want()
            self.overrides += 1
            out = A.retarget(proposal, chosen, chosen_label,
                             reason="off-route: a cheaper decision exists", proposed_name=name or "")
        if chosen in A.CLIMB_ACTIONS:
            self.used.add((self.chart.cell, chosen))
            self.climbs += 1
        self.pending = (chosen, self.travel(chosen, chosen_label))
        self.last_name = chosen
        return out

    # ── speak ───────────────────────────────────────────────────────────────────────────────
    def render_memory(self):
        """The working-memory block, filled from measurements and from nothing else."""
        seen = self.sight
        if seen is not None and seen.ok and seen.face == EXIT and seen.reach >= seen.depth:
            plan = "walk straight onto the GOLD EXIT - it is %d cell(s) ahead" % seen.depth
        elif self.goal is not None:
            plan = self.goal.line()
        else:
            plan = "sweep for any opening at all"
        underfoot = ("you ARE STANDING ON A STAIR (the tile under your feet is never drawn, which "
                     "is exactly why the screen shows none)" if self.chart.on_stair()
                     else "not a stair")
        try:
            return self.template.format(
                ahead=seen.ahead_line() if seen is not None else "not readable this step.",
                beside=seen.flank_line() if seen is not None else "not readable this step.",
                underfoot=underfoot, mapped=self.chart.line(), plan=plan)
        except Exception:
            return ""

    def notes(self):
        """A one-line trace summary, for the proposer that reads this episode afterwards."""
        return ("decisions=%d overrides=%d climbs=%d epoch=%d mapped=%d wipes=%d merges=%d"
                % (self.decisions, self.overrides, self.climbs, self.chart.epoch,
                   len(self.chart.grid()), self.chart.wipes, self.chart.merges))
