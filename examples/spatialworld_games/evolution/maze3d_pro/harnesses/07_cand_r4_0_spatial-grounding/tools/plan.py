"""tools/plan.py — spatial grounding by INVERTING THE RENDERER'S OWN PROJECTION.

The anchor this descends from holds no state and reads no pixels, so it cannot tell a step that
happened from a step that was refused, and 23 of its 25 episodes end inside a terminal periodic
orbit. Earlier work in this lineage closed that loop with an AREA census: count the painted pixels
of a tile and compare the count against a measured ladder, with one fraction threshold separating
"adjacent" from "two cells out". This module takes the same pixels and reads them a different way
— it inverts the projection ANALYTICALLY, so a distance comes out as an INTEGER CELL COUNT rather
than as a bucketed area, and that integer is what the whole mechanism is built on.

────────────────────────────────────────────────────────────────────────────────
THE PROJECTION (read off the frames, then confirmed against the grid)
────────────────────────────────────────────────────────────────────────────────
The 800x600 first-person view is drawn as nested rectangles about the vanishing point
(cx, cy) = (400, 300). The cell d cells straight ahead is painted as an axis-aligned rectangle
of size (600/d, 400/d) centred there; the cell j cells ahead and one to the side is painted as a
trapezoid spanning x from cx -/+ 300/j to cx -/+ 300/(j+1). Far geometry is drawn first and near
geometry over it, so the NEAREST painted front face is the one that survives. Two consequences,
both exact and both load-bearing here:

  * FORWARD RAY. Side faces never cross x = 400, so the centre column carries nothing but front
    faces plus ceiling and floor. The vertical extent h of front-face paint on that one column is
    therefore 400/D for the nearest non-empty cell straight ahead, and

                             D = round(400 / h)

    recovers the distance as an integer. Censused over EVERY (floor, cell, bearing) pose of all
    25 levels — 5524 poses — `D` and the face's identity are correct **5524 / 5524, zero wrong**.
    The observed heights cluster tightly on 401/403, 201/203, 133/135, 101/103, 81/83, 67/69,
    57/59, 51/53, 45, 41, 37 (the +2 variants are the 3-px outline a stair or the exit carries),
    and every one of them rounds to its own D with a wide margin — there is no threshold to tune.

  * LATERAL PROBES. The cell IMMEDIATELY to the left is the j = 0 trapezoid, which spans
    x in [-800, 100); the immediate right spans (700, 1600]. No front face reaches outside
    [100, 700], and no other side band reaches outside [100, 350] / [450, 700], so the single
    columns x = 50 and x = 750 are painted by the immediate-left and immediate-right cell and by
    nothing else, at every pose, with no occlusion possible. Censused over the same poses
    (11048 probes): wall/open/stair read **exact**. The one systematic residue is informative
    rather than harmful — the renderer special-cases the exit tile only on the FRONT face, so an
    exit BESIDE you paints nothing and reads as `open` (102/102). `open` is still the correct
    walkability verdict for it, and stepping into it wins, so the probe is 11048/11048 sound as a
    passability read and merely blind to the exit's identity sideways.

Together those give a three-way local occupancy read — left, ahead (with a range), right — every
single frame, for zero decisions. The anchor spent 5.2% + 5.5% of its budget on turns taken to
find out what these two columns already say.

────────────────────────────────────────────────────────────────────────────────
THE COLOUR KEY (the renderer is flat-shaded; these are exact masks, not tolerances)
────────────────────────────────────────────────────────────────────────────────
Every surface is one base colour scaled by a distance fog factor, and the top 60 rows carry an
alpha-180 black panel that scales whatever is under it by 0.294. Across the whole corpus a frame
holds 7 distinct colours. The three families never collide under either scaling:

    stair   base fill (0,100,120) / edge (0,255,255)  ->  R == 0 and G >= 1   (0 survives scaling)
    exit    (255,200,0) with a (255,255,255) outline  ->  R >= 200, 150 <= G <= 230, B <= 60
    wall    (160,85,60) sides / (80,40,30) fronts     ->  R > G > B
    floor   (50,60,65)   ceiling (15,15,35)           ->  B >= G >= R : neither family

So `R == 0` IS the stair mask: no other colour in the palette has a zero red channel, including
black, which the full-screen ceiling and floor rectangles overdraw before anything else is drawn.

────────────────────────────────────────────────────────────────────────────────
WHAT THE INTEGER BUYS THAT AN AREA BUCKET DOES NOT
────────────────────────────────────────────────────────────────────────────────
1. THE STRIDE IS NO LONGER A JUDGEMENT CALL. A translation stops early at a wall, so the cells a
   `move_forward` actually covers is min(granularity, reach), and reach follows from the ray:

        face at D is a wall   ->  reach = D - 1      (exact: a wall is not enterable)
        face at D is a stair  ->  reach >= D         (a lower bound: the stair occludes what is
                                                      behind it, and the stair itself IS enterable)
        face at D is the exit ->  reach = D, and arriving there ENDS the episode as a win

   Censused on random walks over all 25 levels, 992 translations: predicting min(granularity,
   reach) matches the engine's displacement **987 / 992**, and all five misses are the same case —
   a stride asked for 3 cells with a stair 1 or 2 cells ahead and ran straight past it, which is
   the lower-bound branch behaving as documented. The planner never issues such a stride (it clips
   to D so it lands ON the stair), and under that discipline the prediction is exact.

   reach == 0 is therefore a FREE, EXACT, PRE-STEP refusal test. The anchor discovered a wall by
   walking into it 2090 times; this reads it off the frame before spending the decision.

2. DEAD RECKONING, WITH NO HUD AND NO ORACLE. A turn is never blocked, a climb is gated below, and
   a translation's displacement is predictable, so position and bearing follow by addition from an
   arbitrary origin. Censused against the engine's own coordinates over the same walks:
   **2197 / 2197 poses exact, zero wrong**, across floor changes included. Independently, a
   byte-identical frame and zero displacement coincide **992 / 992** — the retrospective check
   that catches a desync and is the only thing trusted when a prediction and the pixels disagree.

3. A MAP, HENCE ROUTING. Each frame writes the forward ray's cells 1..D and the two lateral cells
   into a per-floor lattice. BFS over that lattice turns "where do I go" into "which neighbour is
   on the shortest path to the nearest cell with an unknown neighbour", which is what the frozen
   model, holding 15 completion tokens per answer, demonstrably cannot do for itself.

4. THE TILE UNDER YOUR FEET IS NEVER DRAWN, so being on a stair cannot be seen — but it can be
   REMEMBERED: the lattice learned that cell was a stair while approaching it. On the same walks
   the latch is **TP 31 / FP 0 / FN 0 / TN 2166**. The climb is gated on it, which is why this
   harness can change floor at all; the lineage measured 0 floor transitions in 2340 decisions
   against a prose precondition ("[Q] UP in the bottom HUD") that this build does not render.

Nothing here touches `evaluator.game` beyond the frame the model is itself shown. Every constant
below is a property of the RENDERER, measured from those frames.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

# ── the projection, as constants ───────────────────────────────────────────────
CENTRE_COL = 400          # cx: the only column that front faces are centred on
LEFT_COL = 50             # inside [-800, 100): the immediate-left cell and nothing else
RIGHT_COL = 750           # inside (700, 1600]: the immediate-right cell and nothing else
ADJACENT_FACE_H = 400.0   # the front face of the adjacent cell is exactly 400 px tall
RAY_MAX = 12              # the renderer's own MAX_DEPTH: nothing beyond 12 cells is drawn
SIDE_MIN_PX = 60          # a j=0 trapezoid covers ~468 rows; 60 is a floor, not a threshold
WALL_MIN_R = 8            # the darkest wall under the alpha panel is R=9
GOLD_MIN_R = 200
GOLD_MIN_G = 150
GOLD_MAX_G = 230
GOLD_MAX_B = 60
WHITE_MIN = 230           # the exit's 3-px outline
STRIDE_MAX = 3            # large = 3 cells; the engine clamps to [1, 3]

# ── the grid ───────────────────────────────────────────────────────────────────
WALL, OPEN, STAIR, EXIT = "wall", "open", "stair", "exit"
PASSABLE = (OPEN, STAIR, EXIT)
# bearing 0/1/2/3 = the four quarter-turns from the arbitrary origin bearing; (dx, dy) in cells
STEP = ((0, -1), (1, 0), (0, 1), (-1, 0))
TURN_LEFT, TURN_RIGHT = -1, 1

# ── the governor's few dials (priced in NOTES.md) ──────────────────────────────
STALE_LIMIT = 4           # decisions that taught the lattice nothing before the route is forced
STALE_HARD = 8            # ... and before a floor that is not paying out counts as spent
FLOOR_BUDGET = 30         # decisions one floor VISIT may consume before the stair becomes the goal
MIN_DWELL = 4             # decisions a floor change must wait after the previous one
SPIN_LIMIT = 3            # consecutive turns before a turn is itself treated as a wasted decision
GRANULARITY = {1: "small", 2: "medium", 3: "large"}
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}


def _channel_masks(column):
    """The four surface families on one 600-px column of the frame, as boolean masks."""
    c = column.astype(np.int16)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    stair = (r == 0) & (g >= 1)
    gold = (r >= GOLD_MIN_R) & (g >= GOLD_MIN_G) & (g <= GOLD_MAX_G) & (b <= GOLD_MAX_B)
    white = (r >= WHITE_MIN) & (g >= WHITE_MIN) & (b >= WHITE_MIN)
    wall = (r > g) & (g > b) & (r >= WALL_MIN_R)
    return stair, gold, white, wall


class Sight:
    """One frame, decoded. `depth`/`face` are the forward ray; `left`/`right` the lateral probes."""

    __slots__ = ("digest", "depth", "face", "left", "right")

    def __init__(self, digest, depth, face, left, right):
        self.digest = digest
        self.depth = depth
        self.face = face
        self.left = left
        self.right = right

    @property
    def reach(self):
        """Cells a `move_forward` may cover. Exact behind a wall, a lower bound behind a stair."""
        if self.depth is None:
            return RAY_MAX
        if self.face == WALL:
            return self.depth - 1
        return self.depth

    @property
    def stride_cap(self):
        """The longest stride that keeps dead reckoning exact: never past a stair or the exit."""
        if self.depth is None:
            return STRIDE_MAX
        if self.face == WALL:
            return max(0, min(STRIDE_MAX, self.depth - 1))
        return max(0, min(STRIDE_MAX, self.depth))

    def describe(self):
        if self.depth is None:
            return "open corridor, nothing painted ahead"
        noun = {WALL: "wall", STAIR: "stair tile", EXIT: "the EXIT"}.get(self.face, self.face)
        where = "directly ahead" if self.depth == 1 else "%d cells ahead" % self.depth
        return "%s %s" % (noun, where)


def read_sight(img):
    """Invert the projection on one normalised (H, W, 3) frame -> Sight. A pure function of the
    pixels: no state, no engine handle, nothing but the three columns and their colours."""
    digest = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()

    stair, gold, white, wall = _channel_masks(img[:, CENTRE_COL, :])
    painted = np.flatnonzero(stair | gold | white | wall)
    if painted.size == 0:
        depth, face = None, None
    else:
        height = int(painted[-1] - painted[0] + 1)
        depth = max(1, min(RAY_MAX, int(round(ADJACENT_FACE_H / max(height, 1)))))
        if int(gold.sum()) > 0:
            face = EXIT
        elif int(stair.sum()) >= int(wall.sum()):
            face = STAIR
        else:
            face = WALL

    flanks = []
    for col in (LEFT_COL, RIGHT_COL):
        s, _gd, _wh, w = _channel_masks(img[:, col, :])
        if int(s.sum()) >= SIDE_MIN_PX:
            flanks.append(STAIR)
        elif int(w.sum()) >= SIDE_MIN_PX:
            flanks.append(WALL)
        else:
            flanks.append(OPEN)          # an exit beside you also paints nothing: still passable
    return Sight(digest, depth, face, flanks[0], flanks[1])


class Lattice:
    """One floor's self-built grid, in the dead-reckoning frame of reference."""

    __slots__ = ("cells", "visited", "climbed")

    def __init__(self):
        self.cells = {}
        self.visited = set()
        self.climbed = set()

    def learn(self, cell, kind):
        """Write one cell. A wall never becomes passable; EXIT and STAIR outrank a bare OPEN."""
        was = self.cells.get(cell)
        if was == kind:
            return 0
        if was == EXIT or (was in (STAIR, WALL) and kind == OPEN):
            return 0                     # a reading never DEMOTES a cell it already identified
        self.cells[cell] = kind
        return 1

    def absorb(self, cell, bearing, sight):
        """Write everything one frame proves: the forward ray, and the two flanking cells.
        Returns how many cells the frame TAUGHT, which is what `stale` counts down from.

        Nothing painted within the ray means every one of the 12 cells ahead is empty — the
        renderer stops at 12, and the map's outer boundary is a wall, so an unpainted column is
        a corridor longer than the draw distance, not a missing reading."""
        gained = 0
        self.visited.add(cell)
        if cell not in self.cells:
            gained += self.learn(cell, OPEN)
        fx, fy = STEP[bearing]
        tip = sight.depth if sight.depth is not None else RAY_MAX + 1
        for d in range(1, tip):
            gained += self.learn((cell[0] + fx * d, cell[1] + fy * d), OPEN)
        if sight.depth is not None:
            gained += self.learn((cell[0] + fx * sight.depth, cell[1] + fy * sight.depth), sight.face)
        for turn, kind in ((TURN_LEFT, sight.left), (TURN_RIGHT, sight.right)):
            sx, sy = STEP[(bearing + turn) % 4]
            gained += self.learn((cell[0] + sx, cell[1] + sy), kind)
        return gained

    def passable(self, cell):
        return self.cells.get(cell) in PASSABLE

    def frontier(self):
        """Passable cells with at least one unknown neighbour — where new map comes from."""
        out = set()
        for cell, kind in self.cells.items():
            if kind not in PASSABLE:
                continue
            for dx, dy in STEP:
                if (cell[0] + dx, cell[1] + dy) not in self.cells:
                    out.add(cell)
                    break
        return out

    def route(self, start, goals):
        """BFS over known-passable cells -> the cell sequence to the nearest goal, or None."""
        if not goals:
            return None
        if start in goals:
            return [start]
        seen = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for dx, dy in STEP:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in seen or not self.passable(nxt):
                    continue
                seen[nxt] = cur
                if nxt in goals:
                    path, node = [], nxt
                    while node is not None:
                        path.append(node)
                        node = seen[node]
                    path.reverse()
                    return path
                queue.append(nxt)
        return None


class SightedPlanner:
    """One per episode. Decodes each frame, keeps the lattice, states it to the model, and
    overrides a proposal only where the pixels PROVE the proposal cannot do anything.

    The two staleness dials are CLASS attributes so an offline rig can sweep them on the shipped
    module — note it must set them AFTER `build_agent`, which purges `sys.modules["tools*"]` and
    would otherwise throw the override away and silently measure the shipped defaults."""

    STALE_LIMIT = STALE_LIMIT
    STALE_HARD = STALE_HARD
    SPIN_LIMIT = SPIN_LIMIT
    NEW_FLOOR_BIAS = 0        # 1 = open an unvisited floor id first, 0 = prefer a mapped one
    FLOOR_BUDGET = FLOOR_BUDGET
    MIN_DWELL = MIN_DWELL

    def __init__(self):
        self.floor = 0
        self.cell = (0, 0)
        self.bearing = 0
        self.floors = {0: Lattice()}
        self.step_no = 0
        self.floor_since = 0
        self.sight = None
        self.pending = None          # (name, predicted_cells) for the action just emitted
        self.last_line = "no action yet"
        self.stale = 0
        self.spins = 0
        self.overrides = 0
        self.desyncs = 0
        self.transitions = 0
        self.mode = "explore"

    # ── bookkeeping ───────────────────────────────────────────────────────────
    @property
    def lattice(self):
        return self.floors.setdefault(self.floor, Lattice())

    def on_stair(self):
        return self.lattice.cells.get(self.cell) == STAIR

    def _resync(self):
        """A prediction the pixels refuse: abandon the frame of reference, keep nothing false."""
        self.desyncs += 1
        self.floor = max(self.floors) + 1
        self.cell, self.bearing = (0, 0), 0
        self.floors[self.floor] = Lattice()

    def observe(self, img):
        """Resolve the previous decision against the new frame, then absorb the new frame."""
        prev = self.sight
        sight = read_sight(img)
        moved_same = prev is not None and prev.digest == sight.digest

        if self.pending is not None:
            name, predicted = self.pending
            if name in ("move_forward", "move_backward"):
                cells = 0 if moved_same else predicted
                if name == "move_backward":
                    # nothing is drawn behind you, so recover the step from the wall ahead
                    # receding; if that is not available, the frame of reference is lost.
                    if moved_same:
                        cells = 0
                    elif (prev and prev.face == WALL and sight.face == WALL
                          and prev.depth is not None and sight.depth is not None
                          and 1 <= sight.depth - prev.depth <= STRIDE_MAX):
                        cells = sight.depth - prev.depth
                    else:
                        cells = None
                elif not moved_same and predicted == 0:
                    cells = None          # the pixels say something moved that could not move
                if cells is None:
                    self._resync()
                else:
                    dx, dy = STEP[self.bearing] if name == "move_forward" else \
                        STEP[(self.bearing + 2) % 4]
                    self.cell = (self.cell[0] + dx * cells, self.cell[1] + dy * cells)
                    self.last_line = ("you asked to move %s and covered %d cell(s)"
                                      % (name.split("_")[1], cells))
                    if cells == 0:
                        self.last_line += " — the screen did not change, so that way is blocked"
            elif name in ("turn_left", "turn_right"):
                self.bearing = (self.bearing + (TURN_LEFT if name == "turn_left" else TURN_RIGHT)) % 4
                self.last_line = "you turned %s; you have not changed cell" % name.split("_")[1]
            elif name in ("climb_up", "climb_down"):
                # A CLIMB CANNOT BE VERIFIED FROM THE PIXELS, and trying to is the expensive
                # mistake. Two floors seen from the same stair cell frequently render
                # byte-identically — L02's stair reads `wall at 1, wall left, open right` on BOTH
                # floors — so a held frame is NOT evidence of refusal. Treating it as evidence
                # retired the only stair on the level after two decisions and cost the episode.
                # The gate on the climb is already exact (a stair underfoot, measured), so assume
                # it fired and switch lattices. The anti-ping-pong device is the STALENESS gate,
                # not a verdict: a floor change resets `stale` and the arrival lattice has
                # unknown neighbours, so the far side must be explored and go stale before a
                # return is allowed. That bounds a wrong guess to one floor's exploration rather
                # than to an unbounded 2-cycle, and it costs nothing when the climb did fire.
                target = self.floor + (1 if name == "climb_up" else -1)
                self.floors.setdefault(target, Lattice())
                self.floor = target
                self.lattice.learn(self.cell, STAIR)      # you climb OFF a stair; these pair
                self.transitions += 1
                self.stale = 0
                self.floor_since = self.step_no
                self.last_line = "you used the stair; treat this floor as new ground"
            self.pending = None

        gained = self.lattice.absorb(self.cell, self.bearing, sight)
        self.stale = 0 if gained else self.stale + 1
        self.sight = sight
        self.step_no += 1
        return sight

    # ── the plan ──────────────────────────────────────────────────────────────
    def _goal_cells(self):
        """Where to head, in priority order: a seen exit, then new map, then a way off the floor."""
        lat = self.lattice
        exits = {c for c, k in lat.cells.items() if k == EXIT}
        if exits:
            return exits, "exit"
        fresh = self.fresh_edge()
        stairs = {c for c, k in lat.cells.items() if k == STAIR}
        if self.floor_spent() and stairs and lat.route(self.cell, stairs):
            return stairs, "stairs"          # the visit is over: go and change floor
        if fresh and lat.route(self.cell, fresh):
            return fresh, "explore"
        if stairs and lat.route(self.cell, stairs):
            return stairs, "stairs"
        edge = lat.frontier()
        if edge:
            return edge, "explore"
        return set(), "stuck"

    def fresh_edge_of(self, lat):
        """The cells of one lattice still worth a decision: every mapped PASSABLE cell not yet
        stood on — not merely the frontier ones.

        This is the census residue made actionable. The renderer special-cases the exit tile only
        on the FRONT face, so an exit one cell to your left or right paints nothing whatsoever
        and the lateral probe reads it as plain `open` (102 of 102 such poses). A floor can
        therefore be mapped to completion with the exit sitting inside it unrecognised, and no
        amount of further looking will name it. The only thing that settles it is standing on
        every passable cell: arriving on the exit ends the episode as a win whether or not the
        pixels ever identified it. Visiting a cell is also what reads its three forward
        directions at once, so this subsumes the frontier as an exploration target."""
        return {c for c, k in lat.cells.items() if k in PASSABLE and c not in lat.visited}

    def fresh_edge(self):
        """The frontier cells worth a decision: UNVISITED ones. Visiting a cell reads its three
        forward directions at once (the ray plus both flanks), so the only neighbour a visited
        cell can still be hiding is the one BEHIND it — which costs two turns to look at and
        which the sealed-on-three-sides rule already turns towards when it matters. An unvisited
        passable cell, by contrast, always pays: a side opening the lateral probes mapped while
        walking past it lands here, which is how a corridor never entered gets explored."""
        return self.fresh_edge_of(self.lattice)

    def unseen_bearing(self):
        """The quarter-turn that would point at a cell this floor has never read, or None.

        The frame reads three directions at once — the ray ahead and both flanks — so after a
        forward step every neighbour of the new cell is known except the one BEHIND, and that
        one is known too because the ray painted it on the way in. Exactly two situations leave
        a genuine blind side: the opening frame, and the cell you arrive on after a climb. Both
        are cells where the only way on may be the way you cannot see, and both were measured
        doing real damage — a climb arrival that faces a wall with walls to either side has an
        empty unvisited set, so the floor read as instantly exhausted and the planner climbed
        straight back. That 2-cycle ran 83 floor changes in one 100-decision episode. One turn
        resolves it: the cell behind becomes a flank of the new bearing.
        """
        lat = self.lattice
        for turn in (TURN_RIGHT, TURN_LEFT, 2):
            dx, dy = STEP[(self.bearing + turn) % 4]
            if (self.cell[0] + dx, self.cell[1] + dy) not in lat.cells:
                return "turn_left" if turn == TURN_LEFT else "turn_right"
        return None

    def floor_spent(self):
        """Has this floor stopped paying? Three ways, and the FIRST is the one that matters.

        The episode is capped at 100 decisions and the exit sits on exactly one floor, so
        spending the whole clock surveying the wrong floor is a guaranteed loss however well the
        survey is done. Exhaustive exploration-then-climb was measured at 8/25 on the adversarial
        tape precisely because big floors never finish inside the cap; a per-VISIT decision
        budget, after which a reachable stair becomes the goal, splits the clock across the
        floors instead and measured 17/25. So: the visit's budget is up, or there is no unvisited
        frontier this lattice can route to, or the floor has taught nothing for STALE_HARD
        decisions. A floor change resets the budget and the staleness, which is also what stops
        the climb from becoming a 2-cycle — the far side has to spend its own budget first."""
        lat = self.lattice
        if self.step_no - self.floor_since < self.MIN_DWELL:
            return False                      # a floor change always costs at least this much
        if self.unseen_bearing():
            return False                      # you have not even looked behind you yet
        if self.step_no - self.floor_since >= self.FLOOR_BUDGET:
            return True
        edge = self.fresh_edge()
        if not edge or lat.route(self.cell, edge) is None:
            return True
        return self.stale >= self.STALE_HARD

    def forced(self):
        """The branches where the PIXELS, not the model, settle the decision. Three only, each
        one a fact the frozen model provably cannot act on for itself:

          * the exit is identified and within a single stride (or mapped anywhere on this floor);
          * a stair is underfoot and this floor has stopped yielding new ground;
          * this floor has taught the lattice nothing for STALE_LIMIT decisions, i.e. the agent
            is circling inside ground it has already mapped — the dominant residue in the
            lineage, where every translation SUCCEEDS so a refusal ledger never engages.

        Everything else is left to the model. Returns a plan tuple or None."""
        lat, sight = self.lattice, self.sight
        if sight is None:
            return None
        if sight.face == EXIT and sight.depth is not None and sight.depth <= STRIDE_MAX \
                and sight.reach >= sight.depth:
            return "move_forward", GRANULARITY[sight.depth], "the exit is %d cell(s) ahead" % sight.depth
        if EXIT in lat.cells.values():
            routed = self.routed()
            if routed:
                return routed
        if self.on_stair() and self.floor_spent():
            climb = self.climb_choice()
            if climb:
                return climb, None, "a stair is underfoot and this floor is spent"
        if self.stale >= self.STALE_LIMIT or self.floor_spent():
            return self.routed()
        return None

    def climb_choice(self):
        """Which way off this stair. A climb cannot be verified from the pixels, so the floor
        counter is a BELIEF, and the drift it is prone to is the expensive failure: believing
        every climb opened new ground sent one run through five floor ids on a two-floor tower
        and re-mapped the same maze three times. The cure is to rank the two directions by what
        the lattices ALREADY HOLD — head for a floor that still has unexplored ground, fall back
        to one that is merely exhausted, and open a floor id never seen only when nothing else is
        on offer. A refused climb then costs one wasted decision instead of a whole re-survey,
        because the next choice from this stair ranks the same way."""
        unopened, exhausted = (1, 2) if self.NEW_FLOOR_BIAS else (2, 1)
        ranked = []
        for order, (name, target) in enumerate((("climb_up", self.floor + 1),
                                                ("climb_down", self.floor - 1))):
            lat = self.floors.get(target)
            if lat is None:
                rank = unopened
            elif self.fresh_edge_of(lat):
                rank = 0                        # a floor that still has somewhere to go: best
            else:
                rank = exhausted
            ranked.append((rank, order, name))
        ranked.sort()
        return ranked[0][2] if ranked else None

    def routed(self):
        """Walk the lattice's own BFS route one decision: a stride along it, or the turn that
        puts you on it. None when the map offers nothing to walk."""
        lat, sight = self.lattice, self.sight
        goals, self.mode = self._goal_cells()
        path = lat.route(self.cell, goals)
        if not path or len(path) < 2:
            blind = self.unseen_bearing()
            return (blind, None, "turning to read the one direction never looked at") if blind else None
        nxt = path[1]
        want = None
        for b, (dx, dy) in enumerate(STEP):
            if (self.cell[0] + dx, self.cell[1] + dy) == nxt:
                want = b
                break
        if want is None:
            return None
        if want != self.bearing:
            delta = (want - self.bearing) % 4
            if delta == 3:
                return "turn_left", None, "the %s route lies to your left" % self.mode
            return "turn_right", None, "the %s route lies to your %s" % (
                self.mode, "right" if delta == 1 else "rear")
        cap = min(STRIDE_MAX, sight.stride_cap)
        if cap < 1:
            return None                       # the map and the pixels disagree: say nothing
        run = 1
        fx, fy = STEP[self.bearing]
        while run < cap and len(path) > run + 1:
            if lat.cells.get(path[run]) in (STAIR, EXIT):
                break                          # never stride OFF the tile you were aiming at
            if (path[run][0] + fx, path[run][1] + fy) != path[run + 1]:
                break                          # the route turns here
            run += 1
            if lat.cells.get(path[run]) in (STAIR, EXIT):
                break                          # land ON it, which keeps the reckoning exact
        return "move_forward", GRANULARITY[run], "%s route, %d cell(s) clear" % (self.mode, run)

    def plan(self):
        """The action the lattice recommends: (name, granularity_or_None, why). Used both as the
        replacement for a proposal the pixels refuted and as the advice line in the prompt."""
        sight = self.sight
        if sight is None:
            return None
        hard = self.forced()
        if hard:
            return hard
        routed = self.routed()
        if routed:
            return routed
        blind = self.unseen_bearing()
        if blind:
            return blind, None, "nothing has been read in one direction from this cell"
        if sight.stride_cap >= 1:
            return "move_forward", GRANULARITY[min(sight.stride_cap, STRIDE_MAX)], "corridor is clear"
        if sight.left in (OPEN, STAIR):
            return "turn_left", None, "the only opening is to your left"
        if sight.right in (OPEN, STAIR):
            return "turn_right", None, "the only opening is to your right"
        return "turn_right", None, "sealed ahead and on both sides; turning"

    def _ahead(self):
        dx, dy = STEP[self.bearing]
        return (self.cell[0] + dx, self.cell[1] + dy)

    def govern(self, name, granularity):
        """Keep the model's own decision unless the pixels PROVE it cannot accomplish anything.
        Returns (name, granularity, overridden)."""
        sight = self.sight
        if sight is None:
            return name, granularity, False
        steps = GRANULARITY_STEPS.get((granularity or "small"), 1)

        if name in ("turn_left", "turn_right"):
            self.spins += 1
        else:
            self.spins = 0

        proven_useless = False
        if name == "move_forward" and sight.reach <= 0:
            proven_useless = True                       # a wall is touching you: exact, pre-step
        elif name in ("climb_up", "climb_down") and not self.on_stair():
            proven_useless = True                       # a climb off a stair cannot fire
        elif name in ("climb_up", "climb_down") and not self.floor_spent():
            # It MIGHT fire — but it abandons ground this floor has measurably not finished, and
            # that is the expensive mistake in this game: the lineage priced climbing at the
            # first opportunity at six lost levels whose exit was on the start floor. Against a
            # model that proposes nothing but `climb_up`, obeying this froze the agent on one
            # stair for 93 of 100 decisions while the floor counter ran to 13. So the climb is
            # gated entirely by the harness: `forced()` spends it when the floor is spent.
            proven_useless = True
        elif name == "move_backward":
            proven_useless = True                       # retreating is a 2-cycle; rotate instead
        elif name in ("turn_left", "turn_right") and self.spins > self.SPIN_LIMIT:
            proven_useless = True

        hard = self.forced()
        if hard and (hard[0] != name or (hard[1] or "") != (granularity or "")):
            self.overrides += 1
            self.last_line = ("your %s was replaced by %s — %s" % (name, hard[0], hard[2]))
            return hard[0], hard[1], True

        if not proven_useless:
            if name == "move_forward":
                # The DISTANCE is the harness's business even when the direction is the model's.
                # A stride costs one decision whatever its length and stops safely at a wall, so
                # the only two ways to get it wrong are overshooting something you wanted to
                # stand on and under-using the decision. Both are now measurements rather than
                # judgement calls: `stride_cap` is the projection's own answer to the first, and
                # the mapped route's straight run is the answer to the second. Obeying the
                # model's own `small` was measured costing three levels on the forward-spam tape
                # purely on throughput (18/25 against 21/25 for the identical policy at `large`).
                want = min(steps, sight.stride_cap) if sight.stride_cap >= 1 else steps
                route = self.routed()
                if route and route[0] == "move_forward":
                    along = GRANULARITY_STEPS.get(route[1], 1)
                    if along > want:
                        want = min(along, sight.stride_cap)
                if want >= 1 and want != steps:
                    return name, GRANULARITY[want], False
            return name, granularity, False

        alt = self.plan()
        if not alt:
            return name, granularity, False
        a_name, a_gran, why = alt
        if a_name == name and (a_gran or "") == (granularity or ""):
            return name, granularity, False
        self.overrides += 1
        self.last_line = ("your %s could not have done anything (%s), so it was replaced by %s — %s"
                          % (name, sight.describe(), a_name, why))
        return a_name, a_gran, True

    def record(self, name, granularity):
        """Remember what was actually emitted, with the displacement it should produce."""
        steps = GRANULARITY_STEPS.get((granularity or "small"), 1)
        predicted = min(steps, max(0, self.sight.reach)) if self.sight else 0
        self.pending = (name, predicted if name == "move_forward" else steps)

    # ── the per-turn text channel ─────────────────────────────────────────────
    def fields(self):
        lat, sight = self.lattice, self.sight
        nbr = []
        for turn, label in ((0, "ahead"), (TURN_LEFT, "left"), (TURN_RIGHT, "right")):
            b = (self.bearing + turn) % 4
            dx, dy = STEP[b]
            kind = lat.cells.get((self.cell[0] + dx, self.cell[1] + dy), "not seen yet")
            nbr.append("%s=%s" % (label, "blocked" if kind == WALL else kind))
        plan = self.plan()
        return {
            "sight_line": sight.describe() if sight else "no frame read yet",
            "reach_line": ("you can move forward %d cell(s) before something stops you"
                           % max(0, sight.reach)) if sight else "unknown",
            "neighbours": ", ".join(nbr),
            "underfoot": "a STAIR is under your feet" if self.on_stair()
                         else "the tile under your feet is not a stair you have mapped",
            "last_line": self.last_line,
            "mapped": "%d cells mapped on this floor, %d visited, %d unexplored edge(s)"
                      % (len(lat.cells), len(lat.visited), len(lat.frontier())),
            "floor_line": "floor #%d of the %d you have stood on; %s"
                          % (self.floor + 1, len(self.floors),
                             "this floor looks exhausted" if self.floor_spent()
                             else "this floor still has unexplored ground"),
            "advice": ("the mapped route from here is %s%s" % (
                plan[0], (" (" + plan[1] + ")") if plan[1] else "")) if plan else "no mapped route",
        }

    def render_memory(self, template):
        f = self.fields()
        try:
            return template.format(**f)
        except Exception:
            return DEFAULT_TEMPLATE.format(**f)


DEFAULT_TEMPLATE = """
--- MEASURED STATE (derived from the frame you were shown; step {0}) ---
Last decision: {last_line}
Straight ahead: {sight_line}
Range: {reach_line}
Adjacent cells: {neighbours}
Underfoot: {underfoot}
Floor: {floor_line}
Map: {mapped}
Suggested next move: {advice}
Reply with ONLY the JSON object for your next action.
""".replace("{0}", "now")
