"""tools/plan.py — EXACT SCENE DECODING + a dead-reckoned occupancy map + a route-planning governor.

THE PROBLEM THIS SOLVES. The parent anchor hands every decision to `evaluator.baseline_step` and
keeps no state, so nothing in the loop can tell a move that happened from a move that did not. Its
own 25 episodes spend 80.3% of all decisions inside terminal periodic orbits (13 episodes press one
refused translation to the step cap; 4 more run a 4-to-16-action limit cycle) and emit 0 floor
transitions on 25 two-floor levels. Both are failures of MEASUREMENT, not of planning: the frame the
agent is already shown says exactly what is in front of it, and nothing was reading it.

WHAT IS MEASURED, AND WHY IT IS EXACT (not a tolerance, not a heuristic).
`draw_3d_view` is a flat-shaded painter with a closed-form geometry, and this module inverts it.
Reading the renderer rather than guessing at it gives three facts per frame:

  * rects[d] = (600/d, 400/d) centred on (400, 300), painted FAR-TO-NEAR, so the nearest non-empty
    cell on the forward ray completely covers every farther one. A 21x21 patch at the frame centre
    is inside rects[12], the smallest front face drawn, so that patch is ALWAYS filled by the
    nearest front face when one exists within MAX_DEPTH=12 cells.
  * every face is `darken(base, fog)` with `fog(d) = max(0.4, 1 - 0.08*d)` and integer truncation,
    so the exact RGB of that patch names BOTH the material and its distance: a front wall is
    (80,40,30)*fog -> (73,36,27) at one cell, (67,33,25) at two, ... ; a stair is (0,100,120)*fog
    with a (0,255,255)*max(fog,0.8) border; the exit is a flat (255,200,0) with a white outline.
    fog saturates at d >= 8, and there the half-width of the painted run recovers d exactly instead.
  * the d=1 side quads span x in [-800, 100] and [700, 1600], so columns 0..89 and 710..799 are
    painted by NOTHING except the cell immediately to the left / right of the agent — a one-cell
    readout with no occlusion and no mixing.

Validated against the engine's own `tower_data` over all 25 levels, both floors, every reachable
cell and all four headings: **forward ray 5624/5624 exact, side cells 11248/11248 exact.** The
`(kind, distance)` of the first non-empty cell ahead and the passability of both lateral neighbours
are therefore measurements, so the number of cells a translation will actually travel is known
BEFORE it is spent (`walkable`), and the heading is exactly dead-reckonable because the engine never
refuses a turn.

WHAT IT BUILDS WITH THEM. A pose (x, y, heading) and a per-floor occupancy map in ONE shared
coordinate frame, both derived only from `frame_data.image` and the agent's own action history.
Each observation writes the whole forward ray plus two lateral cells; BFS over the known-passable
cells then answers the three questions the orbits were failing to answer:
    1. is a known EXIT cell reachable?           -> route onto it and the episode ends
    2. is a FRONTIER (a passable cell with an unknown neighbour) reachable? -> route to it
    3. is this floor finished?                   -> only then route to a stair and change floor
The cell under the agent's feet is never drawn, so "am I on a stair" is unreadable from one frame —
but a stair seen ahead or beside is mapped exactly, so standing on a mapped stair cell IS the
precondition, with no latch and no threshold. (3) is the commitment gate the lineage measured as
load-bearing: climbing at the first opportunity costs the levels whose exit is on the start floor,
while exhausting the floor first cannot.

A floor change is NOT observable: 46 of 240 possible climbs in this benchmark (19.2%) leave the
frame byte-identical even though they succeeded, so frame identity CANNOT decide whether a climb
happened. Every issued climb is therefore treated as a success into a fresh map epoch, which is
sound either way, and the ambiguity is resolved for free afterwards: per-pose frame digests from the
previous epoch are compared against the new epoch's as they arrive, and two matches at mapped poses
retire the new epoch back into the old one (a refused climb), while one mismatch confirms the change.

THE MODEL STILL DECIDES. `baseline_step` is called every step and its proposal is executed unless
the measurement proves it cannot help: a translation with `walkable == 0`, a climb with no mapped
stair underfoot, or (under the default OBEY_MODE) a legal move that adds no coverage while a route
exists. The route action is computed first and the proposal is obeyed whenever it agrees, so
agreement is free. `render_memory()` states every measured fact back to the model in plain text so a
displaced proposal is not simply re-proposed.
"""
from __future__ import annotations

import hashlib
from collections import deque

try:                                     # the engine hands over a numpy frame; never hard-fail
    import numpy as _np
except Exception:                        # pragma: no cover
    _np = None

from . import action as _act

# ── the renderer's own constants (mllm_base_agent/.../maze3d_pro/main.py) ─────────────────────
SCREEN_W, SCREEN_H = 800, 600
CX, CY = SCREEN_W // 2, SCREEN_H // 2
MAX_DEPTH = 12                      # draw_3d_view's loop bound: cells 1..12 ahead are drawn
FRONT_W, FRONT_H = 600, 400         # rects[d] = (FRONT_W / d, FRONT_H / d)
FOG_FLOOR, FOG_RATE = 0.4, 0.08     # fog(d) = max(FOG_FLOOR, 1 - FOG_RATE * d)
EDGE_FOG_FLOOR = 0.8                # the stair border uses max(fog, 0.8)

RGB_WALL_FRONT = (80, 40, 30)       # COLORS['wall_shadow'] — a wall cell's face, dead ahead
RGB_WALL_SIDE = (160, 85, 60)       # COLORS['wall'] — a wall cell's face, to the side
RGB_STAIR_FILL = (0, 100, 120)
RGB_STAIR_EDGE = (0, 255, 255)
RGB_EXIT = (255, 200, 0)
RGB_EXIT_EDGE = (255, 255, 255)
RGB_FLOOR = (50, 60, 65)
RGB_CEILING = (15, 15, 35)

# the centre patch is inside rects[MAX_DEPTH] = 50x33, so the nearest front face always fills it
PATCH_HALF = 10
PATCH_ROWS = slice(CY - PATCH_HALF, CY + PATCH_HALF + 1)
PATCH_COLS = slice(CX - PATCH_HALF, CX + PATCH_HALF + 1)
# the d=1 side quads reach x=100 (left) and x=700 (right); stay clear of their slanted borders
SIDE_ROWS = slice(170, 431)
SIDE_LEFT = slice(0, 90)
SIDE_RIGHT = slice(SCREEN_W - 90, SCREEN_W)


def _fog(d: int) -> float:
    return max(FOG_FLOOR, 1.0 - FOG_RATE * d)


def _darken(rgb, factor: float):
    """The renderer's own `darken`: per-channel int() truncation, floored at 0."""
    return tuple(max(0, int(c * factor)) for c in rgb)


def _packed(rgb) -> int:
    return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]


# depth look-up tables: the exact packed colour of a face -> the cell distance that paints it.
# `setdefault` keeps the NEAREST distance for a shade, which is the only one that can be visible
# (a nearer face covers every farther one), and collapses the d >= 8 fog plateau into one entry.
FRONT_WALL_DEPTH, STAIR_FILL_DEPTH, STAIR_EDGE_DEPTH, SIDE_WALL_DEPTH = {}, {}, {}, {}
for _d in range(1, MAX_DEPTH + 1):
    FRONT_WALL_DEPTH.setdefault(_packed(_darken(RGB_WALL_FRONT, _fog(_d))), _d)
    STAIR_FILL_DEPTH.setdefault(_packed(_darken(RGB_STAIR_FILL, _fog(_d))), _d)
    STAIR_EDGE_DEPTH.setdefault(_packed(_darken(RGB_STAIR_EDGE, max(_fog(_d), EDGE_FOG_FLOOR))), _d)
    SIDE_WALL_DEPTH.setdefault(_packed(_darken(RGB_WALL_SIDE, _fog(_d))), _d)
FOG_PLATEAU = -(-int(round((1.0 - FOG_FLOOR) * 100)) // int(round(FOG_RATE * 100)))
# 8: the first depth at which fog(d) has hit its floor, so the shade stops naming the distance
SIDE_WALL_NEAR = _packed(_darken(RGB_WALL_SIDE, _fog(1)))   # the immediate lateral neighbour only
EXIT_PACKED = _packed(RGB_EXIT)
EXIT_EDGE_PACKED = _packed(RGB_EXIT_EDGE)
BACKGROUND = (_packed(RGB_FLOOR), _packed(RGB_CEILING))

# cell tags. `_RANK` orders the upgrades a later observation may apply to an earlier one: a lateral
# band reports an EXIT cell as merely passable (the renderer draws the exit only straight ahead),
# so "open" must be allowed to become "exit"; nothing may ever be downgraded to "open".
TAG_OPEN, TAG_WALL, TAG_STAIR, TAG_EXIT = "open", "wall", "stair", "exit"
PASSABLE = frozenset((TAG_OPEN, TAG_STAIR, TAG_EXIT))
_RANK = {TAG_OPEN: 0, TAG_WALL: 1, TAG_STAIR: 2, TAG_EXIT: 3}
_SIDE_TAG = {"wall": TAG_WALL, "stair": TAG_STAIR, "open": TAG_OPEN}


# ── the decoder ──────────────────────────────────────────────────────────────────────────────
def _pack_block(block):
    a = block[:, :, :3].astype(_np.int32)
    return (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]


def _tally(codes):
    vals, counts = _np.unique(codes, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


def _run_depth(image, table_or_set) -> int:
    """The depth of the front face at the centre, from the half-width of its painted run.

    Used where fog no longer resolves the distance (a wall at d >= 8) and for the exit, whose face
    is painted with no fog at all. rects[d] has width FRONT_W/d, and those widths (600, 300, 200,
    150, 120, 100, 86, 75, 67, 60, 55, 50) stay separated by more than the rasteriser's rounding."""
    row = _pack_block(image[CY:CY + 1, :, :])[0]
    hit = _np.isin(row, list(table_or_set))
    if not bool(hit[CX]):
        return 1
    left = CX
    while left > 0 and bool(hit[left - 1]):
        left -= 1
    right = CX
    while right < SCREEN_W - 1 and bool(hit[right + 1]):
        right += 1
    width = right - left + 1
    return min(range(1, MAX_DEPTH + 1), key=lambda d: abs(FRONT_W / d - width))


def read_ahead(image):
    """(kind, distance) of the first non-empty cell on the forward ray.

    kind is "wall" / "stair" / "exit" / "open"; "open" means nothing is drawn within MAX_DEPTH, and
    carries distance MAX_DEPTH + 1. The patch is inside the smallest front face the renderer draws,
    so whichever family owns it IS the nearest non-empty cell ahead."""
    tally = _tally(_pack_block(image[PATCH_ROWS, PATCH_COLS, :]))
    wall = {FRONT_WALL_DEPTH[c]: n for c, n in tally.items() if c in FRONT_WALL_DEPTH}
    fill = {STAIR_FILL_DEPTH[c]: n for c, n in tally.items() if c in STAIR_FILL_DEPTH}
    edge = {STAIR_EDGE_DEPTH[c]: n for c, n in tally.items() if c in STAIR_EDGE_DEPTH}
    gold = tally.get(EXIT_PACKED, 0) + tally.get(EXIT_EDGE_PACKED, 0)
    stair_n = sum(fill.values()) + sum(edge.values())
    wall_n = sum(wall.values())
    kind, votes = max((TAG_EXIT, gold), (TAG_STAIR, stair_n), (TAG_WALL, wall_n),
                      key=lambda pair: pair[1])
    if votes == 0:
        return TAG_OPEN, MAX_DEPTH + 1
    if kind == TAG_WALL:
        depth = min(wall)
        if depth >= FOG_PLATEAU:                      # on the fog plateau: recover d from the width
            depth = _run_depth(image, FRONT_WALL_DEPTH)
        return TAG_WALL, depth
    if kind == TAG_STAIR:
        # the stair frame draws both diagonals of its own face, so the exact centre is a BORDER
        # pixel; the fill shades in the patch are what carry the distance.
        if fill:
            return TAG_STAIR, min(fill)
        if edge:
            return TAG_STAIR, min(edge)
        return TAG_STAIR, _run_depth(image, STAIR_FILL_DEPTH)
    return TAG_EXIT, _run_depth(image, (EXIT_PACKED, EXIT_EDGE_PACKED))


def read_side(image, columns) -> str:
    """"wall" / "stair" / "open" for the cell immediately left or right of the agent.

    Only the d=1 side quad reaches these columns, so there is exactly one cell to classify and no
    occlusion to reason about. An EXIT beside the agent is drawn as nothing (the renderer paints the
    exit only as a front face) and correctly reads "open" — it is passable either way."""
    tally = _tally(_pack_block(image[SIDE_ROWS, columns, :]))
    wall = tally.get(SIDE_WALL_NEAR, 0)
    stair = sum(n for c, n in tally.items() if c in STAIR_FILL_DEPTH or c in STAIR_EDGE_DEPTH)
    free = sum(tally.get(c, 0) for c in BACKGROUND)
    kind, votes = max(("wall", wall), ("stair", stair), ("open", free),
                      key=lambda pair: pair[1])
    return kind if votes else "open"


def read_scene(image) -> dict:
    """Everything one frame carries, plus its exact identity digest. Never raises."""
    blank = {"ok": False, "digest": "", "kind": TAG_OPEN, "dist": MAX_DEPTH + 1,
             "left": "open", "right": "open", "walkable": 0}
    if _np is None or image is None:
        return blank
    try:
        arr = _np.asarray(image)
        if arr.ndim != 3 or arr.shape[0] < SCREEN_H or arr.shape[1] < SCREEN_W:
            return blank
        digest = hashlib.md5(_np.ascontiguousarray(arr[:, :, :3]).tobytes()).hexdigest()
        kind, dist = read_ahead(arr)
        left = read_side(arr, SIDE_LEFT)
        right = read_side(arr, SIDE_RIGHT)
    except Exception:
        return blank
    # the engine walks into a cell that is EMPTY or STAIR and wins on the EXIT, so the stair and
    # the exit are themselves reachable; only a wall costs the last cell of the ray.
    walkable = dist if kind in (TAG_STAIR, TAG_EXIT) else dist - 1
    return {"ok": True, "digest": digest, "kind": kind, "dist": dist,
            "left": left, "right": right, "walkable": max(0, walkable)}


#: the working-memory block rendered into the agent's system turn each step. memory.md overrides
#: this when its placeholders match `fields()`, which is what keeps the wording a prose lever.
DEFAULT_TEMPLATE = """## MEASURED STATE (read off the screen you are looking at — not a guess)
- decision {step} of {max_steps}.
- straight ahead: {ahead_line}
- immediately to your left: {left_line}   |   immediately to your right: {right_line}
- under your feet: {foot_line}
- last decision: {last_line}
- this floor so far: {map_line}
- plan: {plan_line}

The cell you are standing on is never drawn, and there is no HUD, no text and no coordinates on
screen. Distances above are counted in grid cells and are exact. A translation into a wall changes
nothing at all, so never spend a decision on one.

Reply with ONLY the JSON object in the format specified above."""


class ActionPlanner:
    """Episode-local perception + map + route planning. One instance per episode (orchestrator.py)."""

    # ── swept levers (class attributes so a rig can set them AFTER build_agent, which purges
    #    sys.modules["tools*"] and would throw away an instance override set before the call) ──
    OBEY_MODE = "route"          # "route": a proposal that disagrees with the mapped route is
                                 # displaced (the model is still obeyed whenever it agrees, whenever
                                 # there is no route, and it still votes on frontier ties);
                                 # "coverage": it also keeps the wheel while reaching new ground,
                                 # and the harness insists only on a PROVEN route (a mapped exit or
                                 # an exhausted floor); "off": only measured-inert proposals move.
                                 # Priced on 6 adversarial tapes x 25 levels: 132 / 76 / 34 of 150.
    FLOOR_GATE = "exhaust"       # "exhaust" (no frontier left) | "stale" (no new cell for a while)
    STALE_LIMIT = 8              # decisions without a newly mapped cell, for FLOOR_GATE == "stale"
    STRIDE_MAX = 3               # the engine's `large`; a stride costs one decision at any length
    OBSERVE_EVERY_CELL = False   # clip a stride so no cell is ever crossed without being stood on
    FRONTIER_RULE = "nearest"    # "nearest" (BFS-closest) | "forward" (prefer one straight ahead)
    MAX_CLIMBS = 8               # a hard bound on floor transitions per episode
    TWIN_MATCHES = 2             # matching per-pose digests that retire an epoch as a refused climb
    TWIN_WINDOW = 6              # decisions the retirement test stays open looking for comparisons
    STALL_LIMIT = 4              # decisions without motion before the blind fallback sweep engages

    def __init__(self, max_steps: int = 100, template: str = ""):
        self.max_steps = int(max_steps or 100)
        self.template = _clean_template(template) or DEFAULT_TEMPLATE
        # pose, dead-reckoned: epoch is a map namespace, one per floor the agent has entered
        self.epoch, self.x, self.y, self.heading = 0, 0, 0, 0
        self.cells = {}              # (epoch, x, y) -> tag
        self.clear = set()           # (epoch, x, y) proven PLAIN floor (a ray went through it)
        self.visits = {}             # (epoch, x, y) -> times stood on
        self.digests = {}            # (epoch, x, y, heading) -> frame digest
        self.stair_tried = {}        # (epoch, x, y) -> set of climb names already spent
        self.links = {}              # (epoch, x, y, climb) -> the epoch it leads to
        self.epochs_seen = {0}
        self.climbs = 0
        self.twin = None             # the pending "did that climb happen?" comparison
        # rolling state
        self.scene = read_scene(None)
        self.step = 0
        self.last = None             # the decision committed on the previous step
        self.prev_digest = None
        self.mapped_mark, self.since_new = 0, 0
        self.since_move = 0
        self.moves, self.overrides, self.desync = 0, 0, 0
        self.lost = False            # dead reckoning contradicted: fall back to the blind ledger
        self.refused = {}            # digest -> translations that provably did nothing (fallback)
        self.plan_note = "nothing measured yet"
        self.base_prompt = None      # stash of the agent's first-call system turn

    # ── pose / map primitives ───────────────────────────────────────────────────────────────
    def cell(self):
        return (self.epoch, self.x, self.y)

    def _neighbour(self, bearing: int, cell=None):
        x, y = (self.x, self.y) if cell is None else cell
        dx, dy = _act.STEP[bearing % 4]
        return (x + dx, y + dy)

    def tag(self, cell, epoch=None):
        return self.cells.get(((self.epoch if epoch is None else epoch),) + tuple(cell))

    def _mark(self, cell, tag) -> None:
        key = (self.epoch,) + tuple(cell)
        old = self.cells.get(key)
        if old is None:
            self.cells[key] = tag
            return
        if old == tag:
            return
        if _RANK[tag] > _RANK[old]:
            if _RANK[old] >= _RANK[TAG_WALL]:
                self.desync += 1          # two different solid readings for one cell
            self.cells[key] = tag
        elif _RANK[tag] >= _RANK[TAG_WALL]:
            self.desync += 1

    def _learn(self, scene) -> None:
        """Write one observation into the map: the whole forward ray plus both lateral cells.

        PROVENANCE IS THE POINT. The ray stops at the FIRST cell that is not plain floor — a wall, a
        stair and the exit all terminate it — so every cell the ray passes THROUGH is proven to be
        ordinary empty floor, i.e. proven not to be the goal. A lateral band proves no such thing:
        the renderer paints the exit only as a front face, so an exit cell beside the agent reads
        exactly like open floor. Those two kinds of "open" are therefore kept apart, in `cells` and
        in `clear`, and the difference is what the floor-exhaustion proof rests on."""
        here = (self.x, self.y)
        if self.tag(here) is None:
            self._mark(here, TAG_OPEN)                # standing on it proves it passable
        self.clear.add((self.epoch,) + here)          # and proves it is not the exit: we are playing
        dx, dy = _act.STEP[self.heading]
        for i in range(1, scene["dist"]):
            cell = (here[0] + dx * i, here[1] + dy * i)
            self._mark(cell, TAG_OPEN)
            self.clear.add((self.epoch,) + cell)      # the ray went through it: plain floor
        if scene["kind"] != TAG_OPEN:
            d = scene["dist"]
            self._mark((here[0] + dx * d, here[1] + dy * d), scene["kind"])
        for offset, side in ((_act.LEFT, scene["left"]), (_act.RIGHT, scene["right"])):
            self._mark(self._neighbour(self.heading + offset), _SIDE_TAG[side])
        self.digests[(self.epoch, self.x, self.y, self.heading)] = scene["digest"]

    def _settled(self, cell, epoch=None) -> bool:
        """Is this cell's identity decided — wall, stair, exit, or proven-plain floor?

        An unmapped cell is not settled. Neither is a cell known only from a lateral band, because
        that reading cannot distinguish the exit from empty floor. Every unsettled cell is cheap to
        settle: one look down the ray that passes through it terminates on it if it is the goal."""
        ep = self.epoch if epoch is None else epoch
        key = (ep,) + tuple(cell)
        tag = self.cells.get(key)
        if tag is None:
            return False
        return tag != TAG_OPEN or key in self.clear

    def _open_bearings(self, cell, epoch=None):
        """The bearings out of `cell` whose neighbour still has to be settled."""
        return [b for b in range(4) if not self._settled(self._neighbour(b, cell), epoch)]

    def _is_frontier(self, cell, epoch=None) -> bool:
        """A reachable cell from which something unsettled can still be looked at.

        Exploration is complete for a floor exactly when no such cell is reachable, and at that
        point every reachable cell is a wall, a stair, the exit, or proven plain floor — so "no exit
        found and no frontier left" PROVES the exit is not on this floor. That proof is what lets the
        floor change be free: the lineage measured that climbing on the first opportunity instead
        costs the levels whose exit is on the start floor."""
        ep = self.epoch if epoch is None else epoch
        if self.cells.get((ep,) + tuple(cell)) not in PASSABLE:
            return False
        return bool(self._open_bearings(cell, ep))

    def _reach(self, epoch=None):
        """BFS over the known-passable cells of one epoch: (parents, depth, cells in depth order).

        Known-passable means the map says floor / stair / exit, so a route never walks into a cell
        the pixels have not already shown to be enterable, and `order` is non-decreasing in depth,
        which makes "the nearest target" the first match in it."""
        ep = self.epoch if epoch is None else epoch
        start = (self.x, self.y)
        parents = {start: None}
        depth = {start: 0}
        order = [start]
        queue = deque(order)
        while queue:
            cur = queue.popleft()
            for b in range(4):
                nxt = self._neighbour(b, cur)
                if nxt in parents or self.cells.get((ep,) + nxt) not in PASSABLE:
                    continue
                parents[nxt] = cur
                depth[nxt] = depth[cur] + 1
                order.append(nxt)
                queue.append(nxt)
        return parents, depth, order

    def _path(self, parents, target):
        path = [target]
        while parents.get(path[-1]) is not None:
            path.append(parents[path[-1]])
        path.reverse()
        return path

    # ── floor logic ─────────────────────────────────────────────────────────────────────────
    def _epoch_has_frontier(self, epoch) -> bool:
        return any(self._is_frontier((x, y), ep) for (ep, x, y) in self.cells if ep == epoch)

    def _climb_choice(self, cell):
        """The climb worth spending at a mapped stair cell, and whether it is worth spending.

        Preference order: a transition never tried from here, into an epoch that does not exist yet
        (new ground) or still has a frontier. `climb_up` is tried before `climb_down` so the first
        transition of an episode goes away from the start floor rather than into the ground."""
        spent = self.stair_tried.get((self.epoch,) + tuple(cell), ())
        best = None
        for name in ("climb_up", "climb_down"):
            if name in spent:
                continue
            dest = self.links.get((self.epoch,) + tuple(cell) + (name,))
            if dest is None:
                rank = 0                                   # unexplored ground
            elif self._epoch_has_frontier(dest):
                rank = 1
            else:
                continue                                   # a finished floor: nothing to go back for
            if best is None or rank < best[0]:
                best = (rank, name)
        return best[1] if best else None

    def _floor_impatient(self) -> bool:
        """May a floor change pre-empt an unfinished floor?

        Under FLOOR_GATE == "exhaust" (the default) it never may: a stair is only spent once every
        reachable cell of this floor has been stood on, which is the point at which "the exit is not
        on this floor" is PROVEN rather than guessed. The lineage measured the alternative — climbing
        at the first opportunity — as the thing that costs the levels whose exit is on the start
        floor. "stale" is the impatient variant, kept as a priced lever rather than a belief."""
        if self.FLOOR_GATE == "stale":
            return self.since_new >= self.STALE_LIMIT
        return False

    # ── routing ─────────────────────────────────────────────────────────────────────────────
    def _route(self, prefer: str = ""):
        """(action_name, cells, note, mode) — the harness's next step, or None when it has no plan.

        `mode` is what the step is FOR, and the governor insists only on the two modes that rest on
        a proof: "exit" (a mapped exit cell, reachable through cells known to be floor) and "stair"
        (this floor has been exhausted, so the exit provably is not on it). In "explore" mode the
        harness has no proof that its guess beats the model's, so the model keeps the wheel for as
        long as its own proposal is reaching somewhere new."""
        if self.lost or not self.scene["ok"]:
            return None
        parents, depth, order = self._reach()
        exits = [c for c in order if self.tag(c) == TAG_EXIT]
        if exits:
            return self._toward(parents, exits[0], "exit", stop_on=True)
        frontier = [c for c in order if self._is_frontier(c)]
        stairs = [c for c in order
                  if self.tag(c) == TAG_STAIR and self._climb_choice(c) is not None]
        if stairs and self.climbs < self.MAX_CLIMBS and (not frontier or self._floor_impatient()):
            target = stairs[0]
            if target == (self.x, self.y):
                name = self._climb_choice(target)
                if name:
                    return (name, 1, "this floor is mapped out; changing floor here", "stair")
            else:
                return self._toward(parents, target, "stair", stop_on=True)
        if frontier:
            ahead = self._frontier_ahead(frontier) if self.FRONTIER_RULE == "forward" else None
            target = ahead or self._pick(parents, depth, frontier, prefer)
            return self._toward(parents, target, "explore", stop_on=False)
        return None

    def _frontier_ahead(self, frontier):
        """The nearest frontier straight ahead down the open ray, or None.

        Preferring it over the BFS-nearest one keeps a corridor walk going in one line instead of
        turning off at the first junction, which both saves the turn and lets the stride grow: the
        route step then spans the whole straight run rather than a single cell."""
        want = set(frontier)
        dx, dy = _act.STEP[self.heading]
        for i in range(1, max(1, self.scene["walkable"]) + 1):
            cell = (self.x + dx * i, self.y + dy * i)
            if self.cells.get((self.epoch,) + cell) not in PASSABLE:
                break
            if cell in want:
                return cell
        return None

    def _pick(self, parents, depth, frontier, prefer: str):
        """Which of several equally-near frontiers to head for — THE MODEL'S FREE VOTE.

        Every candidate here sits at the same BFS distance, so choosing between them cannot cost a
        decision, and the frozen model's own proposal gets to decide: if its action is the first step
        toward one of them, that one wins. Otherwise the tie breaks toward the bearing already being
        faced, since a turn costs a whole decision and the lineage measured turn churn, not path
        length, as what runs these episodes out of clock."""
        best = min(depth[c] for c in frontier)
        tied = [c for c in frontier if depth[c] == best]
        if len(tied) == 1:
            return tied[0]

        def first_bearing(cell):
            path = self._path(parents, cell)
            return _act.bearing_to(path[0], path[1]) if len(path) > 1 else self.heading

        # the vote is taken INSIDE the turn-minimal set, never across it. A vote that could add a
        # turn is not free, and a degenerate proposer would then steer the route with its own
        # repetition: an always-`turn_right` tape swung the choice onto a different tied frontier
        # every step and span in place (11 of 25 levels against 22 when the vote is bounded).
        fewest = min(_act.turns_to(self.heading, first_bearing(c)) for c in tied)
        tied = [c for c in tied if _act.turns_to(self.heading, first_bearing(c)) == fewest]
        if prefer and len(tied) > 1:
            for cell in tied:
                bearing = first_bearing(cell)
                step = (_act.turn_toward(self.heading, bearing) or
                        ("move_forward" if bearing == self.heading else ""))
                if step == prefer:
                    return cell
        return tied[0]

    def _toward(self, parents, target, note, stop_on: bool):
        """One step of the route to `target`: a turn onto the right bearing, or a stride along it.

        A stride is clipped to 1 while it would enter a cell never stood on, because the lateral
        readout only covers the cell the agent is IN — striding past an unvisited cell leaves its
        two side neighbours unmapped and turns it into a frontier the route has to come back for.
        Transit across already-visited cells takes the full stride, which costs the same decision."""
        if target == (self.x, self.y):
            pending = self._open_bearings((self.x, self.y))
            if pending:
                nearest = min(pending, key=lambda b: _act.turns_to(self.heading, b))
                turn = _act.turn_toward(self.heading, nearest)
                if turn:
                    return (turn, 1, "looking down the one bearing here that is still unsettled",
                            note)
            return None
        path = self._path(parents, target)
        if len(path) < 2:
            return None
        bearing = _act.bearing_to(path[0], path[1])
        if bearing is None:
            return None
        turn = _act.turn_toward(self.heading, bearing)
        if turn:
            return (turn, 1, "turning onto the route (%s)" % note, note)
        run = 1
        while run + 1 < len(path) and _act.bearing_to(path[run], path[run + 1]) == bearing:
            run += 1
        cells = min(run, self.STRIDE_MAX, max(1, self.scene["walkable"]))
        if stop_on:
            cells = min(cells, len(path) - 1)             # never stride past the cell we want
        if self.OBSERVE_EVERY_CELL:
            for i in range(1, cells + 1):
                if self.visits.get((self.epoch,) + path[i], 0) == 0:
                    cells = i                              # stop on every newly entered cell
                    break
        return ("move_forward", max(1, cells), "stepping along the route (%s)" % note, note)

    # ── outcome scoring + dead reckoning ────────────────────────────────────────────────────
    def observe(self, image) -> None:
        """Fold the frame now on screen into the ledgers, score the previous decision, and advance
        the dead-reckoned pose. Called once per step before the frozen model is asked."""
        scene = read_scene(image)
        changed = self.prev_digest is None or scene["digest"] != self.prev_digest
        if self.last is not None:
            self._settle(self.last, changed, scene)
        self.scene = scene
        self.prev_digest = scene["digest"]
        self.step += 1
        if scene["ok"] and not self.lost:
            self._learn(scene)
            self._resolve_twin(scene)
            key = self.cell()
            self.visits[key] = self.visits.get(key, 0) + 1
        if len(self.cells) > self.mapped_mark:
            self.mapped_mark, self.since_new = len(self.cells), 0
        else:
            self.since_new += 1
        if self.desync >= 2:
            self.lost = True

    def _settle(self, prev, changed: bool, scene) -> None:
        """Apply the previous decision's measured outcome to the pose, the map and the ledgers."""
        name, cells = prev["name"], prev["cells"]
        if name in _act.TURN_ACTIONS:
            self.heading = (self.heading + (1 if name == "turn_right" else -1)) % 4
            self.since_move += 1
            return
        if name in _act.CLIMB_ACTIONS:
            self.since_move += 1
            return                     # the epoch moved at commit time; see _commit
        if name not in _act.MOVEMENT_ACTIONS:
            self.since_move += 1
            return
        sign = 1 if name == "move_forward" else -1
        bearing = self.heading if sign > 0 else (self.heading + _act.BEHIND) % 4
        if not changed:
            self.since_move += 1
            self.refused.setdefault(prev["digest"], set()).add(name)
            self._mark(self._neighbour(bearing), TAG_WALL)
            if prev["expect"] > 0:
                self.desync += 1       # the measurement said this would move and it did not
            return
        self.moves += 1
        self.since_move = 0
        travelled = prev["expect"] if prev["expect"] > 0 else cells
        if prev["expect"] <= 0:
            self.desync += 1           # it moved where the measurement said it could not
        dx, dy = _act.STEP[bearing]
        self.x += dx * travelled
        self.y += dy * travelled

    def _resolve_twin(self, scene) -> None:
        """Decide for free whether the last climb actually happened.

        19.2% of this benchmark's possible floor changes render byte-identically, so a climb is
        committed optimistically into a fresh epoch and compared against the previous epoch's
        per-pose digests as they arrive: one mismatch confirms the transition, TWIN_MATCHES matches
        retire the new epoch (the climb was refused) and restore the finished map underneath it."""
        twin = self.twin
        if not twin or self.epoch != twin["new"]:
            return
        was = self.digests.get((twin["old"], self.x, self.y, self.heading))
        twin["age"] += 1
        if was is not None:
            if was != scene["digest"]:
                self.twin = None                          # provably a different floor
                return
            twin["same"] += 1
            if twin["same"] >= self.TWIN_MATCHES:
                for key in [k for k in self.cells if k[0] == twin["new"]]:
                    del self.cells[key]
                for key in [k for k in self.visits if k[0] == twin["new"]]:
                    del self.visits[key]
                self.clear -= {k for k in self.clear if k[0] == twin["new"]}
                self.epochs_seen.discard(twin["new"])
                self.epoch = twin["old"]
                self.stair_tried.setdefault((twin["old"],) + twin["cell"], set()).add(twin["name"])
                self.links.pop((twin["old"],) + twin["cell"] + (twin["name"],), None)
                self.climbs = max(0, self.climbs - 1)
                self.twin = None
                self._learn(scene)
                return
        if twin["age"] > self.TWIN_WINDOW:
            self.twin = None                              # nothing comparable: assume it happened

    # ── the governor ────────────────────────────────────────────────────────────────────────
    def _inert(self, name: str):
        """Why the frozen model's proposal provably cannot change anything, or ""."""
        scene = self.scene
        if name == "move_forward" and scene["ok"] and scene["walkable"] == 0:
            return "a wall is in the very next cell, so `move_forward` cannot move at all"
        if name == "move_backward":
            behind = self.tag(self._neighbour(self.heading + _act.BEHIND))
            if behind == TAG_WALL and not self.lost:
                return "the cell behind is a wall, so `move_backward` cannot move at all"
        if name in _act.CLIMB_ACTIONS and not self.lost:
            if self.tag((self.x, self.y)) != TAG_STAIR:
                return "there is no stair under your feet, so `%s` does nothing" % name
        if name in _act.MOVEMENT_ACTIONS and name in self.refused.get(scene["digest"], ()):
            return "`%s` already left this exact screen unchanged" % name
        return ""

    def _adds_coverage(self, name: str, cells: int) -> bool:
        """Does the proposal reach something not yet stood on or not yet mapped?"""
        if self.lost or not self.scene["ok"]:
            return True
        if name in _act.MOVEMENT_ACTIONS:
            bearing = self.heading if name == "move_forward" else (self.heading + _act.BEHIND) % 4
            reach = min(max(1, cells), max(1, self.scene["walkable"]))
            for i in range(1, reach + 1):
                dx, dy = _act.STEP[bearing]
                if self.visits.get((self.epoch, self.x + dx * i, self.y + dy * i), 0) == 0:
                    return True
            return False
        if name in _act.TURN_ACTIONS:
            bearing = (self.heading + (1 if name == "turn_right" else -1)) % 4
            cell = self._neighbour(bearing)
            if ((self.epoch,) + cell) not in self.cells:
                return True
            return self.cells.get((self.epoch,) + cell) in PASSABLE and \
                self.visits.get((self.epoch,) + cell, 0) == 0
        return True                              # a climb is priced by _inert, not by coverage

    def _escape(self):
        """The blind fallback for a lost pose: probe ahead, else rotate consistently."""
        refused = self.refused.get(self.scene["digest"], ())
        if "move_forward" not in refused and self.scene.get("walkable", 1) != 0:
            return ("move_forward", 1, "probing the way ahead", "escape")
        return ("turn_right", 1, "rotating to a bearing this screen has not refused", "escape")

    def govern(self, name: str, cells: int):
        """(action, cells, reason). An empty reason means the model's own proposal is executed.

        Three gates, cheapest first: a proposal the measurement proves inert is always displaced; a
        proposal that disagrees with a route the harness can PROVE (onto a mapped exit, or off an
        exhausted floor) is displaced under every mode but "off"; a proposal that merely re-treads
        covered ground while the harness is exploring is displaced under "coverage" and "route". The
        route step is computed FIRST and the proposal is kept whenever it already agrees, so
        agreeing with the harness never costs the model the wheel."""
        route = self._route(prefer=name)
        mode = route[3] if route else ""
        self.plan_note = route[2] if route else "no route of its own: following you"
        why = self._inert(name)
        if not why and route and self.OBEY_MODE != "off":
            proven = mode in ("exit", "stair")
            if self.OBEY_MODE == "route" or proven:
                why = "the mapped route leads to %s" % ("the exit" if mode == "exit" else
                                                        "a floor change" if mode == "stair" else
                                                        "unexplored ground")
            elif self.OBEY_MODE == "coverage" and not self._adds_coverage(name, cells):
                why = "`%s` only re-treads ground already covered" % name
        if not why:
            return (name, cells, "")
        pick = route or (self._escape() if self.since_move >= self.STALL_LIMIT or self.lost else None)
        if pick is None:
            return (name, cells, "")
        if pick[0] == name and (pick[0] not in _act.MOVEMENT_ACTIONS or pick[1] == cells):
            return (name, cells, "")                      # the model already wants the route step
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
            expect = min(max(1, cells), self.scene["walkable"]) if self.scene["ok"] else cells
        elif name == "move_backward":
            expect = max(1, cells)                        # the backward ray is not rendered
        self.last = {"name": name, "cells": cells, "expect": expect, "displaced": displaced,
                     "digest": self.scene["digest"], "step": self.step}
        if not name:
            return
        if name in _act.CLIMB_ACTIONS and not self.lost:
            self._commit_climb(name)

    def _commit_climb(self, name) -> None:
        """Move into the epoch this climb leads to, creating it when it is new ground."""
        cell = (self.x, self.y)
        key = (self.epoch,) + cell
        self.stair_tried.setdefault(key, set()).add(name)
        self.climbs += 1
        dest = self.links.get(key + (name,))
        fresh = dest is None
        if fresh:
            dest = max(self.epochs_seen) + 1
            self.links[key + (name,)] = dest
            self.links[(dest,) + cell + (_act.inverse_climb(name),)] = self.epoch
        previous = self.epoch
        self.epochs_seen.add(dest)
        self.epoch = dest
        self.cells.setdefault((dest,) + cell, TAG_STAIR)  # a stair link is a stair on both sides
        if fresh:
            self.twin = {"new": dest, "old": previous, "cell": cell, "name": name,
                         "same": 0, "age": 0}

    # ── the per-turn memory block ───────────────────────────────────────────────────────────
    def _ahead_line(self) -> str:
        scene = self.scene
        if not scene["ok"]:
            return "not readable this step."
        if scene["kind"] == TAG_OPEN:
            return "open for at least %d cells." % MAX_DEPTH
        what = {TAG_WALL: "a wall", TAG_STAIR: "a STAIR tile", TAG_EXIT: "the EXIT"}[scene["kind"]]
        where = ("in the very next cell" if scene["dist"] == 1
                 else "%d cells away" % scene["dist"])
        return "%s %s; a forward move travels %d cell(s)." % (what, where, scene["walkable"])

    def _side_line(self, side: str) -> str:
        if not self.scene["ok"]:
            return "not readable this step."
        return {"wall": "a wall.", "stair": "a STAIR tile — one turn and one step away.",
                "open": "open floor."}[self.scene[side]]

    def _foot_line(self) -> str:
        if self.lost:
            return "unknown (the position estimate was dropped)."
        if self.tag((self.x, self.y)) == TAG_STAIR:
            return "a STAIR tile — climbing from here works."
        return "ordinary floor, not a stair."

    def _last_line(self) -> str:
        prev = self.last
        if not prev or not prev.get("name"):
            return "none yet — this is your first decision."
        if prev.get("displaced"):
            head = "you proposed `%s`; `%s` was executed instead" % (prev["displaced"], prev["name"])
        else:
            head = "`%s` was executed as you asked" % prev["name"]
        if prev["name"] in _act.MOVEMENT_ACTIONS:
            head += (" and you moved" if self.since_move == 0 else
                     " and the screen did not change at all, so something solid blocked it")
        return head + "."

    def _map_line(self) -> str:
        mine = [c for c in self.cells if c[0] == self.epoch]
        settled = sum(1 for (e, x, y) in mine if self._settled((x, y)))
        frontier = sum(1 for (e, x, y) in mine if self._is_frontier((x, y)))
        stairs = sum(1 for (e, x, y) in mine if self.cells[(e, x, y)] == TAG_STAIR)
        return ("%d cells identified (%d of them settled), %d still have something unseen next to "
                "them, %d stair(s) found." % (len(mine), settled, frontier, stairs))

    def fields(self) -> dict:
        return {"step": self.step, "max_steps": self.max_steps,
                "ahead_line": self._ahead_line(),
                "left_line": self._side_line("left"), "right_line": self._side_line("right"),
                "foot_line": self._foot_line(), "last_line": self._last_line(),
                "map_line": self._map_line(), "plan_line": self.plan_note,
                "moves": self.moves, "overrides": self.overrides}

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
    """The renderable part of memory.md: everything after the last HTML comment, since the file's
    header documents the template for the next proposer and must never reach the agent. Returns ""
    when there is no body or its placeholders do not match `fields()`."""
    if not text:
        return ""
    body = text.split("-->")[-1].strip() if "-->" in text else text.strip()
    return body if "{ahead_line}" in body and "{step}" in body else ""
