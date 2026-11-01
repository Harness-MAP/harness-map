"""THE CHART — what the frames have proven about this maze, per floor, in the harness's own frame
of reference.

The ancestor of this harness held no state at all, and so the one bit that separates a move that
happened from a move that was refused — *did the frame change?* — never reached anything. 1829 of
its 2340 decisions re-pressed a wall that same episode had already proven at the same cell and the
same bearing. This module is where that bit, and everything `tools/sight.py` reads beside it, is
filed so a route can be planned over it.

THREE KINDS OF OPEN, KEPT STRICTLY APART. A cell the forward ray walked THROUGH is `PLAIN`: the
body can stand there and the harness has seen ALONG one of its axes. A cell a flank band showed
empty is `PASSABLE`: it is enterable, but nothing has looked along it at all, so it may still hold
the exit and it is still a QUESTION. A cell painted teal is `STAIR` and a cell painted gold is `EXIT`.
Collapsing `PASSABLE` into `PLAIN` is the one error that would destroy the frontier, because then
one look down a corridor would mark the whole neighbourhood explored and the planner would have
nothing left to ask.

THE POSE IS RECKONED, NEVER READ. The start cell is called `(0, 0)` on floor `0` facing bearing `0`
and every later pose follows from the harness's own emitted actions, settled against the measured
outcome:

  * a TURN always executes, so a bearing is folded unconditionally;
  * a TRANSLATION is folded by the number of cells the PRE-step ray measured as free
    (`min(requested, reach)`), and only when the frame actually changed. An unchanged frame means
    the press moved nothing — that is the refusal bit, and it is filed as a wall;
  * a CLIMB is folded OPTIMISTICALLY INTO A NEW EPOCH, and settled afterwards. An unchanged frame
    is never read as a failed climb: a climb that succeeds can leave the frame byte-identical, so
    that reading would be wrong about a fifth of the time, and wrong in the expensive direction.

THE EPOCH IS HOW A FLOOR CHANGE IS SETTLED WITHOUT A FLOOR READOUT. Nothing on screen says which
floor the body is on, and a climb pressed on the top floor is simply ignored by the engine — so a
chart that counted floors optimistically would open a fresh grid for a floor that does not exist and
then re-elect the same staircase from it, forever. Instead every climb opens a new EPOCH (a fresh
grid), and each epoch remembers the frame digest of every pose it has looked from. On a fresh epoch's FIRST
look — and only there — the digest at the very pose the climb was pressed from is compared with that
pose's digest in the epoch the press left. Byte-equal means the body never moved and the two epochs
are the same floor, so they are MERGED, which is exactly what a refused climb produces one decision
later with nothing lost. Allowing that match at any LATER pose is measured to be worse than not
settling at all (it traded two levels for four on the offline rig): two different poses on two
different floors can render alike, and merging on that conflates two real floors into one grid.

WHEN THE RECKONING AND THE PIXELS DISAGREE the floor's chart is WIPED and re-learned from the pose
the body is believed to be in. A chart that quietly contradicts the view is worse than no chart,
because the planner would route confidently over cells that are not there.

Nothing here reads engine state. Every fact is either a reading of `frame_data.image` or an
arithmetic consequence of an action this harness itself emitted.
"""
from __future__ import annotations

from tools import action as A
from tools.sight import EXIT, OPEN, STAIR, WALL

PLAIN, PASSABLE, UNKNOWN = "plain", "passable", "unknown"

#: the information content of a reading, and therefore which reading of a cell wins. A goal tile
#: outranks a wall, a wall outranks a cell walked through, and that outranks a cell merely glimpsed
#: from the side — so one ambiguous band can never unmake a corridor.
RANK = {UNKNOWN: 0, PASSABLE: 1, PLAIN: 2, WALL: 3, STAIR: 4, EXIT: 5}

#: cells a body can enter. `PASSABLE` is in; `UNKNOWN` deliberately is not — a route may only be
#: planned over ground the frames have settled.
ENTERABLE = (PLAIN, PASSABLE, STAIR, EXIT)
#: cells the ray has actually looked ALONG. A reversal, and the middle of any multi-cell stride,
#: may only be planned over these.
VERIFIED = (PLAIN, STAIR, EXIT)
#: a flank reading, translated into a chart kind. `OPEN` beside the corridor is PASSABLE and never
#: PLAIN, because an exit beside the corridor paints no side face: unseen is not the same as empty.
FLANK_KIND = {WALL: WALL, STAIR: STAIR, OPEN: PASSABLE}


class Chart:
    """One chart per episode, holding one grid per floor plus the reckoned pose and the ledger."""

    __slots__ = ("grids", "epoch", "minted", "cell", "heading", "blocked", "stood", "views",
                 "arrived", "came", "clashes", "wipes", "merges")

    def __init__(self):
        self.grids = {0: {(0, 0): PLAIN}}
        self.epoch = 0
        self.minted = 1               # the next epoch number a climb will open
        self.cell = (0, 0)
        self.heading = 0
        self.blocked = set()          # (epoch, cell, heading) a translation was refused in
        self.stood = {(0, (0, 0))}    # (epoch, cell) the body has occupied
        self.views = {0: {}}          # epoch -> {(cell, heading): frame digest}
        self.arrived = {}             # (epoch, cell) -> the climb that brought the body here
        self.came = {}                # epoch -> (epoch it was opened from, cell, heading)
        self.clashes = 0
        self.wipes = 0
        self.merges = 0

    # ── the grid ────────────────────────────────────────────────────────────────────────────
    def grid(self, epoch=None):
        """The grid for an epoch, created empty the first time that epoch is asked about."""
        return self.grids.setdefault(self.epoch if epoch is None else epoch, {})

    def kind(self, cell, epoch=None):
        """What the frames say about one cell: PLAIN, PASSABLE, STAIR, EXIT, WALL or UNKNOWN."""
        return self.grid(epoch).get(cell, UNKNOWN)

    def enterable(self, cell, epoch=None):
        """Can a route step onto this cell? Only settled, non-wall ground qualifies."""
        return self.kind(cell, epoch) in ENTERABLE

    def verified(self, cell, epoch=None):
        """Has the ray looked ALONG this cell? The gate a reversal and a long stride both clear."""
        return self.kind(cell, epoch) in VERIFIED

    def mark(self, cell, kind, epoch=None):
        """File one cell, refusing to downgrade what is already settled.

        The ranking is the information content of the reading: a goal tile outranks a wall, a wall
        outranks a cell walked through, and that outranks a cell merely glimpsed from the side. A
        later, weaker reading of the same cell is dropped rather than allowed to erase a stronger
        one — which is what keeps one ambiguous band from unmaking a corridor."""
        g = self.grid(epoch)
        if RANK[kind] >= RANK.get(g.get(cell, UNKNOWN), 0):
            g[cell] = kind

    # ── folding one look ────────────────────────────────────────────────────────────────────
    def fold_sight(self, sight):
        """File everything one frame proved: the corridor, its blocking face, and the flanks."""
        if sight is None or not sight.ok:
            return
        self.mark(self.cell, PLAIN)
        for step in range(1, max(0, sight.reach) + 1):
            self.mark(A.advance(self.cell, self.heading, step), PLAIN)
        if sight.face in (WALL, STAIR, EXIT):
            self.mark(A.advance(self.cell, self.heading, sight.depth), sight.face)
        if sight.blocked():
            self.blocked.add((self.epoch, self.cell, self.heading))
        self.fold_flanks(sight)

    def fold_flanks(self, sight):
        """File the two neighbours of each corridor cell the ladder reported."""
        left = (self.heading - 1) % 4
        right = (self.heading + 1) % 4
        for k, pair in enumerate(sight.flanks):
            spine = A.advance(self.cell, self.heading, k)
            self.mark(A.advance(spine, left, 1), FLANK_KIND[pair[0]])
            self.mark(A.advance(spine, right, 1), FLANK_KIND[pair[1]])

    def fold_outcome(self, name, cells, moved):
        """Advance the reckoned pose by one executed action, settled against the measured outcome.

        `cells` is what the PRE-step ray said the translation could travel; `moved` is whether the
        frame changed at all. A translation that did not move the body files a wall and leaves the
        pose alone — that single fact is what the ancestor harness had no place to put."""
        if name in A.TURN_ACTIONS:
            self.heading = A.heading_after(self.heading, name)
            return
        if name in A.CLIMB_ACTIONS:
            self.change_floor(name)
            return
        if name not in A.MOVEMENT_ACTIONS:
            return
        back = name == "move_backward"
        if not moved or cells <= 0:
            self.blocked.add((self.epoch, self.cell,
                              (self.heading + 2) % 4 if back else self.heading))
            self.mark(A.advance(self.cell, self.heading, 1, back), WALL)
            return
        self.cell = A.advance(self.cell, self.heading, cells, back)
        self.mark(self.cell, PLAIN)
        self.stood.add((self.epoch, self.cell))

    def change_floor(self, name):
        """Commit a floor transition into a FRESH EPOCH, optimistically.

        The stair tile is vertically PAIRED, so the cell the body lands on is a stair on the new
        floor too — which is what makes the way back plannable at all. `arrived` records which press
        brought the body here, so the planner can tell a fresh link from a RETURN down the one it
        just came up."""
        was = self.epoch
        self.epoch = self.minted
        self.minted += 1
        self.grids[self.epoch] = {self.cell: STAIR}
        self.views[self.epoch] = {}
        self.came[self.epoch] = (was, self.cell, self.heading)
        self.arrived[(self.epoch, self.cell)] = name
        self.stood.add((self.epoch, self.cell))

    def settle_epoch(self, sight):
        """Settle a fresh epoch on its FIRST look, and never afterwards.

        One comparison, at the one moment the question is live: the look straight after a climb,
        at the very pose the press was made from, against that pose's own digest in the epoch the
        press left. The renderer is a pure function of the pose, so if the body is still standing
        where it pressed, that frame is byte-identical and the climb changed nothing — the epochs
        are the same floor and they are merged.

        RESTRICTING IT TO THE FIRST LOOK IS WHAT MAKES IT SOUND. A free-floating digest match
        anywhere later is not evidence of anything: two different poses on two different floors can
        render alike, and merging on that conflates two real floors into one grid. Measured on the
        offline rig, the unrestricted version traded two levels for four."""
        if sight is None or not sight.ok:
            return
        key = (self.cell, self.heading)
        opened = self.came.get(self.epoch)
        if opened is not None and not self.views.get(self.epoch):
            was, cell, heading = opened
            if (cell, heading) == key and self.views.get(was, {}).get(key) == sight.digest:
                self.merge(was)
        self.views.setdefault(self.epoch, {})[key] = sight.digest

    def merge(self, into):
        """Fold this epoch wholesale into an earlier one: grid, ledger, footprints and digests."""
        gone = self.epoch
        for cell, kind in self.grids.pop(gone, {}).items():
            self.mark(cell, kind, into)
        self.views.setdefault(into, {}).update(self.views.pop(gone, {}))
        self.blocked = {(into if e == gone else e, c, h) for e, c, h in self.blocked}
        self.stood = {(into if e == gone else e, c) for e, c in self.stood}
        self.arrived = {((into if e == gone else e), c): v
                        for (e, c), v in self.arrived.items()}
        self.came.pop(gone, None)
        self.epoch = into
        self.merges += 1

    def refuted(self, cell, heading, epoch=None):
        """Has a translation out of this pose already been refused this episode?"""
        return ((self.epoch if epoch is None else epoch), cell, heading) in self.blocked

    # ── the consistency check ───────────────────────────────────────────────────────────────
    def clashes_with(self, sight):
        """Does this look contradict the chart at the cell straight ahead?

        One comparison, at the one cell where the reading is unambiguous: the ray says a wall is
        adjacent while the chart has that cell settled as walkable, or the ray walks through a cell
        the chart calls a wall. Anything subtler is left alone — a false wipe is as expensive as a
        stale chart."""
        if sight is None or not sight.ok:
            return False
        ahead = A.advance(self.cell, self.heading, 1)
        settled = self.kind(ahead)
        if sight.blocked() and settled in (PLAIN, STAIR, EXIT):
            return True
        return sight.reach >= 1 and settled == WALL

    def wipe(self):
        """Throw this epoch's grid away and re-seed it at the believed pose.

        The pose survives (it is the one thing the emitted actions determine) and so does the
        ledger of refusals, which is keyed on poses and is the harder fact of the two."""
        self.grids[self.epoch] = {self.cell: PLAIN}
        self.views[self.epoch] = {}
        self.wipes += 1

    def reconcile(self, sight):
        """One look, folded: wipe on a contradiction, file what it proved, then settle the epoch."""
        if self.clashes_with(sight):
            self.clashes += 1
            self.wipe()
        self.fold_sight(sight)
        self.settle_epoch(sight)

    # ── what the planner asks ───────────────────────────────────────────────────────────────
    def questions(self, epoch=None):
        """The open cells nothing has looked out of — the frontier, as a set of cells.

        A question is an ENTERABLE cell that is not yet PLAIN, or a PLAIN cell with an UNKNOWN
        orthogonal neighbour. The second clause is what keeps a corridor's far end a live question
        after a stride flew over it."""
        g = self.grid(epoch)
        out = set()
        for cell, kind in g.items():
            if kind == PASSABLE:
                out.add(cell)
            elif kind in (PLAIN, STAIR):
                for h in range(4):
                    if A.advance(cell, h, 1) not in g:
                        out.add(cell)
                        break
        return out

    def tiles(self, wanted, epoch=None):
        """Every cell of one kind on this floor — how the planner finds the stairs and the exit."""
        return {c for c, k in self.grid(epoch).items() if k == wanted}

    def returning(self, cell, name):
        """Is this climb the UNDO of the press that brought the body to this cell?

        The planner prices a return far above a fresh link: a return is the committed way back off a
        floor that has nothing left to offer, not a competitor to the frontier in front of you."""
        return A.inverse_climb(self.arrived.get((self.epoch, cell), "")) == name

    def on_stair(self):
        """Is a stair believed to be under the body? Not a reading: the tile underfoot is never
        drawn, so this is the chart's answer, from a stair seen earlier and a reckoned pose."""
        return self.kind(self.cell) == STAIR

    def line(self):
        """The pose and the ledger, as the working-memory block states them."""
        g = self.grid()
        refused = sorted(h for h in range(4) if self.refuted(self.cell, h))
        names = ("the way you face", "to your right", "behind you", "to your left")
        where = ", ".join(names[(h - self.heading) % 4] for h in refused) or "none yet"
        return ("%d cells of this floor are mapped and %d are still unexplored; directions already "
                "PROVEN BLOCKED from this exact spot: %s." % (len(g), len(self.questions()), where))
