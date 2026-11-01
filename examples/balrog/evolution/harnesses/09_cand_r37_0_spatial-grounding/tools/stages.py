"""stages.py — the middleware stack. Each stage is a small object with two hooks:

  pre(ctx)          -> action | None   runs BEFORE the model; returning an action short-circuits
                                        the whole stack (the model is not called this step).
  post(ctx, action) -> action          runs AFTER the model, innermost-first; may rewrite it.

Stages share nothing but the StepContext (tools/context.py). Two fields are shared state written by
one stage and read by others: `ctx.blocked` (BumpStage's convicted (square, dir) edges) and
`ctx.chart` (NavStage's per-level remembered map). The decision locus is NavStage: a BFS over the
remembered chart to a drawn `>`, else the nearest unstood frontier square. The model is asked only
when no stage can answer.
"""
from __future__ import annotations

import re
from collections import deque

from context import DELTA, OPPOSITE

WALKABLE = set(".#<>+{$)[=!?/\"*(%_,^")        # boulders `0`/`` ` `` are not planned through
SWING = re.compile(r"\b(You (hit|miss|kill|destroy|bite)|attack|The .* (misses|hits|bites))")
REFUSAL = re.compile(r"(It's a wall|solid stone|in vain|cannot pass|can't move diagonally|"
                     r"You are carrying too much|door is locked|is closed|You can't|squeeze)", re.I)
W, H = 79, 21


def _is_creature(g):
    return g is not None and g.isalpha() and g != "@"


def _modal(m):
    return "[yn" in m or "--More--" in m or "direction?" in m or "What do you want" in m


class ModalStage:
    """An open `[yn]` / `--More--` prompt: answer it deterministically, never spend model turns."""

    def reset(self):
        pass

    def pre(self, ctx):
        m = ctx.message
        if "direction?" in m:
            return None                      # DoorStage owns the direction answer
        if "[yn" in m or "--More--" in m:
            for a in ("y", "more", "esc"):
                if a in ctx.legal and ("[yn" in m or a != "y"):
                    ctx.trail.append(f"modal->{a}")
                    return a
        return None

    def post(self, ctx, action):
        return action


class DoorStage:
    """`This door is locked.` after a move: kick it (a two-step verb: `kick`, then the bearing)."""

    def reset(self):
        self.pending = None
        self.kicks = {}

    def pre(self, ctx):
        if self.pending and "direction?" in ctx.message:
            d, self.pending = self.pending, None
            ctx.trail.append(f"kick-dir {d}")
            return d
        self.pending = None
        last = ctx.shared.get("last_move")
        if ("door is locked" in ctx.message and last and "kick" in ctx.legal
                and ctx.pos is not None and ctx.pos == last[0]):
            key = (ctx.pos, last[1])
            if self.kicks.get(key, 0) < 6:
                self.kicks[key] = self.kicks.get(key, 0) + 1
                ctx.unblock.add(key)           # a locked door is not a wall: keep the edge open
                self.pending = last[1]
                ctx.trail.append(f"kick {last[1]}")
                return "kick"
        if "direction?" in ctx.message and "esc" in ctx.legal:
            ctx.trail.append("modal->esc")
            return "esc"
        return None

    def post(self, ctx, action):
        return action


class FoeStage:
    """A creature drawn on an adjacent square: swing at it (moving into it attacks). Never retreat."""

    def reset(self):
        pass

    def pre(self, ctx):
        if ctx.pos is None:
            return None
        for d in ctx.moves():
            if _is_creature(ctx.cells.get(ctx.target(d))) and not ctx.lethal(d):
                ctx.trail.append(f"swing {d}")
                return d
        return None

    def post(self, ctx, action):
        return action


class BumpStage:
    """Book a (square, direction) edge only when the game REFUSED the move (wall/stone/bars/boulder/
    doorway-diagonal). A swing, a creature on the target or an open prompt is never a refusal."""

    STRIKES = 3                                   # unexplained no-moves before an edge is booked

    def reset(self):
        self.blocked = set()
        self.expect = None
        self.strikes = {}

    def pre(self, ctx):
        e, self.expect = self.expect, None
        m = ctx.message
        if e and ctx.pos is not None and ctx.pos == e[0]:
            tgt = ctx.target(e[1])
            if SWING.search(m) or _modal(m) or _is_creature(ctx.cells.get(tgt)) or "door is locked" in m:
                pass
            elif REFUSAL.search(m) or not m.strip():
                self.blocked.add(e)
            else:                                  # e.g. `The door opens.`: allow a retry, not a loop
                self.strikes[e] = self.strikes.get(e, 0) + 1
                if self.strikes[e] >= self.STRIKES:
                    self.blocked.add(e)
        ctx.blocked = self.blocked
        return None

    def post(self, ctx, action):
        self.blocked -= ctx.unblock
        if action in DELTA and ctx.pos is not None:
            if (ctx.pos, action) in self.blocked and ctx.source == "model":
                alt = [d for d in ctx.moves() if (ctx.pos, d) not in self.blocked and not ctx.lethal(d)]
                if alt:
                    ctx.trail.append(f"bump-veto {action}->{alt[0]}")
                    action = alt[0]
            self.expect = (ctx.pos, action)
            ctx.shared["last_move"] = (ctx.pos, action)
        else:
            ctx.shared["last_move"] = None
        return action


class HazardStage:
    """Never step onto a square named lava/water in prose or drawn as `}`."""

    def reset(self):
        pass

    def pre(self, ctx):
        bad = [d for d in ctx.moves() if ctx.lethal(d)]
        if bad:
            ctx.notes.append("deadly terrain (lava/water) lies " + ", ".join(bad) + " — do not step there")
        return None

    def post(self, ctx, action):
        if action in DELTA and ctx.lethal(action):
            safe = [d for d in ctx.moves() if not ctx.lethal(d)]
            if safe:
                ctx.trail.append(f"hazard-veto {action}->{safe[0]}")
                return safe[0]
        return action


class NavStage:
    """The decision locus. Keeps a per-level chart (last glyph seen per square, squares stood on);
    each step re-elects a goal — a drawn `>` if reachable, else the nearest unstood square that
    borders unseen space, else the nearest unstood square — and emits the first hop of a BFS route
    that respects every convicted edge on EVERY hop. Distance-ascending, (y, x) tie-break: a total
    order, so re-election each step is a fixed point, not an orbit."""

    SEARCHES = 10                                 # searches per square once the chart is exhausted

    def reset(self):
        self.chart, self.stood, self.stairs, self.bars = {}, set(), None, False
        self.searched = {}

    def _absorb(self, ctx):
        if "bars" in ctx.message:
            self.bars = True                      # Boxoban: `#` is iron bars, not corridor
        for p, g in ctx.cells.items():
            if g == "@" or _is_creature(g):
                self.chart.setdefault(p, ".")     # hero/creature hides the terrain: assume floor
            else:
                self.chart[p] = g
                if g == ">":
                    self.stairs = p
        if ctx.pos is not None:
            self.stood.add(ctx.pos)
        ctx.chart = self.chart

    def _walkable(self, p):
        g = self.chart.get(p)
        return g in WALKABLE and not (self.bars and g == "#")

    def _frontier(self, p):
        x, y = p
        for dx, dy in DELTA.values():
            q = (x + dx, y + dy)
            if 0 <= q[0] < W and 1 <= q[1] <= H and q not in self.chart:
                return True
        return False

    def _bfs(self, ctx):
        start = ctx.pos
        moves = [d for d in ctx.moves()]
        prev = {start: None}
        order = deque([start])
        dist = {start: 0}
        while order:
            cur = order.popleft()
            for d in moves:
                if (cur, d) in ctx.blocked:
                    continue
                nxt = (cur[0] + DELTA[d][0], cur[1] + DELTA[d][1])
                if nxt in prev or not self._walkable(nxt):
                    continue
                if cur == start and ctx.lethal(d):
                    continue
                prev[nxt] = (cur, d)
                dist[nxt] = dist[cur] + 1
                order.append(nxt)
        return prev, dist

    def _first_hop(self, prev, goal):
        hop = None
        while prev.get(goal) is not None:
            goal, hop = prev[goal]
        return hop

    def pre(self, ctx):
        self._absorb(ctx)
        if ctx.pos is None:
            return None
        if self.stairs is not None and ctx.pos == self.stairs and "down" in ctx.legal:
            ctx.trail.append("nav:descend")
            return "down"
        prev, dist = self._bfs(ctx)
        goal, why = None, None
        if self.stairs in dist and self.stairs != ctx.pos:
            goal, why = self.stairs, "stairs"
        else:
            unstood = sorted((dist[p], p[1], p[0]) for p in dist if p not in self.stood)
            front = [t for t in unstood if self._frontier((t[2], t[1]))]
            pick = front[0] if front else (unstood[0] if unstood else None)
            if pick:
                goal, why = (pick[2], pick[1]), "frontier" if front else "unstood"
        if goal is None and "search" in ctx.legal and self.searched.get(ctx.pos, 0) < self.SEARCHES:
            # nothing reachable is unexplored: a hidden door/corridor is the only way on
            self.searched[ctx.pos] = self.searched.get(ctx.pos, 0) + 1
            ctx.trail.append(f"nav:exhausted -> search #{self.searched[ctx.pos]}")
            return "search"
        if goal is None:
            ctx.notes.append("no unexplored reachable square is known; pick a move you have not tried")
            return None
        hop = self._first_hop(prev, goal)
        if hop is None:
            return None
        ctx.trail.append(f"nav:{why}{goal} d={dist[goal]} -> {hop}")
        return hop

    def post(self, ctx, action):
        return action


class OrbitStage:
    """Model fallback only: an immediate reversal onto an often-visited square is redirected."""

    def reset(self):
        self.visits = {}
        self.last = None

    def pre(self, ctx):
        if ctx.pos is not None:
            self.visits[ctx.pos] = self.visits.get(ctx.pos, 0) + 1
        return None

    def post(self, ctx, action):
        if (ctx.source == "model" and action in DELTA and self.last == OPPOSITE.get(action)
                and ctx.pos is not None
                and self.visits.get(ctx.target(action), 0) >= 2):
            cands = [d for d in ctx.moves() if d != action and (ctx.pos, d) not in ctx.blocked
                     and ctx.cells.get(ctx.target(d), " ") in (".", "#", "<", ">", "+", " ")]
            if cands:
                pick = min(cands, key=lambda d: self.visits.get(ctx.target(d), 0))
                ctx.trail.append(f"orbit {action}->{pick}")
                action = pick
        self.last = action if action in DELTA else None
        return action


class LegalStage:
    """Outermost: whatever survived must be in the vocabulary, else keep the stock string."""

    def reset(self):
        pass

    def pre(self, ctx):
        return None

    def post(self, ctx, action):
        if ctx.legal and action not in ctx.legal:
            ctx.trail.append(f"illegal {action!r}")
        return action


def default_stack():
    # outer -> inner. pre() runs in this order (first non-None answer wins), post() in reverse.
    # BumpStage.post must see every emitted move, so short-circuit answers are routed through
    # the post hooks too (harness.py).
    return [LegalStage(), ModalStage(), DoorStage(), BumpStage(), HazardStage(), FoeStage(),
            NavStage(), OrbitStage()]
