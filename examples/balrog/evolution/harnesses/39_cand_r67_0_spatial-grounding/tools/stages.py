"""stages.py — the middleware stack around the single model call.

Each stage owns one concern and has two hooks:
  pre(ctx)          -> action | None   BEFORE the model; an action short-circuits (model not called)
  post(ctx, action) -> action          AFTER the model, innermost-first; may rewrite the proposal
Stages share the per-level Ledger (tools/ledger.py) and the StepContext (tools/context.py) only.
Order (outer -> inner): Legal, Modal, Hazard, Route.  post() runs Route -> Hazard -> Legal, so the
hazard veto is the LAST word on any move, whoever proposed it.
"""
from __future__ import annotations

from context import DELTA
from ledger import noun_kind
from route import is_creature, plan, walkable


class LegalStage:
    def reset(self):
        pass

    def pre(self, ctx):
        return None

    def post(self, ctx, action):
        if ctx.legal and action not in ctx.legal:
            ctx.trail.append(f"illegal:{action!r}")
        return action


class ModalStage:
    """An open `--More--` freezes the frame; page it with the legal `more` verb, never a move."""

    def reset(self):
        pass

    def pre(self, ctx):
        if "--More--" in ctx.text and "more" in ctx.legal:
            ctx.trail.append("modal:more")
            return "more"
        return None

    def post(self, ctx, action):
        return action


class HazardStage:
    """Refuse a move onto a deadly/blocked/wall/blank(stone) square or a booked failed edge."""

    def __init__(self, ledger):
        self.ledger = ledger

    def reset(self):
        pass

    def bad(self, ctx, d):
        t = ctx.target(d)
        if t is None:
            return False
        if noun_kind(ctx.nouns.get(d)) or (ctx.pos, d) in self.ledger.failed:
            return True
        if is_creature(ctx, t):
            return False                               # moving into a monster attacks it
        return not walkable(ctx, self.ledger, t)      # an adjacent blank is stone: you would see floor

    def pre(self, ctx):
        bad = [d for d in ctx.moves() if self.bad(ctx, d)]
        if bad:
            ctx.notes.append("Blocked or deadly from here: " + ", ".join(bad) + ".")
        return None

    def post(self, ctx, action):
        if action not in DELTA or not self.bad(ctx, action):
            return action
        ok = [d for d in ctx.moves() if not self.bad(ctx, d)]
        if not ok:
            return action
        v = self.ledger.visits
        pick = min(ok, key=lambda d: (v.get(ctx.target(d), 0), list(DELTA).index(d)))
        ctx.trail.append(f"hazard:{action}->{pick}")
        return pick


class RouteStage:
    """Make the chart binding: the model's move stands only if it is the route's first hop or a
    swing at an adjacent creature; otherwise the BFS first hop (drawn `>`, else best frontier)."""

    def __init__(self, ledger):
        self.ledger = ledger
        self.hop = None

    def reset(self):
        self.hop = None

    def pre(self, ctx):
        self.hop, kind, goal = plan(ctx, self.ledger)
        if self.hop:
            ctx.notes.append(f"Route ({'to the stairs down' if kind == 'stairs' else 'to unexplored ground'}):"
                             f" move {self.hop}.")
            ctx.trail.append(f"route:{kind}@{goal}")
        return None

    def post(self, ctx, action):
        if not self.hop:
            return action
        if action == self.hop:
            ctx.trail.append("model=route")
            return action
        t = ctx.target(action) if action in DELTA else None
        if t is not None and is_creature(ctx, t):
            ctx.trail.append("model:attack")
            return action
        ctx.trail.append(f"route:{action}->{self.hop}")
        return self.hop


def default_stack(ledger):
    return [LegalStage(), ModalStage(), HazardStage(ledger), RouteStage(ledger)]
