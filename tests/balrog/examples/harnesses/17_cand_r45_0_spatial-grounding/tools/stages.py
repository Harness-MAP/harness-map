"""stages.py — the per-step MIDDLEWARE STACK around the single model call.

LEVER (code / wiring). Each stage is a peer with `pre(ctx)` (runs before the model, outer->inner;
may add prompt notes and vetoes) and `post(ctx, action)` (runs after it, inner->outer; may rewrite
the proposal). Stages talk only through the shared StepContext and record provenance in ctx.trail.
Order (outer -> inner): Veto, Swing, Route.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from chart import STEP


@dataclass
class StepContext:
    chart: object
    legal: list
    neigh: list
    route: tuple
    vetoed: set = field(default_factory=set)
    notes: list = field(default_factory=list)
    trail: list = field(default_factory=list)
    fallback: str | None = None


class VetoStage:
    """Deadly (lava/water), terrain-refused, and prose-named wall/bars directions are vetoed."""

    def pre(self, ctx):
        for d, g, v, b, h, w in ctx.neigh:
            if (h or b or w) and d not in ctx.chart.creatures():
                ctx.vetoed.add(d)

    def post(self, ctx, action):
        if action in ctx.vetoed:
            ctx.trail.append(f"veto:{action}")
            return ctx.fallback or action
        return action


class SwingStage:
    """A creature letter on a neighbouring square is a FIGHT target, never a vetoed move: an
    attack that is not answered in kind hands the creature free bites. If the model named a
    creature direction it stands; the route never overrides it."""

    def pre(self, ctx):
        foes = ctx.chart.creatures(ctx.legal)
        if foes:
            ctx.notes.append("CREATURE adjacent: " + ", ".join(foes) +
                             " (moving that way attacks it; keep attacking until it is gone)")
        ctx.foes = foes

    def post(self, ctx, action):
        if action in getattr(ctx, "foes", []):
            ctx.trail.append(f"swing:{action}")
        return action


class RouteStage:
    """Code owns exploration ranking (the model ignores visit tables). If the model's move lands
    on an already-visited square while the BFS route leads elsewhere, take the route hop."""

    def pre(self, ctx):
        hop, kind, dist = ctx.route
        if hop:
            ctx.notes.append(f"ROUTE: nearest {kind} is {dist} step(s) away; first move toward it: {hop}")
            if ctx.fallback is None:
                ctx.fallback = hop

    def post(self, ctx, action):
        hop, ch = ctx.route[0], ctx.chart
        if action in getattr(ctx, "foes", []) or not hop or action == hop or ch.pos is None:
            return action
        if action not in STEP:
            return action
        if ch.visits.get(ch.target(action), 0) > 0 or action in ctx.vetoed:
            ctx.trail.append(f"route:{action}->{hop}")
            return hop
        return action


def default_stack():
    return [VetoStage(), SwingStage(), RouteStage()]
