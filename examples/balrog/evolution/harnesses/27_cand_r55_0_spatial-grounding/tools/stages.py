"""stages.py — the middleware stack. Each stage owns one concern and has two hooks:

  pre(ctx)          -> action | None   BEFORE the model; an action short-circuits (no model call)
  post(ctx, action) -> action          AFTER the model, innermost-first; may rewrite the proposal

Order (outer -> inner): Legal, Modal, Hazard, Route, Bump. The Route stage is the decision locus
for movement: it answers from the chart whenever a BFS route exists and no creature is adjacent;
the model is consulted only for fights, blocked levels (Boxoban pushes, sealed rooms) and prompts.
"""
from __future__ import annotations

from chart import Chart
from context import STEP


class ModalStage:
    """An open `[yn]` / `--More--` prompt freezes the frame: answer it with a legal verb."""

    def reset(self):
        pass

    def pre(self, ctx):
        head = ctx.text.split("map:", 1)[0]
        if "[yn" in head and "y" in ctx.legal:
            ctx.trail.append("modal->y")
            return "y"
        if "--More--" in head and "more" in ctx.legal:
            ctx.trail.append("modal->more")
            return "more"
        return None

    def post(self, ctx, action):
        return action


class RouteStage:
    """Owns the chart. Short-circuits with the BFS first hop toward `>` or the nearest frontier."""

    def __init__(self):
        self.chart = Chart()

    def reset(self):
        self.chart.reset()

    def pre(self, ctx):
        self.chart.fold(ctx)
        ctx.chart = self.chart
        fight = ctx.creature_dirs()
        if fight:
            ctx.notes.append("A creature stands " + ", ".join(fight)
                             + " of you; moving that way attacks it.")
            return None
        hop, kind, dist = self.chart.route(ctx.pos, ctx.moves())
        if hop is None:
            return None
        ctx.trail.append(f"route:{kind}:{dist}->{hop}")
        return hop

    def post(self, ctx, action):
        return action


class BumpStage:
    """The model's proposal: refuse a move already refused from this square or onto a drawn/named
    wall; replace it with the least-visited open neighbour."""

    def reset(self):
        pass

    def post(self, ctx, action):
        chart = getattr(ctx, "chart", None)
        if chart is None or action not in STEP or ctx.pos is None:
            return action
        if action in ctx.creature_dirs():
            return action
        tgt = ctx.target(action)
        bad = (ctx.pos, action) in chart.blocked or (tgt in chart.drawn and not chart.passable(tgt))
        if not bad:
            return action
        alt = [(chart.visits.get(ctx.target(d), 0), i, d) for i, d in enumerate(ctx.moves())
               if (ctx.pos, d) not in chart.blocked and chart.passable(ctx.target(d))]
        if alt:
            pick = min(alt)[2]
            ctx.trail.append(f"bump-veto {action}->{pick}")
            return pick
        return action

    def pre(self, ctx):
        return None


class HazardStage:
    """Never step onto a square drawn `}` or named lava/water (runs after Bump, so it has the last word)."""

    def reset(self):
        pass

    def pre(self, ctx):
        return None

    def post(self, ctx, action):
        chart = getattr(ctx, "chart", None)
        if chart is None or action not in STEP or not chart.deadly(ctx.target(action)):
            return action
        safe = [d for d in ctx.moves() if not chart.deadly(ctx.target(d))]
        if safe:
            ctx.trail.append(f"hazard-veto {action}->{safe[0]}")
            return safe[0]
        return action


class LegalStage:
    def reset(self):
        pass

    def pre(self, ctx):
        return None

    def post(self, ctx, action):
        if ctx.legal and action not in ctx.legal:
            ctx.trail.append(f"illegal {action!r}")
        return action


def default_stack():
    # outer -> inner: pre() runs in this order, post() in reverse.
    return [LegalStage(), ModalStage(), HazardStage(), RouteStage(), BumpStage()]
