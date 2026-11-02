"""stages.py — the decision stack. Each stage owns one concern and has two hooks:

  pre(ctx)          -> action | None   runs BEFORE the model; an action short-circuits the stack
                                        (the model is NOT called this step).
  post(ctx, action) -> action          runs AFTER the model, innermost-first; may rewrite it.

Stages share nothing but the per-step context `ctx` (a dict built by agent.py from tools/ledger.py):
  legal, text, pos, mp, adj, rows, ledger, trail
Order (outer -> inner): Modal, Melee, Route, Arbitrate.
"""
from __future__ import annotations

from action_io import arbitrate
from route import plan


class ModalStage:
    """An open `[yn]` / `--More--` prompt freezes the frame: answer it with a legal verb."""

    def pre(self, ctx):
        text, legal = ctx["text"], ctx["legal"]
        if "[yn" in text and "y" in legal:
            return "y"
        if "--More--" in text and "more" in legal:
            return "more"
        return None

    def post(self, ctx, action):
        return action


class MeleeStage:
    """A creature drawn adjacent (letter glyph): the model decides the fight.
    Marks the context so the route stage stands down; swings are never booked as failures
    (tools/ledger.COMBAT)."""

    def pre(self, ctx):
        ctx["fight"] = any(g.isalpha() for _, g, _, _, _, _ in ctx["rows"])
        if ctx["fight"]:
            ctx["trail"].append("melee->model")
        return None

    def post(self, ctx, action):
        return action


class RouteStage:
    """Code-owned navigation: BFS first hop to a charted `>`, else to the nearest unstood frontier."""

    def pre(self, ctx):
        if ctx.get("fight") or ctx["pos"] is None:
            return None
        led = ctx["ledger"]
        banned = set()
        for d, _, _, _, kind, _ in ctx["rows"]:
            if kind in ("deadly", "blocked"):
                dx, dy = led_step(d)
                banned.add((ctx["pos"][0] + dx, ctx["pos"][1] + dy))
        led.banned |= banned
        res = plan(ctx["mp"], ctx["pos"], led.visits, led.failed, frozenset(led.banned), ctx["legal"])
        if res is None:
            ctx["trail"].append("route:none->model")
            return None
        hop, kind, target, n = res
        ctx["trail"].append(f"route:{kind}{target}/{n}")
        return hop

    def post(self, ctx, action):
        return action


class ArbitrateStage:
    """Innermost post: legal projection + deadly/blocked/failed veto + orbit break on the reply."""

    def pre(self, ctx):
        return None

    def post(self, ctx, action):
        pick, rule = arbitrate(action, ctx["legal"], ctx["rows"])
        ctx["trail"].append(f"arbitrate:{rule}")
        return pick


def led_step(d):
    from ledger import STEP
    return STEP[d]


def default_stack():
    return [ModalStage(), MeleeStage(), RouteStage(), ArbitrateStage()]
