"""stages.py — the middleware stack. Each stage is a small object with two hooks:

  pre(ctx)          -> action | None   runs BEFORE the model; returning an action short-circuits
                                        the whole stack (the model is not called this step).
  post(ctx, action) -> action          runs AFTER the model, innermost-first; may rewrite it.

The pipeline (orchestrator.py) owns ordering; stages own one concern each and keep their own state.
"""
from __future__ import annotations

import re

from context import DELTA, OPPOSITE
from field import Chart

LETHAL_NOUNS = ("lava", "water", "molten")
LETHAL_GLYPHS = ("}",)


class ModalStage:
    """An open `[yn]` / `--More--` prompt: answer it deterministically, never spend model turns."""

    def reset(self):
        pass

    def pre(self, ctx):
        m = ctx.message
        if "[yn" in m or "--More--" in m or m.rstrip().endswith("?"):
            for a in ("y", "more", "esc"):
                if a in ctx.legal and ("[yn" in m or a != "y"):
                    ctx.trail.append(f"modal->{a}")
                    return a
        return None

    def post(self, ctx, action):
        return action


class BumpStage:
    """Remember (square, direction) pairs whose move left the hero in place; refuse repeats."""

    def reset(self):
        self.blocked = set()
        self.expect = None

    def pre(self, ctx):
        if self.expect and ctx.pos is not None and ctx.pos == self.expect[0]:
            self.blocked.add(self.expect)
        self.expect = None
        for d in ctx.moves():
            if (ctx.pos, d) in self.blocked:
                ctx.notes.append(f"moving {d} from here did nothing last time")
        return None

    def post(self, ctx, action):
        if action in DELTA and ctx.pos is not None:
            if (ctx.pos, action) in self.blocked:
                alt = [d for d in ctx.moves() if (ctx.pos, d) not in self.blocked]
                if alt:
                    ctx.trail.append(f"bump-veto {action}->{alt[0]}")
                    action = alt[0]
            self.expect = (ctx.pos, action)
        return action


class HazardStage:
    """Never step onto a square named lava/water in prose or drawn as `}`."""

    def reset(self):
        pass

    def _lethal(self, ctx, d):
        if any(n in noun for noun in ctx.nouns.get(d, ()) for n in LETHAL_NOUNS):
            return True
        t = ctx.target(d)
        return t is not None and ctx.cells.get(t) in LETHAL_GLYPHS

    def pre(self, ctx):
        bad = [d for d in ctx.moves() if self._lethal(ctx, d)]
        if bad:
            ctx.notes.append("deadly terrain (lava/water) lies " + ", ".join(bad) + " — do not step there")
        return None

    def post(self, ctx, action):
        if action in DELTA and self._lethal(ctx, action):
            safe = [d for d in ctx.moves() if not self._lethal(ctx, d)]
            if safe:
                ctx.trail.append(f"hazard-veto {action}->{safe[0]}")
                return safe[0]
        return action


class OrbitStage:
    """Break period-2 orbits: an immediate reversal onto an often-visited square is redirected to
    the least-visited passable neighbour."""

    PASSABLE = set(".#<>{+-|`%$)[=!?/\"*(0_")

    def reset(self):
        self.visits = {}
        self.last = None

    def pre(self, ctx):
        if ctx.pos is not None:
            self.visits[ctx.pos] = self.visits.get(ctx.pos, 0) + 1
        return None

    def post(self, ctx, action):
        if (action in DELTA and self.last == OPPOSITE.get(action) and ctx.pos is not None
                and self.visits.get(ctx.target(action), 0) >= 2):
            cands = [d for d in ctx.moves() if d != action
                     and ctx.cells.get(ctx.target(d), " ") in (".", "#", "<", ">", "+", "`")]
            if cands:
                pick = min(cands, key=lambda d: self.visits.get(ctx.target(d), 0))
                ctx.trail.append(f"orbit {action}->{pick}")
                action = pick
        self.last = action if action in DELTA else None
        return action


class FieldStage:
    """Innermost DECISION OWNER. Folds the frame into a persistent Chart, then — before the model is
    asked — goes `down` on a remembered `>`, swings at an adjacent letter, or steps to the open
    neighbour with the lowest potential (stairs field, else frontier field). Only when every
    neighbour is off-field (flat: e.g. a premapped Boxoban room) does it return None, leaving a
    least-trodden-first move menu in ctx.notes for the model to break the tie."""

    def reset(self):
        self.chart = Chart()
        self.prev = None           # last StepContext; its .action is the FINAL applied action

    def pre(self, ctx):
        pend = None
        if self.prev is not None and self.prev.pos is not None and self.prev.action in DELTA:
            pend = (self.prev.pos, self.prev.action)
        self.prev = ctx
        lv = re.search(r"Dlvl:(\d+)", ctx.raw)
        ch = self.chart
        ch.observe(ctx, lv.group(1) if lv else "?", pend)
        pos = ctx.pos
        dirs = ctx.moves()
        if pos is None or not dirs:
            return None
        if ch.terrain.get(pos) == ">" and "down" in ctx.legal:
            ctx.trail.append("field:down")
            return "down"
        lethal = {d for d in dirs if any(n in noun for noun in ctx.nouns.get(d, ()) for n in LETHAL_NOUNS)
                  or ctx.cells.get(ctx.target(d)) in LETHAL_GLYPHS}
        open_dirs = [d for d in dirs if d not in lethal and (pos, d) not in ch.blocked]
        for d in open_dirs:
            g = ctx.cells.get(ctx.target(d), " ")
            if g.isalpha():
                ctx.trail.append(f"field:swing {d}")
                return d
        for name, pot in ch.fields():
            scored = sorted((pot[ctx.target(d)] + 0.25 * ch.visits.get(ctx.target(d), 0), i, d)
                            for i, d in enumerate(open_dirs) if ctx.target(d) in pot)
            if scored:
                ctx.trail.append(f"field:{name} {scored[0][2]}")
                return scored[0][2]
        menu = sorted(open_dirs, key=lambda d: (ch.visits.get(ctx.target(d), 0), dirs.index(d)))
        if menu:
            ctx.notes.append("open moves, least trodden first: " + ", ".join(menu))
        closed = sorted(set(dirs) - set(open_dirs))
        if closed:
            ctx.notes.append("closed or deadly here: " + ", ".join(closed))
        self.menu = menu
        return None

    def post(self, ctx, action):
        if action in DELTA and ctx.pos is not None and (ctx.pos, action) in self.chart.blocked:
            menu = getattr(self, "menu", [])
            if menu:
                ctx.trail.append(f"field-veto {action}->{menu[0]}")
                return menu[0]
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
    # outer -> inner. pre() runs in this order, post() in reverse.
    # Bump/Orbit are superseded by FieldStage's chart (refusal ledger + visit-weighted field).
    return [LegalStage(), ModalStage(), HazardStage(), FieldStage()]
