"""stages.py — the middleware stack. Each stage is a small object with two hooks:

  pre(ctx)          -> action | None   runs BEFORE the model; returning an action short-circuits
                                        the whole stack (the model is not called this step).
  post(ctx, action) -> action          runs AFTER the model, innermost-first; may rewrite it.

The pipeline (harness.py) owns ordering; stages own one concern each and keep their own state.
"""
from __future__ import annotations

from context import DELTA, OPPOSITE

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
    return [LegalStage(), ModalStage(), HazardStage(), BumpStage(), OrbitStage()]
