"""PoseScript — the 8th module: a per-leg POSE PROGRAM that runs *ahead of*
ActivePerception.

Region (iter 8, steered toward `cand_r006`'s pose-owning node)
--------------------------------------------------------------
The env grades the four legs on three different physical quantities
(`traffic_world.py:396-462`):

  * MOVE_ALONG_ROAD       L1 POSITION only
  * TURN_AT_INTERSECTION  YAW *and* a position box around the junction
  * REACH_DESTINATION     POSITION *and* YAW

Every one of those is a statement about *where the body stands and where it
points*, and the anchor has exactly one predicate for all three ("does the
forward RGB frame look like the hint photo?") and **no component whose job is
to put the body somewhere before the commit fires**. That missing
responsibility — POSE — is the region this candidate lives in.

How this differs from `cand_r006`'s `ViewpointController` (the token-level axis)
-------------------------------------------------------------------------------
Same region, three deliberately different implementation choices:

1. **Delay, don't intercept.** r006 *captured* `ActivePerception`'s
   `[rotate, -1]` burst, held it in `_pending_turn_burst`, and re-issued it
   later through an idempotence latch (`_turn_released`) — which means the
   burst's one-shot latch inside AP has to be reasoned about from two modules
   at once. This module never sees that burst. It runs **before** AP is
   consulted and simply *owns the earlier decisions of the leg*: while the
   approach program is live it returns its own action and AP is not called at
   all; when the program finishes it returns `None` and AP fires its
   relative-turn burst on its own, unmodified, un-latched, on its first
   consultation of the leg. `active_perception.py` stays byte-identical and
   there is no hand-back protocol to get wrong.

2. **One wiring point, not four.** The whole node is a single call at the top
   of the decision body of `run_episode`. No commit-path guard, no `-1` trim-
   loop prefix, no scan re-pointing.

3. **A turn-down (peak) detector over a windowed band trace, not an absolute
   threshold on one frame.** r006 released the turn when the turn-side building
   fraction fell below a fitted `0.10` measured on 32 frames; the
   diagnostician then measured the same band on **743** real frames and found
   the absolute form discriminates nothing (96.2 % of in-window vs 93.5 % of
   out-of-window frames pass it), while *"the named-side building fraction has
   just fallen from its running maximum"* is the actual intersection cue. So
   this module keeps a short rolling **trace** of the lateral band on both
   sides — pushed on **every** decision of the episode, so the window reaches
   back into the tail of the preceding ROAD leg and the detector has real
   support instead of the three frames a leg-scoped statistic would give it —
   and releases on a *relative* drop from that window's maximum. Scale-free,
   fit-free, and the same estimator family the gradient asked for.

Why the TURN leg only
---------------------
The measured failure is unambiguous (diagnosis §2.3): **12/12** TURNING
commits fired at `substeps_in_leg == 1`, i.e. `[rotate, -1]` on the *first*
decision of the leg with **zero** translation since the ROAD waypoint; the
rotation was correct in **7/7** traced failures (yaw error 0) and the body was
**78-980 cm** — one to two 500 cm forwards — short of the junction box. The
exact replay converts 5/12 -> 11/12 at two extra forwards, and an in-place
rotation preserves position so the already-earned ROAD window is never risked.
The gradient is equally unambiguous that this must ship **alone**
(`cand_r001` bundled it with a ROAD-commit rewrite, ROAD collapsed to 0.02,
and the turn change was never observed). So `PoseScript` deliberately has
**one** program: ROAD and FINISH return `None` on every call and every commit
path in `completion_checker.py` is untouched.

Honest inputs only: the STATIC `task_spec['subtasks'][i]['instruction']` text
(via `active_perception`'s own parsers) and `obs['object_mask']` (the
sanctioned channel the perceiver already reads). No coordinates, no `gt.json`,
no analog yaw, no `subtask_max_steps`, no oracle.
"""
from __future__ import annotations
import re
import sys
import types
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple
from collections.abc import Mapping

import numpy as np

# Self-register so @dataclass resolves string annotations when loaded via
# importlib.util.spec_from_file_location (the evaluator / validate_harness
# entry path). Same boilerplate as every other module in this package.
sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))


# ---- high-level action codes (mirror harness.ACTION_TO_STEP) ---------------
_FORWARD = 0


# ---- leg identification: REUSE ActivePerception's parsers -------------------
# Deliberately NOT a second taxonomy. The decision this module makes is "delay
# the burst that AP is about to emit", so it must agree with AP about which
# instructions produce that burst *by construction* — two independent regexes
# over the same text is a drift bug waiting for a phrasing it disagrees on.
# `_parse_target_cardinal` is consulted first for the same reason AP consults
# it first: "Turn to face east" contains the word `turn` but is an ORIENTATION
# leg, whose closed-form cardinal burst is the one path in this harness that
# passes 50/50 and must never be delayed by a single step.
try:                                                     # pragma: no cover
    from .active_perception import (
        _parse_relative_turn as _ap_parse_relative_turn,
        _parse_target_cardinal as _ap_parse_target_cardinal,
    )
except Exception:                                        # pragma: no cover
    # Fallback only for a standalone import of this file; mirrors
    # active_perception.py:125-175 exactly.
    _FALLBACK_CARDINAL_RE = re.compile(
        r'\b(?:face|facing|faced)\b(?:\s+(?:to|toward|towards))?(?:\s+the)?'
        r'\s+(east|west|north|south)\b', re.IGNORECASE)
    _FALLBACK_TURN_RE = re.compile(
        r'\b(?:turn\s+(left|right)|make\s+a\s+(left|right)\s+turn)\b',
        re.IGNORECASE)

    def _ap_parse_target_cardinal(instruction: Any) -> Optional[str]:
        if not isinstance(instruction, str) or not instruction:
            return None
        m = _FALLBACK_CARDINAL_RE.search(instruction)
        return m.group(1).lower() if m else None

    def _ap_parse_relative_turn(instruction: Any) -> Optional[str]:
        if not isinstance(instruction, str) or not instruction:
            return None
        m = _FALLBACK_TURN_RE.search(instruction)
        return (m.group(1) or m.group(2)).lower() if m else None


def turn_leg_direction(instruction: Any) -> Optional[str]:
    """'left' / 'right' iff this instruction is the relative-TURN leg, else None.

    Exactly the predicate `ActivePerception` rule (a'') fires on: a relative
    turn that is *not* a cardinal-orientation instruction.
    """
    if _ap_parse_target_cardinal(instruction) is not None:
        return None
    return _ap_parse_relative_turn(instruction)


# ---- the lateral band: REUSE completion_checker's measured geometry ---------
# One source of truth for where the band is and what a building pixel is. The
# operating point (rows 0.30-0.62, outer 25 % of the columns, purple building
# class with an L1 tolerance of 60) was probe-measured for
# `_named_side_has_structure`; re-typing the numbers here would let the two
# drift. Segmentation, NOT `obs['depth']`: depth is a per-frame min-max
# normalised JET rendering with no absolute scale, and the lever this module
# replaces died precisely because it was gated on a depth predicate that read
# `near_frac == 0.000` on 24/24 real frames and fired 2 times in 50 episodes.
try:                                                     # pragma: no cover
    from .completion_checker import (
        _NAMED_SIDE_ROW as _BAND_ROW,
        _NAMED_SIDE_COL_FRAC as _BAND_COL_FRAC,
        _NAMED_SIDE_BUILDING as _BAND_BUILDING,
    )
except Exception:                                        # pragma: no cover
    _BAND_ROW = (0.30, 0.62)
    _BAND_COL_FRAC = 0.25
    _BAND_BUILDING = np.array([128, 0, 128], dtype=np.int16)

_BAND_CLASS_TOL = 60        # same L1 tolerance completion_checker uses


def side_band_fraction(obs: Any, side: str) -> Optional[float]:
    """Fraction of building-class pixels in the lateral band on `side`.

    Returns None whenever the mask is absent or malformed. Every consumer
    treats None as *no information* and falls through to its own bound, so an
    unavailable channel can never silently hold a pose decision closed.
    """
    if side not in ("left", "right"):
        return None
    m = obs.get("object_mask") if isinstance(obs, Mapping) else None
    if m is None:
        return None
    try:
        a = np.asarray(m)
    except Exception:
        return None
    if a.ndim != 3 or a.shape[-1] < 3 or a.size == 0:
        return None
    h, w = a.shape[:2]
    r0, r1 = int(_BAND_ROW[0] * h), int(_BAND_ROW[1] * h)
    sw = max(1, int(_BAND_COL_FRAC * w))
    band = a[r0:r1, 0:sw, :3] if side == "left" else a[r0:r1, (w - sw):w, :3]
    if band.size == 0:
        return None
    bld = (np.abs(band.astype(np.int16) - _BAND_BUILDING).sum(axis=-1)
           < _BAND_CLASS_TOL)
    return float(bld.mean())


# ---- the turn-down (peak) detector -----------------------------------------
# Physical claim: walking a block, the near facade on one side fills the
# lateral band; at the intersection that block face ENDS and the cross street
# opens, so the fraction falls off its own running maximum. The diagnostician
# measured exactly this inversion on 743 ROAD frames — in-window side fraction
# (mean 0.434) sits BELOW out-of-window (mean 0.583) — which is why an
# absolute floor cannot separate the two and a *relative fall from the recent
# peak* can.
BAND_TRACE_WINDOW = 8       # decisions of history the peak is taken over
BAND_PEAK_MIN = 0.12        # the window peak must exceed this to mean anything
BAND_REL_DROP = 0.35        # current <= (1 - 0.35) * peak  ==  "the side opened"
BAND_MIN_SAMPLES = 3        # never fire off a single frame


@dataclass
class SideBandTrace:
    """Rolling per-side trace of the lateral building fraction.

    Pushed once per decision for the WHOLE episode (not per leg): the TURN leg
    is only a handful of decisions long, so a leg-scoped statistic would have
    no peak to fall from. The window reaches back into the tail of the
    preceding ROAD leg, which is travelled on the SAME heading the TURN leg
    starts on, so the samples are directly comparable.
    """
    _left: Deque[Optional[float]] = field(
        default_factory=lambda: deque(maxlen=BAND_TRACE_WINDOW))
    _right: Deque[Optional[float]] = field(
        default_factory=lambda: deque(maxlen=BAND_TRACE_WINDOW))

    def push(self, obs: Any) -> None:
        self._left.append(side_band_fraction(obs, "left"))
        self._right.append(side_band_fraction(obs, "right"))

    def _series(self, side: str) -> List[float]:
        raw = self._left if side == "left" else self._right
        return [v for v in raw if v is not None]

    def opened(self, side: Optional[str]) -> bool:
        """True iff the band on `side` has fallen off its windowed peak.

        Fails CLOSED (False) on missing data — which defers to the caller's
        forward cap, i.e. to the measured plateau, never to zero motion.
        """
        if side not in ("left", "right"):
            return False
        s = self._series(side)
        if len(s) < BAND_MIN_SAMPLES:
            return False
        peak = max(s)
        if peak < BAND_PEAK_MIN:
            return False
        return s[-1] <= (1.0 - BAND_REL_DROP) * peak


# ---- the TURN-leg approach program -----------------------------------------
# Bounds, not a fit. The exact replay of the 12 episodes that reached a TURNING
# leg converts 5/12 -> 8/12 (1 extra forward) -> 11/12 (2) -> 11/12 (3) and
# decays past 5, so [2, 3] is the measured plateau: the floor keeps the
# detector from releasing on noise at a point that is known worse, the cap
# keeps a silent detector on the plateau rather than walking out the far side.
# The peak detector therefore chooses WHERE ON the plateau to land, and can
# only choose between two values that both measure 11/12.
APPROACH_FLOOR_FORWARDS = 2
APPROACH_CAP_FORWARDS = 3


@dataclass
class PoseScript:
    """Per-episode pose program. One public method, one wiring point.

    `step(instruction, obs)` -> a list of high-level actions to execute NOW
    (short-circuiting perceive+plan and skipping the ActivePerception
    consultation for this decision), or `None` to let the rest of the harness
    run exactly as the anchor does.

    State is per-leg and keyed on the instruction string (the same honest key
    `ActivePerception` resets its own counters on).
    """

    trace: SideBandTrace = field(default_factory=SideBandTrace)

    _leg_key: Optional[str] = field(default=None)
    _approach_forwards: int = field(default=0)
    _released: bool = field(default=False)
    _last_release_reason: Optional[str] = field(default=None)

    # ---- leg bookkeeping ---------------------------------------------
    def _begin_leg_if_new(self, instruction: Any) -> None:
        if instruction == self._leg_key:
            return
        self._leg_key = instruction
        self._approach_forwards = 0
        self._released = False
        self._last_release_reason = None

    @property
    def approach_forwards(self) -> int:
        return self._approach_forwards

    @property
    def release_reason(self) -> Optional[str]:
        """'cue' | 'cap' | None — for the harness's episode notes."""
        return self._last_release_reason

    # ---- the one public entry point ----------------------------------
    def step(self, instruction: Any, obs: Any = None) -> Optional[List[int]]:
        """Run this leg's pose program for one decision.

        The band trace is pushed on EVERY call, including the calls that return
        None, because the TURN leg's release cue needs the preceding leg's
        samples for its peak.
        """
        self.trace.push(obs)
        self._begin_leg_if_new(instruction)

        side = turn_leg_direction(instruction)
        if side is None:
            return None                       # ORIENT / ROAD / FINISH: no program
        if self._released:
            return None                       # AP owns the rest of the leg

        # --- the approach program -------------------------------------
        # Hold the leg for up to APPROACH_CAP_FORWARDS single forwards along
        # the PRE-TURN heading, releasing early once the turn side opens. AP's
        # `[rotate, -1]` burst is never touched: it simply fires on the first
        # consultation after this returns None, from a body that is now inside
        # the junction instead of one to two forwards short of it.
        if self._approach_forwards >= APPROACH_CAP_FORWARDS:
            self._released = True
            self._last_release_reason = "cap"
            return None
        if (self._approach_forwards >= APPROACH_FLOOR_FORWARDS
                and self.trace.opened(side)):
            self._released = True
            self._last_release_reason = "cue"
            return None

        self._approach_forwards += 1
        return [_FORWARD]
