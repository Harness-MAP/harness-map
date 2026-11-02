"""RoadGeometry — the harness's first NON-VLM arrival channel (iter 11).

A new node in the component graph, sitting between the frame and the
CompletionChecker.  Every other arrival readout the harness owns is *language
over raw RGB* — the planner's self-report, the perceiver's free text, and the
binary REACH judge — and the parent diagnosis measured all three to be flat or
inverted in true distance (§3/§4: the judge's positive rate is HIGHER beyond
200 m than inside the 1 000-2 000 cm pass window).  This module answers the same
question "am I at the intersection?" from **geometry** instead, using two
channels the agent already declares and the harness already decodes:

  * `obs['depth']`  — the per-frame-normalised JET colormap, decoded by
    `completion_checker._depth_gray` into a 0 (nearest) .. 255 (farthest) field.
    Its **frame median** is a pure shape statistic: walking a block, a facade
    fills the frame at close range and the median sits low; standing in an
    intersection the cross street and the sky open up and the median rises.
  * `obs['object_mask']` — the building class (purple 128,0,128) in the two
    outer lateral bands of the horizon row band.  At an intersection the
    building rows on BOTH sides stop; mid-block at least one side is a wall.

Both tests are frame-fractions / self-normalised percentile positions — no cm,
no coordinate, no map-tuned constant, and no privileged field.  The component
reads exactly the two obs keys the harness already declares.

MEASURED on the parent's own 50-episode run (743 ROAD-leg perceive frames,
true L1 distance taken from the traces for the measurement only):

    predicate                     in-window   out-of-window   lift
    frame-median depth >= 190       0.593         0.147        4.0x
    ... AND side-building <= 0.25   (per-leg state machine below)

and, run through the per-leg state machine as a COMMIT channel on ROAD legs:

    34 of 44 ROAD legs fire; 22 of those fires land inside the 2 000 cm pass
    window (precision 0.65).  Expected ROAD conversion 9 -> 25 legs.

The state machine is what makes it a commit channel rather than a detector.  A
ROAD instruction says "stop at **the** intersection" — the FIRST one — and a leg
*starts* standing in the intersection the previous leg just committed in, so the
raw predicate is True on the first frame of almost every leg.  Hence:

  1. the leg must first be seen CLOSED (a frame where the predicate is False,
     i.e. the agent is mid-block between facades)  — "armed";
  2. only a RISING EDGE after arming counts as a crossing;
  3. only the FIRST crossing of a leg may commit, and at most once.

Crossings are also counted for the whole leg and exposed to the
CompletionChecker, which uses "this FINISH leg has crossed at least one
intersection" as the far-field guard that the blind judge cannot provide
(the parent's 8 fatal FINISH commits all fired with ZERO crossings behind them;
both of its two real FINISH arrivals had 2).

Fails SILENT (no reading, no event, no commit) on a missing/odd depth or mask,
so an unavailable channel can never manufacture a commit.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:  # the verified JET inverse already lives next door; reuse, never fork it
    from .completion_checker import _depth_gray as _depth_gray
except Exception:  # pragma: no cover - defensive only
    _depth_gray = None  # type: ignore[assignment]


# ---- the horizon band the two lateral tests read ------------------------
# Rows 0.45-0.70 H: below the sky, above the always-near ground immediately in
# front of the robot. Outer 25% of columns on each side: the band a building
# row projects into while the robot walks between facades.
_BAND_ROW = (0.45, 0.70)
_BAND_COL_FRAC = 0.25

# Building class in obs['object_mask'] (purple; palindromic so the BGR-vs-RGB
# order of the mask does not matter). Same constant + tolerance the
# completion_checker's named-side validator already uses.
_BUILDING_BGR = np.array([128, 0, 128], dtype=np.int16)
_BUILDING_TOL = 60

# --- the opening predicate (both operating points measured, see docstring) ---
# `frame_open`: median of the JET-decoded depth over the WHOLE frame, on the
# decoder's own 0 (nearest) .. 255 (farthest) index. This is a position inside
# the frame's own normalised range, NOT a distance: "more than half of this
# frame sits in the far quarter of whatever range this frame spans". Scale-free
# by construction, which is the only way to read a per-frame-normalised depth.
_FRAME_OPEN_MIN = 190.0
# `side_building_max`: the LARGER of the two lateral bands' building-class
# fractions. An intersection needs BOTH sides to have stopped being a wall.
_SIDE_BUILDING_MAX = 0.25


@dataclass
class _Reading:
    """One frame's geometry. `ok=False` == the frame could not be evaluated."""
    ok: bool = False
    frame_open: float = 0.0
    side_building_max: float = 1.0

    @property
    def is_open(self) -> bool:
        return bool(self.ok
                    and self.frame_open >= _FRAME_OPEN_MIN
                    and self.side_building_max <= _SIDE_BUILDING_MAX)


def _lateral_bands(a: np.ndarray):
    """The two outer lateral slices of the horizon band, (left, right)."""
    h, w = a.shape[:2]
    r0, r1 = int(_BAND_ROW[0] * h), int(_BAND_ROW[1] * h)
    sw = max(1, int(_BAND_COL_FRAC * w))
    return a[r0:r1, 0:sw], a[r0:r1, (w - sw):w]


def _building_frac(band: np.ndarray) -> float:
    if band.size == 0:
        return 1.0
    d = np.abs(band[..., :3].astype(np.int16) - _BUILDING_BGR).sum(axis=-1)
    return float((d < _BUILDING_TOL).mean())


def read_geometry(obs: Any) -> _Reading:
    """Decode one frame into a `_Reading`. Never raises; `ok=False` on anything
    it cannot measure (missing depth, missing mask, odd shape, flat frame)."""
    if not isinstance(obs, Mapping) or _depth_gray is None:
        return _Reading()
    try:
        dg = _depth_gray(obs)
    except Exception:
        return _Reading()
    if dg is None or getattr(dg, "ndim", 0) != 2 or dg.size == 0:
        return _Reading()
    try:
        frame_open = float(np.median(dg))
    except Exception:
        return _Reading()

    m = obs.get("object_mask")
    if m is None:
        return _Reading()
    try:
        mask = np.asarray(m)
    except Exception:
        return _Reading()
    if mask.ndim != 3 or mask.shape[-1] < 3 or mask.size == 0:
        return _Reading()
    try:
        bl, br = _lateral_bands(mask)
        side = max(_building_frac(bl), _building_frac(br))
    except Exception:
        return _Reading()
    return _Reading(ok=True, frame_open=frame_open, side_building_max=side)


@dataclass
class _LegState:
    armed: bool = False          # a CLOSED (mid-block) frame has been seen
    prev_open: bool = False      # last evaluable frame's predicate
    crossings: int = 0           # rising edges since arming, this leg
    arrival_claimed: bool = False  # the one commit this leg may raise


@dataclass
class RoadGeometry:
    """Per-episode geometric arrival sensor, keyed by the structural leg index.

    Public API (consumed by agent.py):
      - update(obs, cur_idx) -> dict(ok, event, crossings, arrival)
      - crossings(cur_idx)   -> int
      - summary()            -> str   (one-line digest for notes/traces)
    """
    _legs: Dict[int, _LegState] = field(default_factory=dict)
    _reads: int = 0
    _events: int = 0
    _arrivals: int = 0

    def _state(self, cur_idx: int) -> _LegState:
        st = self._legs.get(cur_idx)
        if st is None:
            st = _LegState()
            self._legs[cur_idx] = st
        return st

    def update(self, obs: Any, cur_idx: int) -> Dict[str, Any]:
        """Fold one frame into this leg's crossing state machine.

        Returns:
          ok        — the frame was evaluable at all
          event     — a rising edge (an intersection opening just appeared)
                      AFTER the leg has been seen closed. Consumed as an
                      off-stride trigger for the (frozen) REACH judge.
          crossings — openings traversed so far on this leg.
          arrival   — True at most ONCE per leg: this is the leg's FIRST
                      crossing, i.e. "the intersection" a ROAD instruction
                      names. The only output allowed to drive a commit.
        """
        st = self._state(int(cur_idx))
        r = read_geometry(obs)
        if not r.ok:
            # Unevaluable frame: do not advance the edge detector at all (a
            # dropped frame must not look like a closing).
            return {"ok": False, "event": False,
                    "crossings": st.crossings, "arrival": False}
        self._reads += 1
        open_now = r.is_open
        event = False
        if not open_now:
            st.armed = True
        elif st.armed and not st.prev_open:
            st.crossings += 1
            event = True
            self._events += 1
        st.prev_open = open_now

        arrival = bool(event and st.crossings == 1 and not st.arrival_claimed)
        if arrival:
            st.arrival_claimed = True
            self._arrivals += 1
        return {"ok": True, "event": event,
                "crossings": st.crossings, "arrival": arrival}

    def crossings(self, cur_idx: int) -> int:
        st = self._legs.get(int(cur_idx))
        return int(st.crossings) if st is not None else 0

    def summary(self) -> str:
        return ("geo reads=%d openings=%d arrivals=%d"
                % (self._reads, self._events, self._arrivals))
