"""RoadGeometry — iter 13.  The 10th node: a FREE, NON-VLM position channel.

WHAT CHANGED IN THE GRAPH
-------------------------
Until now every conjunct of the `-1` license in this harness was a question put
to the frozen VLM.  The parent's diagnosis measured both of them end-to-end and
the split is unambiguous:

  * `ArrivalGate._identity_call` ("is this the same place as the hint photo?",
    two images, no leg text) is EXCELLENT: 39/42 = 93 % recall inside the
    2 000 cm acceptance window on ROAD legs.  It is a LEG-level signal — it
    says *which landmark*, and it saturates at 1 for whole legs (1 on all 13
    asks from 20 212 cm before to 14 288 cm past the waypoint on one trace).
  * `ArrivalGate._place_call` / `_JUNCTION_SYS` ("am I at an intersection?",
    one image) is the bottleneck: 11/38 = 29 % in-window.  Worse, 18 of its 19
    positives across 50 episodes cite ONE disjunct — "a crosswalk is at or just
    in front of the robot" — so the ANY-CUE wording collapsed onto a single
    geometric cue sitting at a roughly fixed half-block OFFSET from the grading
    point: 11 fires inside the window, 5 at 2 074-2 963 cm (all fatal).

So the harness has a good *identity* channel and no usable *position* channel,
and §4 of the diagnosis classifies the binding constraint as exactly that:
"the harness now recognises arrival but still has no arrival controller … on
28/50 legs the robot walks through the acceptance window and keeps going."

This module supplies the missing position channel from GEOMETRY instead of
language, on a sanctioned observation the harness already ships to the
perceiver and has never let decide anything: the ground class in
`obs['object_mask']`.  Walking a block, the forward travel corridor is
sidewalk; when the cross street opens the roadway class floods it.  That is the
intersection — the literal grading point of a ROAD leg ("Continue straight and
stop at the intersection with <building> on your left") — read in closed form,
for zero tokens and zero env substeps.

It is deliberately only ONE CONJUNCT.  Geometry knows *where* but not *which*:
a grid city has many intersections and the cue cannot tell them apart.
`_identity_call` knows *which* but not *where*.  The license is their
conjunction, which is the first time this chain has had both halves.

MEASUREMENT (this iter, on the PARENT'S OWN RUN — 1 326 archived perceive
frames across all 50 episodes; the saved `object_mask` PNG of every perceive
call, scored against the true L1 distance to that leg's graded waypoint.  The
distances are oracle fields read OFFLINE to LABEL frames, exactly as the
diagnostician's §2A probe does; no coordinate appears anywhere in the code
below and none is readable at eval time).

Per-frame discrimination on ROAD legs, `band_fraction > 0.03`:

    in-window (<= 2 000 cm)   fires on 77.3 % of frames   (n = 75)
    2 000-4 000 cm            fires on 29.3 %             (n = 75)
    > 4 000 cm                fires on 25.9 %             (n = 532)

Per-leg, scoring the FIRST rising edge of each of the 45 ROAD legs at the
parent's own loop cadence (one reading per perceive call) — this is the
shipped rule:

    threshold  first-edge IN   2-4 k   > 4 k   never
      0.02         17            5      21       2
      0.03       **25**          4      12       4      <-- shipped
      0.05         19            5      13       8
      0.08         16            8      10      11
      0.12         14            9       7      15

25/45 = 56 % of ROAD legs raise their first edge inside the window, against the
live gate's measured 29 % per-ask and the run's actual ROAD pass rate of
0.26.  The 0.03 peak is broad in the band geometry as well (every row band from
0.40-0.65 through 0.50-0.75 scores 22-25), so what is encoded is "a few percent
of the forward corridor is roadway", not a knife-edge constant.  Allowing the
first TWO edges to be claimable raises leg coverage 25/45 -> 31/45 for 7 extra
far-field opportunities, which the identity conjunct then filters at its
measured 39 % far-field pass rate.

Requiring the leg to be seen CLOSED before the first edge counts ("arming")
was measured and REJECTED: 24/45 instead of 25/45, i.e. it costs a leg and buys
nothing — a ROAD leg does not generally start inside a junction.

FINISH legs are NOT owned by this channel, and that is measured too: the first
edge of a FINISH leg lands in-window 0/17 times and beyond 4 000 cm 16/17, at
every threshold tried.  A FINISH leg *begins* in the intersection the TURNING
leg just committed in, so the cue is true immediately and says nothing.  The
router therefore owns intersection-stop ROAD legs only; ORIENT (cardinal
burst), TURNING (turn-approach latch) and FINISH (frontage question) are
untouched.

Deployment-honest inputs only: `obs['object_mask']` and the static
`task_spec` instruction text.  No coordinate, no gt field, no analog yaw, no
`subtask_max_steps`, no VLM call, no token, no env substep.
"""
from __future__ import annotations
import re
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

from .viewpoint_controller import LEG_ROAD, leg_type  # noqa: E402


# --- the segmentation palette ------------------------------------------------
# Re-probed this iter on 1 326 archived `object_mask` frames of the parent run.
# Four classes dominate every frame and all four are EXACT colours:
#   (255,255,0) sidewalk/crosswalk (~48 %/frame)   (128,0,128) buildings
#   (0,255,0)   sky / (0,128,0) trees              (0,127,255) ROADWAY  <- the cue
# Matched in BOTH channel orders: the roadway colour is not palindromic (unlike
# the purple building class the existing side veto leans on) and nothing in the
# sanctioned surface guarantees RGB over BGR.
_ROADWAY_CLASS_RGB = np.array([0, 127, 255], dtype=np.int16)
_ROADWAY_CLASS_BGR = np.array([255, 127, 0], dtype=np.int16)
_MASK_CLASS_TOL = 20          # the palette is exact; the tol absorbs re-encoding

# The forward travel corridor, as FRACTIONS of the frame — no pixel literal, no
# centimetre, nothing map-tuned.
_BAND_ROW = (0.45, 0.70)      # the ground ahead, below the horizon
_BAND_COL = (0.30, 0.70)      # the lane the body is walking in

# Operating point. See the module docstring for the threshold sweep.
ARRIVAL_FRAC = 0.03

# How many rising edges of one leg may be CLAIMED as arrival.  Measured:
# edge #1 is in-window on 25/45 legs, edge #2 on a further 6, edge #3 on 2 more
# for 5 extra far-field opportunities.  Two is where the marginal in-window leg
# stops being cheap.
MAX_CLAIMABLE_CROSSINGS = 2

# A leg may not claim arrival before the body has translated at all: the edge
# detector is stateless and a leg that happens to start on a corner would
# otherwise commit at `forward_streak == 0`.  Measured: exactly 1 of the 45 ROAD
# legs raised its first edge on the leg's very first reading.
GEO_MIN_TRANSLATIONS = 1

# The ROAD template is "Continue straight and stop at the intersection …".
# Verified on this run's own traces: 40/40 distinct ROAD instructions contain
# the word, and no FINISH instruction does.
_INTERSECTION_RE = re.compile(r'\bintersections?\b', re.IGNORECASE)


def is_intersection_stop_leg(instruction: Any) -> bool:
    """True iff this leg's stop condition is "the intersection ahead of me".

    Two conjuncts, each excluding a leg type that is already owned by a channel
    that works: `leg_type` must be ROAD (which rules out ORIENT's cardinal
    burst, TURNING's approach latch and FINISH's "stop and face" template —
    "Turn left at the intersection" also names an intersection but is graded on
    HEADING), and the word *intersection* must actually be present.
    """
    if not isinstance(instruction, str) or not instruction:
        return False
    if leg_type(instruction) != LEG_ROAD:
        return False
    return _INTERSECTION_RE.search(instruction) is not None


def roadway_band_fraction(obs: Any) -> Optional[float]:
    """Fraction of the forward travel corridor that is roadway class.

    Returns None — "this channel cannot be evaluated", deliberately distinct
    from 0.0 — when the mask is absent or malformed, so the caller hands the
    leg back to the VLM junction question instead of silently reading a missing
    sensor as "no intersection".
    """
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
    c0, c1 = int(_BAND_COL[0] * w), int(_BAND_COL[1] * w)
    band = a[r0:r1, c0:c1, :3]
    if band.size == 0:
        return None
    b = band.astype(np.int16)
    hit = ((np.abs(b - _ROADWAY_CLASS_RGB).sum(axis=-1) < _MASK_CLASS_TOL)
           | (np.abs(b - _ROADWAY_CLASS_BGR).sum(axis=-1) < _MASK_CLASS_TOL))
    return float(hit.mean())


@dataclass
class RoadGeometry:
    """Per-leg rising-edge state machine over `roadway_band_fraction`.

    One instance per episode.  `update` is driven once per loop iteration from
    `run_episode`, on the SAME frame the perceiver sees — that is the cadence
    the 25/45 operating point above was measured at, and sampling it denser
    (e.g. inside the action burst) would move the detector off its measured
    point, so the harness deliberately does not.

    The detector proper is stateless (`frac > ARRIVAL_FRAC`); the state machine
    exists to turn it into an EVENT.  Standing in a junction makes the raw
    predicate true for several consecutive readings, and the measured operating
    point is the RISING EDGE, not the plateau: all frames inside the first two
    above-threshold runs score 52 in-window against 59 far, while their rising
    edges alone score 31 against 19.  So only the edge is claimable.
    """

    # per-leg state, keyed by the structural subtask index
    _above: Dict[int, bool] = field(default_factory=dict)
    _crossings: Dict[int, int] = field(default_factory=dict)
    _owned: Dict[int, bool] = field(default_factory=dict)
    _frac: Dict[int, Optional[float]] = field(default_factory=dict)
    # Whether a CLAIMABLE rising edge is live right now.  Set by `update`, read
    # by the gate in the same loop iteration (both commit sites may read it),
    # cleared by the next `update`.
    _event: Dict[int, bool] = field(default_factory=dict)
    # telemetry for the diagnostician; never read by the agent
    stats: Dict[str, int] = field(default_factory=dict)

    def _bump(self, key: str) -> None:
        self.stats[key] = int(self.stats.get(key, 0)) + 1

    def summary(self) -> str:
        """One compact note token for the trace: this channel's own ledger."""
        if not self.stats:
            return "GEO(no-readings)"
        return "GEO(" + ",".join(
            f"{k}={self.stats[k]}" for k in sorted(self.stats)) + ")"

    # ---- the sensor stage ------------------------------------------------
    def update(self, obs: Any, instruction: Any, cur_idx: int) -> Dict[str, Any]:
        """Read one frame.  Called once per loop iteration from `run_episode`.

        Returns a small dict for the harness's own bookkeeping / note label.
        """
        self._event[cur_idx] = False
        owned = is_intersection_stop_leg(instruction)
        self._owned[cur_idx] = owned
        if not owned:
            return {"owned": False, "event": False, "frac": None,
                    "crossings": int(self._crossings.get(cur_idx, 0))}

        frac = roadway_band_fraction(obs)
        self._frac[cur_idx] = frac
        if frac is None:
            # Missing sensor is NOT evidence of absence: the leg falls back to
            # the VLM junction question for this iteration.
            self._bump("no_mask")
            return {"owned": True, "event": False, "frac": None,
                    "crossings": int(self._crossings.get(cur_idx, 0))}

        above = frac > ARRIVAL_FRAC
        prev = bool(self._above.get(cur_idx, False))
        self._above[cur_idx] = above
        self._bump("above" if above else "below")
        if above and not prev:
            n = int(self._crossings.get(cur_idx, 0)) + 1
            self._crossings[cur_idx] = n
            self._bump("edge")
            if n <= MAX_CLAIMABLE_CROSSINGS:
                self._event[cur_idx] = True
                self._bump("claimable")
            else:
                self._bump("edge_over_cap")
        return {"owned": True, "event": bool(self._event.get(cur_idx)),
                "frac": frac, "crossings": int(self._crossings.get(cur_idx, 0))}

    # ---- what the gate asks ----------------------------------------------
    def event_live(self, cur_idx: int) -> bool:
        """Is a claimable rising edge live on this loop iteration?"""
        return bool(self._event.get(cur_idx))

    def place_verdict(
        self,
        instruction: Any,
        cur_idx: int,
        forward_streak: int,
    ) -> Optional[bool]:
        """The POSITION conjunct of the `-1` license.  Three-valued.

          None  — this channel does not own the leg, or the mask was missing on
                  this iteration.  The caller falls through to `_JUNCTION_SYS`
                  exactly as before: this channel can only ever REPLACE the VLM
                  question on legs it can actually decide.
          True  — a claimable intersection edge is HERE, now.
          False — owned, and the junction is not here.  On a leg whose stop
                  condition IS a junction, "I have not crossed into one" is a
                  positive statement about the world, not a missing answer — so
                  it vetoes, and it does so for free, without spending the
                  identity call.
        """
        if not self._owned.get(cur_idx):
            return None
        if self._frac.get(cur_idx) is None:
            return None
        if int(forward_streak) < GEO_MIN_TRANSLATIONS:
            return False
        return bool(self._event.get(cur_idx))
