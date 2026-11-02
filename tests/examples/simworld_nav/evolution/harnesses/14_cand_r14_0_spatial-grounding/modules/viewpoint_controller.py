"""ViewpointController — the 8th module: POSE arbitration.

Why this node exists (iter 6, `spatial-grounding`)
--------------------------------------------------
Until now the harness had exactly one arrival predicate — "does the forward
RGB frame look like the hint photo?" — and it applied it to three legs that
the env grades on three different quantities:

  * MOVE_ALONG_ROAD      graded on L1 POSITION only (+-2000)
  * TURN_AT_INTERSECTION graded on YAW *and* a position box around the junction
  * REACH_DESTINATION    graded on POSITION *and* YAW ("...stop and face that
                         building")

Every one of those criteria is about WHERE THE BODY IS AND WHERE IT POINTS.
The recognizer only knows what the camera happens to be looking at, and the
harness never had a component whose job was to put the body somewhere before
asking. Five prior candidates re-worked the recognizer (evidence set, look
schedule, criterion wording, in-burst sampling, confirmation counts); none
beat the anchor, because the binding constraint is not what the model is
asked — it is the pose it is asked FROM:

  * the landmark named "on your left/right" is ~90 deg off the travel axis and
    outside the FOV, so the forward-facing question cannot see it (measured
    recall 21%, 9/43 in-window fires);
  * the turn burst fires `[rotate, -1]` on the first frame of the TURNING leg
    with zero forward motion, landing 78-980 cm outside the junction box;
  * the FINISH leg is graded on yaw and nothing in the harness ever turned
    toward the building before committing.

So this module owns three pose decisions, and only pose decisions. It makes
no VLM call and holds no opinion about appearance — `CompletionChecker` stays
the sole authority on "is this the landmark?"; this node decides *from where*
that question gets asked and *in what pose* the resulting `-1` is emitted.

  (1) approach-before-turn  — intercept the zero-motion turn burst, hold it,
      walk into the junction first, release it when the turn side opens.
  (2) face-before-look      — on a side-referenced leg, physically yaw toward
      the named side at the arrival-look cadence so the recognizer sees the
      viewpoint the reference photo was captured from.
  (3) face-before-commit    — a FINISH `-1` is only ever emitted from a pose
      that faces the named side.

Honest inputs only: the STATIC `task_spec['subtasks'][i]['instruction']` text,
`obs['object_mask']` (a sanctioned observation channel the perceiver already
reads), and this leg's own executed-action bookkeeping. No coordinates, no
`gt.json`, no analog yaw, no oracle.
"""
from __future__ import annotations
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, List, Optional
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
_ROTATE_LEFT = 1
_ROTATE_RIGHT = 2
_DONE = -1


# ---- leg taxonomy ----------------------------------------------------------
# Derived from the STATIC instruction string alone. Verified against all 199
# `hard/` task_config.json files: leg 0 always matches _ORIENT_RE (199/199);
# leg 2 always matches _TURN_RE and never _ORIENT_RE (181/181); the terminal
# leg always matches _FINISH_RE (181/181 four-leg tasks, and the 18 two-leg
# tasks whose leg 1 is a destination leg); the ROAD leg matches none of them
# ("Continue straight and stop at the intersection with <building> on your
# left.") and falls through to the default.
LEG_ORIENT = "ORIENT"
LEG_TURN = "TURN"
LEG_FINISH = "FINISH"
LEG_ROAD = "ROAD"

_ORIENT_RE = re.compile(
    r'\b(?:face|facing|faced)\b'
    r'(?:\s+(?:to|toward|towards))?'
    r'(?:\s+the)?'
    r'\s+(east|west|north|south)\b',
    re.IGNORECASE,
)
_TURN_RE = re.compile(
    r'\b(?:turn\s+(left|right)|make\s+a\s+(left|right)\s+turn)\b',
    re.IGNORECASE,
)
_FINISH_RE = re.compile(r'\bstop\s+and\s+face\b', re.IGNORECASE)
_SIDE_LEFT_RE = re.compile(r'\b(?:on|to)\s+(?:your|the|my)\s+left\b', re.IGNORECASE)
_SIDE_RIGHT_RE = re.compile(r'\b(?:on|to)\s+(?:your|the|my)\s+right\b', re.IGNORECASE)


def leg_type(instruction: Any) -> str:
    """Classify a leg from its instruction text. Order matters.

    ORIENT is tested first: "Turn to face east" contains the word `turn` but
    is an orientation leg, and its closed-form cardinal burst is the one path
    in this harness that passes 50/50 — it must never be intercepted.
    """
    if not isinstance(instruction, str) or not instruction:
        return LEG_ROAD
    if _ORIENT_RE.search(instruction):
        return LEG_ORIENT
    if _TURN_RE.search(instruction):
        return LEG_TURN
    if _FINISH_RE.search(instruction):
        return LEG_FINISH
    return LEG_ROAD


def named_side(instruction: Any) -> Optional[str]:
    """Parse 'on your left' / 'on your right' out of the instruction."""
    if not isinstance(instruction, str):
        return None
    if _SIDE_LEFT_RE.search(instruction):
        return "left"
    if _SIDE_RIGHT_RE.search(instruction):
        return "right"
    return None


def turn_direction(instruction: Any) -> Optional[str]:
    """Parse the relative-turn direction ('left'/'right') of a TURN leg."""
    if not isinstance(instruction, str):
        return None
    m = _TURN_RE.search(instruction)
    if m is None:
        return None
    return (m.group(1) or m.group(2)).lower()


def _rotate_code(side: str) -> int:
    return _ROTATE_LEFT if side == "left" else _ROTATE_RIGHT


def _restore_code(side: str) -> int:
    return _ROTATE_RIGHT if side == "left" else _ROTATE_LEFT


# ---- the lateral segmentation band ----------------------------------------
# One geometric primitive serves both pose cues. It reads the SAME band that
# `completion_checker._named_side_has_structure` already uses (rows 0.30-0.62,
# the outer 25% of the columns on one side) off `obs['object_mask']`, which is
# uint8 (600, 720, 3) per the workspace probe, building class = purple.
#
# Deliberately NOT depth: `obs['depth']` is a per-frame min-max normalised JET
# rendering, so it carries no absolute scale — the prior turn-to-face lever was
# gated on a depth predicate (`_side_band_near`) whose own comments record
# `near_frac == 0.000` on 24/24 real frames, and it consequently fired 2 times
# in 50 episodes. Segmentation is scale-free and measurable: the side-building
# fraction runs 0.60-0.72 on an in-window frame and ~0.00 where the cross
# street opens.
_BAND_ROW = (0.30, 0.62)
_BAND_COL_FRAC = 0.25
_BUILDING_RGB = np.array([128, 0, 128], dtype=np.int16)
_CLASS_TOL = 60


def side_building_fraction(obs: Any, side: Optional[str]) -> Optional[float]:
    """Fraction of building-class pixels in the lateral band on `side`.

    Returns None when the mask is missing or malformed — every caller treats
    None as "no information" and falls back to its own motion floor, so a
    missing channel can never silently tighten or loosen a pose decision.
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
    bld = (np.abs(band.astype(np.int16) - _BUILDING_RGB).sum(axis=-1) < _CLASS_TOL)
    return float(bld.mean())


# ---- tuning ----------------------------------------------------------------
# (1) approach-before-turn. The floor and the cap bracket the measured sweep
#     over the 12 episodes that reached a TURNING leg: N extra forwards before
#     the burst gave 5/12 (N=0), 8/12 (N=1), 11/12 (N=2), 11/12 (N=3), and
#     decayed past N=5 as the agent walked out the far side of the box. Both
#     ends of the bracket sit on the plateau, so the adaptive cue only decides
#     WHERE ON the plateau to land, not whether to overshoot. NOTE for the next
#     iter: 2 and 3 are fitted on 12 samples — treat them as a fit, not a law.
APPROACH_FLOOR_FORWARDS = 2
APPROACH_CAP_FORWARDS = 3
# Below this the named side reads OPEN — the facade line has ended and the
# cross street is beside us, i.e. we are in the junction.
TURN_SIDE_OPEN_MAX_FRAC = 0.10

# (2) face-before-look. Cadence mirrors the arrival-look schedule that already
#     governs the forward-facing check (streak floor 4, spacing 4) so the
#     lateral look is a VIEWPOINT swap at the same rhythm, not an extra
#     mechanism firing on its own clock. A scan costs 2 in-place turns (no
#     translation, so a position-graded leg is untouched) plus one VLM call,
#     and commits only on an explicit Match=1 — a wrong look is never fatal.
K_FACE_STREAK = 4
K_FACE_INTERVAL = 4
MAX_FACE_SCANS_PER_LEG = 4
# Veto-only: at this little building class on the named side there is nothing
# there to turn and look at. Fails OPEN (scan allowed) when the mask is absent.
SIDE_STRUCTURE_MIN_FRAC = 0.02


@dataclass
class ViewpointController:
    """Per-episode pose arbiter. All state is per-leg and resets on leg change.

    Public API (consumed by `agent.run_episode`):
      - gate_burst(burst, instruction, obs)  -> Optional[List[int]]
      - request_face_scan(obs, instruction, forward_streak) -> Optional[str]
      - restore_code(side) / note_faced(side) / note_restored()
      - allow_forward_commit(instruction) -> bool
      - commit_pose_prefix(instruction) -> List[int]
      - facing_side -> Optional[str]
    """

    # Leg identity = the instruction string (the harness has no other honest
    # per-leg key; `current_subtask` is structural and already read for the
    # index, but the string is what every other module resets on).
    _leg_key: Optional[str] = field(default=None)

    # (1) approach-before-turn
    _approach_forwards: int = field(default=0)
    _pending_turn_burst: Optional[List[int]] = field(default=None)
    _turn_released: bool = field(default=False)

    # (2) face-before-look
    _face_scans: int = field(default=0)
    _last_scan_streak: Optional[int] = field(default=None)

    # (3) face-before-commit. None == the body is on its travel heading;
    # 'left'/'right' == the body is yawed 90 deg toward that side.
    _facing_side: Optional[str] = field(default=None)

    # ---- leg bookkeeping ---------------------------------------------
    def _begin_leg_if_new(self, instruction: Any) -> None:
        if instruction == self._leg_key:
            return
        self._leg_key = instruction
        self._approach_forwards = 0
        self._pending_turn_burst = None
        self._turn_released = False
        self._face_scans = 0
        self._last_scan_streak = None
        # A leg boundary is a `-1`; every path that emits one either never
        # rotated (travel heading) or restored the heading before committing,
        # except the FINISH scan-commit — which ends the episode. Resetting to
        # None is therefore the physically correct state on a new leg.
        self._facing_side = None

    @property
    def facing_side(self) -> Optional[str]:
        return self._facing_side

    def leg_type(self, instruction: Any) -> str:
        return leg_type(instruction)

    # ---- (1) approach-before-turn ------------------------------------
    def gate_burst(
        self,
        burst: Optional[List[int]],
        instruction: Any,
        obs: Any = None,
    ) -> Optional[List[int]]:
        """Pose-gate ActivePerception's short-circuit burst.

        Pass-through for every leg type except TURN. On a TURN leg the
        `[rotate, -1]` burst is CAPTURED rather than executed: the instruction
        says "turn right AT THE INTERSECTION" and the burst turns wherever the
        previous leg happened to stop, which in 7/7 traced failures was
        78-980 cm short of the junction box (yaw error 0 in all 7 — the
        rotation was always right, only the position was wrong).

        While the burst is held we emit single forwards along the pre-turn
        heading and re-ask the lateral cue each step. The burst is released
        when the turn side OPENS (the facade line on that side gives way to the
        cross street) past the floor, or unconditionally at the cap.

        ActivePerception latches its relative-turn burst to fire once per
        instruction, so once captured the burst can only come back from HERE —
        the controller owns it until release. `_turn_released` makes the
        release idempotent; afterwards the leg behaves exactly as before.
        """
        self._begin_leg_if_new(instruction)

        if leg_type(instruction) != LEG_TURN:
            return burst

        if self._turn_released:
            return burst

        # Capture a burst that commits the leg (`-1`); let any other burst
        # (traffic idle, anti-spin) through untouched when nothing is held.
        if (self._pending_turn_burst is None
                and burst
                and _DONE in burst):
            self._pending_turn_burst = list(burst)
        if self._pending_turn_burst is None:
            return burst

        # --- approach phase: we are holding the commit -----------------
        if self._approach_forwards >= APPROACH_CAP_FORWARDS:
            return self._release_turn_burst()

        if self._approach_forwards >= APPROACH_FLOOR_FORWARDS:
            frac = side_building_fraction(obs, turn_direction(instruction))
            if frac is not None and frac <= TURN_SIDE_OPEN_MAX_FRAC:
                return self._release_turn_burst()

        self._approach_forwards += 1
        return [_FORWARD]

    def _release_turn_burst(self) -> List[int]:
        self._turn_released = True
        burst = self._pending_turn_burst or [_DONE]
        self._pending_turn_burst = None
        return list(burst)

    # ---- (2) face-before-look ----------------------------------------
    def request_face_scan(
        self,
        obs: Any,
        instruction: Any,
        forward_streak: int,
    ) -> Optional[str]:
        """Return 'left'|'right' iff the harness should yaw to face the named
        side and run the arrival look FROM THERE, else None.

        This replaces a trigger that was permanently false. The mechanism it
        gates is the right one — the reference photo is a robot view captured
        FACING the building, so a match needs the building head-on, and an
        in-place yaw preserves position so a position-graded leg is unharmed —
        but it used to be conditioned on a depth proximity predicate that never
        fired (2 firings in 50 episodes). The trigger is now the arrival-look
        CADENCE (a motion floor plus spacing, the schedule the forward-facing
        check already runs on), with the segmentation band as a veto only.

        The cap is charged HERE, on the decision to look, so the cost is
        bounded even if the turn or the subsequent VLM call fails.
        """
        self._begin_leg_if_new(instruction)

        lt = leg_type(instruction)
        if lt not in (LEG_ROAD, LEG_FINISH):
            return None
        side = named_side(instruction)
        if side is None:
            return None
        if self._facing_side is not None:
            return None                      # already off-axis; not re-entrant
        if self._face_scans >= MAX_FACE_SCANS_PER_LEG:
            return None
        if forward_streak < K_FACE_STREAK:
            return None
        if (self._last_scan_streak is not None
                and forward_streak - self._last_scan_streak < K_FACE_INTERVAL):
            return None
        frac = side_building_fraction(obs, side)
        if frac is not None and frac < SIDE_STRUCTURE_MIN_FRAC:
            return None                      # nothing there to look at
        self._face_scans += 1
        self._last_scan_streak = forward_streak
        return side

    @staticmethod
    def restore_code(side: str) -> int:
        """The rotation that undoes a face-turn toward `side`."""
        return _restore_code(side)

    @staticmethod
    def face_code(side: str) -> int:
        """The rotation that faces `side`."""
        return _rotate_code(side)

    def note_faced(self, side: str) -> None:
        self._facing_side = side

    def note_restored(self) -> None:
        self._facing_side = None

    # ---- (3) face-before-commit --------------------------------------
    def allow_forward_commit(self, instruction: Any) -> bool:
        """False on a FINISH leg: a `-1` may not be injected from the travel
        heading there.

        REACH_DESTINATION is the only leg graded on yaw as well as position,
        and the forward-facing arrival look is exactly the mechanism that
        produced every fatal premature finish in the traces (7 commits, 7.8k-
        58.8k cm out, 6 of them additionally 270 deg off the required heading)
        — down a straight city street the destination facade is "clearly
        visible and prominent" from tens of km away, so the question has no
        stopping power on this leg. The lateral face-scan path (which commits
        only from a pose that already satisfies the yaw criterion, and only on
        an explicit Match=1 at a viewpoint that can only exist when the
        building is actually beside the agent) is left as the commit route.
        """
        return leg_type(instruction) != LEG_FINISH

    def commit_pose_prefix(self, instruction: Any) -> List[int]:
        """Rotations to prepend to a `-1` so the commit pose is legal.

        On a FINISH leg the instruction ends "...on your right, stop and face
        that building": the grader wants the body turned toward the landmark.
        When the commit arrives from the travel heading (a planner-driven `-1`
        that survived the cross-check) we prepend the single quarter-turn.
        When it arrives from the face-scan path the body is already there and
        this returns []. Called from the ONE place in the harness where a `-1`
        is materialised, so the pose can never be composed twice.
        """
        if leg_type(instruction) != LEG_FINISH:
            return []
        side = named_side(instruction)
        if side is None or self._facing_side == side:
            return []
        return [_rotate_code(side)]
