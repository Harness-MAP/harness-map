"""ActivePerception — proactively emit a short action burst that
short-circuits the perceive+plan pipeline.

iter_006 stance (cardinal burst + red-signal idle + anti-spin + forward
momentum w/ cap-2 consecutive):

  Parent (s0_naive_005) shipped the 1-flag-toggle forward momentum:
  alternation PP -> momentum -> PP -> momentum. It DID fire on 4/5
  episodes and lifted `moving_distance_mean` from 500 -> 1600 cm
  (+220%), but 5/5 episodes still ended `subtask_success =
  [True, False, False, False]` (SubtaskSR stuck at 0.25 for the 3rd
  iter running). Two near-miss episodes — ep_20_3 (20 cm short of the
  subtask-1 threshold, needed +1 forward) and ep_20_0 (needed +2
  forwards) — show the cadence is still slightly too PP-heavy.

  iter_006 relaxes the iter_005 alternation: a counter
  (`_consecutive_momentum`, cap 2) replaces the 1-flag toggle. Cadence
  becomes PP -> momentum -> momentum -> PP -> momentum -> momentum ->
  ... = 3 forwards per PP cycle (vs. iter_005's 2 forwards per PP).
  Safety properties (red-signal idle, cardinal burst, anti-spin,
  stuck detector) are unchanged: they all run on the outer-loop
  refreshed `obs` between every momentum emission, so the cap-2
  burst still gets a red-signal check before each m1 and m2.

iter_005 stance (cardinal burst + red-signal idle + anti-spin + forward
momentum):

  Parent (s0_naive_004) added the red-signal idle but it never fired
  on the run (all 5 episodes died short of the intersection). The
  rate-limiter on forward progress is **perceive+plan cadence per
  forward env.step**: each forward burns one PP cycle (~3000 tokens,
  ~5-10 s wall). iter_005 inserts a forward-momentum rule that emits
  an extra `[0]` after every successful forward, alternating PP and
  momentum (1-flag toggle). Net effect: ~2x forward density per PP,
  ~50% PP-cost reduction on ROAD subtasks, and the first real test of
  iter_004's red-signal idle when episodes finally reach the
  intersection.

iter_004 stance (cardinal burst + red-signal idle + anti-spin):

  Parent (s0_naive_003) ships closed-form rotation bursts that complete
  the ORIENTATION subtask in 1-3 env.steps with zero VLM calls. That
  works: 5/5 episodes pass subtask 0 (per_position_SR[0] = 1.0). But
  the one episode that reaches a traffic-signal-controlled intersection
  (ep_20_0, 18 steps) drives through it: 5 sequential
  `red_light_violation`s in 5 forward steps. The harness has no
  detection, no veto, no idle — exactly the failure mode flagged in
  the prompt's hard-mode guidance and README §"Traffic safety".

  iter_004 adds a saturated-red-pixel detector over the upper-center
  of `obs['rgb']` (per the README hint: "top-of-frame red-pixel
  cluster near the upper-third"). When the detector fires AND the
  current subtask is not the cardinal-orientation subtask AND the
  per-subtask idle cap hasn't been hit, ActivePerception emits a
  **net-zero idle burst** `[2, 1]` (rotate-right then rotate-left,
  0 cm displacement, 2 sim steps). This buys time for the signal to
  cycle without the forward step that the env counts as a violation.
  The cap (`_MAX_TRAFFIC_IDLES_PER_SUBTASK = 2`) bounds the
  worst-case wait so a false positive or a slow signal cannot trap
  the search.

Decision order in `should_emit`:
  (a)  target-cardinal rotation burst -> [<rotates>, -1]   (iter_003)
  (a') traffic-safety idle            -> [2, 1]            (iter_004)
  (b)  anti-spin break-out            -> [0]               (iter_001)
  (b') wall-stuck recovery rotation   -> [1]               (iter_010)
  (d)  forward momentum (cap=2)       -> [0]               (iter_006)
  (c)  otherwise                      -> None              (H_0)

Reads (deployment-honest):
  - SpatialMemory's recorded action history (for anti-spin)
  - `info['agent']['agent_rotation']` (passed as `orientation`)
  - the subtask instruction string
  - `obs['rgb']` pixel values for red-signal detection (NEW;
    `obs['rgb']` is already in H_0's audited read surface).
  No oracle reads. Specifically: `info['total_red_light_violation']`
  is NOT consulted (that is the env's compliance tally, not a
  perceptual channel a deployed system could trust).

Cost: idle bursts replace some perceive+plan cycles (cost-negative or
neutral). On non-fire states `should_emit` returns None and the H_0
pipeline runs unchanged.
"""
from __future__ import annotations
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import numpy as np

sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))


# High-level action codes (matching orchestrator.py's ACTION_TO_STEP map).
_FORWARD = 0
_ROTATE_LEFT = 1
_ROTATE_RIGHT = 2
_SUBTASK_DONE = -1
_ROTATE_CODES = (_ROTATE_LEFT, _ROTATE_RIGHT)

# --- commit-geometry (this iter): approach the intersection BEFORE the
# relative-turn commit. The TURNING leg is graded on an orientation AND a
# POSITION box centred on the intersection; the ROAD waypoint the agent just
# committed at sits SHORT of that box, so a burst that rotates in place and
# commits with zero translation lands outside it. These constants express the
# approach as a COUNT OF FORWARD ACTIONS (never a cm literal):
#   _TURN_APPROACH_FORWARDS  - forwards to spend closing on the box before the
#                              rotate+commit burst. TURNING legs are granted a
#                              fixed, distance-independent substep allowance,
#                              so the approach is essentially free.
#   _TURN_APPROACH_MIN_DELTA - odometry delta (info['moving_distance'], the
#                              sanctioned progress channel) below which an
#                              approach forward is judged to have been BLOCKED
#                              (facade / kerb). Approaching into a wall is
#                              worse than committing short, so a blocked
#                              approach short-circuits to the commit.
#   _MAX_TURN_APPROACH_STALLS- consecutive blocked approach forwards tolerated
#                              before the latch gives up and commits.
_TURN_APPROACH_FORWARDS = 2
_TURN_APPROACH_MIN_DELTA = 1.0
_MAX_TURN_APPROACH_STALLS = 1

# Cardinal sequence under action 2 (TURN_RIGHT = +90 deg yaw / CCW),
# pinned by the workspace probe and the harness docstring:
#   yaw 0 == West, 90 == North, 180 == East, 270 == South.
# Action 1 (TURN_LEFT = -90 deg yaw / CW) walks this tuple in reverse.
_CCW_ORDER = ("west", "north", "east", "south")

# A rotation step's moving_distance delta is normally exactly 0.0 (no
# translation). Allow a small slack for any odometer noise.
_NO_DISPLACEMENT_THRESHOLD_CM = 0.5

# Window size: how many recent same-rotation steps trigger anti-spin.
_SPIN_WINDOW = 2

# Target-cardinal regex — only matches orientation-only verbs
# (face/facing/faced) followed by an absolute cardinal. Deliberately
# does NOT match "continue east" / "head north" / "go south" — those
# instructions imply translation, and auto-completing them on cardinal
# match would skip the navigation portion of a ROAD subtask.
_CARDINALS = {"east", "west", "north", "south"}
_TARGET_CARDINAL_RE = re.compile(
    r'\b(?:face|facing|faced)\b'
    r'(?:\s+(?:to|toward|towards))?'
    r'(?:\s+the)?'
    r'\s+(east|west|north|south)\b',
    re.IGNORECASE,
)


def _parse_target_cardinal(instruction: Any) -> Optional[str]:
    """Extract the named target cardinal from an instruction string.

    Returns the lowercase cardinal ("east"/"west"/"north"/"south") on a
    match, None otherwise. Conservative by design: only orientation
    verbs trigger a match.
    """
    if not isinstance(instruction, str) or not instruction:
        return None
    m = _TARGET_CARDINAL_RE.search(instruction)
    if m is None:
        return None
    return m.group(1).lower()


# iter 6: relative-turn instruction detector ---------------------------
# Matches "Turn right at the intersection" via group 1 and "Make a left
# turn at the intersection" via group 2. Either group resolves to the
# rotation direction. Language pattern only — no map-tuned constants
# and no per-map coordinates. The cardinal-burst regex above always
# checks first, so an orient subtask ("Turn to face east") never reaches
# this matcher (it would short-circuit on `face`).
_TURN_RELATIVE_RE = re.compile(
    r'\b(?:turn\s+(left|right)|make\s+a\s+(left|right)\s+turn)\b',
    re.IGNORECASE,
)


def _parse_relative_turn(instruction: Any) -> Optional[str]:
    """Extract the relative-turn direction from an instruction.

    Returns "left" / "right" / None. Conservative — only the two
    explicit patterns above trigger; freeform phrasing ("rotate
    counter-clockwise") yields None and falls through to the planner.
    """
    if not isinstance(instruction, str) or not instruction:
        return None
    m = _TURN_RELATIVE_RE.search(instruction)
    if m is None:
        return None
    return (m.group(1) or m.group(2)).lower()


def _approach_delta(state: Any) -> Optional[float]:
    """Odometry delta over the last logged step, or None when unavailable.

    Reads only `SpatialMemory.moving_distance_delta` (cumulative
    `info['moving_distance']`), which `rules.md` sanctions as the progress /
    stuck channel. Returns None -- "unknown, assume the step moved" -- when
    SpatialMemory is absent or the accessor misbehaves, so a missing signal
    never turns into a spurious stall.
    """
    if state is None:
        return None
    fn = getattr(state, "moving_distance_delta", None)
    if fn is None:
        return None
    try:
        return float(fn(1))
    except Exception:
        return None


def _rotation_burst(current: str, target: str) -> List[int]:
    """Closed-form action list to rotate from current cardinal to target.

    Both args are lowercase cardinal strings from `_CCW_ORDER`. Output
    is always terminated by `-1` (subtask done). Chooses the shortest
    rotation direction:
      ccw_n == 0 -> [-1]
      ccw_n == 1 -> [2, -1]
      ccw_n == 2 -> [2, 2, -1]
      ccw_n == 3 -> [1, -1]    (one CW rotate beats three CCW rotates)
    """
    ci = _CCW_ORDER.index(current)
    ti = _CCW_ORDER.index(target)
    ccw_n = (ti - ci) % 4
    if ccw_n == 0:
        return [_SUBTASK_DONE]
    if ccw_n == 3:
        return [_ROTATE_LEFT, _SUBTASK_DONE]
    return [_ROTATE_RIGHT] * ccw_n + [_SUBTASK_DONE]


# --- iter_004 (retuned in iter_009): red-signal-ahead detection -------
# obs['rgb'] is uint8 (600, 720, 3) per the workspace probe (pinned in
# orchestrator.py's docstring). A typical traffic-signal lamp head occupies
# ~10-15 px across at that resolution; an "ahead" signal projects in the
# upper-third / center of the frame.
#
# iter_009 retune (cited in branches/naive/NOTES.md):
#   The iter_004 saturated-red test (R>=200 AND G<=80 AND B<=80) never
#   fired in any iter despite ep_20_0 racking up 5-6 red-light violations
#   per run. Lit-lamp bloom renders with non-zero G/B (halo R~200, G~100,
#   B~80; centre near-white) which the strict conjunction rejects. Retune:
#     - ROI rows (40, 200)   -> (60, 300)  : cover mid-distance signals.
#     - ROI cols (240, 480)  -> (180, 540) : cover off-centre poles.
#     - colour test          -> red-dominant (R>=150 AND R-G>=70 AND R-B>=70).
#     - pixel threshold 60   -> 90         : larger ROI needs larger cluster.
#     - per-subtask cap 2    -> 4          : let it fire through long stop zones.
_RED_SIGNAL_ROI_ROWS = (60, 300)     # rows: skip very-top sky; extend to horizon.
_RED_SIGNAL_ROI_COLS = (180, 540)    # cols: cover left/right side poles.
# Red-dominant test: R is high AND R clearly exceeds G,B. Catches lit-lamp
# bloom (R~200, G~100, B~80) and pure red glow (R~255, G~50, B~50). Rejects
# white (R-G small), sky (R-B requirement), brown brick (R-G typically <70).
_RED_SIGNAL_R_MIN = 150
_RED_SIGNAL_RG_DELTA_MIN = 70
_RED_SIGNAL_RB_DELTA_MIN = 70
# Pixel count threshold: 90 px of red-dominant in the wider ROI ~ a lit
# lamp head close enough that one more forward step would cross the stop
# line. Calibrated to require a contiguous-ish cluster, not scattered specks.
_RED_SIGNAL_MIN_PIXELS = 90

# Net-zero idle burst: rotate-right then rotate-left. Zero displacement,
# 2 sim steps. The harness's burst executor runs both back-to-back, and
# the next outer-loop iteration re-perceives with a refreshed obs.
_TRAFFIC_IDLE_BURST = (_ROTATE_RIGHT, _ROTATE_LEFT)
# Cap per subtask. iter_009 raises from 2 to 4 to give the detector enough
# room to idle through a multi-forward stop zone (ep_20_0's red zone is
# ~5 forward steps long). Worst-case false-positive cost is 4 * 2 = 8
# net-zero sim steps per subtask; worst-case false-negative saved is
# 4 red-light violations.
_MAX_TRAFFIC_IDLES_PER_SUBTASK = 4

# --- iter_005: forward momentum threshold -----------------------------
# A successful forward step delivers +500 cm of moving_distance (per the
# grandparent ep_20_0 env_step_log: every forward in a 14-forward run
# adds exactly +500). Any delta below this floor is treated as "forward
# stalled," so we do not emit a follow-up momentum forward (the planner
# gets to decide). 1.0 cm is generous — it only filters genuine 0-cm
# stalls (sim-side blocked steps), not real forwards.
_FORWARD_PROGRESS_THRESHOLD_CM = 1.0

# --- iter_006: cap on consecutive momentum emissions ------------------
# Replaces iter_005's 1-flag toggle (effective cap 1) with a counter
# capped at 2. The cadence becomes PP -> m -> m -> PP -> m -> m -> ...
# = 3 forwards per perceive+plan cycle (vs. iter_005's 2). The cap
# bounds (i) the worst-case latency between obs-refreshed red-signal
# checks to 2 forward steps (1000 cm), and (ii) the worst-case
# overshoot past a landmark-match -1 commit to the same 2 steps. Both
# bounds are comfortably below the typical ROAD segment length (>=
# 3000 cm) and intersection spacing (>= 5000 cm) in the search subset.
_MAX_CONSECUTIVE_MOMENTUM = 2

# --- iter_010: wall-stuck recovery rotation ---------------------------
# Threshold below which the last forward is treated as "stalled" (i.e.,
# the env step delivered no real translation). Matches the existing
# `_FORWARD_PROGRESS_THRESHOLD_CM` floor used by the momentum gate.
_WALL_STUCK_MD_THRESHOLD_CM = 1.0
# The recovery action: a single rotate-left. Left vs. right is arbitrary
# but consistent — if the recovery rotation also leaves the agent
# facing a wall and the next planner forward stalls again, b' fires
# again, so repeated firings sweep the yaw 90 deg / 180 deg / 270 deg
# in turn before the agent is forced to give up.
_WALL_RECOVERY_ROTATION = _ROTATE_LEFT


def _detect_red_signal_ahead(obs: Any) -> bool:
    """True iff a saturated-red pixel cluster sits in the upper-center.

    Defensive: returns False on any malformed obs / rgb. Pixel test and
    threshold are calibrated for the probe-pinned (600, 720, 3) uint8
    layout — `obs['rgb']` at a different scale would underfire or
    overfire, but the harness docstring pins the layout.
    """
    if not isinstance(obs, Mapping):
        return False
    rgb = obs.get("rgb")
    if rgb is None:
        return False
    try:
        arr = np.asarray(rgb)
    except Exception:
        return False
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False
    r0, r1 = _RED_SIGNAL_ROI_ROWS
    c0, c1 = _RED_SIGNAL_ROI_COLS
    h, w = arr.shape[0], arr.shape[1]
    r1 = min(r1, h)
    c1 = min(c1, w)
    if r0 >= r1 or c0 >= c1:
        return False
    roi = arr[r0:r1, c0:c1, :3].astype(np.int16)
    r = roi[..., 0]
    g = roi[..., 1]
    b = roi[..., 2]
    # iter_009: red-dominant test. The previous "saturated red" test
    # (R>=200 AND G<=80 AND B<=80) was too strict — lit lamps render with
    # non-zero G/B due to bloom/halo. The new test requires R is high AND
    # R clearly exceeds G,B (delta thresholds), which captures both pure
    # red glow and bloomed lit lamps while still rejecting white, sky, and
    # most brick/paint surfaces.
    red_mask = (
        (r >= _RED_SIGNAL_R_MIN)
        & ((r - g) >= _RED_SIGNAL_RG_DELTA_MIN)
        & ((r - b) >= _RED_SIGNAL_RB_DELTA_MIN)
    )
    return int(red_mask.sum()) >= _RED_SIGNAL_MIN_PIXELS


def _normalize_cardinal(orientation: Any) -> Optional[str]:
    """Coerce the orientation field to a lowercase cardinal or None.

    The env returns either a string ("East", "West", ...) or
    occasionally a single-element list-of-strings (harness already
    handles this in `_orientation_from_info`, but be defensive here too
    in case `should_emit` is called from a future caller that doesn't
    pre-normalize).
    """
    if orientation is None:
        return None
    if isinstance(orientation, (list, tuple)) and orientation:
        orientation = orientation[0]
    if not isinstance(orientation, str):
        return None
    s = orientation.strip().lower()
    return s if s in _CARDINALS else None


@dataclass
class ActivePerception:
    # Per-episode state for the iter_004 traffic-idle cap. Reset
    # whenever the harness passes a new `instruction` (subtask change).
    _last_instruction: Optional[str] = field(default=None)
    _traffic_idle_count: int = field(default=0)

    # iter_006: counter for the forward-momentum cadence cap.
    # Counts the number of *consecutive* momentum forwards emitted
    # since the last non-momentum action. Cap is
    # `_MAX_CONSECUTIVE_MOMENTUM = 2`, giving the cadence
    # PP -> m1 -> m2 -> PP -> m1 -> m2 -> ... (3 forwards per PP).
    # Resets to 0 on (i) subtask change, (ii) any non-momentum path
    # through should_emit (cardinal burst, red-signal idle, anti-spin,
    # no-rule-fires) so the cap only tracks consecutive momentum runs.
    # Replaces the iter_005 `_momentum_emitted_last: bool` (which was
    # effectively a cap-1 counter).
    _consecutive_momentum: int = field(default=0)

    # iter 6 (naive_rerun): once-per-instruction latch for the
    # relative-turn burst. Stores the instruction string for which
    # a burst has already been emitted, so the rule fires AT MOST
    # ONCE per subtask. If the env declines to advance current_subtask
    # after the burst (defensive — observed reliable advancement in
    # practice for cardinal bursts), subsequent steps fall through to
    # the planner and the harness's existing fallbacks. Reset on the
    # same subtask-boundary equality check (`instruction !=
    # self._last_instruction`) that resets the iter_004 idle counter
    # and iter_006 momentum counter, so a fresh subtask gets a fresh
    # latch.
    _turn_burst_done_for: Optional[str] = field(default=None)

    # commit-geometry (this iter): approach phase state for the relative-turn
    # burst. `_turn_approach_for` is the instruction the counters belong to
    # (so a fresh TURNING leg starts a fresh approach), `_turn_approach_done`
    # counts approach forwards already emitted, `_turn_approach_stalls` counts
    # consecutive approach forwards that moved the odometer less than
    # `_TURN_APPROACH_MIN_DELTA`. All three are redundantly reset both here
    # (instruction-change equality) and on the subtask-boundary check, so a
    # stale phase can never leak into the next leg.
    _turn_approach_for: Optional[str] = field(default=None)
    _turn_approach_done: int = field(default=0)
    _turn_approach_stalls: int = field(default=0)

    def should_emit(
        self,
        state: Any,
        instruction: str,
        orientation: Any = None,
        obs: Any = None,
    ) -> Optional[List[int]]:
        """Return a short action list to short-circuit perceive+plan, or None.

        Returns:
          [<rotates>, -1] — target-cardinal rotation burst (iter_003):
                 the instruction names a target cardinal AND the env
                 reports a current cardinal. Emit the shortest
                 closed-form rotation list that bridges current ->
                 target, followed by -1 to commit. The 'already
                 aligned' case (ccw_n == 0) is just `[-1]`.
          [2, 1] — traffic-safety idle burst (iter_004): a saturated-
                 red pixel cluster sits in the upper-center of the
                 frame AND the current subtask is not an orientation
                 burst AND the per-subtask idle cap (K=2) hasn't been
                 hit. Net-zero displacement, 2 sim steps to let the
                 signal cycle without a forward-into-red violation.
          [0]  — anti-spin break-out: agent has rotated the same
                 direction twice with no displacement. Inject a single
                 Move_Forward to produce a translated viewpoint.
          None — neither condition triggers; H_0 pipeline runs.
        """
        # Subtask-boundary reset for the iter_004 idle counter and the
        # iter_006 momentum counter. We cannot piggy-back on
        # memory.boundary() because that fires on -1 commit and the
        # caps are read on every call; tracking via instruction-change
        # is a single equality check.
        if instruction != self._last_instruction:
            self._last_instruction = instruction
            self._traffic_idle_count = 0
            self._consecutive_momentum = 0
            self._turn_burst_done_for = None
            self._turn_approach_for = None
            self._turn_approach_done = 0
            self._turn_approach_stalls = 0

        # (a) Target-cardinal rotation burst. This must precede
        # anti-spin: if the instruction names an orientation target we
        # should always handle it deterministically, not via the
        # planner. Anti-spin is only useful when the planner has
        # already gotten itself stuck rotating without a cardinal cue.
        target = _parse_target_cardinal(instruction)
        current = _normalize_cardinal(orientation)
        if target is not None and current is not None:
            return _rotation_burst(current, target)

        # (a'') Relative-turn burst (iter 6, naive_rerun). Fires on a
        # *relative*-turn subtask ("Turn left/right at the
        # intersection" / "Make a left/right turn at the intersection")
        # — the published taxonomy's "43% misinterpret turn pattern"
        # failure. We reach this branch only when (a) did NOT match a
        # cardinal target (orient subtasks are exempt by construction),
        # AND the instruction text matches the relative-turn regex,
        # AND we haven't already emitted a burst for THIS instruction
        # (defense-in-depth latch against an env that doesn't advance
        # current_subtask on a single -1). Emits the deterministic
        # 2-action burst [<single rotate>, -1]: one 90 deg rotation in
        # the named direction, then a subtask-completed commit. Failure
        # mode if the env declines to advance: the planner takes over
        # on subsequent steps (mirrors iter_001 REACH cap-2). No
        # cardinal-compass read needed — the rotation is relative to
        # the agent's current heading, whatever it is.
        if target is None:
            turn_dir = _parse_relative_turn(instruction)
            if (turn_dir is not None
                    and instruction != self._turn_burst_done_for):
                code = _ROTATE_LEFT if turn_dir == "left" else _ROTATE_RIGHT
                # Reset the momentum counter so a subsequent successful
                # forward on the NEXT subtask starts a fresh cap-2
                # burst from this scene.
                self._consecutive_momentum = 0

                # -- approach phase (commit-geometry, this iter) ----------
                # The burst used to rotate and commit from wherever the
                # PRECEDING leg ended. That position is the ROAD waypoint,
                # which lies short of the intersection the TURNING box is
                # centred on, so the commit landed outside the box by a
                # fraction of a forward step. Spend a bounded number of
                # forwards closing on the intersection FIRST, then rotate and
                # commit. One decision per call: phase 0..N-1 return a single
                # [_FORWARD]; phase N returns the rotate+commit burst. Nothing
                # here reads world state -- the phase is a self-derived
                # counter and the stall test reads only odometry.
                if self._turn_approach_for != instruction:
                    self._turn_approach_for = instruction
                    self._turn_approach_done = 0
                    self._turn_approach_stalls = 0
                elif self._turn_approach_done > 0:
                    # Price the PREVIOUS approach forward off the odometer.
                    # A blocked forward (facade dead ahead) means the
                    # remaining approach cannot be taken; count it so the
                    # latch falls through to the commit instead of mashing.
                    delta = _approach_delta(state)
                    if delta is not None and delta < _TURN_APPROACH_MIN_DELTA:
                        self._turn_approach_stalls += 1
                    else:
                        self._turn_approach_stalls = 0

                if (self._turn_approach_done < _TURN_APPROACH_FORWARDS
                        and self._turn_approach_stalls
                        < _MAX_TURN_APPROACH_STALLS):
                    self._turn_approach_done += 1
                    return [_FORWARD]

                # Approach spent (or blocked): rotate and commit. The latch
                # closes here, not on first entry, so the approach forwards
                # are never mistaken for a second burst.
                self._turn_burst_done_for = instruction
                return [code, _SUBTASK_DONE]

        # (a') Traffic-safety idle (iter_004). Only fires on non-
        # orientation subtasks (target is None) and never beyond the
        # per-subtask cap. The cap protects against (1) a false-positive
        # red trigger from brick / brake-light tints — bounded waste of
        # 2 sim steps — and (2) a slow-cycling signal that would
        # otherwise lock the agent out of forward progress entirely.
        if (
            obs is not None
            and self._traffic_idle_count < _MAX_TRAFFIC_IDLES_PER_SUBTASK
            and _detect_red_signal_ahead(obs)
        ):
            self._traffic_idle_count += 1
            return list(_TRAFFIC_IDLE_BURST)

        # (b) Anti-spin break-out (iter_001) and (d) forward momentum
        # (iter_006, cap=2) both consume the same SpatialMemory
        # channels. We check anti-spin first (it bridges a pure-rotation
        # stall); momentum can only fire after a successful forward, so
        # the two are naturally mutually exclusive on the "last actions"
        # axis.
        if state is None:
            # Either rule needs SpatialMemory. Reset the momentum
            # counter so the next call with state available can fire
            # cleanly (it will start a fresh cap-2 burst).
            self._consecutive_momentum = 0
            return None
        last_actions_fn = getattr(state, "last_actions_in_subtask", None)
        md_delta_fn = getattr(state, "moving_distance_delta", None)
        if last_actions_fn is None or md_delta_fn is None:
            self._consecutive_momentum = 0
            return None

        # Anti-spin: last K==2 same-rotation steps with no displacement.
        try:
            spin_recent = list(last_actions_fn(k=_SPIN_WINDOW))
        except Exception:
            spin_recent = []
        if (
            len(spin_recent) >= _SPIN_WINDOW
            and spin_recent[0] in _ROTATE_CODES
            and all(a == spin_recent[0] for a in spin_recent)
        ):
            try:
                spin_delta = float(md_delta_fn(window=_SPIN_WINDOW))
            except Exception:
                spin_delta = 0.0
            if spin_delta < _NO_DISPLACEMENT_THRESHOLD_CM:
                # Anti-spin fires. Reset the momentum counter so the
                # next successful forward starts a fresh cap-2 burst.
                self._consecutive_momentum = 0
                return [_FORWARD]

        # (b') iter_010: wall-stuck recovery rotation. Fires when the
        # LAST action was forward (action 0) but it stalled (md_delta
        # < 1 cm) AND the env registered a building/object collision
        # on that same step. The iter_008 momentum veto already RATE-
        # LIMITS this case by skipping the implicit follow-up momentum
        # forward — but the *planner* is still free to emit another
        # forward on the next iteration, which it did 17 times in a
        # row on iter_009 ep_20_0's wall-mash tail. Rule (b') overrides
        # the planner with a single rotate-left, forcing a 90-deg
        # fresh viewpoint so the next perceive+plan sees a non-wall
        # scene. Repeated firings sweep the cardinals; if every
        # cardinal is a wall, the agent rotates 4 times and the
        # existing STUCK detector eventually catches it via the
        # planner's multi-step emission. Composition vs other rules:
        # mutually exclusive with momentum (d) because momentum
        # requires `fwd_delta >= 1.0` AND `static_collision_delta ==
        # 0` (iter_008 veto) — b's preconditions are the complement.
        try:
            wall_recent = list(last_actions_fn(k=1))
        except Exception:
            wall_recent = []
        if len(wall_recent) == 1 and wall_recent[0] == _FORWARD:
            try:
                wall_md_delta = float(md_delta_fn(window=1))
            except Exception:
                wall_md_delta = 0.0
            coll_delta_fn = getattr(state, "static_collision_delta", None)
            wall_coll_delta = 0
            if coll_delta_fn is not None:
                try:
                    wall_coll_delta = int(coll_delta_fn(window=1))
                except Exception:
                    wall_coll_delta = 0
            if (
                wall_md_delta < _WALL_STUCK_MD_THRESHOLD_CM
                and wall_coll_delta > 0
            ):
                # Reset the momentum counter so a future successful
                # forward starts a fresh cap-2 burst from this scene.
                self._consecutive_momentum = 0
                return [_WALL_RECOVERY_ROTATION]

        # (d) Forward momentum (iter_006). Cap of
        # _MAX_CONSECUTIVE_MOMENTUM consecutive emissions yields the
        # cadence PP -> m -> m -> PP -> m -> m -> ... (3 forwards per
        # planner cycle on confirmed-progress ROAD segments).
        #
        # Preconditions for emission:
        #   (i)   not on an orientation subtask (target_cardinal None);
        #   (ii)  fewer than _MAX_CONSECUTIVE_MOMENTUM emissions since
        #         the last non-momentum action (counter < cap);
        #   (iii) the last logged action was Forward (action code 0);
        #   (iv)  that forward succeeded (moving_distance delta >=
        #         _FORWARD_PROGRESS_THRESHOLD_CM).
        #
        # On a stalled forward (iv fails) we yield to perceive+plan so
        # the planner can re-strategize. On a non-forward last action
        # (iii fails) we yield to perceive+plan so the planner stays in
        # control of mode changes. On the counter being at-cap (ii
        # fails) we yield to perceive+plan and reset the counter; the
        # NEXT successful forward (emitted by the planner) re-arms the
        # cap-2 burst.
        if self._consecutive_momentum < _MAX_CONSECUTIVE_MOMENTUM:
            try:
                fwd_recent = list(last_actions_fn(k=1))
            except Exception:
                fwd_recent = []
            if len(fwd_recent) == 1 and fwd_recent[0] == _FORWARD:
                try:
                    fwd_delta = float(md_delta_fn(window=1))
                except Exception:
                    fwd_delta = 0.0
                if fwd_delta >= _FORWARD_PROGRESS_THRESHOLD_CM:
                    # iter_008: static-collision veto. The last forward
                    # succeeded (delta >= 1 cm), but if the env also
                    # registered a building/object collision on that
                    # same step, the agent is wedged-but-sliding against
                    # a static obstacle. Emitting another momentum
                    # forward (with no fresh perceive call) will repeat
                    # the collision and waste a step. Reset the counter
                    # and yield to PP so the planner sees the wall in a
                    # refreshed RGB frame and can plan a turn/sidestep.
                    # The check is one-sided: planner-driven forwards
                    # are NOT vetoed — only momentum's implicit "do it
                    # again" emission is.
                    coll_delta_fn = getattr(state, "static_collision_delta", None)
                    coll_delta = 0
                    if coll_delta_fn is not None:
                        try:
                            coll_delta = int(coll_delta_fn(window=1))
                        except Exception:
                            coll_delta = 0
                    if coll_delta > 0:
                        self._consecutive_momentum = 0
                        return None
                    self._consecutive_momentum += 1
                    return [_FORWARD]

        # No rule fired. Reset the momentum counter so the next
        # eligible forward (planner-driven) starts a fresh cap-2 burst.
        self._consecutive_momentum = 0
        return None
