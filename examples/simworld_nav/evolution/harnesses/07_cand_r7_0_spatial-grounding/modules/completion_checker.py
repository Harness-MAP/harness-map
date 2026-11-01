"""CompletionChecker — last-line authority on whether the harness should
actually execute a `-1` (subtask completed).

iter 1 stance (REACH check / proactive hint-match commit gate):
  Parent (s0_naiveR_000) shipped the H_0 stub (`-1` iff the planner's
  action list contains `-1`). Parent traces show the dominant failure
  is the planner NEVER proposing `-1` on non-orientation subtasks:
  ep_20_3 emits 38 consecutive `Move_Forward`s after the cardinal
  burst commits subtask 0, walking 10800 cm past the landmark before
  MAX_STEPS. The perceiver-baked Match field, buried in a freeform
  scene description, is too weak to drive timely commits.

  iter 1 adds a dedicated proactive_commit_check method: a single
  binary YES/NO VLM call comparing obs['rgb'] to the per-subtask
  hint image. Fires when the harness has executed K consecutive
  forwards without a planner-driven -1, the current subtask is non-
  orientation, and a hint image exists. Capped at M=2 firings per
  subtask so cost is bounded (~2 extra small calls per subtask).

iter 2 stance (planner-Match commit gate):
  Parent (s0_naiveR_001) showed that REACH check helps but is
  BYPASSED when the planner emits `-1` first. ep_20_4 subtask 1
  reproduces the same false planner `-1` at step 9 across iter_000
  AND iter_001 (identical 10-action sequence), because `verify()`
  trusts `-1 in actions` unconditionally and the harness only
  consults `proactive_commit_check` when verify returns False.

  iter 2 promotes the planner's own `Match` field (already returned
  in `Planner.plan`'s JSON output) from a passive value to an
  active gate: on NON-orientation subtasks, planner-driven `-1` is
  honored only if `planner_match == 1`. Orientation subtasks still
  trust planner `-1` unconditionally (and ActivePerception's
  cardinal burst short-circuits the planner there anyway, so this
  is defense in depth). The REACH check (iter_001) remains the
  fallback when planner `-1` is rejected — verify=False routes the
  control flow through `proactive_commit_check` exactly as before.

H_0 stance (relaxed in iter 2 for non-orient subtasks only):
  `verify(...)` previously trusted any `-1 in actions`. iter 2
  requires `planner_match == 1` for non-orient subtasks to filter
  spurious `-1` commits where the planner's own confidence flag
  contradicts its action list.

Reads (deployment-honest):
  - obs['rgb'] (in H_0's audited surface)
  - hint_image (loaded from task_spec.subtasks[i].visual_hint_path)
  - instruction (task spec)
  - forward_streak (harness-maintained counter over actions_log)
  - vlm_clients (model interface)
  No reads of agent_location, target_location, gt.json, or any
  privileged distance/threshold field.

Constants (not map-tuned; all of them counts of ACTIONS, never cm):
  K_REACH_FORWARD_STREAK = 4   (translations before the first look of a leg)
  REACH_BUCKET_CAPACITY = 4    (looks in hand at the start of a leg)
  REACH_REFILL_FORWARDS = 4    (one look earned per 4 translations walked)
  REACH_MIN_SPACING = 2        (two looks are never closer than this)
  XCHECK_REARM_FORWARDS = 4    (a cross-check answer stands for this far)
"""
from __future__ import annotations
import json
import re
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

# Reuse the baseline's image encoder so the binary check uses the same
# encoding the perceiver uses for the main RGB+hint pair. Avoids a
# second encoder path that could subtly diverge in resize/format.
_BASELINE_DIR = "/home/azureuser/workspace/AutoSpatial/simworld_gym/baseline/single"
if _BASELINE_DIR not in sys.path:
    sys.path.insert(0, _BASELINE_DIR)

from utils import numpy_to_base64  # noqa: E402


K_REACH_FORWARD_STREAK = 4
MAX_REACH_CHECKS_PER_SUBTASK = 2

# --- the arrival-look ALLOCATOR: a RATE, not a quota (this iter) -----------
# The two knobs replaced here were both uniform over a leg whose length the
# harness cannot know, and they failed in OPPOSITE directions at the two ends
# of that unknown length.
#   * `K_REACH_INTERVAL = 5` (fire at streak 4, then every 5 translations)
#     asked the arrival question ~0.97 times per pass through the ~8-
#     translation-wide pass window -- 35 in-window fires across 266 in-window
#     steps in the 36 episodes that physically entered the leg-1 window. At a
#     per-fire in-window recall of 0.143 the leg therefore converted at
#     recall-per-QUESTION (0.167 observed) instead of recall-per-window.
#   * `MAX_REACH_FIRES_PER_SUBTASK = 12` then ran long legs DRY: 12 fires x 5
#     translations is ~36 000 cm of uniform coverage, and 4 of the 5 long
#     (FINISH) legs spent the entire allowance 15-35 km short of the
#     destination -- three episodes stood 211 / 719 / 1 646 cm from it with
#     ZERO looks left to spend.
# A quota cannot be both. One mechanism replaces both: the look is drawn from
# a per-leg TOKEN BUCKET whose only clock is the leg's own executed
# translation count (the harness's `forward_streak`; no coordinate, no map, no
# target, no privileged field, and no absolute distance anywhere):
#   * the bucket starts FULL, so a short leg is sampled densely from its first
#     eligible frame (the old "tight clump", kept);
#   * it earns one look back every REACH_REFILL_FORWARDS translations, so a
#     leg of ANY length still has a look in hand at its far end -- a rate
#     cannot be exhausted early the way a quota can;
#   * two looks are never closer than REACH_MIN_SPACING translations, which is
#     what stops the opening burst from being dumped into one stretch of road.
# Long-run this is one look per 4 translations => ~2 looks inside an 8-step
# window against the old 0.97, and it is paid for, not added: the 2-image
# revert in `_binary_reach_vlm_call` costs ~1 050 tok/call against the
# 4-image call's 2 371, and the FINISH leg (below) no longer spends looks at
# all.
REACH_BUCKET_CAPACITY = 4
REACH_REFILL_FORWARDS = 4
REACH_MIN_SPACING = 2

# iter 8: require N consecutive Match=1 fires (each one K_REACH_INTERVAL
# forwards apart) within the same subtask before committing. Suppresses
# single-frame Match=1 hallucinations that drove the iter_007 regression
# (ep_20_0 committed at mv=4000 cm, ep_20_4 lost subtask 1 after late
# commit). A genuine landmark approached on foot persists across two
# checks 5 forwards apart; a coincidental "looks like a building"
# resemblance does not.
# iter 9: REVERT iter 8's 2-of-2 confirmation requirement back to 1-of-1.
# iter 8 (cap=8, K_INTERVAL=5, 2-of-2) regressed SubtaskSR 0.30 → 0.25 and
# subtask 1 SR 0.20 → 0.00. The 2-of-2 gate over 2500cm-spaced fires made
# Match flicker 1→0 too easily, so the confirms counter reset before
# reaching 2 — REACH never committed on long-walk subtasks. With N=1 this
# restores the iter_007 control flow (cap=8, K_INTERVAL=5, 1-of-1 commit)
# while leaving the consecutive-counter scaffolding inert (any single
# Match=1 commits immediately). The next iter can chase iter_006's
# SubtaskSR=0.40 ceiling by separately reverting cap=8 → cap=2.
N_REACH_CONFIRMATIONS_REQUIRED = 1

# --- cross-check arbitration: ONE displacement-keyed rule (this iter) ------
# The cross-verifier (iter 4) is the same binary REACH call, asked on the
# planner's own `-1` before the harness executes it. It used to be arbitrated
# by three separate mechanisms stacked on top of each other -- a per-leg firing
# cap (MAX_XCHECK_FIRES_PER_SUBTASK), a per-leg refusal allowance
# (MAX_XCHECK_REFUSALS_PER_SUBTASK), and an accept memo bucketed by forward
# progress (XCHECK_CACHE_STREAK_BUCKET) -- each of which answered a different
# question about the SAME thing, and two of which let a spent budget decide a
# commit. All three are deleted. What is left is one rule, shared with the
# proactive look above: the cross-check's last answer STANDS until the agent
# has walked XCHECK_REARM_FORWARDS translations, and walking is what re-asks
# the question.
#
# The value is the arrival channel's own spacing, K_REACH_FORWARD_STREAK: a
# refusal is re-examined once the agent has moved as far as the harness
# requires before it will ask the question unprompted at all. Below that
# distance the frame is the same frame (a blocked commit is answered with an
# in-place hold that translates zero) and both upstream calls run at
# temperature 0.2, so re-asking can only buy the same answer at the price of a
# VLM call -- which is precisely the trade the previous iter made and lost:
# 5 deferred-then-rubber-stamped commits, 1 pass, 4 fatal.
XCHECK_REARM_FORWARDS = K_REACH_FORWARD_STREAK

# iter 3 motion-required floor on planner-driven `-1` commits for
# non-orient subtasks. The planner's self-reported Match=1 alone is
# insufficient: parent (s0_naiveR_002) showed the VLM cheerfully
# returns Match=1 on the FIRST frame of subtask N+1 (forward_streak=0,
# spawn region visually resembles the hint), producing instant false
# commits that collapsed SubtaskSR 0.35 -> 0.25. With this floor, a
# planner -1 on a non-orient subtask requires Match=1 AND at least
# K_MIN forward-step executions WITHIN the current subtask.
# Lowered 3 -> 1 (baseline-leaning): the baseline has NO motion floor before
# honoring a commit; a floor of 1 still blocks the carry-over Match=1 on the first
# frame of a new subtask, but no longer vetoes short approaches. Combined with the
# motion counter now counting strafes (orchestrator.py), short/strafe completions can
# commit instead of overshooting.
K_MIN_FORWARD_BEFORE_PLANNER_COMMIT = 1

# Orientation-subtask regex (mirrors active_perception's _TARGET_CARDINAL_RE
# in spirit but checked at the CompletionChecker layer too, so the REACH
# check never fires on a face/face-the-direction subtask even if the
# harness forgets to gate). Defense in depth.
_ORIENT_RE = re.compile(
    r'\b(?:face|facing|faced)\b'
    r'(?:\s+(?:to|toward|towards))?'
    r'(?:\s+the)?'
    r'\s+(east|west|north|south)\b',
    re.IGNORECASE,
)


def _is_orientation_instruction(instruction: Any) -> bool:
    if not isinstance(instruction, str) or not instruction:
        return False
    return _ORIENT_RE.search(instruction) is not None


def _extract_match(raw: str) -> Optional[int]:
    """Pull a 0/1 Match field from a strict-JSON VLM response.

    Tolerant of markdown fences and stray prose. Returns None on
    unparseable output (caller treats None as "no commit").
    """
    if not raw:
        return None
    s = raw.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1]
        s = s.split("```", 1)[0]
    elif s.startswith("```"):
        s = s.split("```", 1)[1]
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            m = obj.get("Match", obj.get("match"))
            if m is not None:
                try:
                    return int(m)
                except (TypeError, ValueError):
                    if isinstance(m, str):
                        return 1 if m.strip().lower().startswith("y") else 0
    except Exception:
        pass
    m = re.search(r'"Match"\s*:\s*(\d)', raw)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


# ---- iter_002: depth-proximity commit TRIGGER ------------------------
# REPLACES the forward-streak commit trigger (K_REACH_FORWARD_STREAK /
# K_REACH_INTERVAL / cap=2), proven EXHAUSTED by the chain (naive_v3_001
# diagnoses): a step counter fires the arrival check 15k-35k cm too early
# on long walks then goes dormant for the final approach, and never blocks
# the far-field false `-1`s (the anchor's 4 fatal commits at 22k-34k cm).
# Instead we read a MEASURED proximity from obs['depth_map'] so the arrival
# check fires WHEN a large structure is actually close, and the planner-`-1`
# cross-check BLOCKS a commit when the forward scene is clearly open road.
#
# Depth convention (CORRECTED 2026-09-12): obs['depth'] is a per-frame-normalized
# **JET colormap** (cv2.COLORMAP_JET), NOT a grayscale uint8. `_depth_gray` now
# INVERTS the JET LUT to a monotone scalar (0 == NEAREST, 255 == FARTHEST), so the
# "LOWER value == NEARER" convention below is TRUE. (Before the fix, mean(axis=2)
# folded near & far together — both JET extremes are dark — and classified the sky,
# the farthest thing, as NEAREST on 14/14 frames; see diagnosis.md §2.3.) The depth
# tests remain SELF-NORMALIZING per frame (percentile-relative, no absolute cm/uint8
# cutoff) and FAIL OPEN on missing/flat depth. NOTE: the open-city far-veto that used
# to live here (`_strongly_far`) is DISABLED — per-frame normalization has no absolute
# scale, so it cannot tell "near a facade" from "far in open space"; the far-field
# arrival sanity check is now `_named_side_has_structure` (segmentation-based).
# Cone = UPPER-MID center (around/above the horizon), where a building
# dead-ahead projects. Deliberately excludes the lower rows: the ground
# plane right in front of the agent is always near, so a low cone would
# read "near" on every frame. Open road -> the upper-mid center is the
# distant vanishing point / sky (far); a close facade fills it (near).
_CONE_ROW = (0.20, 0.58)        # forward cone rows (fractions of H)
_CONE_COL = (0.34, 0.66)        # forward cone cols (fractions of W)
_SIDE_COL_FRAC = 0.30           # outer width fraction on the named side
_NEAR_BAND = 0.30               # region median <= p_lo + 0.30*range == "near"
_FAR_BAND = 0.62                # cone median >= p_lo + 0.62*range == "far"

_SIDE_LEFT_RE = re.compile(r'\b(?:on|to)\s+(?:your|the|my)\s+left\b', re.IGNORECASE)
_SIDE_RIGHT_RE = re.compile(r'\b(?:on|to)\s+(?:your|the|my)\s+right\b', re.IGNORECASE)


# --- commit-geometry (this iter): the terminal-pose contract ------------
# A FINISH leg is graded on POSITION *and* FACING: the generator's FINISH
# template always asks the robot to "stop and face that building", and the
# building is always on the named side of the travel axis. A `-1` emitted
# while still facing along the street is therefore off by a quarter turn
# whatever the position. Every other leg (ORIENTATION, ROAD, TURNING) is
# graded without a face-the-building clause, so the pose prefix must be
# emitted on FINISH legs ONLY.
#
# The whole inference is a read of the STATIC instruction string from
# `task_spec['subtasks']` -- the same honest, text-only derivation the
# cardinal ORIENTATION burst already makes. No coordinate, no ground-truth
# orientation, no privileged field is touched.
_ROTATE_LEFT_CODE = 1        # rules.md: Rotate_Left,  -90 deg yaw
_ROTATE_RIGHT_CODE = 2       # rules.md: Rotate_Right, +90 deg yaw
_COMMIT_CODE = -1            # rules.md: Subtask_Completed

# "stop and face that building" / "face toward the building" / "facing that
# building". Requires the BUILDING object, so "turn to face east" (the
# ORIENTATION template) cannot match -- and `_is_orientation_instruction` is
# checked as a second, independent guard below.
_FACE_BUILDING_RE = re.compile(
    r'\bfac(?:e|ing)\s+'
    r'(?:(?:to|toward|towards)\s+)?'
    r'(?:it\b'
    r'|(?:that|the|this)\s+(?:\w+[\w-]*\s+){0,5}(?:building|structure|house|tower))',
    re.IGNORECASE,
)
# ROAD / TURNING legs are anchored on an INTERSECTION, never on a facing.
# Any mention of stopping at / reaching an intersection disqualifies a leg
# from the FINISH pose prefix, whatever else the sentence says.
_INTERSECTION_RE = re.compile(r'\bintersections?\b', re.IGNORECASE)


def _is_finish_instruction(instruction: Any) -> bool:
    """True iff this leg is the FINISH leg (stop AND face the side building).

    Three conjuncts, deliberately narrow -- a false positive here prepends a
    quarter turn to a ROAD commit, which would corrupt the heading the
    following relative-turn burst rotates from:
      1. not an ORIENTATION instruction (that path is untouchable and owns
         its own cardinal burst);
      2. the instruction names the side the target sits on ("on your
         left/right") -- ROAD templates are forbidden from mentioning a side;
      3. the instruction contains a face/facing clause -- ROAD templates stop
         *at an intersection* and never ask for a facing.
    """
    if not isinstance(instruction, str) or not instruction:
        return False
    if _is_orientation_instruction(instruction):
        return False
    if _named_side(instruction) is None:
        return False
    if _INTERSECTION_RE.search(instruction):
        return False
    return _FACE_BUILDING_RE.search(instruction) is not None


def _named_side(instruction: Any) -> Optional[str]:
    """Parse 'on your left' / 'on your right' from the instruction."""
    if not isinstance(instruction, str):
        return None
    if _SIDE_LEFT_RE.search(instruction):
        return "left"
    if _SIDE_RIGHT_RE.search(instruction):
        return "right"
    return None


_JET_LUT_BGR = None


def _jet_lut_bgr():
    """cv2.COLORMAP_JET lookup: uint8 value (0=NEAR .. 255=FAR) -> BGR colour. Cached.
    The env renders depth via exactly this LUT (unrealcv._decode_npy), so inverting it
    recovers the original monotone scalar with ~0 residual."""
    global _JET_LUT_BGR
    if _JET_LUT_BGR is None:
        import cv2  # available in the simworld env (utils.numpy_to_base64 already uses it)
        _ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
        _JET_LUT_BGR = cv2.applyColorMap(_ramp, cv2.COLORMAP_JET).reshape(256, 3).astype(np.int16)
    return _JET_LUT_BGR


def _depth_gray(obs: Any) -> Optional[np.ndarray]:
    """obs['depth'] (or 'depth_map' alias) -> 2D float array MONOTONE in true distance:
    **0 == NEAREST, 255 == FARTHEST** (so the module's "LOWER value == NEARER" convention is
    now TRUE). The env ships depth as a per-frame-normalized **JET colormap** (cv2.COLORMAP_JET):
    mean(axis=2) FOLDS near and far together (both extremes render dark, mid-range is brightest),
    which made every depth predicate read the channel backwards (sky classified NEAR on 14/14
    frames). We invert the JET LUT instead. Downsampled for speed — region medians/percentiles are
    downsample-robust. Fails OPEN (None) on missing / non-colour depth; cv2-unavailable falls back
    to the legacy (folded) mean so a missing dep never crashes the gate."""
    if not isinstance(obs, Mapping):
        return None
    d = obs.get("depth_map")
    if d is None:
        d = obs.get("depth")
    if d is None:
        return None
    try:
        arr = np.asarray(d)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[-1] < 3 or arr.size == 0:
        # not a 3-channel colour image (already-gray / odd shape): best-effort 2D, legacy behaviour
        a2 = arr.astype(np.float32)
        if a2.ndim == 3:
            a2 = a2.mean(axis=2)
        return a2 if (a2.ndim == 2 and a2.size) else None
    rgb = arr[..., :3].astype(np.int16)
    h, w = rgb.shape[:2]
    step = max(1, int(round(max(h, w) / 120.0)))   # ~120px longest side: fast, region-stat-safe
    small = rgb[::step, ::step, :].reshape(-1, 3)
    try:
        lut = _jet_lut_bgr()                         # (256,3) BGR
    except Exception:
        return rgb[::step, ::step, :].astype(np.float32).mean(axis=2)   # cv2 missing -> legacy
    # obs['depth'] is BGR (cv2.applyColorMap output, no swap before obs); be robust to a channel
    # swap by picking whichever order gives the smaller nearest-LUT residual.
    best = None
    for cand in (lut, lut[:, ::-1]):
        dist = np.abs(small[:, None, :] - cand[None, :, :]).sum(axis=2)   # (N,256) L1
        idx = dist.argmin(axis=1)
        resid = float(dist[np.arange(len(idx)), idx].mean())
        if best is None or resid < best[0]:
            best = (resid, idx)
    hs = len(range(0, h, step))
    ws = len(range(0, w, step))
    return best[1].reshape(hs, ws).astype(np.float32)   # 0=near .. 255=far


def _looming_state(obs: Any, instruction: Any) -> str:
    """Return 'near' | 'far' | 'unknown' — the commit trigger signal.

      'near'    : a large structure is close in the forward cone OR on the
                  instruction's named side -> the arrival check SHOULD run.
      'far'     : forward cone clearly open/distant AND nothing close on the
                  named side -> a `-1` here is a far-field false commit; BLOCK.
      'unknown' : depth missing / flat / ambiguous -> fall back to legacy
                  behaviour (never tightens the gate below the anchor).
    """
    arr = _depth_gray(obs)
    if arr is None:
        return "unknown"
    h, w = arr.shape
    p_lo = float(np.percentile(arr, 10))
    p_hi = float(np.percentile(arr, 90))
    rng = p_hi - p_lo
    if rng < 1.0:                       # flat frame: depth uninformative
        return "unknown"
    near_thr = p_lo + _NEAR_BAND * rng
    far_thr = p_lo + _FAR_BAND * rng

    r0, r1 = int(_CONE_ROW[0] * h), int(_CONE_ROW[1] * h)
    c0, c1 = int(_CONE_COL[0] * w), int(_CONE_COL[1] * w)
    cone = arr[r0:r1, c0:c1]
    cone_med = float(np.median(cone)) if cone.size else p_hi
    # Forward cone close (a facade fills the upper-mid center)?
    if cone_med <= near_thr:
        return "near"

    # Off-axis side building: check the lateral band on the named side.
    side = _named_side(instruction)
    if side is not None:
        sw = max(1, int(_SIDE_COL_FRAC * w))
        band = arr[r0:r1, (w - sw):w] if side == "right" else arr[r0:r1, 0:sw]
        if band.size and float(np.median(band)) <= near_thr:
            return "near"

    # Forward clearly open (nothing close dead-ahead nor on the named side).
    if cone_med >= far_thr:
        return "far"
    return "unknown"


# ---- iter_003: depth as an ACCEPT-VALIDATOR (not a trigger) -----------
# naive_v3_002 used depth to decide WHETHER to ask the REACH question
# (fire on `near`, block on `far`). The diagnosis proved that `far`-block
# SUPPRESSES the check at the exact off-axis ROAD waypoints where it is
# needed (open intersection ahead = far). iter_003 inverts the role:
# depth NEVER prevents the REACH call; it only VETOES a geometrically
# impossible YES. `_strongly_far` is the veto predicate — True only when
# essentially NOTHING is close in the building-projection band (the agent
# is tens of metres from any structure, as in the anchor's fatal 30k-cm
# commits). It deliberately ignores WHERE the close thing is (forward vs
# side) so it cannot re-suppress an off-axis waypoint commit: any near
# structure anywhere in the band -> not strongly far -> do NOT veto.
# Fail OPEN (return False) on missing/flat depth.
_STRONGLY_FAR_NEAR_FRAC = 0.05   # < 5% of the band is "near" == open scene


def _strongly_far(obs: Any, instruction: Any) -> bool:
    """DISABLED — always returns False (never vetoes). Its intended role (veto a Match=1 when the
    agent is in open city ~30k cm from anything) is UNACHIEVABLE with this depth channel: the env
    normalizes depth PER FRAME (min-max), so the nearest thing in view is always mapped to ~0 and
    there is no absolute scale to tell "close to a facade" from "far in open space". Worse, once
    `_depth_gray` reads the JET channel correctly, the upper-mid cone reads "far" on ~100% of frames
    (near_frac == 0.000 on 24/24 real frames), so the old `near_frac < 0.05` test would veto EVERY
    commit (the "bare-polarity-flip trap" — every episode becomes a never-commits failure). The
    far-field arrival-sanity veto is handled instead by `_named_side_has_structure` (segmentation-
    based, immune to depth's lack of absolute scale); a re-derived depth veto is left to the
    evolution loop now that depth is read correctly. Kept (signature intact) so both call sites and
    any references stay valid without ripping out control flow."""
    return False


# ---- iter_005 (turn_to_face_side_depth): named-side proximity for the
# physical turn-to-face scan. Isolates the named-side branch of `_looming_state`
# (above) so the harness can ask "is a structure NEAR on the instruction's
# side?" WITHOUT also firing on a near forward cone. Same self-normalizing
# per-frame percentile bands (_NEAR_BAND over the p10..p90 range) over the same
# upper-mid rows — carries NO absolute cm / coordinate / map-tuned constant.
# Fails CLOSED (False) on missing/flat depth or no named side, so the look-turn
# only triggers when depth positively reports a side structure (i.e. the agent
# is AT the open-intersection waypoint with the corner building beside it).
def _side_band_near(obs: Any, side: Optional[str]) -> bool:
    """True iff the depth band on `side` ('left'|'right') reads 'near'."""
    if side not in ("left", "right"):
        return False
    arr = _depth_gray(obs)
    if arr is None:
        return False
    h, w = arr.shape
    p_lo = float(np.percentile(arr, 10))
    p_hi = float(np.percentile(arr, 90))
    rng = p_hi - p_lo
    if rng < 1.0:                       # flat frame: depth uninformative
        return False
    near_thr = p_lo + _NEAR_BAND * rng
    r0, r1 = int(_CONE_ROW[0] * h), int(_CONE_ROW[1] * h)
    sw = max(1, int(_SIDE_COL_FRAC * w))
    band = arr[r0:r1, (w - sw):w] if side == "right" else arr[r0:r1, 0:sw]
    return bool(band.size and float(np.median(band)) <= near_thr)


# ---- named-side building precondition (SEGMENTATION-based accept-validator) -----------
# Replaces the disabled depth-based `_strongly_far` as the far-field sanity veto. Rationale:
# the instruction names the side the landmark is on ("... ON YOUR LEFT"); a frame with ZERO
# building-class pixels on that side cannot be the arrival frame, whatever the recognizer says.
# Uses obs['object_mask'] (a different signal from depth) so it is immune to the per-frame-
# normalized depth's lack of absolute scale. Building class = purple (128,0,128), which is
# palindromic so the BGR-vs-RGB order of the mask does not matter. Measured (diagnosis.md §5-A):
# kills the anchor's fatal 2617 cm commit (named-side building fraction 0.000) with 0/29 false
# vetoes on the in-window frames (side fraction 0.595-0.719). FAILS OPEN (True) on no named side
# or a missing/odd mask, so it never tightens the gate when the precondition cannot be evaluated.
_NAMED_SIDE_ROW = (0.30, 0.62)      # rows of the lateral band (diagnosis-measured operating point)
_NAMED_SIDE_COL_FRAC = 0.25         # outer column fraction on the named side
_NAMED_SIDE_BUILDING = np.array([128, 0, 128], dtype=np.int16)   # purple building class
_NAMED_SIDE_MIN_FRAC = 0.02         # > this fraction of building px == the side is NOT empty


def _named_side_has_structure(obs: Any, instruction: Any) -> bool:
    """True unless the instruction names a side AND that side is EMPTY of building-class pixels."""
    side = _named_side(instruction)
    if side is None:
        return True
    m = obs.get("object_mask") if isinstance(obs, Mapping) else None
    if m is None:
        return True
    try:
        a = np.asarray(m)
    except Exception:
        return True
    if a.ndim != 3 or a.shape[-1] < 3 or a.size == 0:
        return True
    h, w = a.shape[:2]
    r0, r1 = int(_NAMED_SIDE_ROW[0] * h), int(_NAMED_SIDE_ROW[1] * h)
    sw = max(1, int(_NAMED_SIDE_COL_FRAC * w))
    band = a[r0:r1, 0:sw, :3] if side == "left" else a[r0:r1, (w - sw):w, :3]
    if band.size == 0:
        return True
    bld = (np.abs(band.astype(np.int16) - _NAMED_SIDE_BUILDING).sum(axis=-1) < 60)
    return float(bld.mean()) > _NAMED_SIDE_MIN_FRAC


# Per-subtask cap on physical turn-to-face scans (cost control, not a distance /
# coordinate). 2 = at most two look-turns per waypoint approach.
MAX_SIDE_SCANS_PER_SUBTASK = 2


@dataclass
class CompletionChecker:
    # Per-subtask arrival-look TOKEN BUCKET (this iter; replaces the fire
    # quota `_per_subtask_reach_fires`, which is deleted rather than left
    # inert -- an unread counter is how three iters of dead schedule logic
    # survived in this file). `_reach_tokens` is the stock of looks in hand,
    # `_reach_refill_mark` the translation count at which the last look was
    # earned. Keyed by `cur_idx` (the structural subtask index passed by the
    # harness), re-instantiated per episode so no cross-episode leakage.
    _reach_tokens: Dict[int, int] = field(default_factory=dict)
    _reach_refill_mark: Dict[int, int] = field(default_factory=dict)
    # The cross-check's STANDING ANSWER for each leg: (forward_streak at
    # which it was given, verdict). It replaces all three of the mechanisms
    # this function used to arbitrate with -- the firing cap, the refusal
    # allowance and the accept memo -- with the one rule the whole of this
    # iter is built on: an answer stands until the agent has physically moved,
    # and then the question is asked again. See `cross_check_planner_commit`.
    _xcheck_standing: Dict[int, Any] = field(default_factory=dict)
    # The forward_streak at which this leg last fired a REACH look. The
    # bucket decides IF a look is affordable; this enforces REACH_MIN_SPACING
    # so an affordable burst is still spread over road rather than dumped
    # into one frame.
    _last_reach_streak: Dict[int, int] = field(default_factory=dict)
    # iter 8: count consecutive Match=1 results per subtask. Reset to 0
    # on any Match=0 (or unparseable None). Commit only when the count
    # reaches N_REACH_CONFIRMATIONS_REQUIRED.
    _consecutive_reach_match: Dict[int, int] = field(default_factory=dict)
    # iter_005 (turn_to_face_side_depth): per-subtask count of physical
    # turn-to-face scans already spent (bounds env-step + VLM cost per waypoint).
    _per_subtask_scan_fires: Dict[int, int] = field(default_factory=dict)

    def commit_burst(
        self,
        instruction: Any,
        already_facing_side: bool = False,
    ) -> List[int]:
        """The action list that ACTUALLY performs a commit on this leg.

        Every `-1` in the harness now goes out through here instead of being
        written as a bare literal, so the terminal pose is decided in one
        place by one rule:

          FINISH leg, target on the LEFT  -> [Rotate_Left,  -1]   (-90 deg)
          FINISH leg, target on the RIGHT -> [Rotate_Right, -1]   (+90 deg)
          anything else                   -> [-1]

        A FINISH leg is graded on the facing as well as the position, and the
        instruction states which side the building is on; a rotation toward
        that side is the quarter turn from "walking past it" to "stopped
        facing it". Rotations are IN-PLACE (rules.md), so prepending one
        cannot move the agent out of a position window it has already earned
        -- the pose prefix is free on the distance criterion.

        `already_facing_side=True` is passed by the side-scan commit path,
        which has PHYSICALLY yawed toward the named side already; prefixing a
        second rotation there would overshoot to 180 deg off. Non-FINISH legs
        always get the bare `[-1]`: a quarter turn before a ROAD commit would
        leave the heading wrong for the relative-turn burst that follows it.
        """
        if already_facing_side:
            return [_COMMIT_CODE]
        if not _is_finish_instruction(instruction):
            return [_COMMIT_CODE]
        side = _named_side(instruction)
        if side == "left":
            return [_ROTATE_LEFT_CODE, _COMMIT_CODE]
        if side == "right":
            return [_ROTATE_RIGHT_CODE, _COMMIT_CODE]
        return [_COMMIT_CODE]

    def verify(
        self,
        actions: List[int],
        planner_match: Optional[int],
        planner_match_history: List[Optional[int]],
        instruction: Any = None,
        forward_streak: int = 0,
    ) -> bool:
        """Return True iff the harness should commit a `-1` step.

        iter 3: planner-Match commit gate + motion-required floor.
          - if `-1` not in planner actions: False.
          - if the current subtask IS an orientation subtask: True.
          - otherwise (non-orientation subtask): require BOTH
            (a) `planner_match == 1` AND
            (b) `forward_streak >= K_MIN_FORWARD_BEFORE_PLANNER_COMMIT`.
            Iter 2 enforced only (a); the VLM cheerfully outputs
            Match=1 on the first frame of subtask N+1 (forward_streak
            = 0) because the spawn region visually resembles the
            hint. Adding (b) blocks those zero-motion false commits
            while leaving legitimate post-walk planner commits to go
            through. The REACH check (the harness's next call after
            verify returns False) remains the fallback path.

        Backwards compatibility: `instruction` defaults to None so a
        caller that omits it falls back to the H_0 rule (`-1 in
        actions`). `forward_streak` defaults to 0 so a caller from
        iter_002-era tests still constructs the call without the
        kwarg — though in iter_3 the harness always passes it.
        """
        if not actions or -1 not in actions:
            return False
        if instruction is None:
            return True
        if _is_orientation_instruction(instruction):
            return True
        return (planner_match == 1
                and forward_streak >= K_MIN_FORWARD_BEFORE_PLANNER_COMMIT)

    def _binary_reach_vlm_call(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        description: Optional[str] = None,
    ) -> Optional[int]:
        """Strict binary VLM call: is the current RGB the hint landmark?

        Returns 1 (match), 0 (no match), or None on any failure
        (missing input, malformed response, transient API error).
        Used by BOTH the proactive REACH check (fallback when planner
        did not -1) AND the cross-verifier (gate on planner -1).
        """
        if hint_image is None or not vlm_clients:
            return None
        if not isinstance(obs, Mapping):
            return None
        rgb = obs.get("rgb")
        if rgb is None:
            return None
        try:
            rgb_b64 = numpy_to_base64(np.asarray(rgb))
            hint_b64 = numpy_to_base64(np.asarray(hint_image))
        except Exception:
            return None

        model_id = next(iter(vlm_clients.keys()))
        client = vlm_clients[model_id]

        # --- CRITERION: the anchor's, restored byte-for-byte --------------
        # The previous iter replaced this wording with a frame-geometry "are
        # you STANDING AT it" criterion. The A/B over the same 50 tasks was
        # decisive and it was decisive in BOTH directions at once: in-window
        # accept rate collapsed 9/44 -> 1/58 while the far-field false-positive
        # rate did not move by a single call (6/138 -> 6/139). Two far-apart
        # wordings, one flat ROC -- and in both runs the model quoted whichever
        # clause it landed on back verbatim in its own Reason field. The
        # wording is therefore NOT a lever and is restored rather than
        # replaced a third time.
        sys_prompt = (
            "You compare two egocentric outdoor images and judge whether "
            "the navigator has REACHED the landmark depicted in the "
            "reference image. Answer with a single integer Match field: "
            "1 if the current view shows the same landmark / place as "
            "the reference (the same building, intersection, or scene "
            "is clearly visible and prominent in the current view); "
            "0 otherwise. Be conservative — output 1 only when the "
            "match is unambiguous. Respond strictly with JSON: "
            '{"Match": 0 or 1, "Reason": "<one short clause>"}.'
        )
        # --- EVIDENCE: REVERTED to the anchor's two RGB frames --------------
        # The previous iter attached the segmentation mask, the JET depth
        # rendering and the perceiver's description of the same frame to this
        # call, on the theory that "the same place" is a question about geometry
        # that RGB alone does not carry. It was A/B'd against the anchor at the
        # identical criterion, the same model and the same 50 tasks, and it is
        # measured NEGATIVE: ROAD in-window recall 5/35 = 0.143 against the
        # 2-image call's 9/43 = 0.209, near-miss (2-5 k cm) false positives
        # 4/61 against 0/52, at 2 371 tok/call against ~1 050 -- +126 % per call
        # and +11.6 points of the whole episode budget. Not one of 388 `Reason`
        # strings cited depth hue or the segmentation layout; every one of them
        # was a pure RGB identity statement. Together with the two criterion
        # wordings already tried (0.209 vs 0.017 in-window at an INVARIANT
        # far-field FP rate) every axis on the QUESTION is now closed, so the
        # budget it was consuming is returned and re-spent on the one axis that
        # is open: how often the question is asked (`_spend_reach_look`).
        # `description` is still accepted so the three call sites stay uniform
        # and the harness needs no edit; it is deliberately unused.
        _ = description
        instr_text = instruction if isinstance(instruction, str) else ""
        user_content: List[Dict[str, Any]] = [
            {"type": "text",
             "text": f"Subtask instruction: {instr_text}"},
            {"type": "text", "text": "Current egocentric view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},
            {"type": "text", "text": "Reference landmark view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{hint_b64}"}},
            {"type": "text",
             "text": ('Output strict JSON: {"Match": 0 or 1, '
                      '"Reason": "<one short clause>"}. No prose.')},
        ]

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                # Baseline parity: even a binary REACH confirm needs <think> room
                # on the Thinking model; 96 truncated before the verdict.
                max_tokens=2048,
            )
        except Exception:
            return None
        try:
            raw = resp.choices[0].message.content or ""
        except Exception:
            return None
        return _extract_match(raw)

    def _spend_reach_look(self, cur_idx: int, forward_streak: int) -> bool:
        """Token bucket over the leg's own executed translations.

        Returns True and debits one look iff the leg can afford one now. The
        only input is `forward_streak`, the harness's count of translations
        executed since this leg began -- self-derived, monotone, and carrying
        no distance, no coordinate and no privileged field.

        Why a bucket and not a stride-plus-quota: the harness does not know how
        long the current leg is, and the two failures of a quota sit at the two
        ends of that unknown. Too coarse a stride under-samples the ~8-step-wide
        pass window of a SHORT leg (0.97 questions per window measured); any
        finite quota is spent long before the far end of a LONG one (4 of 5
        long legs spent all 12 looks 15-35 km short). A rate has neither end:
        the opening stock covers the short leg, and the refill guarantees a look
        in hand at every point of the long one.

          * stock starts at REACH_BUCKET_CAPACITY (full);
          * one look is earned per REACH_REFILL_FORWARDS translations walked,
            clamped at capacity so an idle stretch cannot bank a burst;
          * two looks are never closer than REACH_MIN_SPACING translations.

        Debiting happens HERE, before the VLM call, so a transient API error or
        an unparseable reply cannot retry the same frame every decision.
        """
        try:
            streak = int(forward_streak)
        except Exception:
            return False

        mark = int(self._reach_refill_mark.get(cur_idx, 0))
        stock = int(self._reach_tokens.get(cur_idx, REACH_BUCKET_CAPACITY))
        earned = (streak - mark) // REACH_REFILL_FORWARDS
        if earned > 0:
            stock = min(REACH_BUCKET_CAPACITY, stock + earned)
            mark += earned * REACH_REFILL_FORWARDS
            self._reach_refill_mark[cur_idx] = mark
        self._reach_tokens[cur_idx] = stock
        if stock < 1:
            return False

        last = self._last_reach_streak.get(cur_idx)
        if last is not None and streak - int(last) < REACH_MIN_SPACING:
            return False

        self._reach_tokens[cur_idx] = stock - 1
        self._last_reach_streak[cur_idx] = streak
        return True

    def proactive_commit_check(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        forward_streak: int,
        cur_idx: int,
        description: Optional[str] = None,
    ) -> bool:
        """Return True iff the harness should INJECT a `-1` commit.

        Fires only when:
          - forward_streak >= K_REACH_FORWARD_STREAK
          - hint_image is not None
          - vlm_clients is provided
          - the current subtask is not an orientation subtask
          - the per-subtask firing cap is not yet hit
          - obs has a usable RGB frame

        On True the harness should override `chosen_actions` to
        `[-1]`. On False the harness should leave the planner's
        actions in place.
        """
        # iter_003 (depth_accept_validator): depth is REMOVED from the TRIGGER.
        # naive_v3_002 keyed the fire on `_looming_state == near` and BLOCKED on
        # `far`; the diagnosis proved the `far`-block SUPPRESSES the REACH check
        # at the exact off-axis ROAD waypoints where it is needed (open
        # intersection ahead reads `far`; ep_20_1/14 stood 192-219 cm from the
        # target with ZERO reach calls). So the trigger stays a pure
        # translation-count schedule (un-suppressed: depth can no longer PREVENT
        # a fire); what changed THIS iter is that the schedule is a rate drawn
        # from a refilling bucket instead of a fixed stride against a fixed
        # quota. Depth remains an ACCEPT-VALIDATOR on the Match=1, below.
        if forward_streak < K_REACH_FORWARD_STREAK:
            return False
        if hint_image is None:
            return False
        if not vlm_clients:
            return False
        if _is_orientation_instruction(instruction):
            return False
        # --- the FINISH leg no longer buys looks with this call -------------
        # Measured over every FINISH-leg firing of the previous run: 0 accepts
        # in 15 fires below 5 000 cm of the destination, 5 accepts in 73 fires
        # ABOVE 15 000 cm. It has never once been right on a FINISH leg, and
        # those 5 accepts are 5 of the 6 fatal commits. The cause is structural,
        # not a threshold: the FINISH instruction puts the target "on your
        # left/right" and the reference photo was captured FACING it, so the
        # forward camera at the destination does not contain the reference view
        # -- both the recogniser and the perceiver independently and CORRECTLY
        # report a non-match while standing 719 cm from the target. A call whose
        # every accept is a false positive is not a detector, it is a cost, so
        # the forward look is not spent here at all and the whole of this leg's
        # budget goes to the ROAD legs, where the same call separates in-window
        # from far-field by a likelihood ratio of ~14. FINISH stays reachable
        # through the planner's own `-1` + cross-check path (unchanged); what is
        # removed is only this channel's unprompted injection.
        # `_is_finish_instruction` is the existing predicate, verified 199/199
        # against the static `task_config.json` instruction text in `hard/`.
        if _is_finish_instruction(instruction):
            return False
        if not isinstance(obs, Mapping):
            return False
        if obs.get("rgb") is None:
            return False
        # --- token bucket: earn a look every REACH_REFILL_FORWARDS ----------
        if not self._spend_reach_look(cur_idx, forward_streak):
            return False
        match = self._binary_reach_vlm_call(
            obs=obs,
            hint_image=hint_image,
            vlm_clients=vlm_clients,
            instruction=instruction,
            description=description,
        )
        if match != 1:
            self._consecutive_reach_match[cur_idx] = 0
            return False
        # ACCEPT-VALIDATOR (iter_003): honor a Match=1 only if the scene is not
        # geometrically impossible. `_strongly_far` is True only when essentially
        # nothing is close in the building-projection band -> the matched
        # landmark cannot really be here (the anchor's fatal commit at 33k cm,
        # ep_20_5, fired the proactive near-path on a distant building). VETO it.
        # Fails OPEN (do not veto) on missing/flat depth or whenever anything is
        # near, so a true close-range commit at an off-axis waypoint is never
        # blocked. Depth VALIDATES the answer; it never gated the question.
        if not _named_side_has_structure(obs, instruction):
            self._consecutive_reach_match[cur_idx] = 0
            return False
        # iter 8: require N consecutive Match=1 fires before committing.
        confirms = int(self._consecutive_reach_match.get(cur_idx, 0)) + 1
        self._consecutive_reach_match[cur_idx] = confirms
        return confirms >= N_REACH_CONFIRMATIONS_REQUIRED

    def cross_check_planner_commit(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        cur_idx: int,
        forward_streak: int = 0,
        description: Optional[str] = None,
    ) -> bool:
        """Return True iff a planner-driven `-1` should be allowed to commit.

        iter 4: independent binary REACH-VLM cross-verifier. Called by
        the harness AFTER `verify()` returns True on a non-orient
        subtask, BEFORE the harness actually executes the `-1`. The
        purpose is to filter the planner's first-call Match=1
        hallucinations (parent iter_003 failure mode).

        Semantics:
          - hint_image is None       -> True (cannot cross-check;
                                              defer to verify())
          - vlm_clients missing      -> True (cannot cross-check)
          - orient subtask           -> True (cross-check N/A)
          - obs lacks usable RGB     -> True (defer)
          - a STANDING answer from a pose the agent has not left
                                     -> that same answer, for free
          - VLM returns Match=1      -> True  (commit OK)
          - VLM returns Match=0      -> False (block; the planner's
                                        non-`-1` actions execute, and
                                        the harness answers with an
                                        in-place hold)
          - VLM returns None (error) -> True (transient API failure
                                              must NOT silently veto
                                              the planner — same
                                              defensive bias as
                                              proactive_commit_check
                                              for its return path)

        The "defer" default — return True when the check is INAPPLICABLE
        (no hint, no VLM, orient leg, no RGB, transient error) — is kept: an
        absent grounding channel must not tighten the gate.

        WHAT CHANGED THIS ITER: the three mechanisms this function used to
        arbitrate with — a firing cap, a refusal allowance, and an accept memo
        bucketed by forward progress — are replaced by ONE rule, the same rule
        the proactive look now runs on: **an answer stands until the agent has
        physically moved, and moving is what re-asks the question.**

        Every previous answer to "what happens when the allowance runs out" has
        now been refuted by measurement. `True` (the anchor) let the Nth claim
        of a leg commit with no grounding at all. `False` (iter 1) turned an
        exhausted budget into a free unbounded veto — 117 blocks / 50 episodes,
        62 with no VLM call behind them. `return walked` (iter 2, the motion
        floor) is refuted here: the floor was ALREADY satisfied at the moment of
        the refusal in 5/5 traced instances, so the refusal simply deferred the
        commit by one decision and then rubber-stamped it — 5 fires, 1 pass,
        **4 fatal commits**, each one at the exact position the recogniser had
        refused one decision earlier. It could not have gone otherwise: a block
        is answered with `HOLD_PROBE` (net-zero yaw, zero translation), so the
        next decision perceives the IDENTICAL frame, and perceive and plan both
        run at temperature 0.2. The same claim, from the same pose, with the
        same evidence, deserves the same answer — and that is now exactly what
        it gets, for free, with no VLM call spent. When the agent has walked
        XCHECK_REARM_FORWARDS translations the evidence genuinely is new, the
        standing answer expires, and the question is asked again.

        This makes the cost self-limiting without a cap: the cross-check can
        fire at most once per XCHECK_REARM_FORWARDS translations of a leg, which
        is the same rate the proactive look draws at. A budget therefore no
        longer decides anything, in either direction, anywhere in this file.
        """
        if hint_image is None:
            return True
        if not vlm_clients:
            return True
        if _is_orientation_instruction(instruction):
            return True
        if not isinstance(obs, Mapping):
            return True
        if obs.get("rgb") is None:
            return True
        # iter_003: accept-validator on the planner channel (was naive_v3_002's
        # `_looming_state == "far"` block). VETO a planner `-1` only when the
        # scene is unambiguously far EVERYWHERE in the building-projection band
        # (no structure within reach) — exactly the far-field false commit the
        # anchor fired 4x (22k-34k cm short, all fatal). Same conservative,
        # side-agnostic predicate used on the proactive path; cheap (no VLM
        # call). When anything is near (incl. an off-axis side building at a
        # true waypoint) we defer to the VLM cross-check below.
        if not _named_side_has_structure(obs, instruction):
            return False

        try:
            streak = int(forward_streak)
        except Exception:
            streak = 0

        # -- the standing answer -----------------------------------------
        # Self-derived: the key is this leg's own executed translation count.
        standing = self._xcheck_standing.get(cur_idx)
        if standing is not None:
            answered_at, verdict = standing
            if streak - int(answered_at) < XCHECK_REARM_FORWARDS:
                return bool(verdict)

        match = self._binary_reach_vlm_call(
            obs=obs,
            hint_image=hint_image,
            vlm_clients=vlm_clients,
            instruction=instruction,
            description=description,
        )
        if match is None:
            # API / parse failure: do NOT veto the planner, and do NOT let the
            # failure become a standing answer — there is nothing to stand on.
            return True
        verdict = (match == 1)
        self._xcheck_standing[cur_idx] = (streak, verdict)
        return verdict

    # ---- iter_005 (turn_to_face_side_depth): physical turn-to-face scan -------
    def request_side_scan(
        self,
        obs: Any,
        instruction: Any,
        cur_idx: int,
    ) -> Optional[str]:
        """Return 'left'|'right' iff the harness should PHYSICALLY yaw to face the
        instruction's named side and run a REACH there, else None.

        Binding constraint (parent diagnosis): off-axis FRAMING. At a reached
        ROAD/FINISH waypoint the target building sits ~90 deg off the forward
        travel axis, so the forward-only recognizer never sees it; and the
        robot_view reference hint was captured FACING the building, so a match
        REQUIRES facing it. This predicate fires the look ONLY when:
          - the subtask is non-orientation (a face-subtask must not be perturbed
            by a scan turn; orientation is handled by the cardinal burst),
          - the instruction names a side ('on your left/right'),
          - a structure reads NEAR on that side (`_side_band_near`) — a corner
            building is physically beside the agent = it is AT the waypoint. This
            is the precise TRIGGER fix for the turn mechanism that iter1's
            finish_face_commit got wrong (it turned on a blind forward-streak and
            committed 22k-34k cm short -> fatal), and
          - the per-subtask scan cap is not yet spent.
        The cap is charged HERE (on the decision to scan) so cost is bounded even
        if the subsequent turn or REACH fails.
        """
        if _is_orientation_instruction(instruction):
            return None
        side = _named_side(instruction)
        if side is None:
            return None
        if int(self._per_subtask_scan_fires.get(cur_idx, 0)) >= MAX_SIDE_SCANS_PER_SUBTASK:
            return None
        if not _side_band_near(obs, side):
            return None
        self._per_subtask_scan_fires[cur_idx] = (
            int(self._per_subtask_scan_fires.get(cur_idx, 0)) + 1)
        return side

    def scan_reach(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        cur_idx: int,
        description: Optional[str] = None,
    ) -> Optional[int]:
        """Run the SAME strict binary REACH call on a (post-turn) frame.

        Returns 1 (match), 0 (no match), or None on any failure. The harness
        calls this AFTER physically yawing to face the named side, so `obs` now
        shows the building head-on — the exact viewpoint the robot_view hint was
        captured from. The harness commits (-1) only on an explicit 1; on 0/None
        it restores the travel heading and walks on, so a wrong look NEVER
        produces a false (fatal) commit.
        """
        return self._binary_reach_vlm_call(
            obs=obs,
            hint_image=hint_image,
            vlm_clients=vlm_clients,
            instruction=instruction,
            description=description,
        )
