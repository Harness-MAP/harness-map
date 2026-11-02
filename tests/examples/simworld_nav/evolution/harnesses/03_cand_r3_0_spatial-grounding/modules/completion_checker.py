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

Constants (not map-tuned):
  K_REACH_FORWARD_STREAK = 4  (after 4 consecutive forwards, ask)
  MAX_REACH_CHECKS_PER_SUBTASK = 2  (firing cap per subtask)
"""
from __future__ import annotations
import json
import re
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

# iter 7 interval-based REACH check on long forward walks. Parent
# (s0_naiveR_006) showed the fixed 2-fire cap exhausted in the FIRST
# 2 forward-streak frames of a long walk (streak=4,5), going dormant
# for the remainder of the subtask. On long walks (ep_20_3 subtask 1:
# 37 forwards; ep_20_0 subtask 3: 153 forwards) the agent then ran
# PAST the landmark with no further visual grounding. The new gate
# fires at streak=4 then every K_REACH_INTERVAL=5 forwards within the
# same subtask, capped at MAX_REACH_FIRES_PER_SUBTASK=8 total fires.
# Same content-based binary VLM call; only the firing schedule
# changes. Short-walk subtasks behave identically to iter_006.
K_REACH_INTERVAL = 5
# iter 10: REVERT cap from 8 → 2. iter_007's cap=8 (with K_INTERVAL=5)
# was designed for long-walk subtasks (ep_20_0 subtask 3 at ~22000 cm),
# but the cap-8 coverage still terminates at streak=39 = 19500 cm, LESS
# than the 22000-cm destination distance — so cap=8 does not help on the
# case it was designed for. Meanwhile cap=8 + K_INTERVAL=5 SKIPS streak=5
# (the iter_006 first-opportunity short-walk commit window), HURTING the
# short-walk common case (ep_20_3 subtask 1, ~5000 cm). cap=2 restores
# the iter_006 "tight-clump" fire pattern (streaks 4 and 9, both within
# 4500 cm of subtask start) while leaving K_REACH_INTERVAL=5 in place as
# inert scaffolding for future tuning. Long-walk landmarks (subtask 3
# FINISH at 22000+ cm) remain unaddressed — that needs a depth-grounded
# REACH gate, out of scope for this iter.
# iter_002: raised 2 -> 12. With the depth-proximity TRIGGER (see
# `_looming_state` below) this is now a pure COST cap over the whole
# close-approach window, NOT the firing schedule — proximity decides WHEN
# to fire; this only bounds VLM calls/subtask. 12 rarely binds (only a
# handful of frames read "near" on any single approach).
MAX_REACH_FIRES_PER_SUBTASK = 12

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

# ---------------------------------------------------------------------------
# iter_003 (arrival-window REACH schedule). The look-schedule above is a fixed
# stride: first fire at forward_streak=4, then one every K_REACH_INTERVAL=5
# forwards, cap 12, for the whole leg. Measured over 305 ROAD-leg REACH calls
# (diagnosis.md §2.2): P(Match=1) is 0.21 INSIDE the 2000 cm pass window and
# 0.005 outside it, yet 86 % of the calls are spent outside — the 2500 cm
# stride across a 4000 cm window buys a median of ONE in-window look per
# episode, and 0.21 recall x 1 shot reproduces the observed ROAD 12/45 exactly.
# The budget is not the constraint (7 of 12 fires typically unspent at leg end);
# the PLACEMENT is. So the stride is replaced by an event-driven schedule with
# three modes, and the total cost cap is left where it is:
#
#   far    — cue live, no arrival event yet: stride K_REACH_INTERVAL_FAR (wider
#            than the parent's), and only FAR_FIRE_RESERVE of the cap may be
#            spent here, so the far field cannot eat a budget the window needs.
#   near   — an arrival event has fired: stride K_REACH_INTERVAL_NEAR = 1, i.e.
#            a look on EVERY decision (~1500 cm apart under the momentum burst),
#            and the streak floor drops so a short leg is covered too.
#   sweep  — the cue is unavailable for this leg (the instruction names no side,
#            or no usable object_mask), or the far reserve was spent without an
#            event: fall back to the PARENT schedule verbatim. This mode is the
#            no-regression floor — where the cue cannot speak, nothing changes.
#
# The arrival event is the cross-street gap the ROAD instruction itself names
# ("stop at the intersection with <building> on your left"): walking a block,
# the named-side building band is solid; at the intersection it opens. Measured
# on this run's own masks: named-side building fraction 0.65-0.69 median along
# the block, 0.42 median (p25 0.31) inside the pass window. The test is a drop
# against the leg's OWN running median (below), so it carries no absolute
# threshold, no map constant and no coordinate — it is self-normalizing and
# reads only obs['object_mask'].
#
# The cue is deliberately NOT a commit criterion. It fires 32 times in-window
# against 95 out-of-window across the run: far too noisy to commit on, and it
# is never asked to — it only decides WHERE THE LOOKS GO. The recognizer keeps
# the last word, and its 99.6 % ROAD specificity (1 false accept in 263
# out-of-window calls) is what makes spending extra looks cheap in risk.
K_REACH_INTERVAL_FAR = 8        # stride (forwards) before any arrival event
K_REACH_INTERVAL_NEAR = 1       # stride once the window is open == every decision
K_REACH_STREAK_FLOOR_NEAR = 2   # window-open legs may start looking earlier
FAR_FIRE_RESERVE = 4            # of MAX_REACH_FIRES_PER_SUBTASK, spendable while far

# Arrival-event detector (running-median drop on the named-side band).
GAP_WARMUP_SAMPLES = 3          # samples of history needed before the test can fire
GAP_DROP_RATIO = 0.6            # fire when frac < ratio x median(history so far)
GAP_WINDOW_FORWARDS = 12        # the dense window spans this many forwards, then re-arms
GAP_HISTORY_MAX = 40            # bound the per-leg sample ring

# iter 4 independent binary REACH-VLM cross-verifier on planner-driven
# `-1` commits for non-orient subtasks. Same model+prompt as the
# proactive REACH check, called BEFORE honoring a verify()=True from
# the planner-Match gate. Bounded by a per-subtask firing cap; not a
# distance, not a coordinate, just cost control.
MAX_XCHECK_FIRES_PER_SUBTASK = 2

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
# motion counter now counting strafes (agent.py), short/strafe completions can
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


def _named_side_building_frac(obs: Any, instruction: Any) -> Optional[float]:
    """Fraction of building-class pixels in the named-side lateral band, or None.

    iter_003: extracted from `_named_side_has_structure` so the SAME band is read
    by two consumers with two different questions:

      * `_named_side_has_structure` keeps asking the BOOLEAN question ("is this
        side empty?") as the accept-validator on a Match=1 — behaviour unchanged.
      * `CompletionChecker._update_arrival_window` asks the SHAPE question ("has
        this fraction just DROPPED relative to its own history?") to schedule the
        REACH look — see the running-median cue there.

    Returns None (not a number) whenever the band cannot be evaluated — no named
    side in the instruction, missing/odd `object_mask` — so a consumer can tell
    "no structure there" (0.0) apart from "cannot say" (None). The old boolean
    conflated the two into a fail-open True.
    """
    side = _named_side(instruction)
    if side is None:
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
    r0, r1 = int(_NAMED_SIDE_ROW[0] * h), int(_NAMED_SIDE_ROW[1] * h)
    sw = max(1, int(_NAMED_SIDE_COL_FRAC * w))
    band = a[r0:r1, 0:sw, :3] if side == "left" else a[r0:r1, (w - sw):w, :3]
    if band.size == 0:
        return None
    bld = (np.abs(band.astype(np.int16) - _NAMED_SIDE_BUILDING).sum(axis=-1) < 60)
    return float(bld.mean())


def _named_side_has_structure(obs: Any, instruction: Any) -> bool:
    """True unless the instruction names a side AND that side is EMPTY of building-class pixels."""
    frac = _named_side_building_frac(obs, instruction)
    if frac is None:                      # fails OPEN, exactly as before
        return True
    return frac > _NAMED_SIDE_MIN_FRAC


# Per-subtask cap on physical turn-to-face scans (cost control, not a distance /
# coordinate). 2 = at most two look-turns per waypoint approach.
MAX_SIDE_SCANS_PER_SUBTASK = 2


@dataclass
class CompletionChecker:
    # Per-subtask cap on REACH-check firings. Keyed by `cur_idx` (the
    # structural subtask index passed by the harness). Re-instantiated
    # per episode so no cross-episode leakage.
    _per_subtask_reach_fires: Dict[int, int] = field(default_factory=dict)
    # iter 4: separate per-subtask cap on cross-check firings.
    _per_subtask_xcheck_fires: Dict[int, int] = field(default_factory=dict)
    # iter 7: track the forward_streak at which we last fired REACH for
    # each subtask, to space fires K_REACH_INTERVAL apart across a long
    # walk instead of clumping them in the first 2 streak frames.
    _last_reach_streak: Dict[int, int] = field(default_factory=dict)
    # iter 8: count consecutive Match=1 results per subtask. Reset to 0
    # on any Match=0 (or unparseable None). Commit only when the count
    # reaches N_REACH_CONFIRMATIONS_REQUIRED.
    _consecutive_reach_match: Dict[int, int] = field(default_factory=dict)
    # iter_005 (turn_to_face_side_depth): per-subtask count of physical
    # turn-to-face scans already spent (bounds env-step + VLM cost per waypoint).
    _per_subtask_scan_fires: Dict[int, int] = field(default_factory=dict)
    # iter_003 (arrival-window REACH schedule). Per-subtask state for the
    # running-median gap cue. All three are keyed by `cur_idx` and the checker
    # is re-instantiated per episode, so nothing leaks across legs or episodes.
    #   _side_frac_hist   — the leg's own samples of the named-side building
    #                       fraction, one per proactive_commit_check call, ring-
    #                       bounded at GAP_HISTORY_MAX. The median of this list
    #                       IS the normalizer; there is no tuned absolute level.
    #   _window_opened_at — the forward_streak at which the last arrival event
    #                       fired; absent == no event on this leg yet.
    #   _far_fires        — REACH calls spent on this leg while no window was
    #                       open (bounded by FAR_FIRE_RESERVE).
    _side_frac_hist: Dict[int, List[float]] = field(default_factory=dict)
    _window_opened_at: Dict[int, int] = field(default_factory=dict)
    _far_fires: Dict[int, int] = field(default_factory=dict)

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

    def _update_arrival_window(
        self,
        obs: Any,
        instruction: Any,
        cur_idx: int,
        forward_streak: int,
    ) -> Optional[bool]:
        """Sample the named-side band and update this leg's arrival window.

        Returns True if the dense window is open at this decision, False if it
        is closed, and **None if the cue is unavailable** for this leg (no named
        side in the instruction, or no usable object_mask) — the caller reads
        None as "fall back to the parent schedule", so a leg the cue cannot see
        is never scheduled worse than before.

        The event: the ROAD instruction says "stop at the intersection with
        <building> ON YOUR LEFT". Walking the block, that side is a solid wall
        of building-class pixels; at the intersection the cross street opens and
        the fraction drops. So the test is a DROP against the leg's own running
        median — `frac < GAP_DROP_RATIO * median(samples before this one)` —
        which needs GAP_WARMUP_SAMPLES of history first and compares the frame
        only to other frames from the same leg. No absolute threshold, no map
        constant, no coordinate: the normalizer is the agent's own recent view.

        The window then spans GAP_WINDOW_FORWARDS of travel and RE-ARMS: a
        second gap later on the block re-opens it rather than being swallowed by
        the first. Sampling happens on every call, including the calls that will
        not fire a look, so the median is warm by the time it matters.
        """
        frac = _named_side_building_frac(obs, instruction)
        if frac is None:
            return None
        hist = self._side_frac_hist.setdefault(cur_idx, [])
        # Test BEFORE appending: compare this frame against its own history,
        # never against itself.
        fired = False
        if len(hist) >= GAP_WARMUP_SAMPLES:
            med = float(np.median(np.asarray(hist, dtype=np.float64)))
            if med > 0.0 and frac < GAP_DROP_RATIO * med:
                fired = True
        hist.append(float(frac))
        if len(hist) > GAP_HISTORY_MAX:
            del hist[0]
        if fired:
            self._window_opened_at[cur_idx] = int(forward_streak)
            return True
        opened = self._window_opened_at.get(cur_idx)
        if opened is None:
            return False
        return (int(forward_streak) - int(opened)) <= GAP_WINDOW_FORWARDS

    def _reach_schedule(self, cur_idx: int, window: Optional[bool]) -> Tuple[int, int]:
        """(streak_floor, stride) for this leg's REACH look, given the window state.

        Three modes, in the order they bind:
          * cue unavailable (window is None) -> PARENT schedule, verbatim.
          * window open                      -> dense: stride 1, lower floor.
          * window closed, far reserve left  -> sparse: the wide far stride.
          * window closed, far reserve spent -> PARENT schedule (the safety net:
            a leg whose gap never fires ends up with the coverage it had before,
            just shifted a little later).
        """
        if window is None:
            return K_REACH_FORWARD_STREAK, K_REACH_INTERVAL
        if window:
            return K_REACH_STREAK_FLOOR_NEAR, K_REACH_INTERVAL_NEAR
        if int(self._far_fires.get(cur_idx, 0)) < FAR_FIRE_RESERVE:
            return K_REACH_FORWARD_STREAK, K_REACH_INTERVAL_FAR
        return K_REACH_FORWARD_STREAK, K_REACH_INTERVAL

    def proactive_commit_check(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        forward_streak: int,
        cur_idx: int,
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
        # target with ZERO reach calls). So the trigger reverts to the anchor's
        # pure forward-streak schedule (un-suppressed: depth can no longer
        # PREVENT a fire). Coverage is extended across the whole approach by the
        # K_REACH_INTERVAL spacing + the raised cap, so the check stays live at
        # the waypoint instead of dying after 2 early fires. Depth is repurposed
        # below as an ACCEPT-VALIDATOR on the Match=1, not a gate on the call.
        if hint_image is None:
            return False
        if not vlm_clients:
            return False
        if _is_orientation_instruction(instruction):
            return False
        if not isinstance(obs, Mapping):
            return False

        # iter_003: sample the arrival cue FIRST, on every decision that reaches
        # here — including the ones that will not fire a look. The running median
        # is only as good as its warm-up, and the gates below would starve it.
        window = self._update_arrival_window(
            obs=obs, instruction=instruction,
            cur_idx=cur_idx, forward_streak=forward_streak)
        streak_floor, stride = self._reach_schedule(cur_idx, window)

        if forward_streak < streak_floor:
            return False
        fires = int(self._per_subtask_reach_fires.get(cur_idx, 0))
        if fires >= MAX_REACH_FIRES_PER_SUBTASK:
            return False
        # Space fires `stride` forwards apart. Under the parent this was the flat
        # K_REACH_INTERVAL; it is now whatever mode the leg is in, so the budget
        # is spent where P(Match=1) is 0.21 instead of where it is 0.005.
        last = self._last_reach_streak.get(cur_idx)
        if last is not None and forward_streak - last < stride:
            return False
        if obs.get("rgb") is None:
            return False
        # Count this firing (cost gate) before the call so a transient API
        # error / unparseable response cannot retry every step. A look taken with
        # no window open also draws on the far reserve, which is what keeps the
        # far field from spending a budget the arrival window will need.
        self._per_subtask_reach_fires[cur_idx] = fires + 1
        self._last_reach_streak[cur_idx] = forward_streak
        if window is False:
            self._far_fires[cur_idx] = int(self._far_fires.get(cur_idx, 0)) + 1
        match = self._binary_reach_vlm_call(
            obs=obs,
            hint_image=hint_image,
            vlm_clients=vlm_clients,
            instruction=instruction,
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
          - cap reached for cur_idx  -> True (cost cap; defer)
          - VLM returns Match=1      -> True  (commit OK)
          - VLM returns Match=0      -> False (block; planner's
                                               non-`-1` actions execute)
          - VLM returns None (error) -> True (transient API failure
                                              must NOT silently veto
                                              the planner — same
                                              defensive bias as
                                              proactive_commit_check
                                              for its return path)

        The "defer" default — return True when we can't perform the
        check — preserves the iter_003 behavior and ensures the
        cross-check never tightens the gate compared to verify alone
        when grounding is unavailable.
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
        fires = int(self._per_subtask_xcheck_fires.get(cur_idx, 0))
        if fires >= MAX_XCHECK_FIRES_PER_SUBTASK:
            return True

        # Count this firing before the call so transient failures
        # don't retry every step.
        self._per_subtask_xcheck_fires[cur_idx] = fires + 1
        match = self._binary_reach_vlm_call(
            obs=obs,
            hint_image=hint_image,
            vlm_clients=vlm_clients,
            instruction=instruction,
        )
        if match is None:
            # API / parse failure: do NOT veto the planner. Symmetric
            # with the "defer" defaults above — the cross-check only
            # tightens the gate when it has a confident Match=0.
            return True
        return match == 1

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
        )
