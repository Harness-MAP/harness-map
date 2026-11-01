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

# iter 4 independent binary REACH-VLM cross-verifier on planner-driven
# `-1` commits for non-orient subtasks. Same model+prompt as the
# proactive REACH check, called BEFORE honoring a verify()=True from
# the planner-Match gate. Bounded by a per-subtask firing cap; not a
# distance, not a coordinate, just cost control.
# Restored to the anchor's 2: this is a COST cap again, nothing else. The
# previous iter raised it to 3 and made cap-exhaustion itself decide the
# commit (`return False`), which produced 117 blocks over 50 episodes, 62 of
# them with no VLM call behind them at all. A budget is not evidence, so the
# budget no longer answers the question -- see `MAX_XCHECK_REFUSALS_PER_SUBTASK`.
MAX_XCHECK_FIRES_PER_SUBTASK = 2

# Granularity of the cross-check verdict cache, in forward actions. Two
# claims raised within the same bucket of forward progress are the same
# claim from (approximately) the same place. Expressed in ACTION COUNTS, not
# cm. ONLY `True` verdicts are cached: a memoised refusal at a position the
# agent has not left is a self-fulfilling veto (the agent cannot change the
# evidence without moving, and the refusal is what stopped it moving).
XCHECK_CACHE_STREAK_BUCKET = 2

# --- arrival-channel arbitration (this iter) --------------------------------
# How many times the 2-image cross-check may REFUSE the planner's `-1` on one
# leg before it loses its veto over that leg. The measured discrimination of
# this channel is ~none (in-window accept rate 0.205 vs far-field 0.043 under
# the anchor wording; 0.017 vs 0.043 under the replacement wording -- two
# far-apart operating points on the same flat ROC), while the FIRST planner
# `-1` of a leg is in-window 7/12. A channel that weak may delay a claim once
# so the richer 4-image channel gets another look at a fresh frame; it may not
# hold an unbounded veto. After the allowance is spent the claim is honoured
# iff the agent has actually walked this leg (K_REACH_FORWARD_STREAK, the
# floor that already guards the proactive path), which is derived purely from
# this leg's own action history.
MAX_XCHECK_REFUSALS_PER_SUBTASK = 1

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
# motion counter now counting strafes (harness.py), short/strafe completions can
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


def _aux_b64(obs: Any, keys: tuple) -> Optional[str]:
    """First present `obs[key]` among `keys`, PNG-base64'd, or None.

    Used to attach the segmentation / depth renderings to the arrival call.
    Mirrors `prompt_assembler`'s key-fallback ("depth_map" then "depth" --
    the env actually ships the frame under "depth") and fails soft: a missing
    or unencodable channel simply is not attached, so the call degrades to
    exactly the two-image form it had before.
    """
    if not isinstance(obs, Mapping):
        return None
    for k in keys:
        v = obs.get(k)
        if v is None:
            continue
        try:
            return numpy_to_base64(np.asarray(v))
        except Exception:
            return None
    return None


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


# ---- GEOMETRIC ARRIVAL (this iter): the sidewalk -> roadway class flip ---------------
# The binding constraint in diagnosis.md is not WHICH wording or WHICH evidence the
# hint-photo recognizer gets -- that channel is closed at LR ~= 1.6 over three measured
# operating points -- it is that a leg whose stop condition is a GEOMETRIC event ("stop
# at the intersection") is decided by an APPEARANCE question ("does this frame look like
# the reference photo"). `obs['object_mask']` carries the event directly: the ground
# class in front of the robot flips from sidewalk to roadway the moment the cross street
# opens ahead of it. This is the same move that makes the TURNING leg the only reliable
# one in this harness -- replace "does this look like the place?" with a closed-form read
# of what the leg is actually defined by.
#
# Palette RE-PROBED on this candidate's own traces (the 50-episode parent run, the
# object_mask image of every perceive call; 4 dominant classes, all EXACT colours):
#     (255,255,0) yellow     = SIDEWALK / CROSSWALK   (what the robot walks on, ~48%/frame)
#     (128,0,128) purple     = BUILDING facades
#     (0,255,0)   green      = SKY          (0,128,0) dark green = TREES
#     (0,127,255) blue       = ROADWAY / DRIVEWAY surface          <-- the cue
# Row profile of a mid-block frame: the bottom rows are 100% yellow and the roadway class
# is a thin band near the horizon; walking into a junction floods the forward band with it.
#
# Operating point, measured over the 45 ROAD legs of the parent run at the parent's own
# ~1-decision-per-3-env-steps cadence. Rule = the FIRST frame of the leg whose center
# forward band is more than `_ROAD_ARRIVAL_FRAC` roadway; scored against the true L1 to
# the graded waypoint (analysis only -- no coordinate enters the code below):
#     fires INSIDE the 2000 window : 24/45   (incumbent hint-match channel: 6/45)
#     fires BEFORE the window      : 11/45   (incumbent: 9 far-field commits, all fatal)
#     never fires                  :  8/45   (all 8 of those legs fail today as well)
# The plateau is broad -- every (row band, threshold) pair from (0.30-0.55, 0.03) through
# (0.50-0.75, 0.03) scores 22-25 in-window -- so what is encoded here is "roadway occupies
# a few percent of the forward view", not a knife-edge constant; and both halves of the
# episode set move the same way (9 in / 7 early, and 15 in / 4 early).
#
# Everything below is a FRACTION of the frame: no centimetre, no coordinate, no map-tuned
# literal; the only task input is the STATIC `task_spec` instruction text.
_ROADWAY_CLASS_RGB = np.array([0, 127, 255], dtype=np.int16)   # roadway / driveway surface
_ROADWAY_CLASS_BGR = np.array([255, 127, 0], dtype=np.int16)   # same class, channel order flipped
_MASK_CLASS_TOL = 20             # palette colours are exact; the tol only absorbs re-encoding
_ROAD_BAND_ROW = (0.45, 0.70)    # forward band rows (fractions of H) -- the ground ahead
_ROAD_BAND_COL = (0.30, 0.70)    # center columns (fractions of W) -- the travel corridor
_ROAD_ARRIVAL_FRAC = 0.03        # more than this much roadway in the band == the junction is HERE


def _roadway_band_fraction(obs: Any) -> Optional[float]:
    """Fraction of the center forward band of `obs['object_mask']` that is roadway class.

    Returns None -- "this channel cannot be evaluated", distinct from 0.0 -- when the mask
    is absent or malformed, so the caller can hand the leg back to the VLM path instead of
    silently reading a missing sensor as "no intersection". Matches the class in BOTH
    channel orders: unlike the purple building class the roadway colour is not palindromic,
    and nothing in the sanctioned surface guarantees RGB over BGR.
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
    r0, r1 = int(_ROAD_BAND_ROW[0] * h), int(_ROAD_BAND_ROW[1] * h)
    c0, c1 = int(_ROAD_BAND_COL[0] * w), int(_ROAD_BAND_COL[1] * w)
    band = a[r0:r1, c0:c1, :3]
    if band.size == 0:
        return None
    b = band.astype(np.int16)
    hit = ((np.abs(b - _ROADWAY_CLASS_RGB).sum(axis=-1) < _MASK_CLASS_TOL)
           | (np.abs(b - _ROADWAY_CLASS_BGR).sum(axis=-1) < _MASK_CLASS_TOL))
    return float(hit.mean())


# A relative-turn leg ("Turn right at the intersection") also names an intersection, but it
# is graded on HEADING and is already owned, with zero VLM calls, by the ActivePerception
# turn burst. Mirrors `active_perception._TURN_RELATIVE_RE` deliberately rather than
# importing it: this module must not depend on another module's private matcher.
_TURN_VERB_RE = re.compile(
    r'\b(?:turn\s+(?:left|right)|make\s+a\s+(?:left|right)\s+turn)\b',
    re.IGNORECASE,
)


def _is_intersection_stop_instruction(instruction: Any) -> bool:
    """True iff this leg's stop condition is "the intersection ahead of me".

    The generator's ROAD template is "Continue straight and stop at the intersection with
    <building> on your left/right"; FINISH is "Proceed forward until reaching <building>
    ..., stop and face that building" (never the word intersection); ORIENTATION is the
    cardinal template; TURNING is the turn template. Four conjuncts, each one excluding a
    leg type that is owned by a channel that already works -- a false positive here would
    hand a heading-graded or facing-graded leg to a position-only detector.
    """
    if not isinstance(instruction, str) or not instruction:
        return False
    if _is_orientation_instruction(instruction):
        return False
    if _TURN_VERB_RE.search(instruction):
        return False
    if _is_finish_instruction(instruction):
        return False
    return _INTERSECTION_RE.search(instruction) is not None


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
    # Memo of cross-check verdicts, keyed by (subtask index,
    # forward_streak // XCHECK_CACHE_STREAK_BUCKET). Only confirmed
    # ACCEPTS land here (this iter): a transient API failure is never cached,
    # and neither is a refusal, so nothing negative can harden into a
    # standing veto over a stretch of road the agent is stuck on.
    _xcheck_verdict_cache: Dict[Any, bool] = field(default_factory=dict)
    # How many times the cross-check has refused the planner on each leg.
    # This -- not the cost cap -- is what bounds the veto (this iter).
    _xcheck_refusals: Dict[int, int] = field(default_factory=dict)
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
    # this iter: the geometric arrival latch. True once the roadway-class cue has
    # fired on that leg, so a single leg can raise at most one geometric commit.
    _road_arrival_fired: Dict[int, bool] = field(default_factory=dict)
    # Which channel raised the most recent commit ("road_surface" | "reach_vlm").
    # Written here, read by harness.py only to LABEL the note it appends -- it is a
    # trace-visibility field and nothing in the decision path branches on it.
    last_commit_channel: str = ""

    def road_surface_arrival(
        self,
        obs: Any,
        instruction: Any,
        cur_idx: int,
    ) -> Optional[bool]:
        """The `-1` decision for an intersection-stop leg, read off the segmentation mask.

        Three-valued, and the third value is the point of it:
          - ``None``  -- this channel does not own the leg (not an intersection-stop leg,
                         or the mask is missing/malformed). The caller falls through to
                         the hint-photo VLM path *exactly* as before: this function can
                         only ever REPLACE that path on the legs it can actually decide.
          - ``True``  -- the junction is in front of the robot NOW: commit.
          - ``False`` -- not yet, and the leg is owned, so the VLM path stays suppressed.
                         This is the half that removes the far-field false commits: on a
                         leg defined by a junction, "I have not seen the junction" is a
                         positive statement about the world, not a missing answer.

        Latches once per leg: a `-1` the env grades FAIL ends the episode outright, so a
        leg may spend exactly one geometric claim and may not mash the cue.

        Honest inputs only: `obs['object_mask']` (sanctioned perception) and the static
        `task_spec` instruction text. No coordinate, no ground truth, no VLM call, no cost.
        """
        if not _is_intersection_stop_instruction(instruction):
            return None
        frac = _roadway_band_fraction(obs)
        if frac is None:
            return None
        if self._road_arrival_fired.get(cur_idx):
            return False
        if frac <= _ROAD_ARRIVAL_FRAC:
            return False
        self._road_arrival_fired[cur_idx] = True
        self.last_commit_channel = "road_surface"
        return True

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
        # --- EVIDENCE: what the question is asked OVER (the lever) ---------
        # Until now this call decided every commit in the harness from TWO
        # RGB frames -- no segmentation, no depth, no description -- while the
        # perceive call one stage earlier was handed all four channels and is
        # measurably the better arrival detector. The criterion above is a
        # question about geometry ("the same place", "prominent"), and RGB
        # alone does not carry the geometry: a facade 400 cm away and the same
        # facade 40 000 cm away differ mostly in DEPTH and in how the building
        # class is laid out across the frame, which is exactly what these two
        # renderings encode. The legend is the sensor-truth one the perceiver
        # already ships (prompt_assembler): probe-verified JET polarity, the
        # real 4-class palette, and an explicit ban on reading brightness as
        # distance -- without it the model reads the depth frame backwards.
        sys_prompt += (
            "\n\nYou are given, for the CURRENT view only, two extra sensor "
            "renderings and one text description of the same frame. Use them "
            "to ground the judgement — the Match criterion above is unchanged.\n"
            "- SEGMENTATION mask: purple = buildings, YELLOW = the SIDEWALK / "
            "crosswalk you are walking on, BLUE = the ROADWAY surface of a "
            "street, green = sky, dark green = trees. Yellow and blue are two "
            "DIFFERENT surfaces: if the ground ahead of you has changed from "
            "yellow sidewalk to blue roadway, you have stepped into an "
            "intersection. It also tells you which pixels are facade and how "
            "the facade is arranged around the navigator (ahead across open "
            "road, or alongside and running off the edge of the view).\n"
            "- DEPTH image: a JET colormap normalized PER FRAME — BLUE = "
            "nearest, CYAN/GREEN = mid-range, RED/ORANGE = farthest; BOTH "
            "extremes render dark, so never read brightness as distance. It "
            "is not metric and its scale resets every frame, so use hue for "
            "coarse WITHIN-frame near/far only, never to count steps. A "
            "landmark still down the street reads mid-to-far where its facade "
            "is, with near (blue) road between; a landmark you have arrived "
            "at reads near where its facade is.\n"
            "- DESCRIPTION: what a separate perception pass wrote about this "
            "same frame a moment ago. It is a second opinion on identity and "
            "layout, not a verdict; it does not know the reference image."
        )
        instr_text = instruction if isinstance(instruction, str) else ""
        desc_text = description if isinstance(description, str) else ""
        desc_text = desc_text.strip()[:900]
        mask_b64 = _aux_b64(obs, ("object_mask",))
        depth_b64 = _aux_b64(obs, ("depth_map", "depth"))
        user_content: List[Dict[str, Any]] = [
            {"type": "text",
             "text": f"Subtask instruction: {instr_text}"},
            {"type": "text", "text": "Current egocentric view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},
        ]
        if mask_b64 is not None:
            user_content.append(
                {"type": "text",
                 "text": "Segmentation mask of the current view:"})
            user_content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{mask_b64}"}})
        if depth_b64 is not None:
            user_content.append(
                {"type": "text",
                 "text": "Depth (JET) image of the current view:"})
            user_content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{depth_b64}"}})
        if desc_text:
            user_content.append(
                {"type": "text",
                 "text": f"Description of the current view: {desc_text}"})
        user_content.extend([
            {"type": "text", "text": "Reference landmark view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{hint_b64}"}},
            {"type": "text",
             "text": ('Output strict JSON: {"Match": 0 or 1, '
                      '"Reason": "<one short clause>"}. No prose.')},
        ])

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
        # THIS ITER -- the arrival channel for an intersection-stop leg is the
        # segmentation geometry, not the hint photo. Asked FIRST, before every cost
        # gate below, because it is free (no VLM call, no token) and because the
        # gates below are calibrated for a channel that no longer decides this leg.
        # `None` means "not my leg" (FINISH / ORIENTATION / TURNING, or no usable
        # mask) and falls through to the unchanged VLM path underneath.
        road = self.road_surface_arrival(obs, instruction, cur_idx)
        if road is not None:
            return road
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
        if forward_streak < K_REACH_FORWARD_STREAK:
            return False
        if hint_image is None:
            return False
        if not vlm_clients:
            return False
        if _is_orientation_instruction(instruction):
            return False
        fires = int(self._per_subtask_reach_fires.get(cur_idx, 0))
        if fires >= MAX_REACH_FIRES_PER_SUBTASK:
            return False
        # Space fires K_REACH_INTERVAL forwards apart so the cap is spent across
        # the whole walk (reaching the waypoint) rather than clumped at the start.
        last = self._last_reach_streak.get(cur_idx)
        if last is not None and forward_streak - last < K_REACH_INTERVAL:
            return False
        if not isinstance(obs, Mapping):
            return False
        if obs.get("rgb") is None:
            return False
        # Count this firing (cost gate) before the call so a transient API
        # error / unparseable response cannot retry every step.
        self._per_subtask_reach_fires[cur_idx] = fires + 1
        self._last_reach_streak[cur_idx] = forward_streak
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
        if confirms >= N_REACH_CONFIRMATIONS_REQUIRED:
            self.last_commit_channel = "reach_vlm"
            return True
        return False

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
          - memoized ACCEPT present  -> True  (free re-ask; refusals
                                               are never memoized)
          - refusal allowance spent  -> motion floor decides (see
                                        MAX_XCHECK_REFUSALS_PER_SUBTASK)
          - cost cap reached         -> motion floor decides (a budget
                                        is not evidence: it may not
                                        veto, and it may not rubber-
                                        stamp either)
          - VLM returns Match=1      -> True  (commit OK)
          - VLM returns Match=0      -> False (block, and spend one of
                                        this leg's refusals; the
                                        planner's non-`-1` actions
                                        execute)
          - VLM returns None (error) -> True (transient API failure
                                              must NOT silently veto
                                              the planner — same
                                              defensive bias as
                                              proactive_commit_check
                                              for its return path)

        The "defer" default — return True when the check is INAPPLICABLE
        (no hint, no VLM, orient leg, no RGB, transient error) — is kept: an
        absent grounding channel must not tighten the gate.

        WHAT CHANGED THIS ITER: neither of the two answers this function has
        shipped for budget exhaustion is right. `True` (the anchor) let the
        Nth claim of a leg commit with no grounding at all, at whatever
        position the earlier vetoes had walked the agent to; `False` (the
        previous iter) turned an exhausted budget into a free, unbounded veto
        — 117 blocks / 50 episodes, 62 of them with no VLM call behind them,
        every one of which then fed the harness's empty-action fallback and
        physically carried the agent out of the window it was standing in.
        The cap is a COST bound and must not be the thing that decides. The
        veto is bounded instead by REFUSALS (an evidence event), and once the
        cross-check has had its say the claim falls back to the motion floor
        `K_REACH_FORWARD_STREAK` — the same self-derived "have I actually
        walked this leg" test that already guards the proactive path, and the
        one signal measured to separate the planner's in-window claims (all
        at streak >= 9) from its zero-motion ones (streak 0 and 3).
        """
        # THIS ITER -- on an intersection-stop leg the planner's `-1` is answered by
        # the same geometric detector that raises the harness's own commits, and the
        # VLM cross-check is not spent at all. Measured on the parent run: of the 15
        # planner `-1`s that reached this function, 3 were raised inside the window
        # and 9 out of it were fatal, and all 3 of the good ones sit in episodes where
        # the detector also fires in-window -- so deferring to the detector keeps the
        # planner's correct claims and drops its far-field ones. Returning False here
        # costs the planner nothing but the commit: its remaining actions still run,
        # and the bounded in-place HOLD in harness.py covers the bare-`-1` case.
        road = self.road_surface_arrival(obs, instruction, cur_idx)
        if road is not None:
            return road
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
        # -- the motion floor: what decides when the cross-check cannot ---
        # Derived only from this leg's own executed action history (the
        # harness counts translations since the last `-1`). No coordinate, no
        # target, no privileged field.
        try:
            walked = int(forward_streak) >= K_REACH_FORWARD_STREAK
        except Exception:
            walked = False

        # -- refusal allowance ------------------------------------------
        # The cross-check gets to stop this leg's commit a bounded number of
        # times. The point of the delay is that the harness answers a block
        # with an in-place hold, so the NEXT decision re-perceives the same
        # place with the richer 4-image channel; a second claim from a leg
        # the agent has actually walked is the planner insisting, and the
        # planner is the better-measured detector of the two.
        refusals = int(self._xcheck_refusals.get(cur_idx, 0))
        if refusals >= MAX_XCHECK_REFUSALS_PER_SUBTASK:
            return walked

        # -- accept memo (this iter: ACCEPTS ONLY) -----------------------
        # The planner re-raises the same `-1` on consecutive decisions; a
        # confirmed yes from the same stretch of road is re-answered for
        # free. A memoized NO is deliberately impossible: the agent cannot
        # change the evidence without moving and the refusal is precisely
        # what stops it moving, so caching one makes it self-fulfilling.
        try:
            bucket = int(forward_streak) // XCHECK_CACHE_STREAK_BUCKET
        except Exception:
            bucket = 0
        cache_key = (cur_idx, bucket)
        if self._xcheck_verdict_cache.get(cache_key) is True:
            return True

        fires = int(self._per_subtask_xcheck_fires.get(cur_idx, 0))
        if fires >= MAX_XCHECK_FIRES_PER_SUBTASK:
            # Cost cap: the channel exists, we declined to spend it. That is
            # a statement about the BUDGET, not about the world, so it
            # neither vetoes nor rubber-stamps — it hands the claim to the
            # motion floor, exactly as an exhausted refusal allowance does.
            return walked

        # Count this firing before the call so transient failures
        # don't retry every step.
        self._per_subtask_xcheck_fires[cur_idx] = fires + 1
        match = self._binary_reach_vlm_call(
            obs=obs,
            hint_image=hint_image,
            vlm_clients=vlm_clients,
            instruction=instruction,
            description=description,
        )
        if match is None:
            # API / parse failure: do NOT veto the planner, do NOT memoize,
            # and do NOT spend a refusal. Symmetric with the "defer"
            # defaults above — the cross-check only tightens the gate when
            # it has a confident answer.
            return True
        if match == 1:
            self._xcheck_verdict_cache[cache_key] = True
            return True
        self._xcheck_refusals[cur_idx] = refusals + 1
        return False

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
        # THIS ITER: an intersection-stop leg whose mask is readable is owned by the
        # geometric detector, so the two-turn off-axis scan (1 fire / 0 commits in 50
        # episodes last run, 4 fires / 0 commits the run before) is not spent on it.
        # The scan stays live on the FINISH leg, which is what it was built for --
        # that leg IS graded on facing the named-side building.
        if _is_intersection_stop_instruction(instruction) and _roadway_band_fraction(obs) is not None:
            return None
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
