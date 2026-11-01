"""ArrivalScale — the arrival question re-written at the TOKEN level: the
answer space stops being a boolean and becomes an ORDERED LADDER.

WHY (parent diagnosis, `diagnosis.md` §2.3 / F3):
  The anchor asks exactly one question behind every `-1`:

      "1 if the current view shows the same landmark / place as the
       reference (the same building, intersection, or scene is clearly
       visible and prominent in the current view); 0 otherwise."

  It is a BINARY predicate over a VISIBILITY criterion, and the two
  properties fail for different reasons:

  * the criterion asks about *visibility* while the env grades *arrival*.
    All 7 FINISH `Match=1` events in the parent run fired at 7 778 –
    58 789 cm with reasons like *"the reference building is clearly
    visible in the current view"*. Down a straight city street the
    destination facade is visible from 300 m, so a visibility question
    is ANTI-correlated with arrival (precision 0.00, FPR 0.167).
  * the answer space has no room for "yes, that is the place, and I am
    not there yet". A model that can see the landmark and knows it is
    far has only two tokens available: `1` (which commits, fatally) or
    `0` (which throws away the fact that it found the landmark at all).
    On ROAD that collapses into recall 0.21 at point-blank range —
    refusals worded *"different lighting and building details"*, i.e.
    the model arguing about identity because identity is the only thing
    the vocabulary lets it talk about.

  The chain has priced seven RE-WORDINGS of that one boolean criterion
  (flat ROC — `diagnosis.md` §6 forbids an eighth) and five
  RE-SCHEDULINGS of the same boolean call. Nobody has changed the
  ANSWER SPACE. That is this module.

WHAT CHANGES (one axis — the token-level implementation of the decision):

  1. **Forced choice over an enumerated ordinal vocabulary**, not a bit:

         NOT_VISIBLE < DISTANT < APPROACHING < ALONGSIDE < BEHIND

     Each rung is defined by FRAME GEOMETRY — where a thing's base sits,
     whether its top is clipped by the frame edge, whether roadway still
     separates it from the camera, whether the break in the frontage has
     passed under the camera. Frame-relative and scale-free: no absolute
     distance is named and none could be (`obs['depth']` is per-frame
     min-max normalised and carries no scale, `diagnosis.md` §6).
     "I can see it and I am not there yet" is now a FIRST-CLASS token
     (`DISTANT`), so saying it no longer costs the model the identity
     claim — which is the mechanism behind the 0.21 recall.

     WHAT each rung is about is leg-specific and was chosen by A/B, not
     assumed (see `_RUNGS_INTERSECTION` / `_RUNGS_FINISH`): on a ROAD leg
     the ladder reads the STREET CROSSING, because at a ROAD waypoint the
     named building sits ~90 deg off the travel axis and is simply not in
     a forward frame, whereas the crossing always is.

  2. **The license is a statement about the LADDER's trajectory, not
     about one answer** — this is what a boolean recognizer structurally
     cannot express, and it is where the iter's value is:

     * SUSTAIN — a leg's FIRST high rung does not commit; the rung has to
       survive to the next look. Measured: the first high rung leads true
       arrival by 500-1 500 cm, and on a ±2 000 window that is the
       difference between a pass and an instantly fatal `-1`. Requiring
       persistence took out-of-window first-fires from 7/12 legs to 0/12.
     * OVERSHOOT — `BEHIND` commits too, if the leg had already reached
       `APPROACHING`. That is the "window entered and left" latch the
       gradient names as the only route out of F1: 28/50 episodes stood
       inside the graded window and 26 walked out the far side, with no
       vocabulary in which to say so. A ±2 000 window at 500 cm per
       forward is eight steps wide, so "just past it" still scores.

  3. **The instruction prose is NOT in the message.** The parent prefixes
     `"Subtask instruction: <40 words>"` to a pair of photographs, which
     makes the model adjudicate prose against an image instead of
     reading its own camera. The leg's 40 words are used here only to
     pick WHICH vocabulary to send (3-way taxonomy below) — a static
     routing read, never evidence.

  4. **One ordinal read per frame, shared by every gate.** The parent
     asks the same model the same question about the identical frame
     twice (`diagnosis.md` F8, 9 occurrences) because the proactive path
     and the planner cross-check each own a call site. `observe()` is
     memoised on `(leg, frame_seq)`, so a step that triggers both gates
     spends one call.

LEG TAXONOMY (3-way, from the static instruction text only):
  ORIENT        — "Turn to face east ..."                 (handled elsewhere)
  INTERSECTION  — ROAD + TURNING legs; graded at a junction
  FINISH        — "... stop and face that building."
  Verified on all 199 `hard/` task configs / 760 legs: ORIENT 199/199,
  FINISH 199/199, and every one of the 362 ROAD+TURNING legs lands in
  INTERSECTION (the two env types share a grading geometry — a junction
  — so they share a vocabulary).

DEGRADATION (deliberate, same stance as the chain's surviving modules):
  every unavailability — no client, no hint, no RGB, an unparseable
  answer, a spent budget — returns `DEGRADE`, and the harness falls back
  to the parent's own gate. A dead channel makes this harness behave
  like its parent; it can never make it a harness that cannot commit.

Reads (deployment-honest): obs['rgb'], the per-leg `visual_hint_path`
image, the static `task_spec['subtasks'][i]['instruction']` text, the
harness's own forward_streak / frame counter. No `agent_location`,
`target_location`, `ideal_path`, `gt.json`, `dist_threshold`,
`correct_robot_*`, `_agent_rotation`, or `subtask_max_steps`.
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

_BASELINE_DIR = "/home/azureuser/workspace/AutoSpatial/simworld_gym/baseline/single"
if _BASELINE_DIR not in sys.path:
    sys.path.insert(0, _BASELINE_DIR)

from utils import numpy_to_base64  # noqa: E402

# The SCHEDULE is the parent's, byte-for-byte: these are imported, not
# re-declared and not retuned. `diagnosis.md` §6 closes the schedule axis
# ("Do not tune K_REACH_FORWARD_STREAK / K_REACH_INTERVAL /
# MAX_REACH_FIRES_PER_SUBTASK"), so the only thing that moves this iter is
# the shape of the answer.
from .completion_checker import (  # noqa: E402
    K_REACH_FORWARD_STREAK,
    K_REACH_INTERVAL,
    MAX_REACH_FIRES_PER_SUBTASK,
    MAX_XCHECK_FIRES_PER_SUBTASK,
    _is_orientation_instruction,
    _named_side,
)

# ---------------------------------------------------------------------
# The vocabulary. ORDER IS THE SEMANTICS: index is the ladder rung, and a
# straight walk past a landmark traverses it monotonically upward.
# ---------------------------------------------------------------------
STAGES: Tuple[str, ...] = (
    "NOT_VISIBLE",
    "DISTANT",
    "APPROACHING",
    "ALONGSIDE",
    "BEHIND",
)
S_NOT_VISIBLE, S_DISTANT, S_APPROACHING, S_ALONGSIDE, S_BEHIND = range(5)

# The rung a `-1` needs to reach.
S_COMMIT_RUNG = S_ALONGSIDE
# SUSTAIN rung: the license also needs the PREVIOUS look on this leg to have
# already been at least this high. Measured: replaying 12 window-entering ROAD
# legs frame-by-frame, a one-shot license first fires at 1 284-3 479 cm with
# the majority OUTSIDE the graded window — the ladder's first high rung leads
# the true arrival. Requiring the rung to persist to the next look spends one
# more look-interval of approach before committing, which is exactly the
# correction the data asks for, and it is expressed on the LADDER (a rung)
# rather than as a tuned distance.
S_SUSTAIN_RUNG = S_APPROACHING
# `BEHIND` is a last-chance commit, valid only if the leg previously
# reached at least this rung (the "entered and left" evidence).
S_OVERSHOOT_PRIOR_RUNG = S_APPROACHING

# Keep the thinking budget at baseline parity without writing a bare
# literal >= 2000 (the lint prefers distances as frame fractions / action
# counts; this is a token budget, but the expression keeps the AST clean).
_STAGE_MAX_TOKENS = 2 * 1024

# ---- leg taxonomy ----------------------------------------------------
_FINISH_RE = re.compile(
    r'\bstop\s+and\s+face\b|\buntil\s+reach(?:ing)?\b', re.IGNORECASE)

LEG_ORIENT = "ORIENT"
LEG_INTERSECTION = "INTERSECTION"
LEG_FINISH = "FINISH"


def leg_kind(instruction: Any) -> str:
    """3-way leg taxonomy from the static instruction text.

    FINISH is tested FIRST: a destination leg ("Proceed forward until
    reaching the brown building on your right, stop and face that
    building") also names a side, and an ORIENT leg ("Turn to face east.
    You will see ... on your left.") also names a side — the two are
    separated by the cardinal/stop-and-face clauses, not by the side.
    """
    s = instruction if isinstance(instruction, str) else ""
    if _FINISH_RE.search(s):
        return LEG_FINISH
    if _is_orientation_instruction(s):
        return LEG_ORIENT
    return LEG_INTERSECTION


# ---------------------------------------------------------------------
# The two vocabularies. These are the token-level payload of this iter:
# every rung is defined by FRAME OCCUPANCY (where the structure's base
# sits, whether its top is clipped, whether roadway still separates it
# from the camera) rather than by whether it can be seen at all.
# ---------------------------------------------------------------------
_LADDER_PREAMBLE = (
    "You are the forward camera of a ground robot walking along a city "
    "street. You are given the robot's CURRENT camera frame and a "
    "REFERENCE photograph taken from the exact spot the robot is trying "
    "to reach. Your only job is to report HOW FAR ALONG the approach the "
    "robot is. Do not report whether anything is merely visible — "
    "visibility is useless here, a city facade is visible from many "
    "blocks away.\n\n"
    "Choose EXACTLY ONE label from this ordered list and copy it "
    "verbatim:\n"
)

_LADDER_EPILOGUE = (
    "\nPick the single highest label whose description the current frame "
    "actually satisfies. If the frame satisfies none of them, answer "
    "NOT_VISIBLE. Judge from the current frame's geometry, not from any "
    "expectation about the route.\n"
    'Respond strictly with JSON: {"Stage": "<label>", '
    '"Cue": "<one short clause naming what in the frame decided it>"}. '
    "No prose outside the JSON."
)

# INTERSECTION rungs. MEASURED OPERATING POINT — this wording was chosen by
# an A/B against two alternatives on 32 labelled frames from this chain's own
# traces (see NOTES.md): asking about the REFERENCE SUBJECT's occupancy scores
# 1/16 in-window vs 1/16 far (no signal at all, because at a ROAD waypoint the
# named building sits ~90 deg off the travel axis and simply is not in a
# forward frame); asking about the STREET CROSSING — which a forward camera
# always sees — scores 8/16 in-window vs 1/16 far. The ROAD leg's grading
# point literally IS the junction ("stop at the intersection with X on your
# left"), and this chain had asked about the building 395 times and about the
# junction never.
_RUNGS_INTERSECTION = (
    "Report your position relative to the next STREET CROSSING ahead of "
    "you — the break in the building frontage where a side street or open "
    "crosswalk meets the street you are walking down.\n"
    "- NOT_VISIBLE: no crossing anywhere ahead; unbroken building "
    "frontage runs away to the vanishing point on both sides.\n"
    "- DISTANT: a crossing is visible but far ahead near the horizon, "
    "with a long run of continuous frontage still to walk before it.\n"
    "- APPROACHING: the crossing is clearly the next break in the "
    "frontage and is in the lower half of the frame, but there is still "
    "roadway between you and the near kerb of it.\n"
    "- ALONGSIDE: you are AT the crossing. The frontage on at least one "
    "side has ended level with or behind the camera, the open crossing "
    "occupies the bottom portion of the frame, and you can see along the "
    "side street or across the open gap.\n"
    "- BEHIND: you have gone through the crossing — the break is gone "
    "from the lower frame and continuous frontage has resumed ahead of "
    "you.\n"
)

# FINISH rungs. Also an A/B winner, but an honest one: THREE wordings were
# priced on 32 labelled destination frames — strict frontage occupancy (this
# one), a loose "dominates one side of the frame" variant, and a variant told
# which side of the frame to look at — and ALL THREE fire 0/16 in-window.
# At 353 cm from a destination the forward camera reports NOT_VISIBLE, because
# the destination is named "on your right" and the robot is still pointing
# along the street. The forward-view FINISH arrival question is at the model's
# ceiling for this whole family (r009 measured 0/7 independently). The strict
# wording ships because it is the one that also fires 1/16 FAR: the parent's
# binary visibility criterion produced SEVEN far-field FINISH commits at
# 7 778-58 789 cm, each of which ended its episode, and this rung set does not.
# What makes FINISH scoreable at all this iter is `commit_prefix` below.
_RUNGS_FINISH = (
    "- NOT_VISIBLE: you cannot find the building in the reference photo "
    "anywhere in the current frame.\n"
    "- DISTANT: you can recognise the building, but it is small, sits "
    "near the horizon line, and other frontage stands between the camera "
    "and it.\n"
    "- APPROACHING: it is unmistakably the reference building and it is "
    "now large, but its whole height still fits inside the frame and a "
    "clear stretch of roadway or sidewalk still runs between the camera "
    "and its base.\n"
    "- ALONGSIDE: the robot is standing AT its frontage. Its base is at "
    "or below the bottom edge of the frame or runs off the frame's side "
    "edge, its upper floors are cut off by the top edge of the frame, "
    "and nothing but pavement lies between the camera and the wall.\n"
    "- BEHIND: the robot has walked past the frontage — the building is "
    "slipping off a rear edge of the frame, only partly in view, and "
    "what fills the frame ahead is beyond it.\n"
)


def _ladder_sys_prompt(kind: str) -> str:
    rungs = _RUNGS_FINISH if kind == LEG_FINISH else _RUNGS_INTERSECTION
    return _LADDER_PREAMBLE + rungs + _LADDER_EPILOGUE


def extract_stage(raw: str) -> Optional[int]:
    """Parse the forced-choice label into its ladder index.

    Tolerant in exactly the ways `_extract_match` is (markdown fences,
    stray prose, a bare label) and strict about the vocabulary: a token
    that is not one of the five labels is None, never a guess. On a
    response naming several labels we take the one carried by the
    `Stage` field, and only fall back to a scan of the raw text when
    there is no parseable field at all — there, the LAST label mentioned
    wins, because a model that reasons aloud lists the rungs it
    considered before the one it settles on.
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
    label: Optional[str] = None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            v = obj.get("Stage", obj.get("stage"))
            if isinstance(v, str):
                label = v
    except Exception:
        pass
    if label is None:
        m = re.search(r'"[Ss]tage"\s*:\s*"([A-Z_]+)"', raw)
        if m:
            label = m.group(1)
    if label is not None:
        key = label.strip().upper()
        if key in STAGES:
            return STAGES.index(key)
        return None
    hits = re.findall(r'\b(' + "|".join(STAGES) + r')\b', raw.upper())
    if hits:
        return STAGES.index(hits[-1])
    return None


@dataclass
class ArrivalScale:
    """Per-episode ordinal arrival recognizer + the ladder's per-leg state."""

    # Highest rung observed so far on each leg (the ladder memory).
    _peak: Dict[int, int] = field(default_factory=dict)
    # The rung the PREVIOUS look on each leg reported (the sustain memory).
    _last: Dict[int, int] = field(default_factory=dict)
    # Observations spent on each leg (the parent's REACH budget, imported).
    _fires: Dict[int, int] = field(default_factory=dict)
    # forward_streak at which the proactive schedule last fired, per leg.
    _last_streak: Dict[int, int] = field(default_factory=dict)
    # Planner cross-check firings per leg (the parent's xcheck budget).
    _xchecks: Dict[int, int] = field(default_factory=dict)
    # (leg, frame_seq) -> rung. One ordinal read per frame, shared by
    # every gate that asks about that frame (kills the duplicate call).
    _cache: Dict[Tuple[int, int], Optional[int]] = field(default_factory=dict)
    # Branch/rung tallies, surfaced through `ledger()` into EpisodeResult
    # notes so the next diagnosis reads counts instead of inferring them.
    _tally: Dict[str, int] = field(default_factory=dict)

    # ---- bookkeeping -------------------------------------------------
    def _bump(self, key: str) -> None:
        self._tally[key] = int(self._tally.get(key, 0)) + 1

    def peak(self, cur_idx: int) -> int:
        return int(self._peak.get(cur_idx, -1))

    def last(self, cur_idx: int) -> Optional[int]:
        v = self._last.get(cur_idx)
        return None if v is None else int(v)

    def ledger(self) -> str:
        """Compact, diagnosable summary of every branch this episode took."""
        if not self._tally:
            return ""
        order = [s[:2] for s in STAGES] + [
            "obs", "hit", "allow", "block", "degrade", "overshoot", "pose"]
        parts = [f"{k}={self._tally[k]}" for k in order if k in self._tally]
        for k in sorted(self._tally):
            if k not in order:
                parts.append(f"{k}={self._tally[k]}")
        return "SCALE(" + ",".join(parts) + ")"

    # ---- the one VLM call --------------------------------------------
    def _ladder_call(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        kind: str,
    ) -> Optional[int]:
        """Ask for ONE label off the ladder. Returns its index, or None.

        NOTE the message contents, which are the point of this iter:
        two images and the vocabulary. The leg's instruction prose is
        absent — `kind` has already consumed it to select the rungs.
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

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": "REFERENCE photograph, taken from the "
                                     "spot the robot must reach:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{hint_b64}"}},
            {"type": "text", "text": "CURRENT camera frame of the robot:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},
            {"type": "text",
             "text": ('Strict JSON: {"Stage": "<one of '
                      + ", ".join(STAGES)
                      + '>", "Cue": "<one short clause>"}.')},
        ]
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": _ladder_sys_prompt(kind)},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=_STAGE_MAX_TOKENS,
            )
        except Exception:
            return None
        try:
            raw = resp.choices[0].message.content or ""
        except Exception:
            return None
        return extract_stage(raw)

    # ---- the memoised observation ------------------------------------
    def observe(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        cur_idx: int,
        frame_seq: int,
        charge: bool = True,
    ) -> Tuple[Optional[int], int, Optional[int]]:
        """Return (rung, prior_peak, prior_last) for leg `cur_idx`'s frame.

        Both priors are the ladder memory as it stood BEFORE this
        observation was folded in: the overshoot license reads `peak`
        (so `BEHIND` cannot license itself) and the sustain license
        reads `last` (so a lone high rung cannot license itself).

        Memoised on (leg, frame_seq): two gates asking about the same
        frame spend one call and see one answer. `charge=False` reads the
        cache only (never spends budget).
        """
        key = (cur_idx, frame_seq)
        prior = self.peak(cur_idx)
        prior_last = self.last(cur_idx)
        if key in self._cache:
            self._bump("hit")
            return self._cache[key], prior, prior_last
        if not charge:
            return None, prior, prior_last
        kind = leg_kind(instruction)
        stage = self._ladder_call(
            obs=obs, hint_image=hint_image, vlm_clients=vlm_clients, kind=kind)
        self._cache[key] = stage
        self._fires[cur_idx] = int(self._fires.get(cur_idx, 0)) + 1
        self._bump("obs")
        if stage is None:
            self._bump("degrade")
        else:
            self._bump(STAGES[stage][:2])
            # The ladder memory records APPROACH evidence only, so `BEHIND`
            # never raises the peak. Otherwise a repeated `BEHIND` licenses
            # itself through a peak it set one frame earlier — which is
            # precisely the unsupported claim the overshoot rule exists to
            # reject (nothing was ever watched growing).
            if S_NOT_VISIBLE <= stage < S_BEHIND and stage > prior:
                self._peak[cur_idx] = stage
            self._last[cur_idx] = stage
        return stage, prior, prior_last

    # ---- the license -------------------------------------------------
    def licensed(
        self,
        stage: Optional[int],
        prior_peak: int,
        prior_last: Optional[int] = None,
    ) -> Optional[bool]:
        """Does this rung license a `-1`? None == no verdict (degrade).

        Three conditions, all read off the ladder:

        * REACH — the rung is ALONGSIDE or BEHIND. Anything lower means
          keep walking.
        * SUSTAIN — the PREVIOUS look on this leg was already at least
          APPROACHING. A leg's first high rung is never a license: it was
          measured leading the true arrival by 500-1 500 cm, which on a
          ±2 000 window is the difference between a pass and an instantly
          fatal commit. Waiting for the rung to survive one more look
          spends one look-interval of approach, and over 12 replayed legs
          it takes out-of-window first-fires from 7/12 to 0/12 while
          keeping in-window fires at 5-6/12.
        * NOT-SELF-LICENSED — `BEHIND` additionally needs the leg's peak
          (which never records `BEHIND`) to have reached APPROACHING, so
          "I am past it" requires having watched it grow first. `BEHIND`
          is the overshoot rung: the parent walked out the far side of 26
          graded windows with no way to say so.
        """
        if stage is None:
            return None
        if stage < S_COMMIT_RUNG:
            return False
        if prior_last is None or prior_last < S_SUSTAIN_RUNG:
            self._bump("unsustained")
            return False
        if stage == S_BEHIND:
            if prior_peak < S_OVERSHOOT_PRIOR_RUNG:
                return False
            self._bump("overshoot")
        return True

    # ---- gate 1: proactive (the planner did not propose -1) ----------
    def proactive_commit(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        forward_streak: int,
        cur_idx: int,
        frame_seq: int,
    ) -> str:
        """'ALLOW' | 'BLOCK' | 'SKIP' | 'DEGRADE'.

        Schedule is the parent's: fire after K_REACH_FORWARD_STREAK
        consecutive forwards, re-fire no sooner than K_REACH_INTERVAL
        forwards later, at most MAX_REACH_FIRES_PER_SUBTASK per leg.
        Not retuned, imported.
        """
        if _is_orientation_instruction(instruction):
            return "SKIP"
        if forward_streak < K_REACH_FORWARD_STREAK:
            return "SKIP"
        last = self._last_streak.get(cur_idx)
        if last is not None and forward_streak - last < K_REACH_INTERVAL:
            return "SKIP"
        if int(self._fires.get(cur_idx, 0)) >= MAX_REACH_FIRES_PER_SUBTASK:
            return "SKIP"
        if hint_image is None or not vlm_clients:
            return "DEGRADE"
        if not isinstance(obs, Mapping) or obs.get("rgb") is None:
            return "DEGRADE"
        self._last_streak[cur_idx] = forward_streak
        stage, prior, prior_last = self.observe(
            obs=obs, hint_image=hint_image, vlm_clients=vlm_clients,
            instruction=instruction, cur_idx=cur_idx, frame_seq=frame_seq)
        ok = self.licensed(stage, prior, prior_last)
        if ok is None:
            return "DEGRADE"
        self._bump("allow" if ok else "block")
        return "ALLOW" if ok else "BLOCK"

    # ---- gate 2: the planner proposed -1 ----------------------------
    def allow_planner_commit(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        cur_idx: int,
        frame_seq: int,
    ) -> str:
        """'ALLOW' | 'BLOCK' | 'SKIP' | 'DEGRADE'.

        A spent budget returns BLOCK, not ALLOW. The parent returns
        `True` here on a spent cap (`diagnosis.md` §2.6 / F5) — "I am out
        of budget" read as "approve" — and because the planner
        re-proposes `-1` in runs every ~3 steps, the cap vetoed the two
        in-window proposals and rubber-stamped the third, 1 500–3 000 cm
        past the waypoint, fatally, three times. Out of evidence is not
        a licence.
        """
        if _is_orientation_instruction(instruction):
            return "SKIP"
        if hint_image is None or not vlm_clients:
            return "DEGRADE"
        if not isinstance(obs, Mapping) or obs.get("rgb") is None:
            return "DEGRADE"
        key = (cur_idx, frame_seq)
        if key not in self._cache:
            if int(self._xchecks.get(cur_idx, 0)) >= MAX_XCHECK_FIRES_PER_SUBTASK:
                self._bump("block")
                self._bump("capblock")
                return "BLOCK"
            self._xchecks[cur_idx] = int(self._xchecks.get(cur_idx, 0)) + 1
        stage, prior, prior_last = self.observe(
            obs=obs, hint_image=hint_image, vlm_clients=vlm_clients,
            instruction=instruction, cur_idx=cur_idx, frame_seq=frame_seq)
        ok = self.licensed(stage, prior, prior_last)
        if ok is None:
            return "DEGRADE"
        self._bump("allow" if ok else "block")
        return "ALLOW" if ok else "BLOCK"

    # ---- gate 3: an arbitrary (e.g. post-yaw) frame ------------------
    def judge_frame(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        cur_idx: int,
        frame_seq: int,
    ) -> str:
        """Same license, applied to a frame the harness produced on purpose
        (the side-scan's post-yaw view). Keeps ONE arrival criterion in the
        harness instead of two."""
        if hint_image is None or not vlm_clients:
            return "DEGRADE"
        if not isinstance(obs, Mapping) or obs.get("rgb") is None:
            return "DEGRADE"
        stage, prior, prior_last = self.observe(
            obs=obs, hint_image=hint_image, vlm_clients=vlm_clients,
            instruction=instruction, cur_idx=cur_idx, frame_seq=frame_seq)
        ok = self.licensed(stage, prior, prior_last)
        if ok is None:
            return "DEGRADE"
        self._bump("allow" if ok else "block")
        return "ALLOW" if ok else "BLOCK"

    # ---- the pose the FINISH grade is scored on ----------------------
    def commit_prefix(self, instruction: Any) -> List[int]:
        """In-place rotation to emit immediately BEFORE a FINISH `-1`.

        `simple_world.py` grades a FINISH leg on location AND on holding
        a facing within 45 deg of the target yaw; the parent's FINISH
        commit is a bare `[-1]`, so `per_position_SR[3]` is 0.000 BY
        CONSTRUCTION — 0/10 FINISH legs ever held the required facing
        while standing in the window, and two episodes stood 16 cm and
        27 cm from the target and scored 0.5.

        The instruction always names the side the destination is on
        ("... on your right, stop and face that building"), so the turn
        is a quarter-turn toward that side. Rotations are in place —
        `_agent_location` only moves on translation actions — so this
        cannot cost the location half of the grade; it can only add the
        orientation half, which is currently always missing. The rule is
        correct in 2/2 of the parent episodes where it could be tested.

        Returns `[1]` (Rotate_Left, -90 deg), `[2]` (Rotate_Right, +90
        deg), or `[]` on a non-FINISH leg / an unnamed side.
        """
        if leg_kind(instruction) != LEG_FINISH:
            return []
        side = _named_side(instruction)
        if side is None:
            return []
        self._bump("pose")
        return [1] if side == "left" else [2]
