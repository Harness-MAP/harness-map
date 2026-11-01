"""ArrivalGate — iter 9. The 9th node: it owns the LICENSE for a `-1`.

WHAT CHANGED IN THE GRAPH
-------------------------
Until now every `-1` in this harness was licensed by ONE function,
`CompletionChecker._binary_reach_vlm_call`, re-routed through three different
triggers (the forward proactive check, the planner cross-check, the iter-6
faced look-scan).  The parent's diagnosis measured that single recognizer and
found the failure is in the QUESTION, not in the trigger or the pose:

  * the call prefixes `"Subtask instruction: <the 40-word leg text>"` to a pair
    of images whose job is image identity.  Probed on this run's own frames
    (14 NEAR / 14 FAR, temperature 0, the live model): with the text block the
    recall inside the arrival window is 2/14 (14 %); with the SAME prompt and
    the text block removed it is 14/14 (100 %).  The rejection reasons switch
    to "different building colors and lighting" — the model stops comparing
    the two images and starts adjudicating prose against a photo, which it
    loses.  That is a prompt-composition bug, not a model limit.
  * text-free identity alone is not shippable: it also fires 6/14 on FAR
    frames.  A `-1` is irreversible and episode-fatal, so recall without
    precision converts "reach-but-never-commit" straight into "commit at
    20 000 cm and die".
  * asking ONE call to assert both landmark identity AND arrival scores 0/14
    (the model drops one of the two conjuncts).  The conjunction has to be two
    separate cheap calls: text-free identity AND an independent, hint-free
    spatial question about WHERE THE BODY IS.  Intersected on the same 28
    frames that is 6/14 NEAR and 0/14 FAR — more recall than the live gate at
    a lower false-fire rate.

So this module splits the one recognizer into TWO independent single-purpose
perception nodes and licenses a commit only on their conjunction:

    identity_call(rgb, hint)   -- "same place?"   two images, NO leg text
            AND
    place_call(rgb)            -- "am I THERE?"   one image, NO hint, NO text

`CompletionChecker` keeps its recognizer (it is the degradation path when the
spatial call is unavailable) and its schedule constants — which this module
imports rather than re-declares, so no constant is retuned this iter and the
only thing that moved is the shape of the decision.

The spatial question is asked per leg type (`viewpoint_controller.leg_type`,
structural, read off the static instruction text):

  ROAD / TURN : "are you INSIDE the junction" — the ROAD leg's grading point
                literally is the intersection ("stop at the intersection with
                X on your left"), and the harness has never once asked about
                it; it only ever asked about the building.
  FINISH      : "are you AT the frontage" — hard up against a facade at the
                near end of the block, as opposed to looking down the street
                at it.  Same format (one image, no hint, no leg text) as the
                junction question, which is the format that was measured to
                work; the wording is this iter's bet and is stated as such.

DEGRADATION.  Every unavailability (no client, no RGB, parse failure, cap
spent) falls back to the parent's exact gate — `checker._binary_reach_vlm_call`
with the leg text — so a dead spatial channel can never lock the harness into
"never commits".  That is the failure mode that killed the iter-5 depth gate.

Deployment-honest reads only: `obs['rgb']`, the per-subtask hint image, the
static instruction text, and this module's own call bookkeeping.  No
coordinate, no gt field, no analog yaw, no `subtask_max_steps`.
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

from .completion_checker import (  # noqa: E402
    K_REACH_FORWARD_STREAK,
    K_REACH_INTERVAL,
    MAX_REACH_FIRES_PER_SUBTASK,
    MAX_XCHECK_FIRES_PER_SUBTASK,
    _extract_match,
    _is_orientation_instruction,
    _named_side_has_structure,
)
from .viewpoint_controller import LEG_FINISH, leg_type  # noqa: E402
from .road_geometry import GEO_MIN_TRANSLATIONS  # noqa: E402


# ---------------------------------------------------------------------------
# Call budgets.  Deliberately NOT new tuning surface: the identity call reuses
# CompletionChecker's schedule verbatim (imported above), and the spatial call
# is only ever made AFTER identity has already said 1, so it is already
# bounded by that schedule.  This backstop is therefore DERIVED (one spatial
# call per possible identity call on the leg, across both call sites) rather
# than chosen: if it were smaller it would silently revert the tail of a long
# leg to the parent's gate, which would confound the measurement this iter is
# trying to make.  It is a runaway guard, not an operating point.
# ---------------------------------------------------------------------------
MAX_PLACE_FIRES_PER_SUBTASK = (
    MAX_REACH_FIRES_PER_SUBTASK + MAX_XCHECK_FIRES_PER_SUBTASK)

# Answer budget for both calls.  The live model answers the existing 2-image
# reach call inside 2048; these two are strictly simpler (one asks about two
# images with less text, one asks about a single image), and the parser below
# returns None on a truncated answer, which routes to the legacy fallback
# rather than to a silent 0.  Held under the lint's map-constant threshold.
_MAX_ANSWER_TOKENS = 1024


_IDENTITY_SYS = (
    "You compare two egocentric outdoor photographs and judge whether they "
    "show the SAME place. Answer with a single integer Match field: 1 if the "
    "current view and the reference view show the same building / "
    "intersection / scene (the same structure is visible in both, allowing "
    "for a different viewing angle, lighting or time of day); 0 if they are "
    "different places. Judge the IMAGES ONLY — do not reason about any route "
    "or instruction. Respond strictly with JSON: "
    '{"Match": 0 or 1, "Reason": "<one short clause>"}.'
)

# Operating point chosen by an offline A/B on 14 NEAR (< 2 000 cm of the ROAD
# waypoint) / 14 FAR (> 10 000 cm) frames from the parent's own run, live model,
# temperature 0.  A hard three-way conjunction of the cues scores 3/14 NEAR;
# the diagnosis's own row-C phrasing 2/14; this ANY-CUE phrasing 5/14 NEAR at
# 1/14 FAR, which reproduces the 6/14 : 1/14 the diagnosis measured.  The list
# is deliberately disjunctive: a single frame rarely shows all three cues.
_JUNCTION_SYS = (
    "You look at ONE egocentric photograph from a street-level robot and judge "
    "whether it is AT a street intersection right now, as opposed to partway "
    "down a block. Answer 1 if ANY of these is true of the robot's immediate "
    "surroundings: the unbroken line of building facades beside it has ended "
    "or is ending, a crosswalk / curb cut / stop line is at or just in front "
    "of it, or a cross street or open gap is beside it to the left or right. "
    "Answer 0 only if it is clearly mid-block, with facades running unbroken "
    "on both sides and no opening nearby. Answer with a single integer Here "
    'field. Respond strictly with JSON: {"Here": 0 or 1, '
    '"Reason": "<one short clause>"}.'
)

# FINISH's second conjunct, and the one honest weak spot in this iter.  Probed
# the same way (12 frames > 15 000 cm from the destination, 7 frames < 2 500 cm,
# from the parent's run): this wording fires 0/7 NEAR and 0/12 FAR.  Two
# looser wordings were also tried — an any-cue frontage list and the junction
# question reused — and both stayed 0/7 NEAR while firing 4/12 and 3/12 FAR,
# i.e. strictly worse.  So the strict wording ships: on a FINISH leg this is a
# veto that currently never opens, which is the same net behaviour as the
# parent's blanket `allow_forward_commit` block but arrived at by evidence and
# instrumented (`GATE(place_1=…)` in the notes will say if it ever opens).  It
# CANNOT reproduce the parent's 9 fatal FINISH commits, every one of which was
# 8 666–58 789 cm out: it answered 0 on all 12 far frames.  The NEAR sample is
# 7 frames — small, and drawn from a run that rarely entered the destination
# window — so "0/7" is "low", not "impossible".
_FRONTAGE_SYS = (
    "You look at ONE egocentric photograph taken by a street-level robot and "
    "judge WHERE THE ROBOT IS STANDING RIGHT NOW. Answer with a single "
    "integer Here field: 1 if the robot has arrived at a building's frontage "
    "— a facade is close enough that it fills a large part of the frame, its "
    "ground-floor details (doors, windows, signage, steps) are resolvable, "
    "and the robot is at the near end of the block rather than partway down "
    "it; 0 if every building is still down the street at a distance, however "
    "clearly visible. A building being recognisable is NOT arrival. Respond "
    'strictly with JSON: {"Here": 0 or 1, "Reason": "<one short clause>"}.'
)


# ---------------------------------------------------------------------------
# iter 14 — the FINISH arrival channel.
#
# `_FRONTAGE_SYS` above is RETIRED on the FINISH leg (it is still the wording
# `_place_call` would use if anything ever routed there, and nothing does any
# more).  Measured over 50 episodes: 29 calls, **0 positives**, including three
# asked from inside the success ball at L1 484 / 640 / 1144 cm, on frames where
# every other conjunct had already passed and `commit_pose_prefix` would have
# supplied the exact graded yaw.  Those three were the run's only live chances
# at SR > 0 and a 0/29 veto discarded all three; 31/31 FINISH legs ended with
# the harness never emitting a single `-1`.
#
# Its rejection reasons say exactly what is wrong with the QUESTION, not with
# the model: *"building facade is distant and not close enough to fill the
# frame"*, *"buildings are distant and not close enough to be at the
# frontage"*.  Both are TRUE of the frames they were asked on.  A robot that is
# 484 cm from its destination is walking ALONG the street and the destination
# is BESIDE it, half out of the frame — it is not looking at a facade that
# fills the view, and it never will until `commit_pose_prefix` turns it, which
# only happens AFTER the commit this call is gating.  `_FRONTAGE_SYS` asks for
# a frontal, hint-free composition that the grading geometry forbids.
#
# So the position question for FINISH is re-asked, at the token level, as a
# GRADED field of the two-image call that already works (identity: 90.9 %
# in-window on ROAD this run).  Three things change in the tokens:
#   * it is grounded by the REFERENCE photo — "how far am I from THAT
#     building", not "is a facade filling my frame";
#   * arrival is defined as being LEVEL WITH / PASSING the building, and the
#     prompt says in so many words that off-centre, steeply angled and
#     frame-clipped views are the expected appearance of arrival;
#   * the answer is a 0-3 grade, not a 0/1 — so a "no" still carries HOW FAR,
#     which keys the schedule below instead of a clock.
# One call now carries both conjuncts (WHICH building, HOW FAR), which does
# NOT reproduce the "one call asserting two things scores 0/14" result in this
# module's header: that probe put IDENTITY and PLACE in a single BINARY; here
# identity stays its own binary and place becomes a separate graded field.
# Measured on 120 archived FINISH frames (NOTES.md has the table): in-window
# Match 85 % against the plain identity call's 25 % on the SAME frames — the
# off-axis framing is what fixes the FINISH identity hole the diagnosis
# attributed to pose, and it is a token change, not a pose change.
# ---------------------------------------------------------------------------
_ARRIVAL_SYS = (
    "You compare two egocentric outdoor photographs. The REFERENCE photo shows "
    "a destination building. The CURRENT photo is what a street-level robot "
    "sees while walking ALONG a street, so the destination will usually appear "
    "off to one side, at a steep angle, or partly cut off by the frame edge — "
    "that is still the same building, and when the robot is closest it is the "
    "LEAST centred. Report two integers. "
    "Match: 1 if the building in the reference photo is visible anywhere in "
    "the current view, 0 if it is not there. "
    "Prox: judge ONLY the street geometry of the CURRENT photo. Never compare "
    "the two photos' camera distances or angles — that tells you nothing about "
    "where the robot is standing. How much more street must the robot cover to "
    "draw level with that building? "
    "0 = far down the street: other buildings stand between the robot and it, "
    "or it sits up near the vanishing point where the street meets the "
    "horizon. "
    "1 = about one more building-length of walking. "
    "2 = it is the very next thing along the street, a few steps away, with "
    "nothing between the robot and it. "
    "3 = the robot is already level with it or passing it: it is at the far "
    "left or far right of the frame, or its wall is clipped by the frame edge, "
    "or its ground floor fills one side of the view. "
    "Report 0 when Match is 0. Respond strictly with JSON: "
    '{"Match": 0 or 1, "Prox": 0, 1, 2 or 3, "Reason": "<one short clause>"}.'
)

# The tie-breaker, and it is only ever asked on a `Prox == 2` frame — the one
# grade the probe found ambiguous (in-window 7/20 but also 8/40 beyond 10 km).
# Asked as a position question in its own right, with an explicit base-rate
# nudge, because the probe showed the model will otherwise answer the
# comparison it finds easiest rather than the one it was asked.
_REACHED_SYS = (
    "You look at a photograph taken by a street-level robot walking along a "
    "city street, and a reference photograph of one destination building. "
    "Answer ONE question about the robot's position: has the robot reached "
    "that building yet? "
    "The robot walks down the middle of the street, so when it HAS reached "
    "the building, the building is beside it: far to one side of the frame, "
    "usually clipped by the frame edge, with no street left between the robot "
    "and the building's ground floor. When it has NOT reached it, the "
    "building is still ahead — nearer the centre of the frame, smaller, with "
    "road surface and often other buildings between. "
    "Most of the time the answer is 0: the walk is long and the robot is "
    "almost always still approaching. Answer 1 only if the building is "
    "unmistakably beside the robot now. Respond strictly with JSON: "
    '{"Match": 0 or 1, "Here": 0 or 1, "Reason": "<one short clause>"}, '
    "where Match says whether you can see the reference building at all."
)

# The license rule, read straight off the probe table in NOTES.md:
#   qualify  ==  Match 1 AND ( Prox 3  OR  (Prox 2 AND the tie-breaker says 1) )
# Measured over 120 archived FINISH frames: in-window 65 %, 2-4 km 43 %,
# 4-10 km 17 %, beyond 10 km **2.5 %**, and 7 of the 9 episodes that have an
# in-window frame get at least one qualifying reading.  `_FRONTAGE_SYS`, the
# conjunct this replaces, scored 0/20 in-window on the same frames.
FINISH_PROX_LICENSE = 3      # grade that licenses on its own
FINISH_PROX_CONFIRM = 2      # grade that buys the tie-breaker call

# ...and then a qualifying reading is not yet a commit.  A FINISH leg walks
# in through the 2-4 km band (43 % per-ask) before it reaches the 2 000 cm
# ball, so a first-qualifying-reading rule mostly fires one band early.  The
# gate therefore requires `FINISH_CONFIRM_STREAK` CONSECUTIVE qualifying
# readings, which — because a Prox >= 2 reading re-asks on the very next
# translation — costs about one extra forward step and lands the commit a
# band deeper.  Monte-Carlo over the 31 real FINISH legs' own frame sequences
# (per-frame outcomes resampled from the measured per-band rates): commit
# in-window 8.4 % of legs at k = 1 vs **10.6 % at k = 2**, with beyond-10 km
# commits falling 25.1 % -> 1.0 %.  k = 3 overshoots (4.7 %).
FINISH_CONFIRM_STREAK = 2

K_FINISH_INTERVAL_BY_PROX = {0: 6, 1: 3, 2: 1, 3: 1}
K_FINISH_INTERVAL_DEFAULT = K_REACH_INTERVAL

# Runaway guard, NOT an operating point: a FINISH leg is 20-50 km and the
# whole point of this change is that the gate must still be awake at the end
# of it.  At the Prox-0 spacing of 6 translations this is never the binding
# constraint on any leg length the env actually produces.
MAX_FINISH_ARRIVAL_FIRES_PER_SUBTASK = 48


def _extract_int_field(raw: str, name: str) -> Optional[int]:
    """Pull one integer field out of a possibly-fenced, possibly-chatty answer.

    Same tolerance as `_extract_here` / `completion_checker._extract_match`;
    returns None when nothing parses, which every caller treats as "the
    channel did not answer" rather than as a 0.
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
            v = obj.get(name, obj.get(name.lower()))
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    m = re.search(r'"%s"\s*:\s*(\d)' % name, raw, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _extract_prox(raw: str) -> Optional[int]:
    """The 0-3 approach grade, clamped.  None on a truncated / absent field."""
    v = _extract_int_field(raw, "Prox")
    if v is None:
        return None
    return 0 if v < 0 else 3 if v > 3 else v


def _extract_here(raw: str) -> Optional[int]:
    """Pull a 0/1 `Here` field out of the spatial call's answer.

    Same tolerance as `completion_checker._extract_match` (fenced JSON, stray
    prose, a truncated tail); returns None when nothing parses, which is the
    signal to fall back to the legacy gate rather than to read it as a 0.
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
            v = obj.get("Here", obj.get("here"))
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    if isinstance(v, str):
                        return 1 if v.strip().lower().startswith("y") else 0
    except Exception:
        pass
    m = re.search(r'"Here"\s*:\s*(\d)', raw)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _rgb_of(obs: Any) -> Optional[np.ndarray]:
    if not isinstance(obs, Mapping):
        return None
    rgb = obs.get("rgb")
    if rgb is None:
        return None
    return rgb


@dataclass
class ArrivalGate:
    """Two-call conjunction gate. One instance per episode.

    `checker` is the episode's `CompletionChecker`; the gate borrows its
    recognizer for the degradation path and its per-subtask fire counters so
    the identity schedule stays exactly the parent's (one arrival question per
    `K_REACH_INTERVAL` translations, capped per leg).
    """

    checker: Any = None
    # iter 13: the FREE position channel (`RoadGeometry`, the 10th node).  The
    # gate asks it BEFORE spending any token: on an intersection-stop ROAD leg
    # it answers the "am I THERE?" conjunct in closed form off
    # `obs['object_mask']`, and its False is a veto that costs nothing.  None
    # when it does not own the leg (ORIENT / TURNING / FINISH) or the mask was
    # missing, in which case the VLM `_place_call` runs exactly as before.
    geo: Any = None
    # per-subtask counters (keyed by the structural subtask index)
    _place_fires: Dict[int, int] = field(default_factory=dict)
    _ident_fires: Dict[int, int] = field(default_factory=dict)
    _xcheck_fires: Dict[int, int] = field(default_factory=dict)
    _last_ident_streak: Dict[int, int] = field(default_factory=dict)
    # iter 14: the last FINISH approach grade (0-3) per leg.  This is the
    # gate's own bookkeeping of its own last answer — it keys the next ask.
    _last_prox: Dict[int, int] = field(default_factory=dict)
    # iter 14: consecutive qualifying FINISH readings on this leg.  Reset by
    # any reading that does not qualify, so it is a STREAK, not a tally.
    _fin_streak: Dict[int, int] = field(default_factory=dict)
    # iter 14: which channel licensed the most recent proactive commit, for
    # the trace label only.  Nothing branches on it.
    last_channel: Optional[str] = None
    # (forward_streak, verdict) memo so the two call sites in one step cannot
    # ask the same spatial question twice.
    _place_memo: Dict[int, Tuple[int, Optional[int]]] = field(default_factory=dict)
    # iter 13 (F2): the LAST computed cross-check verdict per leg.  The cap
    # below bounds VLM CALLS; it must not bound the VERDICT.
    _xcheck_memo: Dict[int, bool] = field(default_factory=dict)
    # telemetry for the diagnostician (never read by the agent)
    stats: Dict[str, int] = field(default_factory=dict)

    # ---- bookkeeping -----------------------------------------------------
    def _bump(self, key: str) -> None:
        self.stats[key] = int(self.stats.get(key, 0)) + 1

    def summary(self) -> str:
        """One compact note token for the trace: the gate's own call ledger."""
        if not self.stats:
            return "GATE(no-calls)"
        return "GATE(" + ",".join(
            f"{k}={self.stats[k]}" for k in sorted(self.stats)) + ")"

    # ---- the two perception calls ---------------------------------------
    def _identity_call(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """"Same place?" — two images, and NOTHING ELSE in the user turn.

        This is `_binary_reach_vlm_call` with the `Subtask instruction:` text
        element deleted (the measured 14 % -> 100 % in-window recall lever).
        """
        rgb = _rgb_of(obs)
        if rgb is None or hint_image is None or not vlm_clients:
            return None
        try:
            rgb_b64 = numpy_to_base64(np.asarray(rgb))
            hint_b64 = numpy_to_base64(np.asarray(hint_image))
        except Exception:
            return None

        model_id = next(iter(vlm_clients.keys()))
        client = vlm_clients[model_id]
        user_content: List[Dict[str, Any]] = [
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
                    {"role": "system", "content": _IDENTITY_SYS},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=_MAX_ANSWER_TOKENS,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            self._bump("ident_err")
            return None
        v = _extract_match(raw)
        self._bump("ident_1" if v == 1 else
                   "ident_0" if v == 0 else "ident_none")
        return v

    def _arrival_call(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[int], Optional[int]]:
        """FINISH's single arrival call: "same building?" AND "how far?".

        Byte-for-byte the `_identity_call` message shape (two images, NO leg
        text, same encoder, same temperature, same answer budget) with one
        difference: the system prompt asks for a second, GRADED field.  It
        replaces the identity + `_FRONTAGE_SYS` pair on a FINISH leg, so the
        accept path costs ONE call where it used to cost two, and the reject
        path now returns a distance grade instead of nothing.

        Returns `(match, prox)`.  Either may be None — a transient failure or
        a truncated answer must never be read as a 0 (that is the mode that
        pinned SR at 0.00 this run).
        """
        rgb = _rgb_of(obs)
        if rgb is None or hint_image is None or not vlm_clients:
            return (None, None)
        try:
            rgb_b64 = numpy_to_base64(np.asarray(rgb))
            hint_b64 = numpy_to_base64(np.asarray(hint_image))
        except Exception:
            return (None, None)

        model_id = next(iter(vlm_clients.keys()))
        client = vlm_clients[model_id]
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Current egocentric view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},
            {"type": "text", "text": "Reference landmark view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{hint_b64}"}},
            {"type": "text",
             "text": ('Output strict JSON: {"Match": 0 or 1, '
                      '"Prox": 0, 1, 2 or 3, '
                      '"Reason": "<one short clause>"}. No prose.')},
        ]
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": _ARRIVAL_SYS},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=_MAX_ANSWER_TOKENS,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            self._bump("fin_err")
            return (None, None)
        m = _extract_match(raw)
        pr = _extract_prox(raw)
        self._bump("fin_m1" if m == 1 else
                   "fin_m0" if m == 0 else "fin_mnone")
        self._bump("fin_p%s" % ("none" if pr is None else pr))
        return (m, pr)

    def _reached_call(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """The tie-breaker, asked ONLY on a `Prox == 2` frame.

        Same two images, a different question: not "how far" but "has the
        robot reached it", with the base rate stated.  On the probe set it
        splits the ambiguous grade — it keeps 6 of the 7 in-window `Prox == 2`
        frames and kills 7 of the 8 beyond 10 km.
        """
        rgb = _rgb_of(obs)
        if rgb is None or hint_image is None or not vlm_clients:
            return None
        try:
            rgb_b64 = numpy_to_base64(np.asarray(rgb))
            hint_b64 = numpy_to_base64(np.asarray(hint_image))
        except Exception:
            return None
        model_id = next(iter(vlm_clients.keys()))
        client = vlm_clients[model_id]
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Current egocentric view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},
            {"type": "text", "text": "Reference landmark view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{hint_b64}"}},
            {"type": "text",
             "text": ('Output strict JSON: {"Match": 0 or 1, '
                      '"Here": 0 or 1, "Reason": "<one short clause>"}. '
                      'No prose.')},
        ]
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": _REACHED_SYS},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=_MAX_ANSWER_TOKENS,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            self._bump("reach_err")
            return None
        v = _extract_here(raw)
        self._bump("reach_1" if v == 1 else
                   "reach_0" if v == 0 else "reach_none")
        return v

    def _finish_reading(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        cur_idx: int,
    ) -> bool:
        """One FINISH arrival reading: 1 call, plus the tie-breaker on a 2.

        Updates `_last_prox` (which keys the next ask) and `_fin_streak`
        (which is what a commit is counted against), and returns whether THIS
        reading qualifies.  A transient failure resets the streak: it is a
        non-answer, and a commit must never be built on one.
        """
        match, prox = self._arrival_call(obs, hint_image, vlm_clients)
        if prox is not None:
            self._last_prox[cur_idx] = prox
        qualifies = False
        if match == 1 and prox is not None:
            if prox >= FINISH_PROX_LICENSE:
                qualifies = True
            elif prox >= FINISH_PROX_CONFIRM:
                qualifies = self._reached_call(
                    obs, hint_image, vlm_clients) == 1
        self._fin_streak[cur_idx] = (
            int(self._fin_streak.get(cur_idx, 0)) + 1 if qualifies else 0)
        self._bump("fin_qualify" if qualifies else "fin_noqualify")
        return qualifies

    def _finish_proactive(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        forward_streak: int,
        cur_idx: int,
    ) -> bool:
        """The whole FINISH `-1` license, on the graded channel.

        Replaces, on this leg type only, the `identity AND _FRONTAGE_SYS`
        conjunction (0 licenses in 50 episodes) and the uniform 12-ask clock
        (exhausted before arrival in 9 of the 10 episodes that reached the
        destination).  ROAD / TURN / ORIENT are untouched and do not reach
        here.

        Recall-first, deliberately, and ONLY here: FINISH is the terminal leg,
        so a premature `-1` forfeits a leg that is already scoring 0.00 with
        nothing downstream of it — the precision/recall asymmetry that makes
        the ROAD gate conservative does not exist on this leg.  The only real
        cost of a false fire is the chance of a LATER, true arrival in the
        same episode, which is exactly what the `Prox` conjunct is for.
        """
        if forward_streak < K_REACH_FORWARD_STREAK:
            return False
        if hint_image is None or not vlm_clients:
            return False
        if _rgb_of(obs) is None:
            return False
        last = self._last_ident_streak.get(cur_idx)
        if last is not None and last > forward_streak:
            # The streak was reset by a turn; the spacing rule re-arms rather
            # than going mute until the streak climbs back past its old value.
            last = None
        if last is not None:
            gap = K_FINISH_INTERVAL_BY_PROX.get(
                self._last_prox.get(cur_idx, -1), K_FINISH_INTERVAL_DEFAULT)
            if forward_streak - last < gap:
                return False
        fires = int(self._ident_fires.get(cur_idx, 0))
        if fires >= MAX_FINISH_ARRIVAL_FIRES_PER_SUBTASK:
            self._bump("fin_cap")
            return False
        # Free segmentation veto first on this leg: it costs nothing, so it
        # should not be paid for with a call.  Fails OPEN on a missing mask.
        if not _named_side_has_structure(obs, instruction):
            self._bump("veto_side")
            return False
        self._ident_fires[cur_idx] = fires + 1
        self._last_ident_streak[cur_idx] = forward_streak

        if not self._finish_reading(obs, hint_image, vlm_clients, cur_idx):
            return False
        if int(self._fin_streak.get(cur_idx, 0)) < FINISH_CONFIRM_STREAK:
            # Armed, not licensed.  The grade just written is >= 2, so the
            # next ask is one translation away, not five — the confirmation
            # costs about one forward step.
            self._bump("fin_armed")
            return False
        self._bump("fin_license")
        return True

    def _place_call(
        self,
        obs: Any,
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
    ) -> Optional[int]:
        """"Am I THERE?" — one image, no hint, no leg text.

        Independent of the identity call in every input except the frame: it
        never sees the reference photo and never sees the instruction, so it
        cannot inherit the identity call's error.  Junction wording on
        ROAD/TURN, frontage wording on FINISH.
        """
        rgb = _rgb_of(obs)
        if rgb is None or not vlm_clients:
            return None
        try:
            rgb_b64 = numpy_to_base64(np.asarray(rgb))
        except Exception:
            return None
        sys_prompt = (_FRONTAGE_SYS if leg_type(instruction) == LEG_FINISH
                      else _JUNCTION_SYS)
        model_id = next(iter(vlm_clients.keys()))
        client = vlm_clients[model_id]
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": "The robot's current view:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},
            {"type": "text",
             "text": ('Output strict JSON: {"Here": 0 or 1, '
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
                max_tokens=_MAX_ANSWER_TOKENS,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            self._bump("place_err")
            return None
        v = _extract_here(raw)
        self._bump("place_1" if v == 1 else
                   "place_0" if v == 0 else "place_none")
        return v

    def _place_verdict(
        self,
        obs: Any,
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        cur_idx: int,
        forward_streak: int,
    ) -> Optional[int]:
        """The POSITION conjunct.  Geometry first, VLM only where geometry
        cannot decide.

        iter 13: on an intersection-stop ROAD leg `RoadGeometry` answers this
        in closed form off the segmentation mask, for zero tokens and zero env
        substeps, and it answers it BETTER — measured on the parent run's own
        1 326 archived perceive frames, its first rising edge lands inside the
        2 000 cm window on 25/45 ROAD legs, against the 11/38 = 29 % per-ask
        recall of the `_JUNCTION_SYS` call it displaces.  A `None` means the
        channel does not own this leg (ORIENT / TURNING / FINISH) or the mask
        was missing, and the VLM question below runs exactly as before.

        The VLM path keeps its memo: the body has not moved within a step, so
        the answer cannot change within a step."""
        if self.geo is not None:
            try:
                g = self.geo.place_verdict(instruction, cur_idx, forward_streak)
            except Exception:
                g = None
            if g is not None:
                self._bump("geo_1" if g else "geo_0")
                return 1 if g else 0
        memo = self._place_memo.get(cur_idx)
        if memo is not None and memo[0] == forward_streak:
            return memo[1]
        fires = int(self._place_fires.get(cur_idx, 0))
        if fires >= MAX_PLACE_FIRES_PER_SUBTASK:
            return None            # cost cap -> legacy fallback, not a veto
        self._place_fires[cur_idx] = fires + 1
        v = self._place_call(obs, vlm_clients, instruction)
        self._place_memo[cur_idx] = (forward_streak, v)
        return v

    def _legacy_verdict(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
    ) -> Optional[int]:
        """The parent's exact gate, used only when the spatial channel could
        not answer.  Keeping this path is what makes the rewire safe: a broken
        second call degrades the harness to the parent, never to silence."""
        if self.checker is None:
            return None
        self._bump("legacy")
        try:
            return self.checker._binary_reach_vlm_call(
                obs=obs,
                hint_image=hint_image,
                vlm_clients=vlm_clients,
                instruction=instruction,
            )
        except Exception:
            return None

    # ---- the two decisions the harness asks for --------------------------
    def proactive_commit(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        forward_streak: int,
        cur_idx: int,
    ) -> bool:
        """Harness-injected `-1` (the planner did not propose one).

        Schedule, on a leg the geometric channel does NOT own: the parent's,
        verbatim — fire at `K_REACH_FORWARD_STREAK` translations, then every
        `K_REACH_INTERVAL`, capped at `MAX_REACH_FIRES_PER_SUBTASK` per leg.

        iter 13, on a leg it DOES own (intersection-stop ROAD): the free
        channel keys the schedule instead of the clock.  The parent's gate
        asked ~1.6 times while crossing a window 8 forward steps wide and then
        walked 1 500-3 500 cm blind before asking again — 28 of 50 legs walked
        through the window and kept going a median 13 008 cm.  The fix is not
        to ask MORE often (the diagnosis's F1) but to ask at the RIGHT PLACE:
        the identity call is spent exactly on the frames where the mask says
        the body has just crossed into an intersection, and on every other
        frame of the leg it is not spent at all.  Bounded by construction —
        `RoadGeometry` makes at most `MAX_CLAIMABLE_CROSSINGS` such events per
        leg — so this strictly REDUCES the identity call count while moving
        every remaining call onto a frame that can actually be the answer.
        """
        self.last_channel = None
        if _is_orientation_instruction(instruction):
            return False
        # iter 14: the FINISH leg has its own, single, GRADED call.  Routed
        # out here so that nothing below it — the geometric channel (which
        # does not own FINISH and is measured to have no signature there),
        # the clock, the per-leg cap, `_place_verdict` and therefore
        # `_FRONTAGE_SYS` — is on this leg's path any more.
        if leg_type(instruction) == LEG_FINISH:
            ok = self._finish_proactive(
                obs, hint_image, vlm_clients, instruction,
                forward_streak, cur_idx)
            if ok:
                self.last_channel = "FIN_REACH"
            return ok
        geo_v = None
        if self.geo is not None:
            try:
                geo_v = self.geo.place_verdict(
                    instruction, cur_idx, forward_streak)
            except Exception:
                geo_v = None
        if geo_v is False:
            # Owned leg, junction not here.  Free veto: no token spent, and
            # crucially the identity call is NOT burned against its per-leg cap
            # on a frame that cannot license a commit.
            self._bump("geo_block")
            return False
        geo_event = geo_v is True

        streak_floor = (GEO_MIN_TRANSLATIONS if geo_event
                        else K_REACH_FORWARD_STREAK)
        if forward_streak < streak_floor:
            return False
        if hint_image is None or not vlm_clients:
            return False
        fires = int(self._ident_fires.get(cur_idx, 0))
        if fires >= MAX_REACH_FIRES_PER_SUBTASK:
            return False
        if not geo_event:
            # The clock-keyed spacing only governs the unowned legs; a
            # geometric edge is a one-shot event and waiting out an interval
            # would simply discard it.
            last = self._last_ident_streak.get(cur_idx)
            if last is not None and forward_streak - last < K_REACH_INTERVAL:
                return False
        if _rgb_of(obs) is None:
            return False
        self._ident_fires[cur_idx] = fires + 1
        self._last_ident_streak[cur_idx] = forward_streak

        if self._identity_call(obs, hint_image, vlm_clients) != 1:
            return False
        # Free segmentation veto, kept verbatim from the parent's accept path:
        # nothing built on the named side at all => the landmark is not here.
        # Fails OPEN on a missing mask.
        if not _named_side_has_structure(obs, instruction):
            self._bump("veto_side")
            return False
        here = self._place_verdict(
            obs, vlm_clients, instruction, cur_idx, forward_streak)
        if here is None:
            # Spatial channel unavailable: fall back to the parent's own gate
            # (identity WITH the leg text) rather than committing on the
            # text-free call alone, which is the high-recall/low-precision one.
            return self._legacy_verdict(
                obs, hint_image, vlm_clients, instruction) == 1
        if here != 1:
            self._bump("blocked_by_place")
            return False
        return True

    def allow_planner_commit(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        cur_idx: int,
        forward_streak: int = 0,
    ) -> bool:
        """Cross-check on a planner-proposed `-1`.

        Same defer-on-unavailable contract as the function it replaces (it may
        only ever TIGHTEN the gate when it has a confident answer), now with
        the spatial call as the second conjunct.

        iter 13 fixes the HARNESS-BUG the diagnosis isolated here (§2E).  The
        per-leg cap below used to `return True` once it was spent, so the THIRD
        planner `-1` on a leg executed with NO gate at all.  The run shows the
        signature 10 times (`XCHECK_BLOCK; EMPTY_FALLBACK; XCHECK_BLOCK;
        EMPTY_FALLBACK; …`) and it produced 6 of the 15 fatal commits —
        including ALL FOUR FINISH deaths, at 24 389 / 31 370 / 32 527 /
        44 789 cm, two of them discarding an episode that had already banked
        three legs.  A budget is a bound on CALLS, not on the VERDICT: the cap
        now replays the last verdict it actually computed, and on a FINISH leg
        a `-1` may never pass by deferral at all.
        """
        if hint_image is None or not vlm_clients:
            return True
        if _is_orientation_instruction(instruction):
            return True
        if _rgb_of(obs) is None:
            return True
        # Parent's free veto on the planner channel, unchanged.
        if not _named_side_has_structure(obs, instruction):
            self._bump("veto_side_x")
            return False
        # iter 13: the free geometric conjunct is asked BEFORE the cost cap,
        # because it costs nothing to ask.  On an owned leg a planner `-1`
        # raised mid-block is refused without spending a call, so the cap is
        # reached only by edges that were actually worth a call.
        if self.geo is not None:
            try:
                geo_v = self.geo.place_verdict(
                    instruction, cur_idx, forward_streak)
            except Exception:
                geo_v = None
            if geo_v is False:
                self._bump("geo_block_x")
                self._xcheck_memo[cur_idx] = False
                return False
        # iter 14: a planner-proposed `-1` on FINISH is cross-checked by the
        # same graded call, so the two commit routes on this leg agree by
        # construction.  The cap stays fail-closed (F2, iter 13).
        if leg_type(instruction) == LEG_FINISH:
            xf = int(self._xcheck_fires.get(cur_idx, 0))
            if xf >= MAX_XCHECK_FIRES_PER_SUBTASK:
                cached = self._xcheck_memo.get(cur_idx)
                if cached is not None:
                    self._bump("xcap_replay")
                    return bool(cached)
                self._bump("xcap_closed")
                return False
            self._xcheck_fires[cur_idx] = xf + 1
            # One reading is enough here: the planner's own `-1` is the second
            # opinion that `FINISH_CONFIRM_STREAK` buys on the proactive path.
            verdict = self._finish_reading(
                obs, hint_image, vlm_clients, cur_idx)
            self._xcheck_memo[cur_idx] = verdict
            self._bump("fin_x1" if verdict else "fin_x0")
            return verdict
        fires = int(self._xcheck_fires.get(cur_idx, 0))
        if fires >= MAX_XCHECK_FIRES_PER_SUBTASK:
            cached = self._xcheck_memo.get(cur_idx)
            if cached is not None:
                self._bump("xcap_replay")
                return bool(cached)
            # No verdict was ever computed on this leg (every fire failed
            # transiently).  Defer on a leg where a wrong `-1` is recoverable;
            # on FINISH it is the episode, so block.
            self._bump("xcap_closed")
            return leg_type(instruction) != LEG_FINISH
        self._xcheck_fires[cur_idx] = fires + 1

        ident = self._identity_call(obs, hint_image, vlm_clients)
        if ident is None:
            return True                      # transient failure must not veto
        if ident != 1:
            self._xcheck_memo[cur_idx] = False
            return False
        here = self._place_verdict(
            obs, vlm_clients, instruction, cur_idx, forward_streak)
        if here is None:
            verdict = self._legacy_verdict(
                obs, hint_image, vlm_clients, instruction) != 0
            self._xcheck_memo[cur_idx] = verdict
            return verdict
        if here != 1:
            self._bump("blocked_by_place_x")
            self._xcheck_memo[cur_idx] = False
            return False
        self._xcheck_memo[cur_idx] = True
        return True
