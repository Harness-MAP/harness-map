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
    # per-subtask counters (keyed by the structural subtask index)
    _place_fires: Dict[int, int] = field(default_factory=dict)
    _ident_fires: Dict[int, int] = field(default_factory=dict)
    _xcheck_fires: Dict[int, int] = field(default_factory=dict)
    _last_ident_streak: Dict[int, int] = field(default_factory=dict)
    # (forward_streak, verdict) memo so the two call sites in one step cannot
    # ask the same spatial question twice.
    _place_memo: Dict[int, Tuple[int, Optional[int]]] = field(default_factory=dict)
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
        """Memoised `_place_call`: the body has not moved within a step, so the
        answer cannot change within a step.  Expires the moment the leg's
        translation count changes."""
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

        Schedule is the parent's, verbatim: fire at `K_REACH_FORWARD_STREAK`
        translations, then every `K_REACH_INTERVAL`, capped at
        `MAX_REACH_FIRES_PER_SUBTASK` per leg.  What changed is what the fire
        asks:  identity(no text)  AND  place.
        """
        if forward_streak < K_REACH_FORWARD_STREAK:
            return False
        if hint_image is None or not vlm_clients:
            return False
        if _is_orientation_instruction(instruction):
            return False
        fires = int(self._ident_fires.get(cur_idx, 0))
        if fires >= MAX_REACH_FIRES_PER_SUBTASK:
            return False
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
        fires = int(self._xcheck_fires.get(cur_idx, 0))
        if fires >= MAX_XCHECK_FIRES_PER_SUBTASK:
            return True
        self._xcheck_fires[cur_idx] = fires + 1

        ident = self._identity_call(obs, hint_image, vlm_clients)
        if ident is None:
            return True                      # transient failure must not veto
        if ident != 1:
            return False
        here = self._place_verdict(
            obs, vlm_clients, instruction, cur_idx, forward_streak)
        if here is None:
            return self._legacy_verdict(
                obs, hint_image, vlm_clients, instruction) != 0
        if here != 1:
            self._bump("blocked_by_place_x")
            return False
        return True
