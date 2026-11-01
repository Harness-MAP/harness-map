"""ArrivalGate — iter 9 (rewired iter 15). The 9th node: it owns the LICENSE
for a `-1`.

ITER 15 — THE FINISH LICENCE BECOMES A SEQUENTIAL STOPPING RULE
---------------------------------------------------------------
Nothing moves in the component graph this iter; the change is entirely inside
this module, in HOW the position conjunct of the `-1` licence is computed on a
FINISH leg.

iter 13 shipped the FINISH position conjunct as a VLM question
(`_FRONTAGE_SYS`, "am I at this building's frontage?").  Measured over the
whole run it answered `Here: 0` on **29 of 29** live asks, including on all
**3** frames where the text-free identity call had correctly recognised the
destination from 484 / 640 / 1 144 cm.  A required conjunct that is
identically `False` is not a weak signal — it is a closed gate, and it is why
the run emitted **zero** FINISH `-1` in 31 attempts and scored SR 0.00 with
`per_position_SR[3] = 0.00`.  Four free geometric substitutes for it were
probed offline on all 1 244 archived FINISH frames and all four are flat or
anti-discriminative in true distance, so there is nothing to swap it for
frame-wise.

So the question changes KIND.  A FINISH leg is a **one-shot stopping problem**
(a rejected `-1` terminates the episode), and the evidence the harness has is
not one frame but a *sequence* of identity answers down the block.  The
identity call is a sharp, well-localised detector when it is asked near the
goal — the run contains a clean `0 @2 516 → 1 @484 → 0 @3 484` sweep — so the
position information is in WHERE IN THE SEQUENCE the first positive lands, not
in any single frame.  This module therefore keeps a per-leg log of its own
identity answers and licences a FINISH `-1` on the first `Match=1` that is
preceded by at least `M_FINISH_PRIOR_ZEROS` consecutive `Match=0` answers on
the same leg.  Counterfactual over this run's own 368 recorded FINISH identity
answers ("commit on the first qualifying positive"):

    m=0  in-window 2   far-field 11   never-commits 18
    m=1  in-window 2   far-field 10   never-commits 19
    m=4  in-window 2   far-field  7   never-commits 22
    m=6  in-window 2   far-field  6   never-commits 23

Recall is flat in m, precision is not: the guard is free bookkeeping that
strictly dominates committing on the raw first positive, and it removes the
leg's opening ask (the single most common far-field false positive) outright.
It is deliberately **FINISH-only** — the same guard on a ROAD leg destroys
26 of 30 in-window licences, because `RoadGeometry` already restricts ROAD to
about one ask per leg.

Two consequences ship with it, because the rule does not work without them:

  * `_FRONTAGE_SYS` and the FINISH branch of `_place_call` are **deleted**
    (not disabled — live-looking dead code is how the closed gate survived two
    iterations).  The position conjunct on FINISH is now the history guard;
    the VLM place call survives only on the legs where it is measured to work.
  * the FINISH ask schedule is re-paced inside this module
    (`MAX_IDENT_FIRES_FINISH`).  A stopping rule reads a sequence, so it needs
    the sequence to cover the leg: at the old cap of 12 asks at interval 5 the
    gate stopped asking at translation ~59 of a 126-substep leg and landed
    2.2 % of its asks inside the acceptance window.
  * the planner cross-check may no longer licence a FINISH `-1` by deferral.
    The planner's own arrival claim is 0/19 in-window and 0 of its 37 FINISH
    `-1` proposals were in-window, so on FINISH that channel is held to
    exactly the same stopping rule rather than to the legacy recognizer.

Also this iter, same decision path: the free `_named_side_has_structure` veto
is removed from the PROACTIVE accept path.  It fired exactly once in 50
episodes and the one time it fired it killed a correct in-window ROAD commit
at 113 cm — 1 measured loss, 0 measured saves.  It is kept on the planner
cross-check path, whose contract is "may only ever tighten".

WHAT THE 9TH NODE DID WHEN IT WAS ADDED (iter 9)
------------------------------------------------

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
  FINISH      : asked "are you AT the frontage" in iters 9-13; SUPERSEDED in
                iter 15 by `_finish_history_ok` (see above), because that
                question measured 29/29 `Here: 0` including inside the window.

DEGRADATION.  On the legs that still ask it, every unavailability of the
spatial call (no client, no RGB, parse failure, cap spent) falls back to the
parent's exact gate — `checker._binary_reach_vlm_call` with the leg text — so
a dead spatial channel can never lock the harness into "never commits".  That
is the failure mode that killed the iter-5 depth gate, and it is why the iter-15
FINISH branch replaces the conjunct with another predicate rather than removing
it and inheriting that fallback.

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

# iter 15: `_FRONTAGE_SYS` DELETED.  It was the FINISH position conjunct and it
# answered `Here: 0` on 29/29 live asks — including on the three frames where
# the identity call had already recognised the destination from 484 / 640 /
# 1 144 cm — so the FINISH `-1` was structurally unreachable all run (0 commits
# in 31 legs).  Its replacement is not another wording but a different KIND of
# predicate: `ArrivalGate._finish_history_ok`, a free sequential stopping rule
# over this gate's own per-leg identity answers (see the module docstring).
# The surviving `_place_call` is junction-only, on the legs where it is
# measured to work.


# ---------------------------------------------------------------------------
# The FINISH stopping rule.  Both numbers are read off the parent run's own
# 368 recorded FINISH identity answers, not guessed.
# ---------------------------------------------------------------------------
# How many consecutive `Match=0` answers must precede a `Match=1` on a FINISH
# leg before it may licence the (one-shot, episode-fatal) `-1`.  Counterfactual
# on the parent run: m=0 commits 2 in-window and 11 far-field, m=4 commits the
# SAME 2 in-window and 7 far-field.  Recall is flat in m; precision is not.
# FINISH ONLY — on ROAD the identical guard kills 26 of 30 in-window licences.
M_FINISH_PRIOR_ZEROS = 4

# The stopping rule reads a SEQUENCE, so the sequence has to cover the leg.
# At the inherited cap (`MAX_REACH_FIRES_PER_SUBTASK = 12`, interval 5) the
# gate stopped asking at translation ~59 on FINISH legs of 52-199 translations
# and put 8 of 368 asks (2.2 %) inside the acceptance window, i.e. ~0.8 asks on
# a leg that enters it at all.  Doubling the cap keeps the interval (and so the
# spacing the guard's statistics were measured at) and extends coverage over
# the back half of the leg, where the destination of a long leg actually is.
# Budget: ~+12 identity calls on the longest FINISH legs, against 29 frontage
# calls deleted and an `over_budget: false` flag at the parent's cost.
MAX_IDENT_FIRES_FINISH = 24


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
    # iter 15: the per-leg LOG of this gate's own identity answers, in the
    # order they were asked (0/1 only; an unparsed answer is not evidence and
    # is not recorded).  This is the whole state the FINISH stopping rule
    # reads — self-derived bookkeeping over calls the harness already makes,
    # no new observation and no privileged field.
    _ident_log: Dict[int, List[int]] = field(default_factory=dict)
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

    # ---- the FINISH stopping rule (free: no call, no substep) ------------
    def _note_identity(self, cur_idx: int, verdict: Optional[int]) -> None:
        """Append one identity answer to the current leg's log."""
        if verdict is None:
            return
        self._ident_log.setdefault(int(cur_idx), []).append(
            1 if int(verdict) == 1 else 0)

    def _prior_zero_run(self, cur_idx: int) -> int:
        """Consecutive `Match=0` answers immediately BEFORE the latest one.

        Called straight after `_note_identity` recorded the positive that is
        asking for the licence, so the final entry is that positive and the
        run is counted backwards from the entry before it.
        """
        log = self._ident_log.get(int(cur_idx)) or []
        run = 0
        for v in reversed(log[:-1]):
            if v == 0:
                run += 1
            else:
                break
        return run

    def _finish_history_ok(self, cur_idx: int) -> bool:
        """The FINISH position conjunct — free, and the replacement for the
        deleted `_FRONTAGE_SYS` veto.

        A FINISH leg gets exactly ONE `-1` (a rejection terminates the
        episode), so this is a stopping problem, not a recall problem: the
        question is not "does this frame look like arrival" — the frozen model
        cannot answer that from the travel heading, measured 29/29 `Here: 0` —
        but "is this the first time down this block that the landmark has been
        recognised, after the model has had a fair look at the block and said
        no?".  A positive that follows a run of `M_FINISH_PRIOR_ZEROS` zeros is
        a CHANGE in a sequence the harness itself generated; the leg's opening
        ask, which is the commonest far-field false positive, can never be one.
        """
        ok = self._prior_zero_run(cur_idx) >= M_FINISH_PRIOR_ZEROS
        self._bump("finish_hist_pass" if ok else "finish_hist_block")
        return ok

    # ---- the two perception calls ---------------------------------------
    def _identity_call(
        self,
        obs: Any,
        hint_image: Optional[np.ndarray],
        vlm_clients: Optional[Dict[str, Any]],
        cur_idx: Optional[int] = None,
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
        # iter 15: every answer this gate gets is evidence about the leg, from
        # whichever call site asked for it, so both sites feed one log.
        if cur_idx is not None:
            self._note_identity(cur_idx, v)
        return v

    def _place_call(
        self,
        obs: Any,
        vlm_clients: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """"Am I THERE?" — one image, no hint, no leg text.

        Independent of the identity call in every input except the frame: it
        never sees the reference photo and never sees the instruction, so it
        cannot inherit the identity call's error.

        iter 15: junction wording ONLY.  The FINISH (frontage) wording is
        deleted, not switched off — on FINISH the position conjunct is now
        `_finish_history_ok`, which spends no token at all.
        """
        rgb = _rgb_of(obs)
        if rgb is None or not vlm_clients:
            return None
        try:
            rgb_b64 = numpy_to_base64(np.asarray(rgb))
        except Exception:
            return None
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
                    {"role": "system", "content": _JUNCTION_SYS},
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

        iter 15: a FINISH leg never reaches the VLM path at all — its position
        conjunct is `_finish_history_ok`, which costs nothing.  The guard is
        explicit rather than implied by the call sites so that a future caller
        cannot silently resurrect a frontage question that measured 0/29.

        The VLM path keeps its memo: the body has not moved within a step, so
        the answer cannot change within a step."""
        if leg_type(instruction) == LEG_FINISH:
            return None
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
        v = self._place_call(obs, vlm_clients)
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
        if _is_orientation_instruction(instruction):
            return False
        fires = int(self._ident_fires.get(cur_idx, 0))
        # iter 15: the stopping rule needs its sequence to span the leg, so the
        # FINISH cap is the re-paced one.  ROAD keeps the inherited cap (it is
        # keyed on geometric edges and spends ~1 ask per leg anyway).
        ident_cap = (MAX_IDENT_FIRES_FINISH
                     if leg_type(instruction) == LEG_FINISH
                     else MAX_REACH_FIRES_PER_SUBTASK)
        if fires >= ident_cap:
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

        if self._identity_call(obs, hint_image, vlm_clients, cur_idx) != 1:
            return False
        # iter 15: on a FINISH leg the position conjunct is the free stopping
        # rule over this leg's identity answers.  No place call, no legacy
        # fallback: the legacy recognizer is the identity question WITH the leg
        # text, which is the 14 %-recall composition iter 9 removed, and
        # routing FINISH into it is how the old `None` branch would have
        # leaked far-field commits once the frontage call stopped answering.
        if leg_type(instruction) == LEG_FINISH:
            return self._finish_history_ok(cur_idx)
        # iter 15 DELETED on this path: the `_named_side_has_structure` veto.
        # It fired exactly once in 50 episodes and killed a correct in-window
        # ROAD commit at 113 cm (1 measured loss, 0 measured saves).  It is
        # kept on the cross-check path below, whose contract is tighten-only.
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

        ident = self._identity_call(obs, hint_image, vlm_clients, cur_idx)
        if ident is None:
            return True                      # transient failure must not veto
        if ident != 1:
            self._xcheck_memo[cur_idx] = False
            return False
        # iter 15: on FINISH this channel is held to EXACTLY the proactive
        # path's stopping rule and may not defer its way to a `-1`.  The
        # planner has no measured FINISH arrival signal — its own Match is
        # 0/19 in-window and 0 of 37 FINISH `-1` proposals were in-window —
        # so without this branch the deleted frontage veto would have been
        # replaced by the legacy recognizer, i.e. by the loosest gate in the
        # module, on the one leg where a wrong `-1` is the whole episode.
        if leg_type(instruction) == LEG_FINISH:
            verdict_f = self._finish_history_ok(cur_idx)
            self._xcheck_memo[cur_idx] = verdict_f
            return verdict_f
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
