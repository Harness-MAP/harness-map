"""BurstMonitor — in-burst arrival sampling (the 8th module).

WHY THIS COMPONENT EXISTS (wiring, not wording)
-----------------------------------------------
Until now every arrival question in this harness was asked at a *decision*
frame: the loop perceives, plans, and only then consults
`CompletionChecker.proactive_commit_check`. But the loop does not decide once
per env step — `ActivePerception`'s momentum cadence emits a burst of
translations per decision, and `harness.py`'s burst executor walks that whole
burst without ever re-consulting the checker. Measured on this arm (diagnosis
§2.2): one perceive+plan per **3.28 env steps**, and inside the pass window of
a ROAD leg — **6.61 forward frames deep** — the arrival question is asked
**1.10 times**. The per-frame recognizer answers `Match=1` with probability
0.205 inside the window, so

    P(commit | entered window) = 1 - (1 - 0.205)^1.10 = 0.22

which reproduces the observed ROAD rate (0.24) to two digits. The recall term
is the frozen model's ceiling and seven reformulations of the question have
been refuted; **the exponent is the harness's**. Raising it to the 6.61 frames
that already exist predicts 1 - 0.795^6.61 = 0.75.

Resampling the SAME frame buys nothing — 10/10 byte-identical arrival prompts
in the parent run returned byte-identical verdicts. Only *distinct* frames are
independent draws. So this module does not change the question, the criterion,
the evidence, or any threshold; it adds a **new edge in the component graph**:
the actuation stage now reports back into the arrival channel after every
translation it executes, and can truncate its own burst into a `-1`.

WHAT IT OWNS
------------
The *sampling policy* — when to spend an arrival call, under what per-leg
budget, and with which acceptance rule **per leg type**. The question itself
still belongs to `CompletionChecker._binary_reach_vlm_call` (reached through
the public `binary_reach` delegate), and the mask-based accept-validator is
the same `_named_side_has_structure` the decision-level path applies, so the
operating point of a single sample is unchanged by construction.

LEG-TYPE SPLIT (why one policy cannot serve both)
-------------------------------------------------
The same recognizer has two opposite operating points (diagnosis §2.4):

    leg      fires  Match=1   precision   recall
    ROAD      305      10       0.90       0.21
    FINISH     43       7       0.00       0.00

Every FINISH `Match=1` fired 7 778 - 58 789 cm out, because the destination
building is *visible from across the map*: on a FINISH leg "I can see it" is
true long before "I am at it". Six-fold density on that leg would simply
false-commit earlier. So the two legs get two rules, separated by an exact,
free, sanctioned text test: ROAD instructions say "stop **at the
intersection** with ..." (305/305 of them); FINISH instructions say "Proceed
forward until reaching ... stop and face ..." (0/43). No coordinates, no
ground truth, no leg-type field — `task_spec['subtasks']` text only.

  * ROAD  — a validated `Match=1` commits immediately (precision 0.90).
  * FINISH — the first `Match=1` of the leg only **ARMS** the destination; the
    commit needs a later `Match=1` after `FINISH_APPROACH_FORWARDS` more
    translations. "I have seen it and then kept walking toward it" is the
    closest thing to an approach signal available without reading a
    coordinate. This is what §2.5's two starved episodes needed: they stood
    **16 cm** and **27 cm** from the destination and the harness asked 1 and
    **0** arrival questions there.

ANTI-CHEAT: reads `obs['rgb']`, `obs['object_mask']`, the subtask's
`visual_hint_path` image, this leg's own executed-translation count, and the
static instruction text. No `agent_location`, no `target_location`, no
`gt.json`, no `ideal_path`, no analog yaw, no `subtask_max_steps`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from collections.abc import Mapping

from .completion_checker import (
    K_REACH_FORWARD_STREAK,
    _is_orientation_instruction,
    _named_side_has_structure,
)

# Translations between two arrival samples inside a burst. 1 == ask on every
# frame the burst produces; the whole point of the module is that consecutive
# frames are the only independent draws available (see docstring).
INBURST_SAMPLE_EVERY = 1

# Per-leg budget on in-burst samples. Cost bound only — measured median ROAD
# leg has ~15 translations, so at the motion floor this rarely binds; it exists
# so a pathological leg that never ends cannot spend the episode's token budget
# on one waypoint.
MAX_INBURST_FIRES_PER_SUBTASK = 20

# FINISH legs only: translations that must be executed between the sample that
# ARMS the destination and the sample that may commit it. Expressed in forward
# actions (never centimetres) — the harness has no licence to read distance.
FINISH_APPROACH_FORWARDS = 6

# ROAD legs say "stop at the intersection with <street>"; FINISH legs say
# "Proceed forward until reaching <landmark>, then stop and face it".
# Measured separation on the parent run: 305/305 vs 0/43.
_INTERSECTION_RE = re.compile(r"\bat the intersection\b", re.IGNORECASE)


def is_intersection_leg(instruction: Any) -> bool:
    """True for a TURN-at / stop-at-intersection (ROAD) leg, False for FINISH."""
    return isinstance(instruction, str) and bool(_INTERSECTION_RE.search(instruction))


@dataclass
class BurstMonitor:
    """Samples the arrival question from inside the burst executor.

    One instance per episode (no cross-episode state). The harness calls
    `check(...)` after every translation step it executes; a True return means
    "truncate the rest of this burst and commit `-1` from the frame you are
    standing on".
    """

    # Per-leg count of in-burst samples spent (cost gate).
    _fires: Dict[int, int] = field(default_factory=dict)
    # Per-leg forward_streak at which the last sample was spent (spacing gate).
    _last_streak: Dict[int, int] = field(default_factory=dict)
    # FINISH legs: forward_streak at which the destination was first recognised.
    _armed_streak: Dict[int, int] = field(default_factory=dict)

    # Serial number (== len(actions_log)) of the frame most recently sampled.
    # The harness reads this to avoid asking the decision-level
    # `proactive_commit_check` the same question about the same frame: the
    # recognizer is deterministic per frame (10/10 identical prompts ->
    # identical verdicts), so a duplicate call is a pure token cost.
    last_sampled_serial: int = -1

    def check(
        self,
        checker: Any,
        obs: Any,
        hint_image: Any,
        vlm_clients: Optional[Dict[str, Any]],
        instruction: Any,
        forward_streak: int,
        cur_idx: int,
        step_serial: int,
    ) -> bool:
        """Spend one arrival sample on the frame just produced by a translation.

        Returns True iff the harness should stop the burst and commit `-1`.
        Every failure mode (no hint, no client, no RGB, orientation leg, budget
        spent, transient API error, unparseable verdict) returns False, i.e.
        the monitor can only ever ADD a commit opportunity — it never blocks
        one the decision-level path would have taken.
        """
        # --- applicability -------------------------------------------------
        if hint_image is None or not vlm_clients:
            return False
        if not isinstance(obs, Mapping) or obs.get("rgb") is None:
            return False
        if _is_orientation_instruction(instruction):
            return False

        # --- motion floor: identical to the decision-level path -------------
        # Zero-motion "the spawn looks like the hint" claims are the failure
        # mode this floor was added for; in-burst sampling must not reopen it.
        if forward_streak < K_REACH_FORWARD_STREAK:
            return False

        # --- spacing + per-leg cost bound -----------------------------------
        last = self._last_streak.get(cur_idx)
        if last is not None and forward_streak - last < INBURST_SAMPLE_EVERY:
            return False
        fires = int(self._fires.get(cur_idx, 0))
        if fires >= MAX_INBURST_FIRES_PER_SUBTASK:
            return False

        # Charge the sample BEFORE the call so a transient API error cannot
        # retry on every step of the burst.
        self._fires[cur_idx] = fires + 1
        self._last_streak[cur_idx] = forward_streak
        self.last_sampled_serial = int(step_serial)

        match = checker.binary_reach(
            obs=obs,
            hint_image=hint_image,
            vlm_clients=vlm_clients,
            instruction=instruction,
        )
        if match != 1:
            return False

        # --- accept-validator (unchanged operating point) --------------------
        # The same side-structure veto the decision-level accept applies: if
        # the instruction names a side and that side is EMPTY of building-class
        # pixels, the named landmark cannot be here. Fails OPEN on a missing or
        # unusable mask, and on instructions that name no side.
        if not _named_side_has_structure(obs, instruction):
            return False

        # --- leg-type acceptance rule ----------------------------------------
        if is_intersection_leg(instruction):
            # ROAD: fire-level precision 0.90. A validated positive commits.
            return True

        # FINISH: "visible" != "arrived". Arm on the first positive of the leg,
        # commit only on a positive raised at least FINISH_APPROACH_FORWARDS
        # translations later — i.e. the agent recognised the destination and
        # then actually walked toward it.
        armed = self._armed_streak.get(cur_idx)
        if armed is None:
            self._armed_streak[cur_idx] = forward_streak
            return False
        return forward_streak - armed >= FINISH_APPROACH_FORWARDS
