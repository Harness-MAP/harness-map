"""branches/naive — fresh honest DSL skeleton (H_0).

run_episode wires 10 modules through a per-step loop:

  1. SpatialMemory       — odometry + step / subtask boundary log
  2. PromptAssembler     — chat-completions message construction
  3. Perceiver           — one VLM call: RGB(+hint) -> description
  4. Planner             — one VLM call: description -> action list
  5. AffordanceEstimator — full motion set (stub)
  6. ActivePerception    — no proactive lookaround (stub)
  7. CompletionChecker   — owns the legacy reach recognizer (now the gate's
                           degradation path) + the planner-`-1` verify rule
  8. ViewpointController — POSE arbitration (iter 6). iter 9 keeps only its
                           TURN approach + the FINISH commit pose; its
                           face-before-look edge is deleted (measured 0/128)
  9. ArrivalGate         — iter 9: owns the LICENSE for every non-orient `-1`.
                           identity(two images, no leg text) AND place(one
                           image, no hint) — two independent calls, conjoined
 10. RoadGeometry       — iter 13: a FREE, non-VLM POSITION channel. A new
                           sensor stage reads the roadway class out of
                           `obs['object_mask']` once per loop iteration and
                           raises a rising-edge "I have just crossed into an
                           intersection" event. On an intersection-stop ROAD
                           leg it REPLACES the gate's `_JUNCTION_SYS` call as
                           the position conjunct and KEYS the identity call's
                           schedule; elsewhere it is inert.

Deployment-honest reads only:
  obs:  rgb, depth (env key; NOT depth_map), object_mask
  info: moving_distance, agent.agent_rotation (cardinal), landmark_images
  task_spec: subtasks[i].{instruction, visual_hint_path}
  env.unwrapped.current_subtask  — structural index ONLY (not an oracle).

Hard-coded thresholds resolved by probe (see workspace_single probe):
  depth      : JET colormap (600, 720, 3), per-frame normalized — NEAR=blue, FAR=red,
               NOT grayscale uint8 (decode via JET inverse; see completion_checker._depth_gray).
  yaw conv.  : action 5 (TURN_LEFT) = -90 deg yaw delta (CW);
               action 4 (TURN_RIGHT) = +90 deg yaw delta (CCW).
               yaw 0 == West, 90 == North, 180 == East, 270 == South.
  initial    : West / yaw 0.
  object_mask: uint8 (600, 720, 3); crosswalk shares hue with sidewalk
               (texture is the only disambiguator — out of scope for H_0).

The harness's high-level action codes are the published baseline's:
  0 Move_Forward (env step 0), 1 Rotate_Left (env step 5),
  2 Rotate_Right (env step 4), 3 Move_Left (env step 2),
  4 Move_Right (env step 3), -1 Subtask_Completed (env step -1).

Stuck detector: forward_count > 3 AND delta moving_distance < 0.5 -> break.
"""
from __future__ import annotations
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections.abc import Mapping

import numpy as np

# Self-register so @dataclass below can resolve string annotations when
# loaded via importlib.util.spec_from_file_location (validate_harness.py
# path). Mirrors workspace/branches/react_h0/orchestrator.py lines 30-38, but
# also pins the parent package's `__path__` so `from .modules import ...`
# resolves under the import-by-spec entry path.
sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    _pkg_mod = sys.modules.get(_pkg)
    if _pkg_mod is None:
        _pkg_mod = types.ModuleType(_pkg)
        sys.modules[_pkg] = _pkg_mod
    if not hasattr(_pkg_mod, "__path__"):
        _here = os.path.dirname(os.path.abspath(__file__))
        _pkg_mod.__path__ = [_here]  # type: ignore[attr-defined]

# Reuse the published baseline's `action_history_text` helper (formatter,
# not a model call). Keeps planner prompts identical in surface form to
# what the baseline emits — important for fair H_0 comparison.
_BASELINE_DIR = "/home/azureuser/workspace/AutoSpatial/simworld_gym/baseline/single"
if _BASELINE_DIR not in sys.path:
    sys.path.insert(0, _BASELINE_DIR)

from utils import action_history_text  # noqa: E402

from .modules import (  # noqa: E402
    SpatialMemory,
    Perceiver,
    Planner,
    AffordanceEstimator,
    ActivePerception,
    CompletionChecker,
    PromptAssembler,
    ViewpointController,
    ArrivalGate,
    RoadGeometry,
)


@dataclass
class EpisodeResult:
    success: bool
    subtask_success: List[bool]
    steps: int
    actions: List[int]
    llm_calls: List[Dict[str, Any]]
    notes: str = ""
    observations: List[Dict[str, Any]] = field(default_factory=list)


# Mirror published baseline action -> env step mapping.
# 0 Move_Forward, 1 Rotate_Left, 2 Rotate_Right, 3 Move_Left, 4 Move_Right.
ACTION_TO_STEP = {0: 0, 1: 5, 2: 4, 3: 2, 4: 3}

ACTION_NAMES = [
    "Move_Forward", "Rotate_Left", "Rotate_Right",
    "Move_Left", "Move_Right", "Subtask_Completed",
]

# UnrealCV socket-death substrings (port reused from react_h0 hardening).
# When the sim disconnects mid-episode, env.step raises an exception whose
# message contains one of these. The harness catches and returns the
# partial result so the evaluator's outer catch-all doesn't wipe traces.
_SOCKET_DEAD_SUBSTRS = (
    "socket is closed",
    "remote disconnected",
    "Connection lost during receive",
    "Connection reset",
    "cannot unpack non-iterable NoneType object",
)


def _is_socket_dead(exc: BaseException) -> bool:
    msg = str(exc) or ""
    return any(s in msg for s in _SOCKET_DEAD_SUBSTRS)


def _load_hint_image(visual_hint_path: Optional[str]) -> Optional[np.ndarray]:
    """Load a visual_hint image. Returns None on missing path / read error."""
    if not visual_hint_path:
        return None
    try:
        from PIL import Image  # local import: PIL only needed when path exists.
        return np.asarray(Image.open(visual_hint_path).convert("RGB"))
    except Exception:
        return None


def _current_subtask_idx(env) -> int:
    """Read subtask index from the unwrapped env.

    Structural fact, not an observation — the evaluator already knows the
    episode has N subtasks. We read the index directly rather than via the
    per-step instruction field on info, so instructions come from the static
    `task_spec['subtasks']` list.
    """
    inner = env.unwrapped
    return int(getattr(inner, "current_subtask", 0))


def _orientation_from_info(info: Dict[str, Any]) -> Any:
    """Pull the cardinal compass string from info, honestly.

    Per probe: info['agent']['agent_rotation'] is the cardinal (West/North/
    East/South), NOT a scalar yaw. (Analog yaw lives at
    env.unwrapped._agent_rotation[1] but we keep H_0 free of that read.)
    """
    if not isinstance(info, Mapping):
        return None
    agent_info = info.get("agent") or {}
    orientation = agent_info.get("agent_rotation")
    if isinstance(orientation, (list, tuple)) and orientation:
        orientation = orientation[0]
    return orientation


def run_episode(env, task_spec, vlm_clients, trace_dir=None) -> EpisodeResult:
    """Run one episode through the H_0 module stack.

    The env is already reset by the evaluator; we start from
    `task_spec['_initial_obs']` / `task_spec['_initial_info']`.
    """
    obs = task_spec.get("_initial_obs")
    info = task_spec.get("_initial_info") or {}
    subtasks_spec = task_spec.get("subtasks", [])
    n_subtasks = len(subtasks_spec)
    max_steps = int(task_spec.get("max_steps", 200))

    # ---- module instances (one per episode) -------------------------
    memory = SpatialMemory()
    assembler = PromptAssembler()
    perceiver = Perceiver(assembler=assembler)
    planner = Planner()
    affordance = AffordanceEstimator()
    active = ActivePerception()
    completion = CompletionChecker()
    # iter 6: the 8th node. Owns POSE — where the body stands and points —
    # for every leg. `active`/`completion` keep owning bursts and appearance
    # respectively; `viewpoint` sits between them and decides (a) whether a
    # turn burst may fire yet, (b) from which heading the arrival look is
    # taken, (c) in which pose a `-1` is emitted.
    viewpoint = ViewpointController()
    # iter 9: the 9th node. Owns the LICENSE for a `-1` on every non-orient
    # leg. It replaces BOTH of CompletionChecker's decision methods
    # (`proactive_commit_check`, `cross_check_planner_commit`) with the same
    # schedule asking a different, two-part question: image identity WITHOUT
    # the leg text (the measured 14 % -> 100 % in-window recall lever) AND an
    # independent hint-free spatial call about where the body is standing.
    # `checker=completion` is the degradation edge: if the spatial call cannot
    # answer, the gate falls back to CompletionChecker's original recognizer,
    # so a dead second channel degrades to the parent instead of to silence.
    # iter 13: the 10th node. Owns the POSITION half of the arrival question
    # on intersection-stop ROAD legs, in closed form off `obs['object_mask']`
    # and for zero tokens. The gate keeps the IDENTITY half (93 % in-window
    # recall, this chain's biggest measured perception win) and this node
    # supplies the half the diagnosis measured as the bottleneck (the VLM
    # junction question: 29 % in-window, and 18 of its 19 positives collapsed
    # onto a single crosswalk cue sitting at a fixed half-block offset).
    # Geometry knows WHERE but not WHICH; identity knows WHICH but not WHERE.
    road = RoadGeometry()
    arrival = ArrivalGate(checker=completion, geo=road)

    # ---- per-episode bookkeeping ------------------------------------
    subtask_passed: List[bool] = [False] * n_subtasks
    actions_log: List[int] = []
    llm_calls: List[Dict[str, Any]] = []
    notes_parts: List[str] = []
    parse_failure_count = 0

    last_moving_distance: Optional[float] = None
    chosen_actions: List[int] = []
    action_history_list: List[int] = []
    terminated = False

    # Baseline-parity memory: the rolling semantic History Summary, threaded
    # into both perceive and plan and updated from the planner's output each
    # step. Persists across subtask boundaries (the baseline ReAct agent keeps
    # the summary; only action_history resets on -1).
    rolling_summary: str = ""

    # iter 1 REACH-check counter. Tracks consecutive Move_Forward
    # (action 0) executions within the current subtask. Reset on (i)
    # any non-forward action, (ii) subtask boundary (-1 commit), (iii)
    # a successful proactive REACH commit. Used to gate
    # CompletionChecker.proactive_commit_check.
    forward_streak = 0

    # Per-subtask planner Match history (passed to CompletionChecker).
    planner_match_history: List[Optional[int]] = []

    # Cache hint images per subtask so we don't re-decode every step.
    hint_cache: Dict[int, Optional[np.ndarray]] = {}

    def hint_for(idx: int) -> Optional[np.ndarray]:
        if idx not in hint_cache:
            if 0 <= idx < n_subtasks:
                hint_cache[idx] = _load_hint_image(
                    subtasks_spec[idx].get("visual_hint_path"))
            else:
                hint_cache[idx] = None
        return hint_cache[idx]

    def _scan_step(action_hl: int) -> str:
        """Execute ONE high-level turn action for an in-place look-scan, with the
        SAME socket-death guard + bookkeeping as the burst executor. Mutates
        obs/info/terminated. Returns 'ok' to keep scanning, or 'stop' if the
        episode ended (socket death / MAX_STEPS) and the caller must bail out.
        Turns are in-place yaw (position preserved) and are NOT counted toward
        forward_streak — only the burst executor's translations advance it."""
        nonlocal obs, info, terminated
        action_history_list.append(action_hl)
        actions_log.append(action_hl)
        step_code = ACTION_TO_STEP.get(action_hl)
        if step_code is None:
            return "ok"
        try:
            obs, _, terminated, _, info = env.step(step_code)
        except Exception as e:
            if _is_socket_dead(e):
                notes_parts.append("SOCKET_DEAD")
                terminated = True
                return "stop"
            raise
        memory.log_step(action=action_hl, info=info)
        if len(actions_log) >= max_steps:
            notes_parts.append("MAX_STEPS")
            terminated = True
            return "stop"
        return "stop" if terminated else "ok"

    while not terminated and len(actions_log) < max_steps:
        # iter_007: per-step exception guard. ANY uncaught exception
        # inside the loop body (perceive/plan json crash, malformed obs,
        # transient None response from vLLM, etc.) was previously
        # propagating to evaluator.py's outer except, which constructs a
        # fallback EpisodeResult with `subtask_success=[False]*n` and
        # erases the partial `subtask_passed`, `actions_log`, and
        # `llm_calls` collected so far. This dropped iter_006's ep_20_4
        # from [True, F, F, F] (cardinal burst had already committed
        # subtask 0) to [F, F, F, F], regressing per_position_SR[0]
        # from 1.0 -> 0.8. The guard wraps the body so a non-socket-dead
        # exception is recorded as `CRASH:<type>:<msg>` and the loop
        # exits cleanly, preserving every bit of progress made so far.
        # BaseException (KeyboardInterrupt, SystemExit) is intentionally
        # NOT caught so Ctrl-C still aborts. SOCKET_DEAD detection is
        # handled BEFORE the guard so the socket-death note stays
        # specific (matches the substring set in _SOCKET_DEAD_SUBSTRS).
        try:
            # Guard against silent obs=None returns from env.step (observed in
            # react_h0 iter_010 — env.step occasionally returns None without
            # raising). Without this guard the perceiver crashes on obs['rgb'].
            if obs is None or not isinstance(obs, Mapping):
                notes_parts.append("SOCKET_DEAD")
                break

            # ---- stuck detector (forward_count > 3 AND delta < 0.5) ----
            md = info.get("moving_distance", 0.0) if isinstance(info, Mapping) else 0.0
            try:
                current_moving_distance = float(
                    np.asarray(md).flat[0]) if md is not None else 0.0
            except Exception:
                current_moving_distance = 0.0
            forward_count = sum(1 for a in chosen_actions if a == 0)
            if (last_moving_distance is not None
                    and forward_count > 3
                    and (current_moving_distance - last_moving_distance) < 0.5):
                notes_parts.append("STUCK")
                break
            last_moving_distance = current_moving_distance

            # ---- subtask bookkeeping (read structural index, not oracle) -
            cur_idx = _current_subtask_idx(env)
            if cur_idx >= n_subtasks:
                break
            instruction = subtasks_spec[cur_idx].get("instruction", "")

            orientation = _orientation_from_info(info)

            # ---- 6. ActivePerception (iter_004: red-signal idle burst on
            # top of iter_003 cardinal burst + iter_001 anti-spin. obs is
            # passed for the saturated-red detector over the upper-center
            # of obs['rgb']; orientation is the already-coerced cardinal
            # string from info['agent']['agent_rotation']) ----
            ap_actions = active.should_emit(
                state=memory,
                instruction=instruction,
                orientation=orientation,
                obs=obs,
            )

            # ---- 6b. ViewpointController: POSE gate on the burst (iter 6) --
            # Pass-through on every leg but TURN_AT_INTERSECTION. There the
            # `[rotate, -1]` burst is captured and held while the controller
            # walks the agent INTO the junction (the instruction says "turn
            # right AT THE INTERSECTION"; the burst used to fire on the first
            # frame of the leg with zero forward motion, landing 78-980 cm
            # outside the position box in 7/7 traced failures with yaw error 0
            # in every one). It is released when the lateral segmentation band
            # on the turn side opens, past a 2-forward floor, or at a 3-forward
            # cap. ActivePerception latches that burst to once per instruction,
            # so after capture the controller is the only thing that can emit
            # it — `_turn_released` makes the hand-back idempotent.
            ap_actions = viewpoint.gate_burst(
                ap_actions, instruction=instruction, obs=obs)

            if ap_actions:
                # Reserved for proposers: a non-None list short-circuits the
                # perceive+plan pipeline.
                chosen_actions = list(ap_actions)
            else:
                # ---- 2b. RoadGeometry: the FREE sensor stage (iter 13) ----
                # Deliberately placed INSIDE the perceive branch rather than
                # at the top of the loop: these are exactly the frames the
                # 25/45 operating point was measured on (one reading per
                # perceive call, 1 326 of them across the parent's 50
                # episodes). Sampling it on the ActivePerception short-circuit
                # frames too — or, worse, at env-step rate inside the burst —
                # would make the edge detector denser than anything that has
                # been measured, and a rising-edge rule is sensitive to its
                # sampling rate in a way a threshold rule is not. That denser
                # cadence is the next lever, not this one.
                geo_read = road.update(obs, instruction, cur_idx)

                # ---- 3. Perceiver: single VLM call --------------------
                perceive_out = perceiver.perceive(
                    obs=obs,
                    vlm_clients=vlm_clients,
                    hint_image=hint_for(cur_idx),
                    instruction=instruction,
                    orientation=orientation,
                    # Baseline parity: feed the semantic running summary (the
                    # baseline perceive step receives the History Summary).
                    memory_summary=rolling_summary,
                )
                description = perceive_out.get("description", "") or ""
                p_usage = perceive_out.get("usage") or {"input": 0, "output": 0}
                # Log the perceiver's actual description so a future proposer
                # can see WHAT the VLM said about each frame, not just how
                # many tokens it took.
                llm_calls.append({
                    "input_tokens": int(p_usage.get("input", 0)),
                    "output_tokens": int(p_usage.get("output", 0)),
                    "subtask_idx": cur_idx,
                    "stage": "perceive",
                    "raw_response": description[:600],
                })

                # ---- 4. Planner: single VLM call ----------------------
                ah_text = action_history_text(action_history_list, ACTION_NAMES)
                plan_out = planner.plan(
                    vision_description=description,
                    instruction=instruction,
                    orientation=orientation,
                    action_history_text=ah_text,
                    vlm_clients=vlm_clients,
                    memory_summary=rolling_summary,
                )
                # Thread the planner's updated summary forward (baseline ReAct
                # keeps a running History Summary; persists across subtasks).
                _new_summary = plan_out.get("summary")
                if isinstance(_new_summary, str) and _new_summary.strip():
                    rolling_summary = _new_summary.strip()[:600]
                r_usage = plan_out.get("usage") or {"input": 0, "output": 0}
                # Log the planner's parsed output so a future proposer can
                # see WHAT the planner decided per tick, not just that it
                # spent N tokens. Planner contract returns `summary` (1-line
                # gloss) and `reason` (raw Qwen text ≤800 chars); both are
                # the channel where the VLM explained ITSELF.
                planner_response = {
                    "actions": list(plan_out.get("actions") or []),
                    "match":   plan_out.get("match"),
                    "summary": (plan_out.get("summary") or "")[:200],
                    "reason":  (plan_out.get("reason")  or "")[:400],
                }
                llm_calls.append({
                    "input_tokens": int(r_usage.get("input", 0)),
                    "output_tokens": int(r_usage.get("output", 0)),
                    "subtask_idx": cur_idx,
                    "stage": "plan",
                    "parsed": planner_response,
                })

                planner_actions = list(plan_out.get("actions") or [])
                planner_match = plan_out.get("match")
                planner_match_history.append(planner_match)

                if not planner_actions:
                    parse_failure_count += 1
                    notes_parts.append("NO_ACTION_RETURNED")
                    if parse_failure_count > 10:
                        notes_parts.append("TOO_MANY_ERRORS")
                        break
                    continue

                # ---- 5. AffordanceEstimator (stub: full motion set) ---
                allowed = set(affordance.estimate(obs=obs, info=info))

                # ---- 7. CompletionChecker: gate -1 commit ---------------
                # iter 3: also pass `forward_streak` so verify() can
                # apply the motion-required floor on planner-driven
                # commits for non-orient subtasks (parent iter_002's
                # Match-only gate was bypassed at forward_streak=0).
                commit_done = completion.verify(
                    actions=planner_actions,
                    planner_match=planner_match,
                    planner_match_history=planner_match_history,
                    instruction=instruction,
                    forward_streak=forward_streak,
                )

                # iter 4: independent binary REACH-VLM cross-verifier on
                # planner-driven `-1`. If verify() honored a planner -1
                # on a NON-orient subtask, route the claim through an
                # independent strict-binary VLM call (same prompt as
                # the proactive REACH check). On Match=0, REVOKE the
                # commit — the planner's surviving non-`-1` actions
                # (typically forwards) execute instead, and the fallback
                # proactive_commit_check path remains live on subsequent
                # steps. The cross-check defers (returns True) when it
                # cannot run: no hint, no VLM, orient subtask, missing
                # RGB, or per-subtask cap reached — preserving the
                # iter_003 control flow exactly when grounding is
                # unavailable.
                #
                # iter 9 REWIRE: the cross-check is now `ArrivalGate`'s, not
                # `CompletionChecker`'s. Same defer-on-unavailable contract
                # (it may only tighten the gate when it has a confident
                # answer), same free segmentation veto, same per-leg cap — but
                # the confident answer is now a CONJUNCTION of two independent
                # calls instead of one recognizer's verdict. 5 of the 17 fatal
                # commits in the parent's run came through this path.
                if commit_done:
                    xcheck_ok = arrival.allow_planner_commit(
                        obs=obs,
                        hint_image=hint_for(cur_idx),
                        vlm_clients=vlm_clients,
                        instruction=instruction,
                        cur_idx=cur_idx,
                        forward_streak=forward_streak,
                    )
                    if not xcheck_ok:
                        commit_done = False
                        notes_parts.append("XCHECK_BLOCK")

                # iter 1: proactive REACH check. Only consults the VLM when
                # the planner did NOT propose -1, we've executed
                # K_REACH_FORWARD_STREAK consecutive forwards within the
                # current subtask, and a hint image exists. The per-subtask
                # firing cap inside CompletionChecker bounds cost.
                # iter 9: the injection is `ArrivalGate.proactive_commit`, on
                # the parent's schedule verbatim (streak floor, interval, cap
                # — all imported from CompletionChecker, none retuned).
                #
                # The iter-6 `allow_forward_commit` gate is REMOVED from this
                # site and the FINISH leg is live again. That gate was a
                # workaround for a criterion with no stopping power ("clearly
                # visible and prominent" is true of a destination facade from
                # tens of km down a straight street), and blocking the path
                # without fixing the question just moved the 7 fatal finishes
                # onto the face-scan path, where they became 9. The question
                # now carries an explicit standing-position conjunct
                # (`_FRONTAGE_SYS` on a FINISH leg), and `commit_pose_prefix`
                # below still supplies the yaw the leg is graded on — so this
                # is the only remaining FINISH commit route, and it is the one
                # that has a distance predicate in it. The method is left in
                # `ViewpointController` (unreferenced) so a future iter can
                # A/B the block against the conjunction.
                if not commit_done:
                    if arrival.proactive_commit(
                        obs=obs,
                        hint_image=hint_for(cur_idx),
                        vlm_clients=vlm_clients,
                        instruction=instruction,
                        forward_streak=forward_streak,
                        cur_idx=cur_idx,
                    ):
                        commit_done = True
                        # Label which channel raised the commit so the next
                        # diagnosis can attribute per-leg outcomes without
                        # substring-matching system prompts out of the trace
                        # (the parent's §2 had to recover 595 gate calls that
                        # way). Pure telemetry — nothing branches on it.
                        notes_parts.append(
                            "GEO_REACH" if geo_read.get("event")
                            else "REACH_COMMIT")
                        # Override planner: emit a pure -1 burst.
                        planner_actions = [-1]

                # iter 9 DELETED: the iter-6 face-before-look edge.
                # `ViewpointController.request_face_scan` -> in-place yaw ->
                # `CompletionChecker.scan_reach` -> +/- commit is removed from
                # the graph. It was measured end-to-end in the parent run and
                # it is the single worst edge in this harness:
                #   ROAD   128 faced reach calls, 0 matches (0/20 in-window,
                #          against the forward view's 8/34) -> 256 substeps of
                #          pure cost, 10 % of all ROAD substeps and 12-18 % of
                #          the budget on tight legs; 5 legs died at
                #          used == budget EXACTLY with 4-8 of those substeps
                #          spent on scans that returned 0.
                #   FINISH 35 calls, 9 matches, 9/9 fatal premature commits at
                #          8 666-58 789 cm (threshold 2 000) — every one of the
                #          nine Reasons asserts VISIBILITY, not arrival, which
                #          is what the criterion literally asks for. Five of
                #          them threw away an episode that had banked 3 legs
                #          after spending 8-18 of a 173-333 substep budget.
                # The premise it was built on ("the hint was captured facing
                # the building, so a match needs the building head-on") is
                # refuted by its own numbers: turning to face made recognition
                # strictly worse. Those 256 substeps go back to walking, and
                # the FINISH commit route moves to the conjunction gate above.
                # `scan_reach` / `request_face_scan` / `face_code` /
                # `restore_code` are left in their modules, unreferenced, so a
                # future iter can re-enable the edge behind a better question.
                if terminated:
                    break

                # Materialize the action sequence to execute. -1 always wins
                # if commit_done; otherwise filter through `allowed`.
                if commit_done:
                    # Drop everything after the first -1 (subtask boundary).
                    # iter 6: this is the ONE place a `-1` is materialised, so
                    # it is the one place the terminal pose is composed —
                    # `commit_pose_prefix` prepends the quarter-turn toward the
                    # named side on a FINISH leg that is about to commit from
                    # the travel heading ("...on your right, stop and face that
                    # building" is graded on yaw, and no code path ever turned).
                    # It returns [] on every other leg and on a commit that
                    # already arrived from the faced pose, so the rotation can
                    # never be applied twice.
                    trimmed: List[int] = []
                    for a in planner_actions:
                        if a == -1:
                            trimmed.extend(
                                viewpoint.commit_pose_prefix(instruction))
                            trimmed.append(-1)
                            break
                        if a in allowed:
                            trimmed.append(a)
                    chosen_actions = trimmed
                else:
                    chosen_actions = [a for a in planner_actions if a in allowed]

                if not chosen_actions:
                    # iter 5: NO_AFFORDANCE fallback. The parent
                    # (s0_naiveR_004) terminated ep_20_0 and ep_20_4
                    # after only 3 steps each because the planner
                    # stochastically emitted a bare `[-1]` on the
                    # first frame of subtask 1; verify() correctly
                    # rejected it (forward_streak=0 < K_MIN=3), the
                    # affordance filter stripped `-1`, and the harness
                    # broke. A single anomalous planner output should
                    # not end the whole episode. When forward (action
                    # 0) is in the allowed motion set, inject one
                    # forward and let the next perceive+plan cycle
                    # work on the new frame. The stuck detector
                    # (forward_count > 3 AND delta moving_distance
                    # < 0.5) still bounds runaway forwards into a
                    # wall, so cost on a stubborn planner is at most
                    # ~6 extra cycles before STUCK fires. The
                    # original `break` is preserved as the fallback
                    # for a future affordance estimator that vetoes
                    # forward (defense in depth).
                    if 0 in allowed:
                        chosen_actions = [0]
                        notes_parts.append("EMPTY_FALLBACK")
                    else:
                        notes_parts.append("NO_AFFORDANCE")
                        break

            # ---- execute the burst ---------------------------------------
            for chosen_action in list(chosen_actions):
                action_history_list.append(chosen_action)
                actions_log.append(chosen_action)

                if chosen_action == -1:
                    pre_idx = _current_subtask_idx(env)
                    try:
                        obs, _, terminated, _, info = env.step(-1)
                    except Exception as e:
                        if _is_socket_dead(e):
                            notes_parts.append("SOCKET_DEAD")
                            terminated = True
                            break
                        raise
                    memory.log_step(action=-1, info=info)
                    memory.boundary()
                    post_idx = _current_subtask_idx(env)
                    if post_idx > pre_idx and pre_idx < n_subtasks:
                        subtask_passed[pre_idx] = True
                    # Reset per-subtask buffers.
                    action_history_list = []
                    chosen_actions = []
                    planner_match_history = []
                    forward_streak = 0
                    if terminated:
                        break
                else:
                    step_code = ACTION_TO_STEP.get(chosen_action)
                    if step_code is None:
                        continue
                    try:
                        obs, _, terminated, _, info = env.step(step_code)
                    except Exception as e:
                        if _is_socket_dead(e):
                            notes_parts.append("SOCKET_DEAD")
                            terminated = True
                            break
                        raise
                    memory.log_step(action=chosen_action, info=info)
                    # Baseline-leaning: count ALL translation (forward + strafes)
                    # toward the motion floor, not just forward; and don't let a
                    # reorientation turn wipe accrued progress — a turn mid-approach
                    # would otherwise reset the streak below K_MIN and block a
                    # legitimate short/strafe completion the planner tried to commit.
                    if chosen_action in (0, 3, 4):   # Move_forward / left / right
                        forward_streak += 1
                    if terminated:
                        break

                if len(actions_log) >= max_steps:
                    notes_parts.append("MAX_STEPS")
                    terminated = True
                    break
        except Exception as e:
            # Non-socket-dead exception inside the per-step body. Record
            # the type and a truncated message; preserve all state
            # gathered so far (subtask_passed, actions_log, llm_calls,
            # notes_parts) and exit the loop. The evaluator's outer
            # try/except only fires on exceptions that propagate OUT of
            # run_episode — by breaking here we keep our partial result.
            msg = str(e)[:80].replace("\n", " ")
            if _is_socket_dead(e):
                # Defense in depth: if an env.step socket-death somehow
                # slipped past the inner catches (e.g. a future code
                # path that calls env.step outside the burst executor),
                # still classify it as SOCKET_DEAD for the FAILURE_MODES
                # taxonomy rather than CRASH.
                notes_parts.append("SOCKET_DEAD")
            else:
                notes_parts.append(f"CRASH:{type(e).__name__}:{msg}")
            break

    # If the env grades the whole episode as success, all subtasks pass.
    success = False
    try:
        success = bool(info.get("success", False)) if isinstance(info, Mapping) else False
    except Exception:
        success = False
    if success and n_subtasks > 0:
        subtask_passed = [True] * n_subtasks

    # iter 9: the gate's own call ledger into the trace notes. Pure telemetry
    # for the diagnostician (identity 1/0/none, place 1/0/none, how often the
    # place call vetoed an identity match, how often the legacy fallback ran)
    # — it is written AFTER the rollout and never enters a decision.
    try:
        notes_parts.append(arrival.summary())
    except Exception:
        pass
    # iter 13: the geometric channel's own ledger (readings above/below the
    # operating point, rising edges raised, edges over the per-leg claim cap,
    # frames with no mask). Written AFTER the rollout; never a decision input.
    try:
        notes_parts.append(road.summary())
    except Exception:
        pass

    return EpisodeResult(
        success=success,
        subtask_success=subtask_passed,
        steps=len(actions_log),
        actions=actions_log,
        llm_calls=llm_calls,
        notes="; ".join(notes_parts),
    )
