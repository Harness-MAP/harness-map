"""SpatialMemory — honest odometry + step / subtask boundary bookkeeping.

H_0 stance: no privileged location reads. We log only what the deployment
channel honestly provides:
  - `info['moving_distance']` (cumulative L2 path length, cm) — proprio-
    ceptive scalar; NOT a displacement. Stuck detection consumes deltas.
  - the discrete action that produced the step (so a downstream proposer
    can mine action-history bursts without re-deriving them).
  - a monotone step counter and the last subtask-advance step (so
    `since_subtask_advance()` can answer "how many env.steps since the
    last `-1` emission" without leaking the grader's `current_subtask`).

Nothing else is logged. A proposer wanting yaw / cardinal deltas /
crosswalk-strip history can extend `log_step()` honestly — those signals
live in `info['agent']['agent_rotation']` (cardinal) and
`env.unwrapped._agent_rotation[1]` (analog yaw, per the probe), neither
of which is an oracle read.
"""
from __future__ import annotations
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Self-register so @dataclass below can resolve string annotations when
# imported via importlib.util.spec_from_file_location (validate_harness.py
# path). Matches the boilerplate at workspace/branches/react_h0/orchestrator.py
# lines 30-38.
sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))


@dataclass
class _StepRecord:
    step_idx: int
    action: int
    moving_distance_cm: float


@dataclass
class SpatialMemory:
    """Per-episode pose/step log.

    Public API (consumed by orchestrator.py + downstream proposers):
      - log_step(action, info)
      - boundary()
      - since_subtask_advance() -> int
      - summary() -> str   (a one-line text digest the perceiver can paste)
    """
    history: List[_StepRecord] = field(default_factory=list)
    _step_idx: int = 0
    _last_boundary_step: int = 0

    # ---- write side ---------------------------------------------------
    def log_step(self, action: int, info: Dict[str, Any]) -> None:
        """Record one env.step. `info` is the post-step info dict."""
        from collections.abc import Mapping as _M
        md = 0.0
        if isinstance(info, _M):
            raw = info.get("moving_distance", 0.0)
            try:
                md = float(raw) if raw is not None else 0.0
            except (TypeError, ValueError):
                md = 0.0
        self.history.append(_StepRecord(
            step_idx=self._step_idx,
            action=int(action) if action is not None else 0,
            moving_distance_cm=md,
        ))
        self._step_idx += 1

    def boundary(self) -> None:
        """Mark a subtask boundary (called when harness emits -1)."""
        self._last_boundary_step = self._step_idx

    # ---- read side ----------------------------------------------------
    def step_count(self) -> int:
        return self._step_idx

    def since_subtask_advance(self) -> int:
        return self._step_idx - self._last_boundary_step

    def last_moving_distance(self) -> Optional[float]:
        if not self.history:
            return None
        return self.history[-1].moving_distance_cm

    def forward_count_since_boundary(self) -> int:
        """Forward (action 0) calls in this subtask. Used by stuck detector."""
        boundary = self._last_boundary_step
        return sum(
            1 for r in self.history
            if r.step_idx >= boundary and r.action == 0
        )

    def last_actions_in_subtask(self, k: int = 2) -> List[int]:
        """Return the last `k` actions logged since the last `boundary()`.

        Used by ActivePerception to detect rotation-spin loops. Returns a
        list of length min(k, n_actions_in_subtask). All ints come from
        log_step's recorded action field.
        """
        if k <= 0 or not self.history:
            return []
        boundary = self._last_boundary_step
        acts = [int(r.action) for r in self.history if r.step_idx >= boundary]
        return acts[-k:]

    def moving_distance_delta(self, window: int = 1) -> float:
        """Cumulative-path delta over the last `window` steps. 0.0 if empty."""
        if len(self.history) < 2:
            return 0.0
        end = self.history[-1].moving_distance_cm
        start_idx = max(0, len(self.history) - 1 - window)
        start = self.history[start_idx].moving_distance_cm
        return float(end - start)

    # ---- blockage, read off odometry alone (iter 4) --------------------
    # The previous version of this section exposed `static_collision_delta`,
    # a cumulative building/object collision count sourced from
    # `info["agent_collision"]`. That read was never wired (log_step passed a
    # hard-coded 0) and it can never BE wired: `agent_collision` is a
    # FORBIDDEN field in rules.md, so the accessor was identically 0 and both
    # of its consumers in ActivePerception were unreachable by construction.
    # It is deleted rather than left looking live. What the harness actually
    # needs — "did that forward move me?" — is fully answerable from
    # `info['moving_distance']`, the sanctioned progress channel, and is now
    # a first-class memory primitive instead of a delta recomputed ad hoc at
    # each call site.

    def step_progress(self, back: int = 1) -> Optional[float]:
        """Odometry delta produced by the `back`-th most recent logged step.

        `back=1` is the last step. Returns None when the step has no
        predecessor to difference against (i.e. it is the first record).
        """
        n = len(self.history)
        if n < 2 or back < 1 or back > n - 1:
            return None
        return float(self.history[n - back].moving_distance_cm
                     - self.history[n - back - 1].moving_distance_cm)

    def forward_stall_run(self, threshold_cm: float = 1.0) -> int:
        """Trailing run of consecutive BLOCKED forwards in this subtask.

        Walks back from the most recent logged step, counting steps that are
        (a) inside the current subtask, (b) Move_Forward (action 0), and
        (c) produced an odometry delta below `threshold_cm` — i.e. the sim
        refused the translation (facade, kerb, static actor). Stops at the
        first non-forward action, the first forward that actually moved, or
        the subtask boundary. 0 means "the agent is moving, or is not
        currently trying to".

        This is the honest replacement for the collision channel: a forward
        that produces no displacement IS the observable event that the
        forbidden collision counter was standing in for.
        """
        n = len(self.history)
        run = 0
        for back in range(1, n):
            rec = self.history[n - back]
            if rec.step_idx < self._last_boundary_step:
                break
            if rec.action != 0:
                break
            delta = self.step_progress(back)
            if delta is None or delta >= threshold_cm:
                break
            run += 1
        return run

    def summary(self) -> str:
        """One-line text digest. Shape proposers paste this into prompts."""
        return (
            f"step={self._step_idx} "
            f"steps_in_subtask={self.since_subtask_advance()} "
            f"cum_path_cm={self.last_moving_distance() or 0.0:.0f}"
        )
