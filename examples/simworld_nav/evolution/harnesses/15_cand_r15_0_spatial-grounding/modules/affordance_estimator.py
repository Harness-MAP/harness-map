"""AffordanceEstimator — gates which actions are physically meaningful
right now.

H_0 stance: a permissive stub. We always return the full motion set
[0, 1, 2, 3, 4] (forward, rot-left, rot-right, strafe-left, strafe-right).
We explicitly DO NOT include -1 here — the subtask-completion action is
the planner's call (see `CompletionChecker`), not an affordance.

Why a stub at H_0:
  - Honest affordance estimation from `obs['depth_map']` /
    `obs['object_mask']` requires resolving the depth-units question
    (per probe findings: uint8 [0,255] normalized — usable, but the
    conversion-to-cm policy is a proposer choice, not an H_0 fact).
  - A non-stub estimator that vetoes forwards into close depth would
    couple H_0 to safety semantics, which is exactly the layer a
    proposer should be free to redesign.

Proposers can extend `estimate()` to:
  - drop 0 when depth-median in central cone < threshold (forward block)
  - drop 1/2 when angular-velocity stuck-detector fires
  - add per-step affordance scores rather than a binary set
"""
from __future__ import annotations
import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List

sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))


@dataclass
class AffordanceEstimator:

    def estimate(self, obs: Dict[str, Any], info: Dict[str, Any]) -> List[int]:
        """Return the high-level action codes the agent is allowed to emit.

        H_0: full motion set (no -1; that is `CompletionChecker`'s decision).
        """
        return [0, 1, 2, 3, 4]
