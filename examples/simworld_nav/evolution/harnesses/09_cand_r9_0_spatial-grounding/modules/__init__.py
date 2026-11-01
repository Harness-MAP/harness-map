"""Naive H_0 module package — exports the modules wired by harness.py.

iter 6 added an 8th node, `ViewpointController` (POSE arbitration). iter 9
adds a 9th, `ArrivalGate`, which takes the `-1` LICENSE away from
`CompletionChecker` and makes it the conjunction of two independent
single-purpose VLM calls (image identity without the leg text AND a hint-free
"where am I standing" call). `CompletionChecker` keeps its recognizer as the
gate's degradation path, and `ViewpointController` keeps only its TURN
approach + FINISH commit pose — its face-before-look edge is deleted.
"""
from .spatial_memory import SpatialMemory  # noqa: F401
from .perceiver import Perceiver  # noqa: F401
from .planner import Planner  # noqa: F401
from .affordance_estimator import AffordanceEstimator  # noqa: F401
from .active_perception import ActivePerception  # noqa: F401
from .completion_checker import CompletionChecker  # noqa: F401
from .prompt_assembler import PromptAssembler  # noqa: F401
from .viewpoint_controller import ViewpointController  # noqa: F401
from .arrival_gate import ArrivalGate  # noqa: F401
