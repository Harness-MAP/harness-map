"""Naive H_0 module package — exports the modules wired by agent.py.

iter 6 added an 8th node, `ViewpointController` (POSE arbitration). iter 9
adds a 9th, `ArrivalGate`, which takes the `-1` LICENSE away from
`CompletionChecker` and makes it the conjunction of two independent
single-purpose VLM calls (image identity without the leg text AND a hint-free
"where am I standing" call). `CompletionChecker` keeps its recognizer as the
gate's degradation path, and `ViewpointController` keeps only its TURN
approach + FINISH commit pose — its face-before-look edge is deleted.

iter 13 adds a 10th, `RoadGeometry`: a FREE, non-VLM POSITION channel read off
the roadway class of `obs['object_mask']`. It owns the "am I THERE?" conjunct
on intersection-stop ROAD legs (displacing `ArrivalGate._JUNCTION_SYS`, which
the parent run measured at 29 % in-window) and it KEYS the identity call's
schedule, so the gate's one good question is asked at the junction instead of
on a clock.
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
from .road_geometry import RoadGeometry  # noqa: F401
