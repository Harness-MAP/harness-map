"""Naive H_0 module package — exports the modules wired by agent.py.

iter 6 adds an 8th node, `ViewpointController` (POSE arbitration: approach
before a turn, face before a look, face before a FINISH commit). It sits
between ActivePerception and CompletionChecker in `run_episode`'s graph.
"""
from .spatial_memory import SpatialMemory  # noqa: F401
from .perceiver import Perceiver  # noqa: F401
from .planner import Planner  # noqa: F401
from .affordance_estimator import AffordanceEstimator  # noqa: F401
from .active_perception import ActivePerception  # noqa: F401
from .completion_checker import CompletionChecker  # noqa: F401
from .prompt_assembler import PromptAssembler  # noqa: F401
from .viewpoint_controller import ViewpointController  # noqa: F401
