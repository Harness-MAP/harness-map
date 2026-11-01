"""Naive H_0 module package — exports the modules wired by orchestrator.py.

iter 11 adds an 8th node to the component graph: `RoadGeometry`, the harness's
first non-VLM arrival channel (see road_geometry.py).
"""
from .spatial_memory import SpatialMemory  # noqa: F401
from .road_geometry import RoadGeometry  # noqa: F401
from .perceiver import Perceiver  # noqa: F401
from .planner import Planner  # noqa: F401
from .affordance_estimator import AffordanceEstimator  # noqa: F401
from .active_perception import ActivePerception  # noqa: F401
from .completion_checker import CompletionChecker  # noqa: F401
from .prompt_assembler import PromptAssembler  # noqa: F401
