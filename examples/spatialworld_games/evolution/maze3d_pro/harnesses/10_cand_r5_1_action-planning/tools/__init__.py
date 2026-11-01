"""Executable CODE helpers the maze harness CALLS (not model-facing function tools).

tools/frame.py  the frame accessor and the exact one-frame decode (the corridor and both flanks)
tools/action.py the engine key table, the dead-reckoned chart, and the decision-cost route planner
tools/vlm.py    the per-step loop that threads them around the evaluator's own pristine VLM call
"""
