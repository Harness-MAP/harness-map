"""Executable CODE helpers the Block3D harness CALLS — not model-facing function tools.

One concern per module, and the import order is the dependency order:

    frame       the observation channel: the frame as an array, and nothing else
    action      the legal action set, the key table, the clamp algebra, the published spawn
    silhouette  the three target panels, the envelope they back-project to, and the censuses
    optics      the FITTED panel ruler and the CALIBRATED cursor projection
    route       the press tape (proof first, slice by slice, core before rim) and the free-leg rule
    book        self-derived episode state: cursor, placed set, refutation book, press grades
    vlm         the per-turn decision: ask the frozen model, measure the board, price the answer

Every one of them reads the picture the frozen agent is already shown and the actions the harness
itself chose. None of them touches `evaluator.game`, re-renders, or opens a file other than the
harness's own append-only trace.
"""
