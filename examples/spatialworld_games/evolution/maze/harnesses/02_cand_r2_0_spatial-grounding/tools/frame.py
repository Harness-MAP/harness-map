"""Frame capture tool — reuses the evaluator's pygame-surface grab (no new rendering path).

The ledger's per-step digest is taken from `frame_data.image`, the same rendered surface already
handed to the decision fn, so this helper stays the base64 encoding of the frame. It now answers
the empty string rather than None when the grab yields nothing.
"""


def capture(evaluator) -> str:
    """The current maze frame as a base64 image, via the evaluator's engine screenshot."""
    return evaluator.capture_game_screenshot() or ""
