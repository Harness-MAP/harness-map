"""Frame capture tool — reuses the evaluator's pygame-surface grab (no new rendering path)."""


def capture(evaluator) -> str:
    """The current maze frame as a base64 image, via the evaluator's engine screenshot."""
    return evaluator.capture_game_screenshot()
