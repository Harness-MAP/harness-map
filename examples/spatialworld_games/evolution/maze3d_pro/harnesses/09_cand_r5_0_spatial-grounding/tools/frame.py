"""Frame capture — reuses the evaluator's own pygame-surface grab, so there is no second rendering
path and nothing here can disagree with the frame the model is shown."""


def capture(evaluator) -> str:
    """The current maze frame as a base64 image, via the evaluator's engine screenshot."""
    return evaluator.capture_game_screenshot()
