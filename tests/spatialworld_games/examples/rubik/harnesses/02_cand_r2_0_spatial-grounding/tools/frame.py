"""Frame capture -- reuses the evaluator's pygame-surface grab (no new rendering path) -- plus the
frame FINGERPRINT the committed step uses as a RENDER.

A face turn is animated and the adapter refuses a new turn while one is in flight. A capture
renders, and rendering is what advances the animation, so capturing between two attempts is the
one honest cure for a refused move. The digest itself is no longer reported to the agent: the live
run measured it structurally near-constant (every face turn and every view turn changes the
picture) and the retry path fired 0 times in 1901 steps, so the brief spends its words on the
search instead.
"""
import hashlib


def capture(evaluator) -> str:
    """The current rubik frame as a base64 PNG, via the evaluator's engine screenshot."""
    return evaluator.capture_game_screenshot()


def fingerprint(evaluator) -> str:
    """A short digest of the current frame. Calling it renders, which is the point."""
    return hashlib.md5(capture(evaluator).encode()).hexdigest()[:12]
