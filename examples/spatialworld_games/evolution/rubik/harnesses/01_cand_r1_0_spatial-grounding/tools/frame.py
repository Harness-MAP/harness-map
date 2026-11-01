"""Frame capture — reuses the evaluator's pygame-surface grab (no new rendering path) — plus the
frame FINGERPRINT the ledger uses to tell a move that changed the picture from one that did not.

A still cube renders byte-identically every time, so two equal consecutive digests are a sound
"that action moved nothing on screen" signal: a view turn the camera was already at, or a rotation
the adapter refused while an animation was still in progress.
"""
import hashlib


def capture(evaluator) -> str:
    """The current rubik frame as a base64 PNG, via the evaluator's engine screenshot."""
    return evaluator.capture_game_screenshot()


# Module-level so `capture` stays the one-line adapter it has always been.
fingerprint = lambda evaluator: hashlib.md5(capture(evaluator).encode()).hexdigest()[:12]
