"""Frame access — the single place the harness turns an observation into pixels.

`evaluator.capture_game_screenshot()` is the evaluator's own grab (it re-renders the pygame
surface and base64-encodes it); `evaluator.baseline_step` calls it itself every step, so calling
it again here would only cost a second render of an identical scene. The decode path therefore
uses `frame_data.image`, which the engine's frame callback has already handed over as an
(H, W, C) uint8 array of exactly that surface — the frame the model is about to be shown and
nothing else.

`pixels` is the normalisation gate in front of the projection decode in tools/plan.py: it is the
one function that decides what counts as a readable frame, so the planner can treat its input as
a 3-channel array of the expected size and the harness falls back to the pristine step whenever
that is not what arrived.
"""
import numpy as np

EXPECTED = (600, 800)          # the renderer's own SCREEN_HEIGHT, SCREEN_WIDTH


def capture(evaluator) -> str:
    """The current maze frame as a base64 image, via the evaluator's engine screenshot. Kept for
    a caller that needs the encoded form; the decode path uses `pixels` instead."""
    return evaluator.capture_game_screenshot()


def pixels(frame_data):
    """The observation as an (H, W, 3) uint8 array, or None if it is not one.

    Returning None rather than raising is deliberate: tools/vlm.py reads it as "this frame cannot
    be measured", skips the whole mechanism for that step and delegates to the pristine VLM step,
    so an unexpected frame shape degrades the harness to the baseline for one decision instead of
    costing the episode."""
    img = getattr(frame_data, "image", None)
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[:2] != EXPECTED:
        return None
    return arr[:, :, :3]
