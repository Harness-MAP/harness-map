"""THE OBSERVATION CHANNEL, and the only one.

`frame_data.image` is already the rendered frame as a uint8 HWC array in DISPLAY orientation — the
same pixels the frozen agent is shown, needing no transpose and no second render. Reading it costs
nothing and never touches `evaluator.game`, so it is unaffected by the anti-cheat guard (which
wraps a `game` reference snapshotted before the adapter exists, making even the whitelisted
`.screen` unreadable).

`raster` is deliberately the ONLY way into the picture: every decoder in this harness takes the
array it returns, so there is exactly one place where the harness could possibly be reading
something it should not, and that place is two lines long. `capture` stays available for the
base64 PNG the evaluator itself sends to the model and is not used by any decoder here.
"""
from __future__ import annotations


def raster(frame_data):
    """The current frame as a uint8 HWC array — the harness's whole view of the world."""
    return frame_data.image


def capture(evaluator) -> str:
    """The current frame as a base64 PNG, via the evaluator's own screenshot method."""
    return evaluator.capture_game_screenshot()
