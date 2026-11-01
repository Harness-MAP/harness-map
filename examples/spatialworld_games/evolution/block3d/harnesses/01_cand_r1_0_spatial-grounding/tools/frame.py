"""FRAME — the single pixel source for this harness.

The only observation channel the agent and the harness share is the rendered screen. The
evaluator already produces it for the model (`capture_game_screenshot`, a base64 PNG of the
pygame surface), so the harness grabs the very same bytes rather than opening a second rendering
path. Everything downstream (the lattice fit, the panel readout, the cursor readout) is a pure
function of the array returned here, which is what keeps every derived fact strictly observational.
"""

import base64
import io

RGB = "RGB"


def capture_frame(evaluator):
    """The frame the frozen model is shown, as an image, or None when it is unavailable.

    A missing or unreadable grab is reported as None rather than raised: every reader above this
    treats "no picture" as a refusal to decode and falls back to the pristine pass-through, which
    is the only safe behaviour when the harness cannot see what the model sees.
    """
    from PIL import Image

    try:
        blob = evaluator.capture_game_screenshot()
    except Exception:                                  # noqa: BLE001 — a grab failure is a refusal
        return None
    if not blob:
        return None
    return Image.open(io.BytesIO(base64.b64decode(blob))).convert(RGB)


def pixels_of(img):
    """(width, height, pixel-accessor) for an image, or None for a missing frame."""
    if img is None:
        return None
    wide, high = img.size
    return wide, high, img.load()
