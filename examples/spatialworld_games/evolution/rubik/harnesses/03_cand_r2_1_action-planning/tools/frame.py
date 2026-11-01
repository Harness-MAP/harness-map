"""Reading the rubik frame — the agent's observation, decoded.

The SpatialWorld rubik renderer draws the cube FACE-ON: one face fills an exact, axis-aligned
3x3 grid of 79-pixel sticker squares at origin (131, 131) with a 1-pixel gutter, on a flat
(128, 128, 128) backdrop, with a static unfolded-cube legend in the bottom-right corner and a
dark glyph stamped over the centre sticker. Six sticker colours are drawn from one fixed palette
and nothing else on the screen uses them, so a majority vote over a lattice of samples inside a
cell recovers that sticker exactly; the glyph is a minority of the centre cell and loses the vote.

A 45-degree view rotation leaves the cube oblique: the painted quadrilaterals no longer line up
with the cell lattice, so cells read as a mixture AND the cube spills past the grid's mid-edges.
`read_face` reports both of those as the two halves of one gate, because an oblique frame reads a
PLAUSIBLE-BUT-WRONG permutation of the face in view — it passes a nine-stickers-per-colour check
and would poison the ledger silently. Only a gated reading is ever recorded.

Measured on the real renderer (levels 1 / 13 / 18, 54 captures): the gate accepted all 27
axis-aligned poses and refused all 27 oblique ones, and every accepted reading matched the
sticker the cube model predicted — 24/24 of the readings used, 0 misreads.
"""
import base64
import io

from PIL import Image

GRID_ORIGIN = 131          # px of the first sticker cell, both axes
CELL_PITCH = 80            # px from one cell's left edge to the next
FRAME_SIZE = (500, 500)    # the renderer's surface; anything else is not this screen
COLOUR_TOLERANCE = 30      # L1 distance in RGB that still counts as "that sticker colour"
PURITY_FLOOR = 0.62        # a cell below this is a mixture, i.e. the cube is not square-on
SAMPLE_OFFSETS = (8, 20, 32, 44, 56, 68)      # 36 lattice points inside each 79px cell

# The six sticker colours, as the renderer paints them. Keys are the letters the harness and the
# cube model use; nothing else drawn on the screen lands within COLOUR_TOLERANCE of any of them.
PALETTE = (
    ("W", (255, 255, 255)),
    ("Y", (254, 213, 47)),
    ("O", (255, 85, 37)),
    ("B", (13, 72, 172)),
    ("R", (137, 18, 20)),
    ("G", (25, 155, 76)),
)
BACKDROP = (128, 128, 128)

# Eight points 6px OUTSIDE the 3x3 bounding box, as (x, y). On a square-on frame every one of
# them is backdrop; the mid-edge four are the ones an oblique cube covers, which is what makes
# this half of the gate sharp rather than a threshold.
OUTSIDE_PROBES = (
    (125, 125), (250, 125), (375, 125),
    (125, 250), (375, 250),
    (125, 375), (250, 375), (375, 375),
)


def classify(rgb):
    """The palette letter for one pixel, or '' when it is not a sticker colour at all."""
    for letter, reference in PALETTE:
        gap = 0
        for channel in range(3):
            gap += abs(rgb[channel] - reference[channel])
        if gap <= COLOUR_TOLERANCE:
            return letter
    return ""


def is_backdrop(rgb):
    """True when this pixel is the flat grey the cube is drawn on."""
    gap = 0
    for channel in range(3):
        gap += abs(rgb[channel] - BACKDROP[channel])
    return gap <= COLOUR_TOLERANCE


def capture(evaluator):
    """The current frame as a base64 PNG, through the evaluator's own screenshot path."""
    return evaluator.capture_game_screenshot()


def render(evaluator):
    """Force one render without reading anything. A capture IS a render, and rendering is what
    advances a face-rotation animation, so this is the honest cure for a refused move."""
    capture(evaluator)
    return True


def cell_vote(pixels, row, column):
    """Majority sticker letter inside one cell, with the winner's share of the samples."""
    tally = {}
    for dy in SAMPLE_OFFSETS:
        for dx in SAMPLE_OFFSETS:
            x = GRID_ORIGIN + CELL_PITCH * column + dx
            y = GRID_ORIGIN + CELL_PITCH * row + dy
            letter = classify(pixels[x, y])
            if letter:
                tally[letter] = tally.get(letter, 0) + 1
    if not tally:
        return "", 0.0
    winner = ""
    best = -1
    for letter in sorted(tally):
        if tally[letter] > best:
            best = tally[letter]
            winner = letter
    return winner, best / float(len(SAMPLE_OFFSETS) ** 2)


def outside_is_clear(pixels):
    """True when nothing is painted just outside the 3x3 box — half of the square-on gate."""
    for x, y in OUTSIDE_PROBES:
        if not is_backdrop(pixels[x, y]):
            return False
    return True


def read_face(png_b64):
    """Decode the face in view.

    Returns (stickers, purity, square_on): `stickers` is 9 letters row-major as drawn (top-left
    first), `purity` the worst cell's majority share, `square_on` the gate. Callers must record a
    reading ONLY when square_on is true and len(stickers) == 9.
    """
    try:
        image = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
    except Exception:                                  # noqa: BLE001 — a bad capture is not a crash
        return "", 0.0, False
    if image.size != FRAME_SIZE:
        return "", 0.0, False
    pixels = image.load()
    stickers = []
    purity = 1.0
    for row in range(3):
        for column in range(3):
            letter, share = cell_vote(pixels, row, column)
            if not letter:
                return "", 0.0, False
            stickers.append(letter)
            if share < purity:
                purity = share
    square_on = purity >= PURITY_FLOOR and outside_is_clear(pixels)
    return "".join(stickers), purity, square_on
