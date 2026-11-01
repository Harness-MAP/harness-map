"""Frame access and the two pixel readings this harness trusts: the frame's IDENTITY and the cursor.

The per-step callback hands over a `FrameData` whose `.image` is already the (H, W, 3) RGB array the
model is shown, so nothing here renders anything new; `capture_game_screenshot` is the evaluator's own
published helper and is used only when that array is missing.

TWO READINGS, each with a stated precondition, because the 3-D viewport is self-occluding:

  * `digest_of` — an md5 over the viewport columns only (the static panel strip is excluded so the
    digest tracks the SCENE, not the level). The frame is a pure function of (cursor, placed set), so a
    digest that HELD across a move press is a one-bit proof that the press was clamped. The converse is
    NOT available: a newly placed cube can be fully occluded, so a changed digest is not required of a
    real placement. The ledger therefore books placements from its own toggles and uses the digest only
    to audit MOVES.
  * `cursor_from_ink` — the highlighted cell, inverted from the highlight ink. The highlight is painted
    in `HILITE` and the sidebar legend uses the same colour, so the mask is restricted to the viewport
    (`x >= VIEW_X0`). Over the 216 poses of an empty grid the ink is one solid footprint of a FIXED
    area whose centroid is an EXACT affine image of the cell — so inverting the centroid is sound
    EXACTLY WHILE THE FOOTPRINT IS WHOLE. Once cubes stand in front of the cursor the ink is clipped,
    the centroid walks off the lattice, and the inversion becomes a guess. So this returns a cell only
    when the ink area matches the whole footprint and the inverse lands within `SNAP` of a lattice
    point; otherwise it returns None and the dead-reckoned ledger stands unchallenged.
"""
import base64
import hashlib
import io

HILITE = (100, 200, 250)       # the cursor highlight fill
INK_SLACK = 10                 # L1 tolerance on the highlight colour
VIEW_X0 = 300                  # the 3-D viewport starts here; left of it is the panel strip
O_COL, O_ROW = 643.101, 278.236          # highlight centroid of cell (0, 0, 0), in (col, row)
X_COL, X_ROW = 40.0, 20.0                # the +X centroid step
Y_COL, Y_ROW = -34.0, 17.0               # the +Y centroid step
Z_COL, Z_ROW = 0.0, -40.0                # the +Z centroid step
WHOLE = 1024                   # footprint area in pixels when nothing occludes the cursor
AREA_SLACK = 24
SNAP = 2.0                      # how close the inverse must land to a lattice point; the closest two
                               # cells of this grid are 9 px apart in L1, so 2 px is unambiguous
GRID = 6
# Module-level one-liners: the pixel arithmetic, written once, out of the way of the readable code.
_SEATS = [(x, y, z) for x in range(GRID) for y in range(GRID) for z in range(GRID)]
_GAP = lambda view: (abs(view[:, :, 0] - HILITE[0]) + abs(view[:, :, 1] - HILITE[1])
                     + abs(view[:, :, 2] - HILITE[2]))
_OFF = lambda seat, col, row: abs(seat[0] - col) + abs(seat[1] - row)


def capture_array(evaluator, frame_data):
    """The frame as an (H, W, 3) array, or None if neither channel yields one."""
    img = getattr(frame_data, "image", None)
    if getattr(img, "ndim", 0) == 3:
        return img
    try:
        from PIL import Image
        import numpy as np
        raw = evaluator.capture_game_screenshot()
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB"))
    except Exception:                               # noqa: BLE001 — a missing frame is never fatal
        return None


def digest_of(image):
    """md5 of the 3-D viewport. Equal digests across a move press prove the press was clamped."""
    if image is None:
        return ""
    try:
        return hashlib.md5(image[:, VIEW_X0:].tobytes()).hexdigest()
    except Exception:                               # noqa: BLE001
        return ""


def highlight_mask(image):
    """(area, mean_col, mean_row) of the viewport highlight ink, or None when there is none."""
    if image is None:
        return None
    try:
        import numpy as np
        view = image[:, VIEW_X0:, :].astype("int16")
        rows, cols = np.nonzero(_GAP(view) <= INK_SLACK)
    except Exception:                               # noqa: BLE001
        return None
    area = int(rows.size)
    if area < 2:
        return None                                  # a stray pixel or two is not a footprint
    return (area, float(cols.mean()) + VIEW_X0, float(rows.mean()))


def lattice_ink(cell):
    """Where the highlight centroid lands for `cell` — the forward map, stated in one convention."""
    x, y, z = cell
    return (O_COL + x * X_COL + y * Y_COL + z * Z_COL,
            O_ROW + x * X_ROW + y * Y_ROW + z * Z_ROW)


def cursor_from_ink(image):
    """The highlighted cell, or None when the ink is clipped or the centroid misses the lattice.

    The three centroid steps are integer vectors in a 2-D picture, so the forward map is not injective
    over the integers in general — its kernel direction is (17, 20, 17), which leaves the 6-box, so on
    THIS grid every cell has its own centroid and the inverse is a nearest-lattice-point lookup. The
    lookup is required to be UNAMBIGUOUS: the best cell must sit within `SNAP` pixels and the runner-up
    must be further than `SNAP` away, otherwise this refuses and the dead-reckoned ledger stands.
    """
    ink = highlight_mask(image)
    if ink is None:
        return None
    area, col, row = ink
    if abs(area - WHOLE) > AREA_SLACK:
        return None                                  # a clipped footprint moves the centroid
    rank = sorted((_OFF(lattice_ink(seat), col, row), seat) for seat in _SEATS)
    near, cell = rank[0]
    if near > SNAP:
        return None
    if len(rank) > 1 and rank[1][0] <= SNAP * 2:
        return None                                  # two cells could explain the ink: never guess
    return cell
