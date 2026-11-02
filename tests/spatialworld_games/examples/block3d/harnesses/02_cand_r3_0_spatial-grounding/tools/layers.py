"""THE PLANNER: the three panels FACTOR BY HEIGHT, so each layer is a minimum bipartite EDGE COVER.

This is the mechanism this candidate exists to test. The parent chased the target as one 216-cell
search; the hull-first harnesses chase it as one 3-D set and descend. Neither uses the structure the
projections actually have:

    for a fixed height z, FRONT fixes EXACTLY which y's that layer must light
                          SIDE  fixes EXACTLY which x's that layer must light
                          TOP   is the ONLY coupling between layers

So layer z, read on its own, is a bipartite graph: columns `xs` on one side, rows `ys` on the other, an
edge for every (x, y) TOP square the layer is allowed to use. "Light every row and every column" is
precisely an EDGE COVER of that graph, and the smallest one is |xs| + |ys| - (maximum matching) — a
Kuhn augmenting-path computation, exact and cheap at this size. The union over layers settles FRONT and
SIDE completely and leaves one residue: a TOP square no layer happened to use. That debt is settled at
the lowest admissible height, and the result is a MINIMAL structure consistent with all three panels.

Why minimal matters: the admitted family is upward-closed inside the hull and the engine re-tests set
equality after every executed action, so a monotone build order starting from a minimal member makes
every prefix a distinct candidate the engine checks for free. Starting from the hull instead tests one
hypothesis — the largest — and has to shrink to find any other.

`anchored` names the cells that no other cell could explain; they are in every admitted structure, so
pressing them first costs nothing and settles the most debt per press. Nothing here reads engine state:
the only input is the decoded strip.
"""
from collections import Counter

from tools import panels

GRID = 6
# Module-level one-liners: the two projections the planner reads its mates and its tally through.
_MATES = lambda pairs, half, v: [pair[1 - half] for pair in pairs if pair[half] == v]
_TALLY = lambda cells: Counter(sq for cell in cells for sq in panels.shadows_of(cell))


def span_at(masks, z):
    """What layer z owes on its own: (columns from SIDE, rows from FRONT)."""
    xs = sorted({pair[0] for pair in masks["side"] if pair[1] == z})
    ys = sorted({pair[0] for pair in masks["front"] if pair[1] == z})
    return (xs, ys)


def pairs_at(masks, z):
    """The (x, y) cells layer z is ALLOWED to use: TOP squares inside the layer's own spans."""
    xs, ys = span_at(masks, z)
    wide = set(xs)
    deep = set(ys)
    return sorted((x, y) for (x, y) in masks["top"] if x in wide and y in deep)


def pairing(xs, ys, pairs):
    """A MAXIMUM matching of columns to rows by augmenting paths (Kuhn), as {row: column}.

    Each column is offered its admitted rows in turn; a row already taken yields only if the column
    holding it can be re-seated along an alternating path. `seen` is per-column, which is what keeps
    one augmentation from revisiting a row and makes the search terminate.
    """
    if not pairs:
        return {}
    near = {x: [] for x in xs}
    for (x, y) in pairs:
        if x in near:
            near[x].append(y)
    taken = {}

    def widen(x, seen):
        for y in near.get(x, []):
            if y not in seen:
                seen.add(y)
                held = taken.get(y)
                if held is None or widen(held, seen):
                    taken[y] = x
                    return True
        return False

    for x in xs:
        widen(x, set())
    return taken


def cover_at(masks, z):
    """The SMALLEST cell set that lights every row and column layer z owes — a minimum edge cover.

    Take a maximum matching, then hand every still-unlit row or column one of its admitted edges.
    That is optimal whenever no vertex is isolated; an isolated row or column means the three panels
    disagree about this layer, and the honest answer is None — stand down rather than invent a cell.
    """
    xs, ys = span_at(masks, z)
    if len(xs) + len(ys) < 2:
        return []
    pairs = pairs_at(masks, z)
    taken = pairing(xs, ys, pairs)
    picked = {(x, y) for (y, x) in taken.items()}
    for half, span in ((0, xs), (1, ys)):
        for v in span:
            if v in {seat[half] for seat in picked}:
                continue
            mates = _MATES(pairs, half, v)
            if not mates:
                return None                          # an isolated row or column: the panels disagree
            picked.add((v, mates[0]) if half == 0 else (mates[0], v))
    return sorted((x, y, z) for (x, y) in picked)


def thin_shape(masks):
    """The SMALLEST structure the three panels admit: per-layer edge covers plus the TOP residue.

    The per-layer covers settle FRONT and SIDE exactly. What they can leave behind is a TOP square no
    layer happened to use, because TOP is the one constraint that does not live inside a single layer.
    Each such square is settled at its lowest admissible height — lowest because a low cell is both
    nearer the spawn plane and more likely to be reusable by a later repair.
    """
    shape = set()
    heights = sorted({pair[1] for pair in masks["front"]})
    if len(heights) < 1:
        return []
    for z in heights:
        layer = cover_at(masks, z)
        if layer is None:
            return None
        shape.update(layer)
    owed = sorted(set(masks["top"]) - {(cell[0], cell[1]) for cell in shape})
    while owed:
        x, y = owed.pop()
        seats = [z for z in heights if (y, z) in masks["front"] and (x, z) in masks["side"]]
        if not seats:
            return None
        shape.add((x, y, seats[0]))
    return sorted(shape)


def settles(cells, wanted):
    """Does this set light every square the panels want? The plan's own acceptance test."""
    cast = set()
    for cell in cells:
        cast.update(panels.shadows_of(cell))
    return wanted <= cast


def support(cell, cells):
    """How many of `cells` share this cell's THINNEST shadow — 1 means nothing else explains it.

    A graded forcedness. Support 1 is a sole explainer and therefore a member of every structure the
    panels admit; support 2 or more means each of the cell's three shadows is accounted for elsewhere,
    so it may be a phantom crossing of the projections rather than a real block.
    """
    mine = panels.shadows_of(cell)
    counts = []
    for i in range(3):
        counts.append(sum(1 for other in cells if panels.shadows_of(other)[i] == mine[i]))
    return min(counts) if counts else 0


def phantoms(cells, wanted):
    """The cells whose every shadow is accounted for elsewhere — the ones the panels do not require.

    Support 2 or more means each of the three squares this cell lights is also lit by another admitted
    cell, so it may be a crossing of the projections rather than a block. These are exactly the cells a
    tape should press LAST, and the ones a surplus peel is allowed to take back.
    """
    return sorted(cell for cell in cells if support(cell, cells) >= 2)


def anchored(cells, wanted):
    """The cells no other cell in `cells` could stand in for — sole explainers of a wanted square.

    A sole explainer belongs to EVERY admitted structure, so pressing it early can never be wasted
    work, and it settles more debt per press than a cell whose shadows are covered elsewhere.
    """
    tally = _TALLY(cells)
    sole = set()
    for cell in cells:
        for square in panels.shadows_of(cell):
            if square in wanted and tally[square] == 1:
                sole.add(cell)
    return sorted(sole)
