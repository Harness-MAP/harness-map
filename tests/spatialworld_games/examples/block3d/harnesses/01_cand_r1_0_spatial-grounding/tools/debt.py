"""DEBT — the goal cell, re-derived every decision from what the panels still owe.

The mechanism this harness contributes is a RESIDUAL-PROJECTION-DEBT goal, not a compiled tape.
A tape is decided once, at the first clean read, and then walked; it cannot notice that a square
has already been explained by a block placed for another reason, that a stray toggle has made a
panel over-explained, or that the cheapest remaining cell has changed because the cursor moved.
Re-deriving the goal each turn from the CURRENT debt makes all three self-correcting, and costs
nothing: the debt is a set difference over at most three six-by-six panels.

The debt of a reading is the lit panel squares that the booked cells do not yet account for.
A candidate cell's WORTH is how many of those squares it would retire at once — a cell sitting
under three unexplained squares buys three, a cell under none buys nothing and is deferred even
if it is inside the admitted volume. Ties break the way the picture's own structure suggests:

  * a FORCED cell first — one that is the only admitted cell able to explain some lit square, so
    it belongs to every structure consistent with the panels and can never be a wasted placement;
  * then the lower layer, because the structure is grown upward from a minimal member and every
    prefix of that growth is itself a candidate the engine will test for free;
  * then the cell nearer the cursor, so the walk between commitments stays short.

A STRAY — a booked cell the panels exclude — outranks everything: the picture proves it wrong,
and lifting it costs one decision while leaving it costs the episode.
"""

NO_WORTH = 0
LONE = 1
PANELS = 3
_reach = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
_hub = lambda cells: (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))
_rim = lambda c, hub: round((c[0] - hub[0]) ** 2 + (c[1] - hub[1]) ** 2)
_marks = lambda c: ((c[0], c[1]), (c[1], c[2]), (c[0], c[2]))
_worth = lambda c, owed: sum(1 for i in range(PANELS) if _marks(c)[i] in owed[i])
_owners = lambda cells, i: {s: [c for c in cells if _marks(c)[i] == s]
                            for c in cells for s in [_marks(c)[i]]}
_lonely = lambda cells, i: {v[0] for v in _owners(cells, i).values() if len(v) == LONE}


def residual_debt(shadows, placed):
    """The lit squares of each panel that the booked cells do not yet explain."""
    cast = shadows.shadow_of(placed)
    return (shadows.xy - cast[0], shadows.yz - cast[1], shadows.xz - cast[2])


def forced_set(shadows):
    """Admitted cells that are the ONLY admitted cell able to explain some lit square.

    Such a cell is in every structure the panels allow, so committing to it first can never be
    wrong and it shortens the candidate family the engine has to be walked through.
    """
    out = set()
    panel = 0
    while panel < PANELS:
        out |= _lonely(shadows.admitted, panel)
        panel += 1
    return frozenset(out)


def rank_cells(shadows, placed, cursor, forced, lifted=frozenset()):
    """Admitted-but-unplaced cells, best first: forced, then still-owed, then layer, then rim,
    then the shortest walk from where the cursor actually is.

    A cell this harness has already LIFTED is never offered again. That is what makes the endgame
    monotone: once the admitted volume has been built and pruned, a pruned cell cannot come back,
    so the harness cannot fall into the place-remove-place orbit the parent spent whole episodes in.
    """
    owed = residual_debt(shadows, placed)
    hub = _hub(shadows.admitted)
    scored = [(cell not in forced, _worth(cell, owed) == NO_WORTH, cell[2],
               _rim(cell, hub), _reach(cell, cursor), cell)
              for cell in shadows.admitted if cell not in placed and cell not in lifted]
    scored.sort()
    return [row[-1] for row in scored]


def pick_cell(shadows, ledger, forced):
    """The cell this decision is working toward.

    In order: a cell the panels PROVE wrong and the book has placed (lift it); the best remaining
    debtor; and, once every admitted cell is filled and the structure was still not accepted, a
    cell whose removal keeps all three silhouettes intact — the only move left when the admitted
    volume turns out to be a strict over-approximation of the real structure.
    """
    stray = sorted(c for c in ledger.placed if c not in shadows.admitted)
    if stray:
        return stray[0]
    wanted = rank_cells(shadows, ledger.placed, ledger.cursor, forced, ledger.lifted)
    if wanted:
        return wanted[0]
    loose = [c for c in shadows.droppable(ledger.placed) if c not in ledger.lifted]
    if not loose:
        return None
    return min(loose, key=lambda c: (_reach(c, ledger.cursor), c))
