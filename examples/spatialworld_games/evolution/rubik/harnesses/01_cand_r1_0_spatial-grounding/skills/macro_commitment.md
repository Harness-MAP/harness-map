# Committing to a macro, and reading the move ledger

Treat face letters as permanent cube coordinates, never as screen positions. A prime (`'`) means
anticlockwise; an unprimed letter means clockwise; `Q2` is the same `Q` turn issued twice.
Clockwise is judged while looking straight at the named face. A whole-cube view turn only moves
the camera and must never be substituted for a face turn inside an algorithm.

## The move ledger

A HARNESS STATE block is appended to these instructions on every turn after the first. It lists
the moves executed so far in order, the reduced move word, how many distinct configurations have
been stood on, whether the last action changed the picture at all, which of the eighteen actions
are refused this turn and why, and which are allowed. All of it is derived from the moves this
harness itself played, so it is working memory you do not have to reconstruct from the
conversation. Read it before you read the frame, and name an action the block lists as ALLOWED —
a refused name does not stop the step; the harness plays its own first allowed action instead.

If the reduced word is empty the cube is back at its scrambled start and nothing has been
accomplished. If it is short the scramble is nearly undone. A long word means the cube has been
walked away from the start rather than toward the solution.

## Compile relative faces

Choose a working front `X` from the observed case. Its right and left neighbours come from the
ring `F -> R -> B -> L -> F`:

- `X=F`: `XR=R`, `XL=L`
- `X=R`: `XR=B`, `XL=F`
- `X=B`: `XR=L`, `XL=R`
- `X=L`: `XR=F`, `XL=B`

U and D stay fixed. Before acting, substitute `X`, `XR` and `XL` into the chosen macro and
translate each symbol into the matching legal `rotate_<face>_<cw|acw>` action.

## Choosing the case

Only one face is in the picture, so a solid front proves nothing about the other five, and the
letter on the centre sticker names the face you are looking at however the cube has been turned.
If the frame shows the cube one quarter turn from solved, play that turn now rather than opening
a macro. Otherwise work the earliest milestone that is not visibly proved — D cross, then D
corners, then the middle edges, then U orientation, then U permutation — and pick the macro whose
precondition that case satisfies.
## Macro vocabulary

- Right trigger `RT(X)`: `XR U XR' U'`
- Left trigger `LT(X)`: `XL' U' XL U`
- Middle insert right `MIR(X)`: `U XR U' XR' U' X' U X`
- Middle insert left `MIL(X)`: `U' XL' U XL U X U' X'`
- Make the U cross `UC(X)`: `X XR U XR' U' X'`
- Orient U corners `SUN(X)`: `XR U XR' U XR U2 XR'`
- Position U corners `CP(X)`: `U XR U' XL' U XR' U' XL`
- Cycle U edges `UA(X)`: `XR U' XR U XR U XR U' XR' U' XR2`

Choose `X` by the visible case. For a D corner, place its target slot at front-right and repeat
`RT(X)` until that corner inserts. For a middle edge, align its side colour with its centre and
use `MIR` or `MIL` according to the destination. For `UC`, hold an L-shaped pair at U-back and
U-left, or hold a line horizontally. For `SUN`, place an already U-facing corner at U-front-left.
For `CP`, place a correctly positioned corner at U-front-right. For `UA`, place an already correct
U edge at U-back and predict which way the other three must cycle.

## Execute one symbol per turn

Fix the milestone, the macro and the compiled sequence once, then emit only the next symbol. On
the following turn recover your place from the executed move list: the symbols you have already
issued are listed in order, so the next one is the one after them. Intermediate frames inside a
macro normally look worse than the frame before — that is expected and is not a reason to abandon
the sequence, to insert a view turn, to repeat a symbol or to improvise.

The refusals protect exactly this. A turn that provably returns the cube to a configuration it has
already stood on is refused, and that is what an undo or a fourth identical turn does; a
commutator such as `R U R' U'` is never refused, because it genuinely reaches somewhere new. If
your next symbol comes back refused, your bookkeeping has drifted and the sequence you believe you
are in the middle of is not the one that was played: re-read the move list and recompile rather
than repeating the symbol.

If the move list shows a face turn you did not intend, undo only the unmatched tail of the macro,
in exact reverse order, and then replan. Never undo a macro that completed merely because it
temporarily disturbed stickers that looked correct.
