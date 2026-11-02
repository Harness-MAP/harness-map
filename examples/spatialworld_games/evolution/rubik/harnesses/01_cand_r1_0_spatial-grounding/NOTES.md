# Rubik candidate — refuse the turns that undo the algorithm

## change:
Added `skills/macro_commitment.md`, routed by the existing sorted prompt composer between the role
prior and the legal-action rules, and a per-turn move ledger the skill can be executed against.
The skill replaces isolated move selection with a five-milestone goal stack, a relative-face
compiler over the intrinsic `F -> R -> B -> L` ring, a beginner-method macro vocabulary covering
insertion, orientation and final-layer permutation, and a program counter the agent recovers from
its own executed move list rather than from memory. The ledger is rendered into the prompt every
turn: the moves played in order, the reduced move word, the count of distinct configurations stood
on, whether the last action changed the picture, and which of the eighteen turns are refused and
why. A turn that provably returns the cube to a configuration already stood on is refused, proved
by reducing the played word with the only two identities a single face has — a turn followed by
its inverse, and four quarter turns of that face. A commutator such as `R U R' U'` is untouched by
that reduction, so an algorithm whose setup moves temporarily disturb solved-looking stickers runs
to its end. A refused pick does not stop the step: the harness plays the first turn the ledger
still allows and records both words on the Action, so the override rate is readable in the trace.
The one-call decision path, the parser and the legal action set are unchanged.

## axis: token-level implementation — control flow moved inside the components, graph held fixed

## module map

| file | role |
|---|---|
| `agent.py` | Unchanged evaluator-facing `build_agent(evaluator)` factory and sorted Markdown composer. |
| `AGENT.md` | Unchanged Rubik role and fixed face-centre facts. |
| `skills/macro_commitment.md` | Milestone ladder, relative-face macros, case gates, atomic program counter. |
| `rules.md` | Unchanged legal face and view move taxonomy and baseline guidance. |
| `memory.md` | The executed-move ledger the agent plans against, rendered every turn. |
| `tools/vlm.py` | One-call pristine VLM delegation plus the commit step. |
| `tools/action.py` | The public 18-action Rubik vocabulary and the rules over it. |
| `tools/frame.py` | Rendered-frame adapter and still-frame digest. |

## why:
Rubik algorithms need setup moves and make the cube look worse before it looks better, so a
reactive frame-by-frame policy abandons a useful sequence or changes goal mid-algorithm. The
parent does exactly that at the extreme: eight of twenty episodes played only view rotations and
never touched the cube, the median episode visited four distinct configurations in a hundred
steps, and 1137 of 1901 steps went on view turns, fourth identical turns and immediate undos.
Stable milestone invariants and an explicit program counter turn long-horizon intent into
executable single-action decisions, and the ledger is what makes the program counter recoverable.
The long version of the implementation, the rule justifications and the validation runs are in
`traces_notes/MECHANISM.md`.

## anti-cheat and validation

The skill uses only rendered stickers, fixed public centre conventions and the harness's own
action history. It does not access cube internals, scramble data, solver output, completion state
or score. Python compilation, harness import and composed prompt order were checked locally, and
the harness was driven over the real evaluator and the real renderer with a stubbed reply stream;
the frozen model was not called, as proposer scope requires.

## files:
- `skills/macro_commitment.md` — the milestone ladder, the macro vocabulary and the program counter.
- `memory.md` — the move ledger template, rendered into the prompt every turn.
- `tools/action.py` — the 18-action algebra plus the rule table and the word reduction.
- `tools/vlm.py` — the turn: inform, delegate, commit.
- `tools/frame.py` — the still-frame digest the ledger reports.
- `agent.py` — docstrings retargeted from maze to rubik; the factory and composer are unchanged.
- `traces_notes/MECHANISM.md` — the implementation record; outside the agent's prompt.
