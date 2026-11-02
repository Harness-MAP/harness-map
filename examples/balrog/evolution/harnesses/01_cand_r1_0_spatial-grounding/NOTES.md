# cand_r001_0_spatial-grounding

## axis:
Component graph / decision locus (code). The anchor is `prompt -> one model call -> regex -> env`.
This candidate adds a perception → spatial-memory → grounding → arbiter pipeline around the call,
and moves the final decision from the raw model reply to a code ARBITER.

## change:
New modules: `tools/perception.py` parses each text observation into message, cursor pos, a
row-aligned map grid and per-direction prose nouns ("lava adjacent east").
`tools/spatial_memory.py` keeps self-derived episode state: visit counts, a persistent seen-map, and
a refusal ledger keyed on (cell, dir). A move that leaves the position unchanged with a refusal
message or no message convicts that pair, never the glyph. The state resets on reset(), on a new
Dlvl, and on a system-prompt change (the agent is reused across tasks).
`tools/grounding.py` builds a per-direction move menu with vetoes (lethal lava/water nouns or `}`,
wall/stone/bars nouns, refused pairs, bare wall glyphs). It also runs a BFS over the remembered map
to a drawn `>`, or else to the nearest unvisited frontier square, whose first hop is the
`recommended` move. The menu is rendered into the user turn via `tools/prompting.build_messages(grounding=...)`.
`tools/action_io.py` now parses the legal set from the system prompt and snaps replies to a legal
action with whole-word matching. It answers open modal prompts (`--More--`, `[yn]`, "What do you
want to…") with `more`/`esc` WITHOUT a model call. `arbitrate()` overrides a vetoed reply with the
recommended move, and falls back to the route instead of BALROG's silent `north`.
`agent.py` orchestrates all of this and writes the deciding component into `response.reasoning`.
`skills/spatial.md` explains the grounding block to the model.

## why:
The diagnosis's binding constraint is a stateless policy: period-2 orbits (27% of steps), wall-bump
fixed points (14%), walking into lava that is named adjacent (4 deaths), and 134 steps coerced by
modal prompts or bad verbs. Each of these is fixed by code: visit memory, the refusal ledger, the
hazard veto, the modal handler and the arbiter. For coverage, the harness is structurally unlike the
anchor: 5 tool modules instead of 2, a stateful memory object, a planner, and a decision made
after the model call.
Replay smoke test over the anchor's 2874 recorded steps (stub model echoing the recorded actions):
model 1810, override-veto 919, modal 134, fallback-route 11. validate_harness: PASS.

## files:
agent.py (rewritten), tools/perception.py (new), tools/spatial_memory.py (new),
tools/grounding.py (new), tools/action_io.py (rewritten), tools/prompting.py (grounding slot),
skills/spatial.md (new), NOTES.md
