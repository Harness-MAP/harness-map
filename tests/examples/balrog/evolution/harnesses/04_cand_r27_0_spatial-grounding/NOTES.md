## axis
Code: the component graph and where the decision is made. A potential field now makes the decision; the model breaks ties only when the field is flat.

## change
`tools/field.py` holds `parse_tty_frame`, which decodes the frame into a per-level chart and then calls `flood_field`. That function runs a reverse multi-source Dijkstra from a drawn `>`, or from every unvisited open-edge cell when there is none. Each cell's cost grows with how often it has been visited. `agent.py` has three parts: `build_agent`, the hub `GradientAgent`, and `choose_action`. The hub answers modal prompts, goes `down` when standing on `>`, and attacks an adjacent letter. Otherwise it steps to the neighbour with the lowest field value. A move that leaves the position unchanged closes that (square, direction) pair, and lava is always closed. The model is called only when no neighbour has a field value, for example in Boxoban. It is then offered a menu of open moves, least visited first, and its reply is matched against the legal set as whole words.

## why
The diagnosis puts most of the loss on 2-cycles and wall-bumping. A field whose cost rises with visits has no fixed-point 2-cycle. Lava and prompt freezes are handled before the field is used.

## files
agent.py, tools/field.py (new), tools/prompting.py and tools/action_io.py (removed), AGENT.md, rules.md
