### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `move_backward`: Step backward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall. This is a legal action and it is often the cheapest one: it reaches the cell behind you for ONE action, where turning around and walking costs THREE.
- `turn_left`: Turn your view 90 degrees left. Travels nowhere.
- `turn_right`: Turn your view 90 degrees right. Travels nowhere.
- `climb_up`: Ascend to the higher floor. Works only while standing ON a stair tile. **This build draws no prompt to confirm that — the `underfoot` line of the working-memory block is the only thing that can tell you, and when it says you are on a stair, the precondition is met.**
- `climb_down`: Descend to the lower floor. Same condition, same source of truth.

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up`, and `climb_down`, do not include a `granularity` field.

### What an action costs:
Every action you name costs exactly ONE of your limited steps, whether it travels three cells or none. So distance is nearly free and **turning is the expensive thing.** Two consequences:
- When the way ahead is clear for several cells, use `large` or `medium`. A `small` step down an open corridor wastes the step you just spent.
- Never answer with a turn and then its opposite. That pair spends two steps and leaves you exactly where you started, facing exactly where you started, and it is how most lost episodes end.

### Decide from the observation only:
Choose your action from the frame you are shown and from the measured `working memory` block beside it. Nothing else is available to you, and nothing else is needed.
