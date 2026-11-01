"""The maze action / key algebra + the engine Action contract (reused, unchanged).

Each step returns an engine Action:
    Action(type="key_press", key=<"w"|"a"|"d">, metadata={..., "granularity", "repeat"})
Movement (`move_forward`) carries a granularity (small=1 / medium=2 / large=3 grid cells); turns do
not. The evaluator's PygameInputSource maps the key -> game.execute_mapped_action, so this table
must stay in step with rules.md's legal action set.

Both lookups are now total on the inputs they actually receive: the upstream JSON parser hands back
whatever the model typed, so `key_for` normalises first, and the granularity fall-back is a named
constant rather than an inline number.
"""

KEY_MAPPING = {"move_forward": "w", "turn_left": "a", "turn_right": "d"}
MOVEMENT_ACTIONS = {"move_forward"}
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}
_MIN_CELLS = GRANULARITY_STEPS["small"]       # the cautious default: one grid cell


def key_for(action_name: str):
    """The engine key for a legal maze action, or None if the name is not a legal action.
    A padded or capitalised label is the model's typing, not an illegal action."""
    name = str(action_name).strip().lower()
    if name in KEY_MAPPING:
        return KEY_MAPPING[name]
    return None


def granularity_steps(name: str) -> int:
    """Cells moved for a granularity label, defaulting to `_MIN_CELLS` (= `small`, one cell)."""
    return GRANULARITY_STEPS.get((name or "small").strip().lower(), _MIN_CELLS)
