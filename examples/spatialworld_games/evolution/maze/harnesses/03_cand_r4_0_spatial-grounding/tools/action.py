"""The maze action / key algebra + the engine Action contract.

Each step returns an engine `Action(type="key_press", key=<"w"|"a"|"d">, metadata={...})`. Movement
carries a granularity (small=1 / medium=2 / large=3 cells); turns do not. The evaluator's
PygameInputSource maps the key to the game's own mapped action, so this table must stay in step
with rules.md AND with the key table `tools/vlm.py` stamps onto a corrected action.
"""

KEY_MAPPING = {"move_forward": "w", "turn_left": "a", "turn_right": "d"}
MOVEMENT_ACTIONS = {"move_forward"}
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}


def key_for(action_name: str):
    """The engine key for a legal maze action, or None if the name is not a legal action."""
    label = (action_name or "").strip().lower()
    if label in KEY_MAPPING:
        return KEY_MAPPING[label]
    return None


def granularity_steps(name: str) -> int:
    """Cells moved for a granularity label (an unknown label falls back to `small`)."""
    label = (name or "small").strip().lower()
    if label in GRANULARITY_STEPS:
        return GRANULARITY_STEPS[label]
    return GRANULARITY_STEPS["small"]
