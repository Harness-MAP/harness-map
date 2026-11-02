"""The maze action / key algebra + the engine Action contract (reused, unchanged).

Each step returns an engine Action:
    Action(type="key_press", key=<"w"|"a"|"d">, metadata={..., "granularity", "repeat"})
mapping a legal maze action to an engine key. Movement (`move_forward`) carries a granularity
(small=1 / medium=2 / large=3 grid cells); turns do not. The evaluator's PygameInputSource maps the
key -> game.execute_mapped_action, so this table must stay in step with rules.md's legal action set.
This is an evolvable lever: a proposer can add richer action shaping here without touching the engine.
"""

KEY_MAPPING = {"move_forward": "w", "turn_left": "a", "turn_right": "d"}
MOVEMENT_ACTIONS = {"move_forward"}
GRANULARITY_STEPS = {"small": 1, "medium": 2, "large": 3}


def key_for(action_name: str):
    """The engine key for a legal maze action, or None if the name is not a legal action."""
    return KEY_MAPPING.get(action_name)


def granularity_steps(name: str) -> int:
    """Cells moved for a granularity label (defaults to 1 = `small`)."""
    return GRANULARITY_STEPS.get((name or "small").strip().lower(), 1)
