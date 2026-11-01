"""The perceive -> VLM -> COMMIT step.

The anchor delegated the whole turn to `evaluator.baseline_step` (the pristine SpatialWorld
per-step fn that eval_runner stashes before it reassigns `evaluator.openai_ai_model`) and returned
whatever came back. This harness still makes exactly ONE model call per step through that same
pristine path -- same parser, same JSON contract, same legal action set -- but it owns the two ends
of the turn:

  BEFORE the call it splices a harness-authored BRIEF into the one durable prompt slot. The brief
  is now a MENU: how far the cube is from where the episode started, how much of the near
  neighbourhood has been stood on, and the at-most-four quarter turns that still make progress.
  The parent printed fourteen of the eighteen names as "allowed" and the live run measured that
  channel inert (P(pick refused) 0.195 against a 0.178 chance baseline); the one channel that did
  work was a static, unconditional ban, so the invariants now live in `skills/` as prose and the
  per-turn text carries a short menu instead of a long list.
  AFTER the call it reads the proposed action word off the returned Action and asks
  `tools/action.py:choose` for the move to commit: the pick itself whenever it shortens the walk's
  distance to the nearest configuration of the start's neighbourhood not yet stood on, otherwise
  the first menu row -- with the refused pick REMEMBERED, so a turn that is merely illegal here is
  tried as soon as it becomes the best thing on the board rather than being discarded.

`baseline_step` EXECUTES what it parses, so the veto cannot come afterwards: `mute` swaps
`evaluator.execute_action` for a sink that only records the attempt, `unmute` puts the real one
back, and `commit` plays the one move this harness actually commits to. One decision, one engine
action, one model call -- the step accounting the evaluator scores is untouched.

WRITTEN AS PLAIN FUNCTIONS. The parent kept every branch in a module-level lambda; this file is
ordinary `def`s with explicit statements, which is the same turn in the staged anchor's idiom. The
behaviour that the live run verified (one engine action per step, the four metadata fields
rewritten together, the slot-0 splice landing after the engine's own tail) is preserved statement
for statement; only the shape of the code and the policy it consults have changed.

The brief's wording lives in `memory.md`, read at import, so the prose of the working memory stays
an ordinary prose lever and not a string buried in code.
"""
import json
from pathlib import Path

from tools import action
from tools.frame import fingerprint

HOME = Path(__file__).resolve().parent.parent
FALLBACK_TEMPLATE = ("HARNESS STATE (self-derived): step {{step}}; {{out}} quarter-turns from the "
                     "start; menu {{menu}}")
MENU_WIDTH = 4                 # how many candidate moves the brief prints (14 was measured inert)
TAIL_WIDTH = 6                 # how many executed words the "last turns" line echoes back


def load_template():
    """The brief's prose, from `memory.md`, with a one-line fallback if the file is missing."""
    path = HOME / "memory.md"
    if path.exists():
        return path.read_text()
    return FALLBACK_TEMPLATE


TEMPLATE = load_template()


# --- the ledger -------------------------------------------------------------------------------
# One dict per episode, cached on the evaluator. eval_runner runs one episode per process, so it
# cannot leak between levels. It holds only what this harness itself emitted.

def ledger(evaluator):
    """The per-episode ledger, created on first use and cached on the evaluator."""
    book = getattr(evaluator, "rubik_book", None)
    if book is None:
        book = action.new_book()
        evaluator.rubik_book = book
    return book


# --- the brief --------------------------------------------------------------------------------

def menu_line(row):
    """One menu row as a line of prose: the move, and the one fact that distinguishes it."""
    name, landing_depth, fresh = row
    if fresh and landing_depth == 1:
        return "  %s - a configuration ONE turn from the start that you have never stood on" % name
    if fresh:
        return "  %s - a configuration %d turns from the start, never stood on" % (name, landing_depth)
    return "  %s - %d turns from the start, stood on already, but it opens the next new one" % (
        name, landing_depth)


def menu_text(rows):
    """The at-most-MENU_WIDTH menu, best first."""
    if not rows:
        return "  (no move makes progress this turn - the harness will pick)"
    lines = []
    for row in rows[:MENU_WIDTH]:
        lines.append(menu_line(row))
    return "\n".join(lines)


def last_line(book):
    """The turns actually executed, most recent last -- working memory the agent would otherwise
    have to reconstruct from 29 replayed copies of its own answers."""
    if not book["played"]:
        return "nothing executed yet"
    return ", ".join(book["played"][-TAIL_WIDTH:])


def spent_line(book):
    """Whether the previous turn went the way the agent named it, in the agent's own terms."""
    if not book["last_pick"]:
        return "this is the first decision of the episode"
    if not book["last_code"]:
        return "the harness played exactly what you named (%s)" % book["last_played"]
    reason = action.REASONS.get(book["last_code"], "that turn made no progress")
    return "you named %s and the step was spent on %s instead: %s" % (
        book["last_pick"], book["last_played"], reason)


def fields(book, rows):
    """The brief's token values, all of them derived from this harness's own executed moves."""
    stood_one, total_one = action.swept(book, 1)
    stood_two, total_two = action.swept(book, 2)
    return {
        "step": str(book["step"]),
        "out": str(action.turns_from_start(book)),
        "shell1": "%d of %d" % (stood_one, total_one),
        "shell2": "%d of %d" % (stood_two, total_two),
        "last": last_line(book),
        "spent": spent_line(book),
        "menu": menu_text(rows),
    }


def fill(text, values):
    """Token replacement (not str.format: the template contains braces of its own)."""
    out = text
    for key in sorted(values):
        out = out.replace("{{%s}}" % key, values[key])
    return out


def brief(book, rows):
    """The HARNESS STATE block spliced into the durable prompt slot."""
    return "\n\n" + fill(TEMPLATE, fields(book, rows)).strip() + "\n"


# --- the prompt slot --------------------------------------------------------------------------
# `openai_utils.build_conversation_history` copies `previous_history[0]` verbatim forever, so slot
# 0 is the only durable channel a harness owns. It is captured ONCE, pristine, on the first turn
# that has a history; after that the brief is re-spliced onto that pristine copy each turn, which
# appends it AFTER the engine's AVAILABLE ACTIONS + RESPONSE FORMAT tail and never inside it. On
# step 0 there is no history and nothing to report, so the composed prior is left exactly as it is.

def slot(evaluator):
    """The one durable prompt block: `conversation_history[0]`'s first content part."""
    return evaluator.conversation_history[0]["content"][0]


def remember(evaluator, book):
    """Keep the pristine composed prior, once, before anything is ever spliced onto it."""
    if not book["prior"]:
        book["prior"] = getattr(evaluator, "current_prompt", "")
    if not book["slot"] and evaluator.conversation_history:
        book["slot"] = slot(evaluator)["text"]


def inform(evaluator, book, rows):
    """Re-splice this turn's brief onto the pristine slot-0 text."""
    remember(evaluator, book)
    if evaluator.conversation_history:
        slot(evaluator)["text"] = book["slot"] + brief(book, rows)


# --- muting the delegate's own execution ------------------------------------------------------

def sink(book):
    """A stand-in for `execute_action` that records the delegate's attempt and plays nothing."""
    def note_attempt(action_object):
        book["sunk"].append(name_of(action_object))
        return True
    return note_attempt


def mute(evaluator, book):
    """Swap `execute_action` for the sink for the duration of the delegation."""
    book["real"] = evaluator.execute_action
    evaluator.execute_action = sink(book)


def unmute(evaluator, book):
    """Put the real `execute_action` back."""
    evaluator.execute_action = book["real"]


# --- committing -------------------------------------------------------------------------------

def metadata_of(action_object):
    """The Action's metadata dict, created if the parser left it None."""
    if action_object.metadata is None:
        action_object.metadata = {}
    return action_object.metadata


def name_of(action_object):
    """The action word the shipped parser read out of the model's JSON."""
    return metadata_of(action_object).get("function_called", "")


def report(book, played, pick, code, deferred):
    """The one-step record: what the model asked for, what the harness committed, and where the
    walk stood when it decided. It goes onto the Action AND into the journal, so the override rate
    and the search's coverage are READ from the run rather than reconstructed from it."""
    stood_one, _total_one = action.swept(book, 1)
    stood_two, _total_two = action.swept(book, 2)
    return {"step": book["step"], "function_called": played, "model_pick": pick,
            "harness_override": played != pick, "refused": code, "deferred_pick": deferred,
            "turns_from_start": action.turns_from_start(book),
            "shell1_stood": stood_one, "shell2_stood": stood_two}


def rewrite(action_object, played, record):
    """Point the Action at the move this harness commits to.

    `key` AND the four metadata fields have to move together: `execute_action` dispatches on
    metadata["action_type"] / ["face"] / ["direction"], never on the name. The live run corrected
    107 disagreements this way (the model answers `{"action":"rotate_cube_y_cw","face":"Y"}`,
    where `face` is the axis letter and not the engine's axis face)."""
    face, direction, key, kind = action.SPEC[played]
    action_object.key = key
    metadata = metadata_of(action_object)
    metadata.update(record)
    metadata["face"] = face
    metadata["direction"] = direction
    metadata["action_type"] = kind
    return action_object


def play(evaluator, action_object):
    """Execute the committed move once, with one honest retry.

    A face turn is animated and the adapter REFUSES both families while one is in flight,
    returning False. A capture renders, and rendering is what advances the animation -- so the one
    cure for a refused move is to capture and try again. It never fired in 1901 live steps."""
    if evaluator.execute_action(action_object):
        return True
    fingerprint(evaluator)
    return bool(evaluator.execute_action(action_object))


def decide_move(book, pick):
    """The policy call, isolated so a fault in it degrades to the model's own pick instead of
    killing the episode. A fault would be invisible otherwise; `policy_fault` reports it."""
    try:
        return action.choose(book, pick)
    except Exception as failure:                            # pragma: no cover - a guard, not flow
        book["fault"] = repr(failure)[:120]
        if pick in action.FACE_ROTATION_ACTIONS:
            return pick, "", False
        return action.FACE_MOVES[0], "policy-fault", False


def journal(evaluator, record):
    """Append the step record beside the episode's own trace.

    The shipped response record keeps five fields and the committed move is only in the session
    log, so without this the next reader has to replay the decision table to learn what the
    harness did. Nothing here is privileged -- it is the harness's own bookkeeping -- and the whole
    write is guarded: a journal that cannot be written must never cost an episode."""
    try:
        folder = getattr(evaluator, "log_dir", "")
        if not folder:
            return
        label = getattr(evaluator, "config_name", "episode")
        path = Path(folder) / ("rubik_harness_ledger_%s.jsonl" % label)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:                                       # pragma: no cover - a guard, not flow
        return


def commit(evaluator, book, action_object):
    """Choose, rewrite, play once, record the move in the ledger and in the journal."""
    pick = name_of(action_object)
    played, code, deferred = decide_move(book, pick)
    record = report(book, played, pick, code, deferred)
    rewrite(action_object, played, record)
    executed = play(evaluator, action_object)
    if code:
        book["overrides"] += 1
    if deferred:
        book["deferred"] += 1
    book["last_pick"] = pick or "nothing the parser could read"
    book["last_code"] = code
    book["last_played"] = played
    action.note_move(book, played, executed)
    record["executed"] = executed
    record["turns_after"] = action.turns_from_start(book)
    if book["fault"]:
        record["policy_fault"] = book["fault"]
        metadata_of(action_object)["policy_fault"] = book["fault"]
    journal(evaluator, record)
    return action_object


async def decide(evaluator, frame_data):
    """One decision step -> engine Action: inform, delegate, then commit through the router."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    book = ledger(evaluator)
    inform(evaluator, book, action.menu_rows(book))
    mute(evaluator, book)
    try:
        proposed = await step(frame_data)
    finally:
        unmute(evaluator, book)          # a raising delegate must not leave the engine muted
    return commit(evaluator, book, proposed)
