"""The turn: read the frame, inform the agent, make the one model call, commit the plan.

The anchor delegated the whole turn to `evaluator.baseline_step` (the pristine SpatialWorld
per-step fn that eval_runner stashes before it reassigns `evaluator.openai_ai_model`) and returned
whatever came back. This candidate still makes exactly ONE model call per step through that same
pristine path — same parser, same JSON contract, same legal action set, one engine action per step
— but the turn now has three parts around it:

  OBSERVE  — decode the face in the frame (`tools/frame.py`), file it against the camera pose, and
             either name the configuration outright or check it against the one the plan assumes.
  INFORM   — splice a short harness-authored brief into the one durable prompt slot: what the
             frame showed, how far from solved the configuration is, the next move, and the menu.
  COMMIT   — play the agent's own pick when the menu allows it, else the planned move; rewrite the
             engine metadata so the Action and the log agree; advance the ledger.

`baseline_step` EXECUTES what it parses, so the harness's choice cannot come afterwards: `arm`
swaps `evaluator.execute_action` for a sink that only records the attempt, `disarm` puts the real
one back, and `commit` plays the single move the harness stands behind.

The brief's wording lives in `memory.md`, read once at import, so the prose of the working memory
stays an ordinary prose lever and not a string buried in code.
"""
import json
import os
import time
from pathlib import Path

from tools import action, frame

HARNESS_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = HARNESS_DIR / "memory.md"
FALLBACK_TEMPLATE = ("HARNESS STATE (self-derived): step {{step}} · {{phase}} · {{cube}} · "
                     "NEXT {{next}} · ALLOWED {{allowed}}")
TEMPLATE = TEMPLATE_PATH.read_text() if TEMPLATE_PATH.exists() else FALLBACK_TEMPLATE
JOURNAL_DIR = HARNESS_DIR / "traces_journal"
JOURNAL_LIMIT = 400


# ─── the ledger, one per episode ────────────────────────────────────────────────────────────
def ledger(evaluator):
    """The book for this episode, cached on the evaluator — eval_runner gives each episode its
    own process, so it cannot leak between levels."""
    book = getattr(evaluator, "rubik_book", None)
    if book is None:
        book = action.new_book()
        setattr(evaluator, "rubik_book", book)
    return book


# ─── observe ────────────────────────────────────────────────────────────────────────────────
def observe(evaluator, book):
    """Decode the frame the agent is about to be shown and file it against the camera pose."""
    stickers, purity, square = frame.read_face(frame.capture(evaluator))
    action.record_reading(book, stickers, purity, square)
    book["seen"] = stickers
    return stickers


# ─── inform ─────────────────────────────────────────────────────────────────────────────────
def describe_frame(book):
    """One line about what the harness read off the frame the agent is looking at."""
    stickers = book.get("seen") or ""
    if not book["square"] or len(stickers) != 9:
        return "the cube is mid-turn in this frame, so nothing was read from it"
    rows = [stickers[0:3], stickers[3:6], stickers[6:9]]
    return "the face in view reads %s, top row first" % " / ".join(rows)


def describe_cube(book):
    """One line about how well the configuration is known, in moves rather than in states."""
    if book["state"] is not None:
        gap = len(book["plan"])
        if gap == 1:
            return "the cube is ONE quarter turn from solved"
        return "the cube is %d quarter turns from solved and the way home is proved" % gap
    live = len(book["live"])
    if live == 0:
        return "the configuration is not yet readable from what has been seen"
    if live == 1:
        return "the configuration has just been named"
    return "%d configurations still fit what has been seen, so the survey continues" % live


def describe_plan(book):
    """The remaining word, in the move letters the skill file explains."""
    if not book["plan"]:
        return "no solution word yet"
    return " ".join(book["plan"])


def describe_last(book):
    """Whether the previous turn's pick was played or substituted."""
    if not book["played"]:
        return "nothing played yet"
    last = book["journal"][-1] if book["journal"] else None
    if last is None:
        return "last played %s" % book["played"][-1]
    if last.get("override"):
        return "last turn you named %s, which was not on the menu, so the harness played %s" % (
            last.get("pick") or "nothing legal", last.get("played"))
    return "last turn you named %s and it was played" % last.get("played")


def fields(book):
    """The substitutions the brief template asks for."""
    return {
        "step": str(book["step"]),
        "phase": {"survey": "SURVEY — turning the camera to read the other faces",
                  "solve": "SOLVE — executing the proved solution",
                  "sweep": "SWEEP — perception unavailable, walking the nearest configurations"
                  }.get(book["phase"], book["phase"]),
        "frame": describe_frame(book),
        "cube": describe_cube(book),
        "plan": describe_plan(book),
        "next": book["want"] or "nothing",
        "allowed": ", ".join(book["menu"]) or book["want"],
        "last": describe_last(book),
    }


def fill(text, values):
    """Substitute {{name}} placeholders in the memory template."""
    for name in sorted(values):
        text = text.replace("{{%s}}" % name, values[name])
    return text


def brief(book):
    """The per-turn HARNESS STATE block, rendered from memory.md."""
    return "\n\n" + fill(TEMPLATE, fields(book)).strip() + "\n"


def prompt_slot(evaluator):
    """`openai_utils.build_conversation_history` copies previous_history[0] verbatim forever, so
    slot 0 is the only durable channel a harness owns."""
    return evaluator.conversation_history[0]["content"][0]


def inform(evaluator, book):
    """Re-splice the brief onto a pristine copy of slot 0, after the engine's own ACTIONS +
    RESPONSE FORMAT tail and never inside it. Step 0 has no history and nothing to report."""
    if not book["prior"]:
        book["prior"] = getattr(evaluator, "current_prompt", "") or ""
    if not evaluator.conversation_history:
        return False
    slot = prompt_slot(evaluator)
    if not book["slot"]:
        book["slot"] = slot["text"]
    slot["text"] = book["slot"] + brief(book)
    return True


# ─── muting the delegate's own execution ────────────────────────────────────────────────────
def arm(evaluator, book):
    """Swap the engine's executor for a sink, so the delegated parse cannot pre-empt the plan."""
    book["real"] = evaluator.execute_action

    def sink(candidate):
        book["sunk"].append(name_of(candidate))
        return True

    evaluator.execute_action = sink
    return True


def disarm(evaluator, book):
    """Put the real executor back."""
    evaluator.execute_action = book["real"]
    return True


# ─── commit ─────────────────────────────────────────────────────────────────────────────────
def metadata(candidate):
    """The Action's metadata dict, created if the parser left it None."""
    if candidate.metadata is None:
        candidate.metadata = {}
    return candidate.metadata


def name_of(candidate):
    """The action word the model's JSON asked for."""
    return metadata(candidate).get("function_called", "")


def rewrite(book, candidate, played, pick):
    """Point the Action at the move the harness commits to, all four engine fields together.

    The engine dispatches on metadata rather than on the name, and the model's own `face` field is
    the AXIS letter for a view rotation, not the engine's axis face — so key, face, direction and
    action_type must be set from SPEC as a set or the engine and the trace diverge.
    """
    spec = action.SPEC[played]
    candidate.key = spec[2]
    metadata(candidate).update({
        "function_called": played, "face": spec[0], "direction": spec[1], "action_type": spec[3],
        "model_pick": pick, "harness_override": played != pick, "harness_phase": book["phase"],
        "harness_menu": list(book["menu"]), "harness_plan": list(book["plan"]),
    })
    return candidate


def play(evaluator, candidate):
    """Execute the committed move. A face turn is animated and the adapter refuses both families
    while one is in flight; a capture renders, and rendering is what advances the animation, so
    the one honest cure for a refused move is to render and try once more."""
    if evaluator.execute_action(candidate):
        return True
    frame.render(evaluator)
    return bool(evaluator.execute_action(candidate))


def record(book, pick, played, ok):
    """One journal line per step: the pick, the move, the phase and why they differ."""
    line = {"step": book["step"], "phase": book["phase"], "pick": pick, "played": played,
            "override": played != pick, "ok": bool(ok), "menu": list(book["menu"]),
            "pose": list(book["pose"]), "square": book["square"],
            "purity": round(book["purity"], 3), "seen": book.get("seen", ""),
            "live": len(book["live"]), "plan": list(book["plan"]),
            "surveys": book["surveys"], "faults": book["faults"]}
    book["journal"].append(line)
    if len(book["journal"]) > JOURNAL_LIMIT:
        book["journal"].pop(0)
    try:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        target = JOURNAL_DIR / ("episode_%d.json" % os.getpid())
        target.write_text(json.dumps({"pid": os.getpid(), "at": int(time.time()),
                                      "overrides": book["overrides"],
                                      "honoured": book["honoured"],
                                      "steps": book["journal"]}, indent=1))
    except Exception:                                  # noqa: BLE001 — a journal must never fail a run
        pass
    return line


def commit(evaluator, book, proposed):
    """Play the agent's pick when the menu allows it, else the planned move; advance the ledger."""
    pick = name_of(proposed)
    played, _honoured = action.choose(book, pick)
    rewrite(book, proposed, played, pick)
    ok = play(evaluator, proposed)
    action.note_move(book, played, ok)
    record(book, pick, played, ok)
    return proposed


async def decide(evaluator, frame_data):
    """One decision step -> engine Action: observe, plan, inform, delegate, commit."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    book = ledger(evaluator)
    observe(evaluator, book)
    action.think(book)
    inform(evaluator, book)
    arm(evaluator, book)
    try:
        proposed = await step(frame_data)
    finally:
        disarm(evaluator, book)          # a raising delegate must not leave the engine muted
    return commit(evaluator, book, proposed)
