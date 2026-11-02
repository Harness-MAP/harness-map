"""agent.py — ENTRYPOINT. Field walker: code descends a charted potential field; the model is
consulted only where the field is flat, and then sees the grounded NEIGHBOUR TABLE.

Rungs, first answer wins (the rung name goes to `response.reasoning`):
  modal    an open prompt (`[yn]`, `--More--`, cursor on row 0, `In what direction?`) is answered in code:
           the pending kick bearing, else `more` (the one verb legal on every task that escapes it)
  down     standing on a charted `>`
  kick     a door the game called locked, adjacent and orthogonal: `kick`, then its bearing (<=4 tries)
  swing    a creature letter adjacent (not a peaceful whose `Really attack?` was already raised)
  descend  argmin of the field over legal, non-vetoed neighbours (tools/field.Chart)
  consult  one model call on the naive transcript + priors + NEIGHBOUR TABLE + open-move menu;
           reply projected whole-word onto the legal set (tools/action_io)
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prompting import build_messages                  # noqa: E402
from action_io import legal_actions, mentions, rank_open  # noqa: E402
from ledger import STEP, Ledger                        # noqa: E402
from field import Chart                                # noqa: E402

FOE = re.compile(r"[a-zA-Z&']")
KICK_TRIES = 4


def section(text, name):
    m = re.search(rf"(?m)^{name}:[ \t]*\n?", text)
    if not m:
        return ""
    rest = text[m.end():]
    k = re.search(r"(?m)^(message|language observation|cursor|map):", rest)
    return rest if name == "map" else (rest[:k.start()] if k else rest)


class FieldHarness:
    def __init__(self, client_factory, prompt_builder, config=None):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.cfg = dict(config or {})
        self.ledger = Ledger()
        self.chart = Chart()
        self.kick_dir = None

    def reset(self):
        self.prompt_builder.reset()
        self.ledger.reset()
        self.chart.reset()
        self.kick_dir = None

    # ---- responses --------------------------------------------------------------------------
    def reply(self, action, why):
        try:
            from balrog.client import LLMResponse
            return LLMResponse(model_id="harness", completion=action, stop_reason="stop",
                               input_tokens=0, output_tokens=0, reasoning=f"[{why}]")
        except Exception:
            from collections import namedtuple
            R = namedtuple("R", "model_id completion stop_reason input_tokens output_tokens reasoning")
            return R("harness", action, "stop", 0, 0, f"[{why}]")

    def consult(self, rows, legal, banned, menu):
        messages = build_messages(self.prompt_builder, rows)
        if messages and messages[-1].role == "user":
            if menu:
                messages[-1].content += "\nOpen moves, least trodden first: " + ", ".join(menu) + "."
            if banned:
                messages[-1].content += "\nClosed or deadly here: " + ", ".join(sorted(banned)) + "."
        out = copy.deepcopy(self.client.generate(messages))
        pick, rule = None, "consult"
        for name in mentions(out.completion, legal):
            if name not in banned:
                pick = name
                break
        if pick is None:
            ms = mentions(out.completion, legal)
            pick = menu[0] if menu else (ms[0] if ms else re.sub(r"[^a-zA-Z\s:]", "", out.completion or ""))
            rule = "consult:fallback"
        out = out._replace(completion=pick)
        if "reasoning" in getattr(out, "_fields", ()):
            out = out._replace(reasoning=f"[{rule}] {getattr(out, 'reasoning', '') or ''}".strip())
        return out

    def move(self, pos, d, why):
        self.chart.pending = (pos, d)
        self.ledger.commit(pos, d)
        return self.reply(d, why)

    # ---- the step ---------------------------------------------------------------------------
    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        try:
            text = obs["text"]["long_term_context"] if isinstance(obs, dict) else str(obs)
        except (KeyError, TypeError):
            text = str(obs)
        legal = legal_actions(getattr(self.prompt_builder, "system_prompt", "") or "")
        msg = section(text, "message").strip()
        cm = re.search(r"\(x=(\d+),\s*y=(\d+)\)", section(text, "cursor") or text)
        pos = (int(cm.group(1)), int(cm.group(2)) - 1) if cm else None
        modal = (("[yn" in msg) or msg.endswith("--More--") or "What do you want" in msg
                 or "In what direction" in msg or (pos is not None and pos[1] < 0))

        # modal rung: never read geometry off a prompt frame
        if modal:
            if "Really attack" in msg and self.chart.pending:
                self.chart.peaceful.add(self.chart.pending)
                self.chart.blocked.add(self.chart.pending)
                self.chart.pending = None
            if "In what direction" in msg and self.kick_dir and self.kick_dir in legal:
                d, self.kick_dir = self.kick_dir, None
                return self.reply(d, "modal:kick-bearing")
            if "more" in legal:
                return self.reply("more", "modal:more")
            return self.consult([], legal, set(), [])

        dl = re.search(r"Dlvl:(\d+)", text)
        rows_map = section(text, "map").split("\n")[:21]
        pos_l, _, adj = self.ledger.observe(text)
        self.chart.fold(dl.group(1) if dl else "?", pos, rows_map, adj, msg, modal)
        rows = self.ledger.rows(pos_l, rows_map, adj)
        dirs = [d for d in STEP if d in legal]
        if pos is None or not dirs:
            return self.consult(rows, legal, set(), [])

        ch = self.chart
        if (ch.terrain.get(pos) == ">" or "staircase down here" in text) and "down" in legal:
            return self.reply("down", "down")

        kinds = {r[0]: r[4] for r in rows}
        banned = {d for d in dirs if kinds.get(d) in ("deadly",)
                  or (pos, d) in ch.blocked or (pos, d) in ch.peaceful
                  or ch.terrain.get((pos[0] + STEP[d][0], pos[1] + STEP[d][1])) == "}"}

        if re.search(r"crashes open|door opens|shatter", msg):
            for d in STEP:
                ch.locked.pop((pos[0] + STEP[d][0], pos[1] + STEP[d][1]), None)

        # kick rung: a locked door adjacent on a cardinal
        if "kick" in legal:
            for d in ("north", "east", "south", "west"):
                c = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
                if c in ch.locked and d in legal:
                    ch.locked[c] += 1
                    if ch.locked[c] > KICK_TRIES:
                        ch.locked.pop(c)
                        ch.blocked.add((pos, d))
                        continue
                    self.kick_dir = d
                    return self.reply("kick", "kick")
        # swing rung
        for d in dirs:
            if d in banned:
                continue
            x, y = pos[0] + STEP[d][0], pos[1] + STEP[d][1]
            g = rows_map[y][x] if 0 <= y < len(rows_map) and 0 <= x < len(rows_map[y]) else " "
            if FOE.fullmatch(g):
                return self.move(pos, d, "swing")

        # descend rung
        field = ch.flood()
        open_dirs = [d for d in dirs if d not in banned and ch.edge(pos, d)]
        scored = []
        for i, d in enumerate(open_dirs):
            c = (pos[0] + STEP[d][0], pos[1] + STEP[d][1])
            if c in field:
                scored.append((field[c] + 0.25 * ch.visits.get(c, 0), len(d) > 5, i, d))
        if scored:
            return self.move(pos, min(scored)[3], "descend")

        # consult rung: the field is flat
        menu = [d for d in rank_open(rows, legal) if d not in banned]
        resp = self.consult(rows, legal, banned, menu)
        if resp.completion in STEP:
            ch.pending = (pos, resp.completion)
            self.ledger.commit(pos, resp.completion)
        return resp


def build_agent(client_factory, prompt_builder, config=None):
    return FieldHarness(client_factory, prompt_builder, config)
