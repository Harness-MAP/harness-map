"""Planner — single VLM call: text-only reasoning over the perceiver's
description -> action list + match flag.

H_0 stance: keep the published baseline's two-stage Re-Act flow (perceive
then reason) but force the reason stage to be text-only — no images, no
re-perception. The planner outputs the harness's high-level action codes
(0..4 plus -1 for subtask-done). This isolates the "what to do" choice
from raw pixels, which (a) caps tokens, and (b) makes the failure mode
"description was wrong" vs "reasoning was wrong" cleanly separable for
downstream proposers.

Single-call contract:
  Input: vision_description, instruction, orientation, action_history_text, vlm_clients
  Output: {"actions": List[int], "summary": str, "match": int|None, "reason": str, "usage": {...}}
"""
from __future__ import annotations
import json
import re
import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

# Reuse the published baseline's reasoning prompt verbatim. This is the
# H_0 contract: same prompt as ReAct, but consumed via a single chat-
# completions call instead of through the `ReActAgent.act` wrapper.
_BASELINE_DIR = "/home/azureuser/workspace/AutoSpatial/simworld_gym/baseline/single"
if _BASELINE_DIR not in sys.path:
    sys.path.insert(0, _BASELINE_DIR)

from prompt_template import reasoning_template  # noqa: E402


_VALID_ACTIONS = {-1, 0, 1, 2, 3, 4}


def _finish_reason(resp: Any) -> Optional[str]:
    """`"length"` when the model ran out of output budget mid-answer."""
    try:
        return getattr(resp.choices[0], "finish_reason", None)
    except Exception:
        return None


def _coerce_actions(raw: Any) -> List[int]:
    """Sanitize a model-returned action list.

    Accepts: list of ints, list of strings, JSON-string of a list. Drops
    out-of-range values. Falls back to `[]` (no actions) which the
    harness treats as a planner-no-op and breaks out cleanly.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    out: List[int] = []
    if isinstance(raw, (list, tuple)):
        for x in raw:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if xi in _VALID_ACTIONS:
                out.append(xi)
    return out


def _extract_json_block(raw: str) -> Optional[dict]:
    if not raw:
        return None
    s = raw.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1]
        s = s.split("```", 1)[0]
    elif s.startswith("```"):
        s = s.split("```", 1)[1]
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # last resort: find the first {...} block.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


@dataclass
class Planner:

    def plan(
        self,
        vision_description: str,
        instruction: str,
        orientation: Any,
        action_history_text: str,
        vlm_clients: Dict[str, Any],
        memory_summary: str = "",
    ) -> Dict[str, Any]:
        empty = {
            "actions": [],
            "summary": "",
            "match": None,
            "reason": "",
            "usage": {"input": 0, "output": 0},
        }
        if not vlm_clients:
            return empty

        model_id = next(iter(vlm_clients.keys()))
        client = vlm_clients[model_id]

        sys_prompt = reasoning_template(full=False, strip=False)
        # Baseline parity: the published ReAct agent keeps a running History
        # Summary and feeds it back so the model UPDATES it each step. Without
        # a prior summary the planner is amnesiac — it "updates" a summary it
        # never saw. Thread the rolling summary in (empty on the first step).
        user_text = (
            f"Actions taken: {action_history_text}\n\n"
            f"History Summary (update this): {memory_summary or '(none yet)'}\n\n"
            f"Current Subtask: {instruction}\n\n"
            f"Current Orientation: {orientation}\n\n"
            f"Current Vision Description: {vision_description}\n\n"
            "Provide your reason, update the summary, and emit the next "
            "action list as strict JSON: "
            '{"Summary": "<text>", "Match": 0|1, "Actions": [int, ...]}.'
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text},
        ]

        def _ask(msgs: List[Dict[str, Any]], budget: int):
            return client.chat.completions.create(
                model=model_id,
                messages=msgs,
                temperature=0.2,
                max_tokens=budget,
            )

        try:
            resp = _ask(
                messages,
                # Baseline parity: the 30B-A3B-Thinking model burns output tokens
                # on its <think> block before answering; 384 truncated it mid-think
                # (-> NO_ACTION / degraded plans). Baseline floors at 8192.
                8192,
            )
        except Exception as e:
            return {**empty, "reason": f"plan_failed:{type(e).__name__}"}

        try:
            raw = resp.choices[0].message.content or ""
        except Exception:
            raw = ""
        obj = _extract_json_block(raw) or {}

        actions = _coerce_actions(obj.get("Actions") or obj.get("actions"))

        # iter 14 (diagnosis F6), bundled bug fix. 23 of this run's 1 650 plan
        # calls came back with `finish_reason == "length"`: the model talked
        # for the whole 8 192-token budget and was cut mid-JSON, so the parse
        # found no action list, the whole perceive+plan cycle was wasted, and
        # one episode was killed by TOO_MANY_ERRORS while holding three banked
        # legs. The budget is NOT lowered (that would truncate the healthy
        # calls too); instead a run-on is retried ONCE, with the format demand
        # restated and a short budget so the retry cannot run on in turn. Only
        # reached on a truncated call that parsed to nothing, so the 1 627
        # healthy calls are byte-identically unaffected.
        retried = False
        if not actions and _finish_reason(resp) == "length":
            retried = True
            try:
                resp2 = _ask(
                    messages + [{
                        "role": "user",
                        "content": (
                            "Your previous reply ran past the output limit "
                            "and was cut off. Reply with ONLY the JSON "
                            "object, on one line, nothing before or after "
                            "it. Keep \"Summary\" under 20 words and do not "
                            "explain your reasoning."),
                    }],
                    1024,
                )
                raw2 = resp2.choices[0].message.content or ""
            except Exception:
                resp2, raw2 = None, ""
            obj2 = _extract_json_block(raw2) or {}
            actions2 = _coerce_actions(obj2.get("Actions") or obj2.get("actions"))
            if actions2:
                resp, raw, obj, actions = resp2, raw2, obj2, actions2
        summary = obj.get("Summary") or obj.get("summary") or ""
        try:
            match_raw = obj.get("Match", obj.get("match"))
            match: Optional[int] = int(match_raw) if match_raw is not None else None
        except (TypeError, ValueError):
            match = None

        usage_obj = getattr(resp, "usage", None)
        try:
            in_tok = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        except Exception:
            in_tok, out_tok = 0, 0

        return {
            "actions": actions,
            "summary": summary if isinstance(summary, str) else "",
            "match": match,
            "reason": (("PLAN_RETRY_TRUNC " if retried else "") + raw)[:800],
            "usage": {"input": in_tok, "output": out_tok},
        }
