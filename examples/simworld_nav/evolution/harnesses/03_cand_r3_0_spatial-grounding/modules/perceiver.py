"""Perceiver — single VLM call: RGB (+ hint) -> JSON description.

H_0 stance: the cheapest honest perception channel. One call per step,
text-only output, no chain-of-thought beyond the published
`perception_template`'s "first describe, then judge match" framing. The
returned `description` is the only structured-perception input the
planner sees in H_0; proposers can layer more (segmentation digest,
depth bands, detection lists) by extending here.

Single-call contract (matches the published baseline's perceive step):
  Input:  obs, vlm_clients, hint_image, instruction, orientation, memory_summary
  Output: {"description": str, "usage": {"input": int, "output": int}}
"""
from __future__ import annotations
import json
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from collections.abc import Mapping

import numpy as np

from .prompt_assembler import PromptAssembler

sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))


def _parse_description(raw: str) -> str:
    """Extract a `Description` field from the model's text output.

    Robust to (a) bare JSON, (b) markdown-fenced JSON, (c) free text. On
    any failure returns the raw string truncated to 800 chars — better
    than crashing the per-step loop.
    """
    if not raw:
        return ""
    s = raw.strip()
    # Strip ```json fences if present.
    if "```json" in s:
        s = s.split("```json", 1)[1]
        s = s.split("```", 1)[0]
    elif s.startswith("```"):
        s = s.split("```", 1)[1]
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            for key in ("Description", "description", "scene", "summary"):
                v = obj.get(key)
                if isinstance(v, str) and v:
                    return v
    except Exception:
        pass
    return raw.strip()[:800]


@dataclass
class Perceiver:
    assembler: PromptAssembler = field(default_factory=PromptAssembler)

    def perceive(
        self,
        obs: Dict[str, Any],
        vlm_clients: Dict[str, Any],
        hint_image: Optional[np.ndarray],
        instruction: str,
        orientation: Any,
        memory_summary: str,
    ) -> Dict[str, Any]:
        """Single VLM call returning `{description, usage}`.

        On any client failure returns an empty description with zero
        usage; planner is responsible for tolerating empty descriptions.
        """
        if not vlm_clients:
            return {"description": "", "usage": {"input": 0, "output": 0}}

        model_id = next(iter(vlm_clients.keys()))
        client = vlm_clients[model_id]

        messages = self.assembler.assemble_perceiver_messages(
            obs=obs,
            hint=hint_image,
            instruction=instruction,
            orientation=orientation,
            memory_summary=memory_summary,
        )

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.2,
                # Baseline parity: Thinking model needs room for its <think> block
                # before the description; 384 truncated it. Baseline floors at 8192.
                max_tokens=8192,
            )
        except Exception as e:
            return {
                "description": "",
                "usage": {"input": 0, "output": 0},
                "error": f"perceive_failed:{type(e).__name__}",
            }

        try:
            raw = resp.choices[0].message.content or ""
        except Exception:
            raw = ""
        description = _parse_description(raw)

        usage_obj = getattr(resp, "usage", None)
        try:
            in_tok = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        except Exception:
            in_tok, out_tok = 0, 0

        return {
            "description": description,
            "usage": {"input": in_tok, "output": out_tok},
        }
