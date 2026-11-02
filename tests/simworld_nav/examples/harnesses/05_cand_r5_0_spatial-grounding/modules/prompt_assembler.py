"""PromptAssembler — converts (obs, hint, instruction, ...) into VLM
chat-completions `messages` payload for the perceiver.

H_0 stance: a single user message containing the egocentric RGB plus
(optionally) the visual hint image, framed by the same
`perception_template` text the published baseline uses. Object_mask /
depth_map are NOT attached here; the perceiver is a single-call freeform
seed and adding them would multiply per-call tokens for marginal H_0
gain. A proposer can extend assembly by appending mask/depth images or
by switching to strip-split RGB without touching `Perceiver.perceive`.
"""
from __future__ import annotations
import os
import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections.abc import Mapping

import numpy as np

sys.modules.setdefault(__name__, types.ModuleType(__name__))
_pkg = __name__.rsplit(".", 1)[0] if "." in __name__ else None
if _pkg:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

# Reuse the published single-baseline's text helpers (perception_template)
# and image-encoder utilities. Both live under
# /home/azureuser/workspace/AutoSpatial/simworld_gym/baseline/single
# and are imported honestly — they are scoring-side neutral utilities.
_BASELINE_DIR = "/home/azureuser/workspace/AutoSpatial/simworld_gym/baseline/single"
if _BASELINE_DIR not in sys.path:
    sys.path.insert(0, _BASELINE_DIR)

from prompt_template import perception_template  # noqa: E402
from utils import numpy_to_base64  # noqa: E402


@dataclass
class PromptAssembler:
    """Assemble the perceiver's user messages."""

    def assemble_perceiver_messages(
        self,
        obs: Dict[str, Any],
        hint: Optional[np.ndarray],
        instruction: str,
        orientation: Any,
        memory_summary: str,
    ) -> List[Dict[str, Any]]:
        """Build the OpenAI/vLLM chat-completions messages list.

        Returns a list ready to drop straight into `client.chat.completions
        .create(messages=...)`. H_0 wires:
          - role=system: `perception_template()` with depth + segmentation ON —
            BASELINE PARITY. The published baseline feeds these to GROUND the VLM's
            perception (the prompt then explains the seg colour legend and
            "judge distance by the depth image"); the naive anchor had stripped
            them to RGB-only, which left the model eyeballing depth/identity from a
            single frame.
          - role=user:   short text frame + RGB image + segmentation mask + depth
            map (+ hint image if any).
        """
        sys_prompt = perception_template(strip=False, depth=True, segment=True)
        # Nav-harness SENSOR-LEGEND correction appended to the shared baseline template. We do NOT edit
        # perception_template itself (baseline parity); we override its depth/segmentation claims here.
        # The env delivers depth as a JET colormap (cv2.COLORMAP_JET), NOT grayscale, and the shared
        # template's "judge distance / how many forward steps" over-promises a metric read that a
        # per-frame-normalized JET image cannot support. Encoding probe-verified (AGENT.md, diagnosis.md
        # §2.3-2.4): ground dead-ahead reads blue, far/sky reads red.
        sys_prompt += (
            "\n\nSENSOR LEGEND (authoritative — overrides any conflicting description above):\n"
            "- The DEPTH image is a JET colormap, normalized PER FRAME: BLUE = nearest, "
            "CYAN/GREEN = mid-range, RED/ORANGE = farthest; BOTH extremes render dark, so do NOT "
            "read brightness as distance. It is NOT metric and its scale resets every frame — use it "
            "only for COARSE, WITHIN-FRAME relative near/far by hue, never to count forward steps.\n"
            "- The SEGMENTATION mask classes are: purple = buildings, yellow = ground/road, "
            "green = SKY, blue = other. There is no 'trees' and no 'driveway' class."
        )

        text_frame = (
            f"Memory: {memory_summary}\n"
            f"Current subtask: {instruction}\n"
            f"Current orientation: {orientation}\n"
            "Describe the egocentric view in detail. Compare it against the "
            "expected view (if provided). Return strict JSON: "
            '{"Description": "<text>"}.'
        )

        rgb = obs.get("rgb") if isinstance(obs, Mapping) else None
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": text_frame},
        ]
        if rgb is not None:
            try:
                rgb_b64 = numpy_to_base64(np.asarray(rgb))
                user_content.append({"type": "text",
                                     "text": "The current view"})
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{rgb_b64}",
                    },
                })
            except Exception:
                pass

        if hint is not None:
            try:
                hint_b64 = numpy_to_base64(np.asarray(hint))
                user_content.append({"type": "text",
                                     "text": "The expected view"})
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{hint_b64}",
                    },
                })
            except Exception:
                pass

        # Baseline-parity auxiliary channels: attach the segmentation mask + depth
        # map as extra images so the VLM can GROUND perception in real geometry
        # (the published baseline feeds these; the naive anchor had stripped them to
        # RGB-only). observation_type="all" renders all three, but the env returns the
        # depth frame under key "depth" — the old single-key lookup "depth_map" matched
        # nothing and silently dropped depth on 100% of perceive calls, even though the
        # system prompt promises it. Try each candidate key in order (mirrors
        # completion_checker._depth_gray's "depth_map" or "depth" fallback).
        for _aux_keys, _aux_label in (
            (("object_mask",), "The object segmentation mask of the current view "
             "(purple=buildings, yellow=ground/road, green=SKY, blue=other; "
             "there is no trees or driveway class)"),
            (("depth_map", "depth"), "The depth image of the current view — a JET colormap "
             "normalized per frame: BLUE=nearest, CYAN/GREEN=mid-range, RED/ORANGE=farthest, "
             "both extremes dark. Use hue for coarse relative near/far only, NOT brightness, "
             "NOT a metric distance"),
        ):
            _aux = None
            if isinstance(obs, Mapping):
                for _k in _aux_keys:
                    if obs.get(_k) is not None:
                        _aux = obs.get(_k)
                        break
            if _aux is not None:
                try:
                    _aux_b64 = numpy_to_base64(np.asarray(_aux))
                    user_content.append({"type": "text", "text": _aux_label})
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_aux_b64}",
                        },
                    })
                except Exception:
                    pass

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
