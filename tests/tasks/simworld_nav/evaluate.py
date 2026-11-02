"""evaluate.py — run one harness on SimWorld-MMNav task records and write score.json.

The evaluator of the paper's navigation runs, ported from the AutoSpatial workspace it ran in:
the SimWorld UE simulator is driven through the SimWorld-Robotics gym package, each task's
`task_config.json` is split into what the harness may see (instructions, visual hints, step cap,
distance threshold, starting compass) and the ground truth the checker keeps (destination,
ideal path, per-subtask checks), and the harness's `run_episode` is called once per task.

    python evaluate.py --harness DIR --tasks TASKS.json --out OUT [--setting hard|easy]

TASKS.json is a list of {"map": "20_10", "task": "task_dist_21_0_1"} records (the split format of
tests/examples/simworld_nav/config.json; tests/tasks/simworld_nav/*.json are single-task examples).
Outputs: OUT/score.json (SubtaskSR, SR, per_position_SR, distance_progress_mean, cost_per_episode,
n_episodes, per_episode rows), OUT/traces/ep_<map>_<task>/ (actions, model calls with the frames
the agent saw, verdict, ground truth), OUT/source/ (the harness as evaluated), OUT/read_audit.json.

Environment: SIMWORLD_ROBOTICS_REPO (the SimWorld-Robotics checkout; default ~/workspace/SimWorld-Robotics),
SIMWORLD_SETTINGS_DIR (the extracted test settings; default <repo>/simworld_gym/envs/setting/single_agent_world),
SIMWORLD_BASELINE_DIR (default <repo>/baseline/single), NAV_MODEL and NAV_API_BASE (the frozen base model on a
vLLM server; default Qwen3-VL-30B-A3B-Instruct at http://127.0.0.1:8000/v1) or EVAL_ALLOWED_MODELS (a JSON pool
file), UE_PORT (UnrealCV port, 9000), EVAL_STALL_LIMIT_S, EVAL_RESET_TIMEOUT_S, VLM_CALL_TIMEOUT_S,
MAX_VLM_CALLS_PER_STEP, EVAL_SAVE_FRAMES; SIM_LAUNCHER with EVAL_SIM_RESTART_EVERY restarts the simulator
between episodes (see the comments below).
"""
from __future__ import annotations

# Vendor-patch unrealcv with a finite per-RPC timeout BEFORE any sim code
# imports it. See unrealcv_patch.py for the rationale + the destroy()-hang
# wedge it works around. Must come before `import simworld_gym` /
# `gym.make('simworld_gym/TrafficWorld')` anywhere downstream.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
import unrealcv_patch  # noqa: F401

import argparse
import base64
import csv
import importlib.util
import io
import json
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


class ResetTimeout(Exception):
    """Raised when env.reset() blocks past its wall-clock budget.

    UnrealCV connections can wedge (TCP socket stays ESTABLISHED, sim
    silently stops responding). The blocking socket recv() can't be
    interrupted via normal API. SIGALRM aborts the blocked call so the
    evaluator's recovery path (close env, wait sim, remake env) can run.
    """


def _reset_timeout_handler(signum, frame):
    raise ResetTimeout("env.reset() exceeded wall-clock budget")

import numpy as np

WORKSPACE = Path(__file__).resolve().parent
# The SimWorld-Robotics checkout (gym package, baselines, test data); see tests/examples/simworld_nav/README.md.
SIMWORLD_ROBOTICS = Path(os.environ.get(
    "SIMWORLD_ROBOTICS_REPO", str(Path.home() / "workspace" / "SimWorld-Robotics"))).expanduser()
# The task settings tree: <SETTINGS_DIR>/<simple|hard>/map_road_<id>/<task>/task_config.json, ...
SETTINGS_DIR = Path(os.environ.get(
    "SIMWORLD_SETTINGS_DIR",
    str(SIMWORLD_ROBOTICS / "simworld_gym" / "envs" / "setting" / "single_agent_world"))).expanduser()

# The harnesses import the published baseline's helpers as a top-level module
# (`from utils import action_history_text / numpy_to_base64`). Those live in
# simworld_gym/baseline/single/utils.py, so put that dir on sys.path here.
_BASELINE_SINGLE = Path(os.environ.get(
    "SIMWORLD_BASELINE_DIR", str(SIMWORLD_ROBOTICS / "baseline" / "single"))).expanduser()
if _BASELINE_SINGLE.is_dir() and str(_BASELINE_SINGLE) not in sys.path:
    sys.path.insert(0, str(_BASELINE_SINGLE))


# ----------------------------- helpers --------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _save_thumb(arr: np.ndarray, path: Path, max_side: int = 256) -> None:
    try:
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            a = a - a.min()
            a = a / (a.max() + 1e-8) * 255
            a = a.astype(np.uint8)
        else:
            a = arr
        if a.ndim == 2:
            img = Image.fromarray(a, mode="L")
        elif a.shape[-1] == 1:
            img = Image.fromarray(a[:, :, 0], mode="L")
        else:
            # SimWorld delivers BGR via cv2-fed pipelines; agent_controller's
            # get_image("lit") returns BGR-ordered uint8. The .png on disk
            # ends up swapped iff we don't convert. To match what the LLM
            # sees inside the harness (PIL.Image.fromarray on the raw numpy),
            # save without conversion — harness uses the raw bytes.
            img = Image.fromarray(a[:, :, :3])
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        img.save(path, format="PNG", optimize=True)
    except Exception:
        # Thumbnails are diagnostic; never block on failure.
        pass


# setting → (env dir component, gym env id)
SETTING_MAP = {
    "easy": ("simple", "simworld_gym/SimpleWorld"),
    "hard": ("hard",   "simworld_gym/TrafficWorld"),
}


def _build_task_spec(map_id: str, task_id: str, setting: str = "easy") -> Dict[str, Any]:
    """Construct the harness-visible task_spec from the task's
    task_config.json — stripping GT fields the harness must not see."""
    setting_dir, _ = SETTING_MAP[setting]
    setting_root = SETTINGS_DIR / setting_dir / f"map_road_{map_id}" / task_id
    cfg_path = setting_root / "task_config.json"
    cfg = _load_json(cfg_path)

    instructions = cfg.get("instruction", []) or []
    robot_views  = cfg.get("robot_view_images", []) or []

    subtasks = []
    for i, instr in enumerate(instructions):
        hint = robot_views[i] if i < len(robot_views) else None
        hint_path = str((setting_root / hint).resolve()) if hint else None
        subtasks.append({"instruction": instr, "visual_hint_path": hint_path})

    criteria = cfg.get("criteria", {}) or {}
    spawn_yaw = (cfg.get("spawning_point", {})
                   .get("orientation", {}).get("yaw", 0.0))

    return {
        "subtasks": subtasks,
        "max_steps": int(criteria.get("max_steps", 200)),
        "dist_threshold_cm": float(criteria.get("dist_threshold", 2000)),
        "compass_at_start": float(spawn_yaw),
        # Internal helpers the harness uses but the proposer should not
        # rely on as part of the public spec.
        "_map_id": map_id,
        "_task_id": task_id,
    }


def _build_gt(map_id: str, task_id: str, setting: str = "easy") -> Dict[str, Any]:
    setting_dir, _ = SETTING_MAP[setting]
    setting_root = SETTINGS_DIR / setting_dir / f"map_road_{map_id}" / task_id
    cfg = _load_json(setting_root / "task_config.json")
    return {
        "destination":            cfg.get("destination"),
        "ideal_path":             cfg.get("ideal_path"),
        "evaluation_of_subtask":  cfg.get("evaluation_of_subtask"),
        "spawning_point":         cfg.get("spawning_point"),
        "criteria":               cfg.get("criteria"),
    }


ENTRY_FILES = ("agent.py", "harness.py")          # the harness entry point; the second is its older name


def _import_harness(harness_dir: Path):
    """Import the harness and return the module that exports `run_episode`.

    A harness directory with an `__init__.py` is imported as a package (its `__init__` re-exports
    `run_episode` from the entry module); otherwise, or when the package does not export it, the
    entry file itself is loaded with the directory on sys.path so its `modules/` resolve."""
    pkg_dir = harness_dir.resolve()
    parent = pkg_dir.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    mod = None
    if (pkg_dir / "__init__.py").exists():
        pkg_name = pkg_dir.name
        if pkg_name in sys.modules:
            del sys.modules[pkg_name]
        mod = __import__(pkg_name)
    if mod is None or not hasattr(mod, "run_episode"):
        entry = next((pkg_dir / n for n in ENTRY_FILES if (pkg_dir / n).exists()), None)
        if entry is None:
            raise FileNotFoundError(f"{pkg_dir} has no {' or '.join(ENTRY_FILES)}")
        if str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
        spec = importlib.util.spec_from_file_location(f"_nav_harness_{pkg_dir.name}", entry)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return mod


def _make_openai_client(base_url: str):
    import openai
    return openai.OpenAI(base_url=base_url, api_key="EMPTY")


# ----------------------------- main runner ----------------------------------

def run(args):
    iter_idx        = args.iter
    out_dir         = Path(args.out).resolve()
    harness_dir     = Path(args.harness).resolve()
    tasks_file      = Path(args.tasks).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    # Trace dir: use --trace-dir if given, else default to traces/iter_<NNN>
    # (back-compat) for legacy callers. Phase1 orchestrator passes an explicit
    # path so seed_A and seed_B chains don't collide on traces/iter_005.
    if getattr(args, "trace_dir", None):
        trace_root = Path(args.trace_dir).resolve()
    else:
        trace_root = (out_dir / "traces").resolve()
    trace_root.mkdir(parents=True, exist_ok=True)

    # Task records: a JSON list of {"map": ..., "task": ...} (the split format of config.json), or a
    # split dict with "items" or "maps" x "tasks_per_map".
    items = _load_json(tasks_file)
    if isinstance(items, dict):
        items = items.get("items") or [{"map": m, "task": t}
                                        for m in items["maps"] for t in items["tasks_per_map"]]
    budget = {"per_episode_token_ceiling": None}
    # Base model pool: NAV_MODEL on NAV_API_BASE (one vLLM server), or EVAL_ALLOWED_MODELS, a JSON file
    # {"models": [{"id": ..., "base_url": ...}, ...]} naming several servers.
    _models_env = os.environ.get("EVAL_ALLOWED_MODELS")
    if _models_env:
        models = _load_json(Path(_models_env))
        print(f"[evaluator] model pool: {_models_env}")
    else:
        models = {"models": [{"id": os.environ.get("NAV_MODEL", "Qwen3-VL-30B-A3B-Instruct"),
                              "base_url": os.environ.get("NAV_API_BASE", "http://127.0.0.1:8000/v1")}]}

    # Episode fan-out: EVAL_MAP_SLICE="i/K" restricts this process to items[i::K]
    # (every K-th episode from offset i). K evaluator PROCESSES — each with its
    # own sim port (UE_PORT) and its own Python globals (so the trace context +
    # VLM-call guard never race) — together cover the subset CONCURRENTLY, all
    # sharing the one batched vLLM. The runner merges the K score slices.
    # EVAL_MAP_LIMIT caps the subset to the first N episodes (applied BEFORE the
    # slice) — used to bound a quick fan-out benchmark.
    _limit = os.environ.get("EVAL_MAP_LIMIT")
    if _limit:
        try:
            items = items[:int(_limit)]
        except Exception:
            pass
    _slice = os.environ.get("EVAL_MAP_SLICE")
    if _slice:
        try:
            _si, _sk = (int(x) for x in _slice.split("/"))
            items = items[_si::_sk]
            _maps = ",".join(str(it["map"]) for it in items)
            print(f"[evaluator] EVAL_MAP_SLICE={_slice} -> {len(items)} episodes ({_maps})")
        except Exception as _e:
            print(f"[evaluator] bad EVAL_MAP_SLICE={_slice!r}: {_e}")

    # Smoke mode: meta_harness sets EVAL_FORCE_EPISODE to run ONE episode as a
    # fast crash-check before the full subset. Restrict the episode list to the
    # matching ep_<map>_<task>; fall back to the first episode if the requested
    # id is not in this subset (e.g. a different subset/setting).
    _force_ep = os.environ.get("EVAL_FORCE_EPISODE")
    if _force_ep:
        _tags = [f"ep_{it['map']}_{it['task']}" for it in items]
        matched = [it for it, tag in zip(items, _tags) if tag == _force_ep]
        items = matched or items[:1]
        _ran = [f"ep_{it['map']}_{it['task']}" for it in items]
        print(f"[evaluator] EVAL_FORCE_EPISODE={_force_ep} -> running "
              f"{len(items)} episode(s): {_ran}")

    # Build vlm_clients dict.
    vlm_clients = {}
    for m in models["models"]:
        try:
            vlm_clients[m["id"]] = _make_openai_client(m["base_url"])
        except Exception as e:
            print(f"[evaluator] WARN: client init failed for {m['id']}: {e}")

    # Per-step VLM-call guard. Protects against harness inner-loops that
    # keep calling the VLM and never advance env.step (cf. iter_006 hang on
    # ep_20_1). Counter is reset to 0 inside _step_capture on every env.step,
    # so total VLM calls per episode remain uncapped — only the planning
    # cycle between two consecutive env.step calls is bounded.
    MAX_VLM_CALLS_PER_STEP = int(os.environ.get("MAX_VLM_CALLS_PER_STEP", "30"))
    vlm_call_counter: Dict[str, int] = {"n": 0, "max_seen_per_step": 0}
    # Progress heartbeat: bumped on every VLM call AND every env.step. The
    # watchdog thread (started below) force-exits the process if this stops
    # advancing for STALL_LIMIT_S — catches mid-episode UnrealCV wedges that
    # block inside env.step (which SIGALRM on reset() doesn't cover).
    progress_hb: Dict[str, float] = {"t": time.time()}

    # ---- VLM trajectory logger ----------------------------------------------
    # The chat.completions.create wrapper below also records EVERY VLM call from
    # ANY harness (the baseline, the starting harness, proposer candidates) with its full
    # reasoning trajectory: prompt, the model's <think> chain (reasoning_content,
    # surfaced by the deepseek_r1 parser), the response, tokens, latency, and the
    # frames the agent saw. This is the harness-diagnosis reference, written to
    # <ep_dir>/vlm_trajectory.jsonl with deduped frames under <ep_dir>/frames/.
    # _traj_ctx is repointed at each episode's dir in the loop below.
    import base64 as _b64, hashlib as _hl
    _save_frames = os.environ.get("EVAL_SAVE_FRAMES", "1") == "1"
    _traj_ctx: Dict[str, Any] = {"ep_dir": None, "call_idx": 0, "step": 0,
                                 "records": []}

    def _save_frame_b64(ep_dir: Path, data_url: str):
        try:
            raw = _b64.b64decode(data_url.split(",", 1)[1])
        except Exception:
            return None
        h = _hl.md5(raw).hexdigest()[:12]
        fdir = ep_dir / "frames"
        fdir.mkdir(parents=True, exist_ok=True)
        fp = fdir / f"{h}.png"
        if not fp.exists():
            try:
                fp.write_bytes(raw)
            except Exception:
                return None
        return f"frames/{h}.png"

    def _log_vlm_call(messages, model, result, latency):
        ep_dir = _traj_ctx["ep_dir"]
        if ep_dir is None:
            return
        idx = _traj_ctx["call_idx"]
        _traj_ctx["call_idx"] = idx + 1
        prompt_parts, image_refs = [], []
        for m in (messages or []):
            if not isinstance(m, dict):
                continue
            role = m.get("role", "?")
            content = m.get("content")
            if isinstance(content, str):
                prompt_parts.append(f"[{role}] {content}")
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    pt = part.get("type")
                    if pt in ("text", "input_text"):
                        prompt_parts.append(f"[{role}] {part.get('text', '')}")
                    elif pt in ("image_url", "input_image"):
                        url = part.get("image_url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        if _save_frames and isinstance(url, str) and url.startswith("data:image"):
                            ref = _save_frame_b64(ep_dir, url)
                            if ref:
                                image_refs.append(ref)
        content_txt, reasoning_txt, in_tok, out_tok = "", "", None, None
        finish_reason = None
        try:
            finish_reason = getattr(result.choices[0], "finish_reason", None)
            msg = result.choices[0].message
            content_txt = msg.content or ""
            # vLLM surfaces the <think> chain under "reasoning" (this build) or
            # "reasoning_content" (other parsers); the OpenAI SDK may expose it
            # as a direct attribute or stash it in model_extra — check all.
            reasoning_txt = (getattr(msg, "reasoning", None)
                             or getattr(msg, "reasoning_content", None))
            if not reasoning_txt:
                _extra = getattr(msg, "model_extra", None) or {}
                reasoning_txt = (_extra.get("reasoning")
                                 or _extra.get("reasoning_content") or "")
            reasoning_txt = reasoning_txt or ""
            usage = getattr(result, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None)
            out_tok = getattr(usage, "completion_tokens", None)
        except Exception:
            pass
        rec = {
            "call_idx": idx,
            "step": _traj_ctx.get("step"),
            "kind": "perceive" if image_refs else "plan",
            "model": model,
            "latency_s": round(latency, 2),
            "images": image_refs,
            "prompt": "\n".join(prompt_parts),
            "reasoning": reasoning_txt,
            "response": content_txt,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "finish_reason": finish_reason,
            "truncated": bool(finish_reason == "length" or (out_tok and not content_txt)),
        }
        try:
            _traj_ctx["records"].append(rec)
        except Exception:
            pass
        try:
            with open(ep_dir / "vlm_trajectory.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # Patch chat.completions.create at the CLASS level, not per vlm_client:
    # harnesses don't always call through the passed vlm_clients — the baseline's
    # BaseModel builds its own OpenAI client internally, and only a class-level
    # patch catches every client in this process (baseline + naive + future
    # candidates). Guard + heartbeat + trajectory logging all live here.
    from openai.resources.chat.completions import Completions as _Completions
    _orig_create = _Completions.create

    def _patched_create(_self, *args, **kwargs):
        progress_hb["t"] = time.time()
        vlm_call_counter["n"] += 1
        if vlm_call_counter["n"] > vlm_call_counter["max_seen_per_step"]:
            vlm_call_counter["max_seen_per_step"] = vlm_call_counter["n"]
        if vlm_call_counter["n"] > MAX_VLM_CALLS_PER_STEP:
            raise RuntimeError(
                f"VLM-call guard tripped: > {MAX_VLM_CALLS_PER_STEP} calls "
                f"between two env.step() calls — harness is looping "
                f"without progress. Edit MAX_VLM_CALLS_PER_STEP env var "
                f"or fix the harness's inner planning loop.")
        _t0 = time.time()
        # A stalled vLLM response must NOT hang the whole eval forever: the
        # openai/httpx client's read can block indefinitely with no timeout,
        # and the progress watchdog only kills the SIM (which can't unblock a
        # hung HTTP read). Cap each call + retry a few times; on persistent
        # failure raise so the episode fails gracefully instead of wedging.
        kwargs.setdefault("timeout", float(os.environ.get("VLM_CALL_TIMEOUT_S", "180")))
        _vlm_retries = int(os.environ.get("VLM_CALL_RETRIES", "3"))
        result = None
        for _vlm_attempt in range(_vlm_retries):
            try:
                result = _orig_create(_self, *args, **kwargs)
                break
            except Exception as _vlm_e:
                if _vlm_attempt + 1 >= _vlm_retries:
                    raise
                print(f"[evaluator] VLM call {type(_vlm_e).__name__} "
                      f"(retry {_vlm_attempt + 1}/{_vlm_retries}): "
                      f"{str(_vlm_e)[:120]}", flush=True)
                progress_hb["t"] = time.time()
                time.sleep(2)
        progress_hb["t"] = time.time()  # bump again after the call returns
        try:
            _log_vlm_call(kwargs.get("messages"), kwargs.get("model"),
                          result, time.time() - _t0)
        except Exception:
            pass
        return result

    _Completions.create = _patched_create

    # ---- progress watchdog -------------------------------------------------
    # A daemon thread that detects mid-episode stalls (neither a VLM call nor
    # an env.step for STALL_LIMIT_S). This is the case SIGALRM-on-reset doesn't
    # cover: the sim wedges *inside* env.step, blocking the harness's step loop.
    #
    # Two-tier response:
    #   Tier 1 (stall > STALL_LIMIT_S): kill the sim process. The hung env.step
    #     is blocked on a socket recv; killing the sim closes the TCP conn so
    #     recv raises, the exception unwinds out of run_episode, and the
    #     evaluator's per-episode try/except records a crashed episode and moves
    #     on. The next episode's reset hits the sim-recovery path (external
    #     watchdog restarts the sim within ~60s; _wait_for_sim picks it up).
    #     => the iter still produces a valid score.json with a partial set of
    #        real episodes, instead of losing the whole iter to the 60-min cap.
    #   Tier 2 (still stalled after 2 sim-kills): os._exit(7) — give up cleanly
    #     so the master retries; better than hanging.
    # sim_kills resets whenever genuine progress resumes.
    import subprocess
    import threading
    # Default 900s: under heavy vLLM contention (TP=8 saturating the shared
    # GPUs), env.step + the harness's VLM-call cycle can run 5-15 min between
    # actual env.step returns without being truly hung. The old 360s default
    # was tuned for low-contention conditions and triggers false kill cycles
    # whenever another user's job is also on the box.
    STALL_LIMIT_S = float(os.environ.get("EVAL_STALL_LIMIT_S", "900"))
    _wd_stop = {"stop": False}

    def _kill_sim_processes():
        # Only kill the SimWorld instance listening on OUR `ue_port`. The
        # earlier `comm in (SimWorld, SimWorld.sh)` filter killed every UE
        # process on the box — fine when only one sim ran, fatal when a
        # second sim served a parallel eval on a different port (sim 1 on
        # :9000 and sim 2 on :9001 — killing both took down the master too).
        try:
            ss_out = subprocess.run(
                ["ss", "-Hltnp", f"sport = :{ue_port}"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except Exception:
            return
        import re
        # ss row format: ... users:(("SimWorld",pid=12345,fd=N))
        listener_pid = None
        m = re.search(r"pid=(\d+)", ss_out)
        if m:
            listener_pid = int(m.group(1))
        if listener_pid is None:
            return  # nothing listening on our port; nothing to kill
        try:
            ppid = None
            with open(f"/proc/{listener_pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
        except Exception:
            ppid = None
        for pid in filter(None, (listener_pid, ppid)):
            try:
                os.kill(pid, 9)
            except Exception:
                pass

    def _progress_watchdog():
        sim_kills = 0
        last_seen = progress_hb["t"]
        while not _wd_stop["stop"]:
            time.sleep(30)
            hb = progress_hb["t"]
            if hb > last_seen + 1.0:        # genuine progress since last check
                last_seen = hb
                sim_kills = 0
                continue
            stalled = time.time() - hb
            if stalled <= STALL_LIMIT_S:
                continue
            if sim_kills < 2:
                sim_kills += 1
                msg = (f"[evaluator] PROGRESS WATCHDOG: stalled {stalled:.0f}s "
                       f"(limit {STALL_LIMIT_S:.0f}s) — killing sim (kill "
                       f"#{sim_kills}) to unblock the hung env.step; episode "
                       f"will be recorded as crashed and the next one recovers.")
                print(msg, flush=True)
                sys.stderr.write(msg + "\n"); sys.stderr.flush()
                _kill_sim_processes()
                progress_hb["t"] = time.time()   # fresh window for recovery
                last_seen = progress_hb["t"]
            else:
                msg = (f"[evaluator] PROGRESS WATCHDOG: still stalled "
                       f"{stalled:.0f}s after {sim_kills} sim-kills — "
                       f"force-exiting so the master can retry.")
                print(msg, flush=True)
                sys.stderr.write(msg + "\n"); sys.stderr.flush()
                os._exit(7)

    threading.Thread(target=_progress_watchdog, daemon=True).start()

    # Import the harness.
    harness_mod = _import_harness(harness_dir)
    run_episode = harness_mod.run_episode
    ACTION_NAMES = getattr(harness_mod, "ACTION_NAMES", {})

    # Build env. Lazy import so this file is cheap to syntax-check.
    import warnings
    warnings.filterwarnings("ignore")
    import gym
    import simworld_gym  # noqa: F401  (registers env ids)

    setting = getattr(args, "setting", "easy")
    setting_dir, env_id = SETTING_MAP[setting]
    ue_port = int(os.environ.get("UE_PORT", "9000"))
    print(f"[evaluator] harness sees the full env info dict — "
          f"honesty rules in workspace/README.md §'Rules of engagement'.")

    def _make_env():
        return gym.make(
            env_id,
            port=ue_port,
            resolution=(720, 600),
            render_mode="rgb_array",
            observation_type="all",
            record_video=False,
            log_dir=str(out_dir / "_env_logs"),
            reward_setting={
                "human_collision_penalty": -1,
                "object_collision_penalty": -0.1,
                "action_penalty": -0.1,
                "success_reward": 10.0,
                "off_track_penalty": -0.01,
            },
            disable_env_checker=True,
        )

    def _wait_for_sim(timeout_s: float = 120.0, port: int = ue_port) -> bool:
        import socket
        t_start = time.time()
        while time.time() - t_start < timeout_s:
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=2.0)
                s.close()
                return True
            except Exception:
                time.sleep(2.0)
        return False

    # Per-episode sim restart (opt-in). On this box, reusing one UnrealCV
    # connection across episodes wedges the 2nd+ episode (episode 0 runs fine,
    # then the next reset's clear_env of the prior city drops the client). So
    # give each episode a freshly-launched sim (empty map → nothing to clear).
    # Enable with EVAL_SIM_RESTART_EACH_EP=1 + SIM_LAUNCHER (+ SIM_GPU). The
    # caller launches a clean sim for episode 0; we restart before 1..N.
    _sim_launcher = os.environ.get("SIM_LAUNCHER", "")
    _sim_gpu = os.environ.get("SIM_GPU", "")
    _restart_each_ep = os.environ.get("EVAL_SIM_RESTART_EACH_EP", "0") == "1"
    # EVAL_SIM_RESTART_EVERY=N restarts the sim every N episodes (N<~13 sidesteps the cross-episode
    # wedge with FAR less overhead than restarting every episode, which blew past the eval timeout).
    _restart_every = int(os.environ.get("EVAL_SIM_RESTART_EVERY", "0") or "0")

    def _port_is_free(port: int) -> bool:
        import subprocess as _sp
        chk = _sp.run(["ss", "-ltn"], capture_output=True, text=True)
        return f":{port} " not in (chk.stdout or "")

    def _restart_sim() -> bool:
        import subprocess as _sp
        if not (_sim_launcher and Path(_sim_launcher).exists()):
            print("[evaluator] WARN: EVAL_SIM_RESTART_EACH_EP set but SIM_LAUNCHER "
                  "missing/not found; reusing current sim")
            return _wait_for_sim(timeout_s=120.0)
        (out_dir / "_env_logs").mkdir(parents=True, exist_ok=True)
        senv = os.environ.copy()
        if _sim_gpu:
            senv["SIM_GPU"] = _sim_gpu
        # Robust relaunch. The SIGKILLed UE process can hold its UnrealCV listening
        # socket for >3s under GPU/CPU contention (node1 packs 2 sims/GPU alongside a
        # co-tenant campaign), so the old fixed sleep(3)+single-launch raced into
        # EADDRINUSE: the replacement came up "deaf" (never bound the port),
        # _wait_for_sim timed out, _restart_sim returned False with NO retry, and the
        # whole eval dead-walked returning valid=False for ~16h. Mirror the campaign
        # lane_worker: WAIT until the port is actually free, then retry up to 3x.
        for _attempt in range(1, 4):
            _sp.run(["fuser", "-k", f"{ue_port}/tcp"], capture_output=True)
            _t0 = time.time()
            while time.time() - _t0 < 60.0 and not _port_is_free(ue_port):
                _sp.run(["fuser", "-k", f"{ue_port}/tcp"], capture_output=True)
                time.sleep(2.0)
            _sp.Popen(["setsid", "nohup", "bash", _sim_launcher],
                      stdout=open(out_dir / "_env_logs" / "sim_restart.log", "a"),
                      stderr=_sp.STDOUT, env=senv, start_new_session=True)
            if _wait_for_sim(timeout_s=150.0):
                return True
            print(f"[evaluator] sim relaunch attempt {_attempt}/3 failed to bind "
                  f"port {ue_port}; retrying")
        return False

    env = _make_env()

    # ---- per-episode loop ----------------------------------------------------
    _spawned_map = None  # map currently spawned in the (warm) sim; re-pass
                         # world_json whenever it changes -> clear_env level-reload + spawn
    per_episode_records: List[Dict[str, Any]] = []
    t0_total = time.time()

    for idx, item in enumerate(items):
        map_id  = item["map"]
        task_id = item["task"]
        ep_tag  = f"ep_{map_id}_{task_id}"
        ep_dir  = trace_root / ep_tag
        (ep_dir / "observations").mkdir(parents=True, exist_ok=True)
        # Point the VLM trajectory logger at this episode (resets call index).
        _traj_ctx["ep_dir"] = ep_dir
        _traj_ctx["call_idx"] = 0
        _traj_ctx["step"] = 0
        _traj_ctx["records"] = []

        # Fresh sim for every episode after the first (caller gives ep 0 a clean
        # sim). Sidesteps the cross-episode reset wedge described above.
        if idx > 0 and (_restart_each_ep or (_restart_every > 0 and idx % _restart_every == 0)):
            print(f"[evaluator] restarting sim before {ep_tag} (fresh map)")
            try:
                env.close()
            except Exception:
                pass
            if not _restart_sim():
                print(f"[evaluator] sim restart failed before {ep_tag}; skipping")
                continue
            env = _make_env()
            _spawned_map = None  # fresh sim → re-pass world_json on this reset

        task_root_rel = os.path.join("single_agent_world", setting_dir,
                                     f"map_road_{map_id}", task_id)
        options = {
            "task_path":  task_root_rel,
            "agent_json": os.path.join(task_root_rel, "task_config.json"),
        }
        # Hard / TrafficWorld needs traffic.json *loaded as a dict* — upstream
        # baseline passes a path string but TrafficWorld.reset() indexes it as
        # a dict (traffic_world.py L477 num_vehicles = traffic_json[...]).
        # Resolve via the same envs/setting/ root the env uses for everything
        # else (cf. misc.get_settingpath()).
        if setting == "hard":
            import simworld_gym as _swg
            traffic_cfg_path = (Path(_swg.__file__).parent / "envs" / "setting"
                                / task_root_rel / "traffic.json")
            # A few task dirs in the dataset are empty/incomplete (no traffic.json
            # or task_config.json). Skip them gracefully instead of crashing the
            # whole subset — one broken task must not abort the remaining episodes.
            if not os.path.isfile(traffic_cfg_path):
                print(f"[evaluator] SKIP {ep_tag}: missing task data "
                      f"({traffic_cfg_path}) — broken/incomplete task dir")
                continue
            options["traffic_json"] = _load_json(traffic_cfg_path)
        if map_id != _spawned_map:
            options["world_json"] = os.path.join(task_root_rel,
                                                 "progen_world.json")
            _spawned_map = map_id

        print(f"\n[evaluator] === {ep_tag} ===")
        t0 = time.time()
        # Reset with in-process sim recovery: on failure, close the dead env,
        # wait for the sim to come back (the external watchdog typically
        # restarts it in <10s), and rebuild the env. Up to 3 attempts.
        obs = info = None
        reset_err = None
        for reset_attempt in range(3):
            progress_hb["t"] = time.time()  # each attempt counts as progress
            try:
                # Hard wall-clock cap on a single reset attempt. UnrealCV
                # wedges (TCP open, sim silent) can otherwise block forever.
                # 180s was generous under low contention; with vLLM TP=8
                # saturating the shared GPUs, env.reset takes 200-400s.
                # Bumped to 600s + override via EVAL_RESET_TIMEOUT_S env var.
                reset_budget = int(os.environ.get("EVAL_RESET_TIMEOUT_S", "600"))
                old_handler = signal.signal(signal.SIGALRM, _reset_timeout_handler)
                signal.alarm(reset_budget)
                try:
                    obs, info = env.reset(options=options)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                if reset_attempt > 0:
                    print(f"[evaluator] reset recovered on attempt {reset_attempt+1}")
                break
            except Exception as e:
                reset_err = e
                print(f"[evaluator] reset attempt {reset_attempt+1} failed for {ep_tag}: {e}")
                traceback.print_exc()
                if reset_attempt >= 2:
                    break
                # Tear down likely-dead env, poll sim, rebuild.
                print(f"[evaluator] attempting sim recovery (close+wait+remake env)")
                try:
                    env.close()
                except Exception:
                    pass
                if not _wait_for_sim(timeout_s=120.0):
                    # Sim is genuinely dead. WAITING alone never revives it — this is the
                    # trap that turned a transient relaunch bind-race into a 16h wedge:
                    # every fresh candidate eval starts at idx=0 against a dead port and
                    # can never reach the idx%N restart gate, so it dead-walks valid=False.
                    # Relaunch robustly (port-free poll + 3x retry) so idx=0 self-heals.
                    # Only fires when the sim is confirmed down, so a live sim is never
                    # needlessly bounced (a transient reset hiccup keeps the old behavior).
                    print(f"[evaluator] sim not back on :{ue_port} after 120s; relaunching")
                    if not (_sim_launcher and _restart_sim()):
                        continue
                try:
                    env = _make_env()
                    # First reset after env recreate must include world_json so
                    # the env initializes its world state from scratch.
                    options["world_json"] = os.path.join(task_root_rel,
                                                         "progen_world.json")
                    print(f"[evaluator] env recreated; retrying reset")
                except Exception as ee:
                    print(f"[evaluator] env recreate failed: {ee}")
                    traceback.print_exc()

        if obs is None:
            # Recovery exhausted — record and STOP the iter (don't dead-walk
            # through remaining episodes against a dead/uncooperative sim).
            per_episode_records.append({
                "episode": ep_tag, "success": False,
                "subtask_success": [], "subtask_sr": 0.0,
                "n_subtasks": 0, "steps": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "wall_s": round(time.time() - t0, 1),
                "reset_error": str(reset_err),
                "human_collision": 0, "object_collision": 0,
                "building_collision": 0, "vehicle_collision": 0,
                "static_collision": 0, "dynamic_collision": 0,
                "red_light_violation": 0, "moving_distance": 0.0,
                "initial_dist_to_target": 0.0, "final_dist_to_target": 0.0,
                "distance_progress": 0.0,
            })
            print(f"[evaluator] reset unrecoverable after 3 attempts; ending iter early")
            break

        task_spec = _build_task_spec(map_id, task_id, setting)
        task_spec["_initial_obs"]  = obs
        task_spec["_initial_info"] = info
        n_subtasks = len(task_spec["subtasks"])

        # ===== paper-metric capture =====
        # Wrap env.step so the LAST info dict is preserved after run_episode
        # returns. The harness sees an unchanged env interface; the evaluator
        # gets info["agent_collision"], info["moving_distance"], target etc.
        # for free. Also keep a per-step snapshot list for retroactive scoring.
        last_info_ref: Dict[str, Any] = {"info": info, "per_step": []}
        last_info_ref["initial_info"] = info
        _orig_step = env.step
        # Per-step frame counter for the `--render` mode below
        step_counter = {"n": 0}

        def _step_capture(action):
            # Reset the per-step VLM-call counter — total calls per episode
            # are uncapped; only the planning cycle between two env.steps is.
            vlm_call_counter["n"] = 0
            progress_hb["t"] = time.time()  # bump before the (blocking) step
            obs2, r2, term2, trunc2, info2 = _orig_step(action)
            progress_hb["t"] = time.time()  # and after it returns
            last_info_ref["info"] = info2
            try:
                ac = info2.get("agent_collision") or {}
                ag = info2.get("agent", {}) or {}
                tg = info2.get("target", {}) or {}
                last_info_ref["per_step"].append({
                    "action": int(action),
                    "agent_location": list(ag.get("agent_location", [])),
                    "target_location": list(tg.get("target_location", [])),
                    "moving_distance": float(info2.get("moving_distance", 0.0)),
                    "human":    int(ac.get("human", 0)),
                    "object":   int(ac.get("object", 0)),
                    "building": int(ac.get("building", 0)),
                    "vehicle":  int(ac.get("vehicle", 0)),
                    "red_light_violation": int(info2.get("total_red_light_violation", 0)),
                })
            except Exception:
                pass
            # Advance the trajectory step counter: every VLM call logged AFTER
            # this point plans the NEXT action, so it is stamped with the new
            # step index. This is the call->step->action->outcome join key.
            try:
                _traj_ctx["step"] = _traj_ctx.get("step", 0) + 1
            except Exception:
                pass
            # Save per-step RGB thumbnail so a future proposer can VISUALLY
            # inspect what the agent saw at any step (correlated with the
            # action it emitted next + the VLM call meta written elsewhere).
            # Only sample every N-th step to bound disk: every step on
            # short episodes, every 2nd on long ones. ~10 KB per thumb.
            if getattr(args, "render", False):
                try:
                    step_counter["n"] += 1
                    n = step_counter["n"]
                    # Sample every step for the first 30, then every 2nd
                    if n <= 30 or n % 2 == 0:
                        rgb = obs2.get("rgb") if isinstance(obs2, dict) else None
                        if rgb is not None:
                            _save_thumb(rgb, ep_dir / "observations" / f"step_{n:03d}.png")
                except Exception:
                    pass
            return obs2, r2, term2, trunc2, info2

        env.step = _step_capture

        # Runtime taint tracker: wrap env+task_spec so the harness's reads on
        # info/obs/task_spec are logged. Defense against cheats that bypass
        # the AST scan (string-built keys, sub-helper reads, getattr tricks).
        # The wrapper forwards env.step / env.reset / env.close transparently.
        # Per-episode dump goes to ep_dir/read_audit.json; per-iter aggregate
        # is written at out_dir/read_audit.json after the loop.
        try:
            import evaluator_taint as _taint
            _taint.reset_log()
            tainted_env = _taint.wrap_env(env)
            tainted_task_spec = _taint.TaintedDict(task_spec, "task_spec")
        except Exception:
            tainted_env = env
            tainted_task_spec = task_spec
            _taint = None

        # Run the harness.
        try:
            result = run_episode(tainted_env, tainted_task_spec, vlm_clients,
                                 trace_dir=str(ep_dir))
        except Exception as e:
            print(f"[evaluator] run_episode crashed for {ep_tag}: {e}")
            traceback.print_exc()
            result = harness_mod.EpisodeResult(
                success=False,
                subtask_success=[False] * n_subtasks,
                steps=0, actions=[], llm_calls=[],
                notes=f"crash: {e}",
                observations=[],
            )

        wall = time.time() - t0
        # Restore env.step (idempotent across episodes; defensive).
        env.step = _orig_step

        # ===== paper-metric extraction (from last_info_ref) =====
        final_info = last_info_ref.get("info") or {}
        initial_info = last_info_ref.get("initial_info") or {}
        per_step_log = last_info_ref.get("per_step") or []
        ac = final_info.get("agent_collision") or {}
        ag0 = initial_info.get("agent", {}) or {}
        ag1 = final_info.get("agent", {}) or {}
        tg0 = initial_info.get("target", {}) or {}
        ic_loc = list(ag0.get("agent_location", []))
        fc_loc = list(ag1.get("agent_location", []))
        tg_loc = list(tg0.get("target_location", []))
        # Distance progress: 1 - (final_dist / initial_dist), clipped to [0, 1].
        # Paper aligns this with "fraction of the gap to the target closed".
        dp = 0.0
        if len(ic_loc) >= 2 and len(fc_loc) >= 2 and len(tg_loc) >= 2:
            d0 = ((ic_loc[0] - tg_loc[0])**2 + (ic_loc[1] - tg_loc[1])**2) ** 0.5
            d1 = ((fc_loc[0] - tg_loc[0])**2 + (fc_loc[1] - tg_loc[1])**2) ** 0.5
            if d0 > 1e-3:
                dp = max(0.0, min(1.0, 1.0 - d1 / d0))
        paper_metrics = {
            "human_collision":    int(ac.get("human", 0)),
            "object_collision":   int(ac.get("object", 0)),
            "building_collision": int(ac.get("building", 0)),
            "vehicle_collision":  int(ac.get("vehicle", 0)),
            "static_collision":   int(ac.get("object", 0)) + int(ac.get("building", 0)),
            "dynamic_collision":  int(ac.get("human", 0))  + int(ac.get("vehicle", 0)),
            "red_light_violation": int(final_info.get("total_red_light_violation", 0)),
            "moving_distance":    float(final_info.get("moving_distance", 0.0)),
            "initial_dist_to_target": round(d0 if 'd0' in dir() else 0.0, 1),
            "final_dist_to_target":   round(d1 if 'd1' in dir() else 0.0, 1),
            "distance_progress":      round(dp, 4),
        }
        # Persist the full per-step env log alongside the harness's
        # observations.jsonl (the env-side log is independent and never
        # filtered by the harness).
        with open(ep_dir / "env_step_log.jsonl", "w") as f:
            for r in per_step_log:
                f.write(json.dumps(r, default=_json_default) + "\n")

        # Save thumbnails of the initial + final RGB so a future proposer
        # has a sense of the trajectory without paying for full frames.
        try:
            _save_thumb(obs["rgb"], ep_dir / "observations" / "step_000_initial.png")
        except Exception:
            pass

        # Resolve an action index to its name. The harness may export
        # ACTION_NAMES as a list (index->name) OR a dict (int->name); handle
        # both so the name is never an unresolved "?".
        # Canonical TrafficWorld action space — fixed by the ENV, not the
        # harness. Most candidate harnesses don't re-export ACTION_NAMES (so the
        # getattr above is {}), so this is the fallback that keeps names resolved.
        _ACTION_NAMES_DEFAULT = {
            0: "Move_Forward", 1: "Rotate_Left", 2: "Rotate_Right",
            3: "Move_Left", 4: "Move_Right", 5: "Subtask_Completed",
            -1: "Auto_Orient",
        }

        def _action_name(a):
            try:
                ai = int(a)
            except Exception:
                return "?"
            name = None
            if isinstance(ACTION_NAMES, dict):
                name = ACTION_NAMES.get(ai)
            elif isinstance(ACTION_NAMES, (list, tuple)) and 0 <= ai < len(ACTION_NAMES):
                name = ACTION_NAMES[ai]
            return name or _ACTION_NAMES_DEFAULT.get(ai, "?")

        # Write traces.
        with open(ep_dir / "actions.jsonl", "w") as f:
            for step_idx, a in enumerate(result.actions):
                f.write(json.dumps({
                    "step": step_idx, "action": int(a),
                    "action_name": _action_name(a),
                }) + "\n")
        with open(ep_dir / "llm_calls.jsonl", "w") as f:
            for call in result.llm_calls:
                f.write(json.dumps(call, default=_json_default) + "\n")

        verdict = {
            "episode":         ep_tag,
            "success":         bool(result.success),
            "subtask_success": [bool(b) for b in result.subtask_success],
            "n_subtasks":      n_subtasks,
            "subtask_sr":      (sum(result.subtask_success) / n_subtasks
                                if n_subtasks else 0.0),
            "steps":           int(result.steps),
            "wall_s":          round(wall, 1),
            "notes":           result.notes,
        }
        _save_json(ep_dir / "verdict.json", verdict)
        # GT — diagnosis only, NOT for the harness.
        _save_json(ep_dir / "gt.json", _build_gt(map_id, task_id, setting))
        # Per-step observation log written by the harness.
        with open(ep_dir / "observations.jsonl", "w") as f:
            for o in result.observations:
                f.write(json.dumps(o, default=_json_default) + "\n")

        # ===== per-episode JOIN: trajectory.json =====
        # One record per env step joining the action (+name), the env outcome
        # (location, distance-to-target, moving_distance, collisions, red-light),
        # and EVERY VLM call that planned that step (frames, full reasoning,
        # response, tokens, truncation). This is the single human + proposer
        # diagnosis reference. Fully defensive: a failure here must NEVER fail
        # the episode/eval — the score and the other traces are already written.
        try:
            recs = _traj_ctx.get("records") or []
            by_step: Dict[int, list] = {}
            for r in recs:
                by_step.setdefault(r.get("step"), []).append(r)
            steps_in_calls = [r.get("step") for r in recs if r.get("step") is not None]
            n_steps = max(len(per_step_log), len(result.actions),
                          (max(steps_in_calls) + 1) if steps_in_calls else 0)
            steps_join = []
            for i in range(n_steps):
                ps = per_step_log[i] if i < len(per_step_log) else {}
                a = (result.actions[i] if i < len(result.actions)
                     else ps.get("action"))
                d2t = None
                try:
                    al = ps.get("agent_location") or []
                    tl = ps.get("target_location") or []
                    if len(al) >= 2 and len(tl) >= 2:
                        d2t = round(((al[0] - tl[0])**2 + (al[1] - tl[1])**2)**0.5, 1)
                except Exception:
                    pass
                calls = [{
                    "call_idx": r.get("call_idx"), "kind": r.get("kind"),
                    "images": r.get("images"), "reasoning": r.get("reasoning"),
                    "response": r.get("response"),
                    "input_tokens": r.get("input_tokens"),
                    "output_tokens": r.get("output_tokens"),
                    "finish_reason": r.get("finish_reason"),
                    "truncated": r.get("truncated"),
                    "latency_s": r.get("latency_s"),
                } for r in by_step.get(i, [])]
                steps_join.append({
                    "step": i,
                    "action": (int(a) if a is not None else None),
                    "action_name": (_action_name(a) if a is not None else None),
                    "agent_location": ps.get("agent_location"),
                    "target_location": ps.get("target_location"),
                    "dist_to_target": d2t,
                    "moving_distance": ps.get("moving_distance"),
                    "collisions": {"human": ps.get("human"), "object": ps.get("object"),
                                   "building": ps.get("building"), "vehicle": ps.get("vehicle")},
                    "red_light_violation": ps.get("red_light_violation"),
                    "n_vlm_calls": len(calls),
                    "vlm_calls": calls,
                })
            traj = {
                "episode": ep_tag,
                "model": (recs[0].get("model") if recs else None),
                "success": bool(result.success),
                "subtask_success": [bool(b) for b in result.subtask_success],
                "n_steps": n_steps,
                "n_vlm_calls": len(recs),
                "n_truncated_calls": sum(1 for r in recs if r.get("truncated")),
                "steps": steps_join,
            }
            _save_json(ep_dir / "trajectory.json", traj)
        except Exception as _je:
            print(f"[evaluator] trajectory.json join skipped for {ep_tag}: {_je}")

        # Cost ledger for this episode.
        in_toks  = sum((c.get("input_tokens")  or 0) for c in result.llm_calls)
        out_toks = sum((c.get("output_tokens") or 0) for c in result.llm_calls)
        record = {
            "episode":      ep_tag,
            "success":      bool(result.success),
            "subtask_sr":   verdict["subtask_sr"],
            "subtask_success": verdict["subtask_success"],
            "steps":        verdict["steps"],
            "input_tokens": in_toks,
            "output_tokens": out_toks,
            "total_tokens": in_toks + out_toks,
            "wall_s":       verdict["wall_s"],
            # Paper-aligned metrics (added per user request 2026-05-24).
            **paper_metrics,
        }
        per_episode_records.append(record)

        # Per-episode taint dump (best-effort, never fail the eval).
        if _taint is not None:
            try:
                _taint.dump_audit(ep_dir / "read_audit.json")
            except Exception:
                pass

        print(f"[evaluator] {ep_tag}: success={record['success']} "
              f"subtask_sr={record['subtask_sr']:.2f} steps={record['steps']} "
              f"tokens={record['total_tokens']} wall={record['wall_s']}s")

    env.close()

    # ---- per-iter taint audit aggregation ----------------------------------
    # Union the per-episode read_audit.json files into one out_dir/read_audit.json
    # so PARETO and downstream tooling can see whether ANY episode in this iter
    # touched a forbidden field.
    try:
        import evaluator_taint as _taint_agg
        all_reads_union: dict = {}
        per_ep_forbidden: dict = {}
        for r in per_episode_records:
            ep_audit = trace_root / r["episode"] / "read_audit.json"
            if not ep_audit.exists():
                continue
            try:
                audit = json.loads(ep_audit.read_text())
            except Exception:
                continue
            for path, count in audit.get("read_counts", {}).items():
                all_reads_union[path] = all_reads_union.get(path, 0) + count
            if audit.get("forbidden_reads"):
                per_ep_forbidden[r["episode"]] = audit["forbidden_reads"]
        paths_sorted = sorted(all_reads_union.keys())
        agg = {
            "n_unique_paths": len(paths_sorted),
            "n_total_reads": sum(all_reads_union.values()),
            "all_reads": paths_sorted,
            "read_counts": all_reads_union,
            "allowed_reads": [p for p in paths_sorted if _taint_agg._is_allowed(p)],
            "forbidden_reads": [p for p in paths_sorted if _taint_agg._is_forbidden(p)],
            "undeclared_reads": [p for p in paths_sorted if not _taint_agg._is_allowed(p) and not _taint_agg._is_forbidden(p)],
            "per_episode_forbidden": per_ep_forbidden,
        }
        # Verdict
        if agg["forbidden_reads"]:
            agg["verdict"] = "FAIL"
        elif agg["undeclared_reads"]:
            agg["verdict"] = "WARN"
        else:
            agg["verdict"] = "PASS"
        (out_dir / "read_audit.json").write_text(json.dumps(agg, indent=2))
        print(f"[evaluator] taint audit: {agg['verdict']}  "
              f"forbidden={len(agg['forbidden_reads'])}  "
              f"undeclared={len(agg['undeclared_reads'])}")
    except Exception as _e:
        print(f"[evaluator] taint aggregation skipped: {_e}")

    # ---- aggregate ----------------------------------------------------------
    n = len(per_episode_records)
    if n == 0:
        print("[evaluator] no episodes ran.")
        return
    SR        = sum(r["success"] for r in per_episode_records) / n
    SubtaskSR = sum(r["subtask_sr"] for r in per_episode_records) / n
    mean_tok  = sum(r["total_tokens"] for r in per_episode_records) / n
    mean_wall = sum(r["wall_s"] for r in per_episode_records) / n

    # Per-subtask SR by position (0..k-1).
    max_k = max((len(r["subtask_success"]) for r in per_episode_records),
                default=0)
    per_position_sr: List[float] = []
    for k in range(max_k):
        rs = [r for r in per_episode_records
              if len(r["subtask_success"]) > k]
        if rs:
            per_position_sr.append(sum(r["subtask_success"][k] for r in rs) / len(rs))

    ceiling = budget.get("per_episode_token_ceiling")
    over_budget = bool(ceiling and mean_tok > ceiling)

    # Paper-aligned aggregates (added per user request 2026-05-24).
    def _mean(field: str) -> float:
        vals = [r.get(field) for r in per_episode_records if r.get(field) is not None]
        return (sum(vals) / len(vals)) if vals else 0.0

    paper_aggregate = {
        "distance_progress_mean":  round(_mean("distance_progress"), 4),
        "moving_distance_mean":    round(_mean("moving_distance"), 1),
        "static_collision_mean":   round(_mean("static_collision"), 3),
        "dynamic_collision_mean":  round(_mean("dynamic_collision"), 3),
        "red_light_violation_mean": round(_mean("red_light_violation"), 3),
        # Per-episode rates: fraction of eps with any collision in each class.
        "static_collision_rate":   round(sum(1 for r in per_episode_records
                                              if (r.get("static_collision") or 0) > 0) / n, 3),
        "dynamic_collision_rate":  round(sum(1 for r in per_episode_records
                                              if (r.get("dynamic_collision") or 0) > 0) / n, 3),
    }

    score = {
        "iter":             iter_idx,
        "tasks":            str(tasks_file),
        "setting":          setting,
        "n_episodes":       n,
        "SR":               round(SR, 4),
        "SubtaskSR":        round(SubtaskSR, 4),
        "per_position_SR":  [round(v, 4) for v in per_position_sr],
        "cost_per_episode": int(mean_tok),
        "wall_per_episode_s": round(mean_wall, 1),
        "over_budget":      over_budget,
        "model_pool":       [m["id"] for m in models["models"]],
        # Paper-aligned aggregates flattened at top level for easy citation.
        **paper_aggregate,
        "per_episode":      per_episode_records,
        "wall_total_s":     round(time.time() - t0_total, 1),
    }
    _save_json(out_dir / "score.json", score)

    # Copy NOTES.md and snapshot harness source.
    src_notes = harness_dir / "NOTES.md"
    if src_notes.exists():
        shutil.copy2(src_notes, out_dir / "NOTES.md")
    snapshot = out_dir / "source"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(harness_dir, snapshot,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    print(f"\n[evaluator] DONE  SR={SR:.2f}  SubtaskSR={SubtaskSR:.2f}  "
          f"dist_progress={paper_aggregate['distance_progress_mean']:.2f}  "
          f"static_coll={paper_aggregate['static_collision_mean']:.2f}  "
          f"dyn_coll={paper_aggregate['dynamic_collision_mean']:.2f}  "
          f"cost/ep={int(mean_tok)} tok  iter={iter_idx}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harness", type=str, required=True,
                   help="harness directory (agent.py, or the older harness.py)")
    p.add_argument("--tasks", type=str, required=True,
                   help="JSON list of {map, task} records (a split of config.json, or tests/tasks/simworld_nav/*.json)")
    p.add_argument("--out", type=str, required=True,
                   help="output directory: score.json, traces/, source/, read_audit.json")
    p.add_argument("--setting", type=str, default=os.environ.get("NAV_SETTING") or os.environ.get("SETTING") or "hard",
                   choices=["easy", "hard"], help="easy = SimpleWorld; hard = TrafficWorld (the paper's main setting)")
    p.add_argument("--trace-dir", type=str, default=None, help="where to write the episode traces (default: <out>/traces)")
    p.add_argument("--iter", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
