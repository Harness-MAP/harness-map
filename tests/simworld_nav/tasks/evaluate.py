#!/usr/bin/env python
"""Run one harness on SimWorld-MMNav tasks and write score.json.

    python evaluate.py --harness DIR --tasks TASKS.json --out OUT [--setting hard|easy]   # or --subset search_hard_50
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

TASK_SPEC = Path(__file__).resolve().parent / "task_spec" / "search_val_test_split.json"   # the paper's splits
ENTRY_FILES = ("agent.py", "harness.py")          # the harness entry point; the second is its older name
SIMWORLD_ROBOTICS = Path(os.environ.get("SIMWORLD_ROBOTICS_REPO", Path.home() / "workspace" / "SimWorld-Robotics")).expanduser()
BASELINE_DIR = Path(os.environ.get("SIMWORLD_BASELINE_DIR", SIMWORLD_ROBOTICS / "baseline" / "single")).expanduser()
SETTINGS = {"easy": ("simple", "simworld_gym/SimpleWorld"), "hard": ("hard", "simworld_gym/TrafficWorld")}
REWARD = {"human_collision_penalty": -1, "object_collision_penalty": -0.1, "action_penalty": -0.1,
          "success_reward": 10.0, "off_track_penalty": -0.01}
ACTION_NAMES = {0: "Move_Forward", 1: "Rotate_Left", 2: "Rotate_Right", 3: "Move_Left", 4: "Move_Right",
                5: "Subtask_Completed", -1: "Auto_Orient"}                  # the environment's action space


def _load(path) -> dict:
    return json.loads(Path(path).read_text())


def _plain(o):
    """JSON default: numpy arrays and scalars become lists and numbers, anything else its text."""
    return o.tolist() if hasattr(o, "tolist") else str(o)


def _dump(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=_plain))


def _jsonl(path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_plain) + "\n")


# ── a finite wait on every simulator call ──────────────────────────────────────────────────────
def bound_unrealcv_rpc(timeout_s: float) -> None:
    """Make unrealcv.Client.request wait at most `timeout_s` for a reply: upstream waits forever, so one
    dropped reply (seen under GPU contention) would hang the whole evaluation."""
    import queue
    import unrealcv
    Client = unrealcv.Client
    if getattr(Client, "_bounded", False):
        return

    def request(self, message, timeout=5):
        if timeout < 0:                                                  # the asynchronous path is untouched
            (self.request_batch_async if isinstance(message, list) else self.request_async)(message)
            return True
        if isinstance(message, list):
            return self.request_batch(message)
        if not isinstance(message, bytes):
            message = message.encode("utf-8")
        raw = b"%d:%s" % (self.send_message_id, message)
        if not self.send(raw):
            raise ConnectionError("failed to send: the socket is closed")
        self.send_message_id += 1
        self.recv_num_q.put(-1)
        try:
            return self.recv_data_q.get(timeout=max(float(timeout), timeout_s))
        except queue.Empty:
            raise TimeoutError(f"unrealcv: no reply within {timeout_s:.0f} s to {raw[:60]!r}") from None

    Client.request, Client._bounded = request, True


# ── the task as the harness sees it, and the ground truth the checker keeps ────────────────────
def harness_task_spec(tdir: Path, map_id: str, task: str) -> dict:
    """Instructions, visual hints, step cap, arrival radius and starting compass; no ground truth."""
    cfg = _load(tdir / "task_config.json")
    hints = cfg.get("robot_view_images") or []
    subtasks = [{"instruction": ins, "visual_hint_path": str((tdir / hints[i]).resolve()) if i < len(hints) else None}
                for i, ins in enumerate(cfg.get("instruction") or [])]
    crit = cfg.get("criteria") or {}
    yaw = ((cfg.get("spawning_point") or {}).get("orientation") or {}).get("yaw", 0.0)
    return {"subtasks": subtasks, "max_steps": int(crit.get("max_steps", 200)),
            "dist_threshold_cm": float(crit.get("dist_threshold", 2000)), "compass_at_start": float(yaw),
            "_map_id": map_id, "_task_id": task}


def ground_truth(tdir: Path) -> dict:
    cfg = _load(tdir / "task_config.json")
    return {k: cfg.get(k) for k in ("destination", "ideal_path", "evaluation_of_subtask", "spawning_point", "criteria")}


# ── read audit: which environment fields the harness touched ───────────────────────────────────
ALLOWED = ("obs.rgb", "obs.depth_map", "obs.depth", "obs.object_mask", "info.moving_distance", "info.landmark_images",
           "info.agent", "info.agent.agent_rotation", "info.agent.agent_orientation", "info.success",
           "task_spec.subtasks", "task_spec.max_steps", "task_spec.compass_at_start", "task_spec.dist_threshold_cm",
           "task_spec._initial_obs", "task_spec._initial_info", "task_spec._map_id", "task_spec._task_id")
FORBIDDEN = ("info.agent.agent_location", "info.target", "info.agent_collision", "info.buildings_info",
             "info.current_instruction")                                    # the answer key
READS: dict[str, int] = {}


def _under(path: str, roots) -> bool:
    return any(path == r or path.startswith(r + ".") for r in roots)


class Audited(Mapping):
    """A read-only view of a dict that counts every key the harness reads, nested dicts included."""
    __slots__ = ("_d", "_p")

    def __init__(self, d: dict, prefix: str):
        self._d, self._p = d, prefix

    def _read(self, key):
        path = f"{self._p}.{key}"
        READS[path] = READS.get(path, 0) + 1
        v = self._d[key]
        return Audited(v, path) if isinstance(v, dict) else v

    def __getitem__(self, key):
        return self._read(key)

    def __getattr__(self, name):                     # info.agent.x style reads
        if name.startswith("_") or name not in self._d:
            raise AttributeError(name)
        return self._read(name)

    def __iter__(self):
        READS[self._p + ".<keys>"] = READS.get(self._p + ".<keys>", 0) + 1
        return iter(self._d)

    def __len__(self):
        return len(self._d)


class AuditedEnv:
    """The gym environment with the obs and info of reset() and step() wrapped in Audited views."""

    def __init__(self, env):
        self._env = env

    def reset(self, *a, **k):
        obs, info = self._env.reset(*a, **k)
        return _audited(obs, "obs"), _audited(info, "info")

    def step(self, *a, **k):
        obs, reward, term, trunc, info = self._env.step(*a, **k)
        return _audited(obs, "obs"), reward, term, trunc, _audited(info, "info")

    def __getattr__(self, name):
        return getattr(self._env, name)


def _audited(d, prefix):
    return Audited(d, prefix) if isinstance(d, dict) else d


def read_audit() -> dict:
    paths = sorted(READS)
    forbidden = [p for p in paths if _under(p, FORBIDDEN)]
    undeclared = [p for p in paths if not _under(p, ALLOWED) and not _under(p, FORBIDDEN)]
    return {"verdict": "FAIL" if forbidden else "WARN" if undeclared else "PASS",
            "forbidden_reads": forbidden, "undeclared_reads": undeclared, "read_counts": dict(READS)}


# ── the harness ─────────────────────────────────────────────────────────────────────────────────
def import_harness(hdir: Path):
    """The module exporting run_episode: the directory as a package when it has an __init__.py, else its entry
    file loaded with the directory on sys.path so that its modules/ resolve."""
    hdir = hdir.resolve()
    sys.path.insert(0, str(hdir.parent))
    if (hdir / "__init__.py").exists():
        sys.modules.pop(hdir.name, None)
        mod = __import__(hdir.name)
        if hasattr(mod, "run_episode"):
            return mod
    entry = next((hdir / n for n in ENTRY_FILES if (hdir / n).exists()), None)
    if entry is None:
        raise FileNotFoundError(f"{hdir} has no {' or '.join(ENTRY_FILES)}")
    sys.path.insert(0, str(hdir))
    spec = importlib.util.spec_from_file_location(f"_harness_{hdir.name}", entry)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def action_name(names, a) -> str:
    """The name of action `a` from the harness's ACTION_NAMES (a list or a dict), else the environment's."""
    try:
        a = int(a)
    except (TypeError, ValueError):
        return "?"
    own = (names.get(a) if isinstance(names, dict)
           else names[a] if isinstance(names, (list, tuple)) and 0 <= a < len(names) else None)
    return own or ACTION_NAMES.get(a, "?")


# ── the base model: every call bounded, counted per step and logged with its frames ────────────
class ModelCalls:
    """Wraps openai's chat.completions.create for the whole process: a timeout and retries per call, a cap on
    the calls between two env.step() calls (a looping harness fails its episode instead of hanging the
    search), and a record of every call with the frames it carried, appended to the episode's trace."""

    def __init__(self, timeout_s: float, retries: int, per_step_cap: int):
        from openai.resources.chat.completions import Completions
        self.timeout_s, self.retries, self.cap = timeout_s, retries, per_step_cap
        self.ep_dir, self.step, self.in_step, self.records = None, 0, 0, []
        original = Completions.create
        Completions.create = lambda completions, *a, **kw: self._call(original, completions, *a, **kw)

    def episode(self, ep_dir: Path) -> None:
        self.ep_dir, self.step, self.in_step, self.records = ep_dir, 0, 0, []

    def _call(self, original, completions, *a, **kw):
        self.in_step += 1
        if self.in_step > self.cap:
            raise RuntimeError(f"{self.cap} base-model calls without an env.step(): the harness is looping")
        kw.setdefault("timeout", self.timeout_s)
        t0 = time.time()
        for attempt in range(1, self.retries + 1):
            try:
                result = original(completions, *a, **kw)
                break
            except Exception as e:                                           # noqa: BLE001
                if attempt == self.retries:
                    raise
                print(f"[evaluator] base-model call failed ({type(e).__name__}); retry {attempt}", flush=True)
                time.sleep(2)
        try:
            self._log(kw.get("messages") or [], kw.get("model"), result, time.time() - t0)
        except Exception:                                                    # noqa: BLE001 — logging never fails an episode
            traceback.print_exc()
        return result

    def _log(self, messages, model, result, latency) -> None:
        if self.ep_dir is None:
            return
        text, images = [], []
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            for part in ([content] if isinstance(content, str) else content or []):
                if isinstance(part, str):
                    text.append(f"[{m.get('role')}] {part}")
                elif part.get("type") in ("text", "input_text"):
                    text.append(f"[{m.get('role')}] {part.get('text', '')}")
                elif part.get("type") in ("image_url", "input_image"):
                    url = part.get("image_url")
                    url = url.get("url") if isinstance(url, dict) else url
                    if isinstance(url, str) and url.startswith("data:image"):
                        images.append(self._frame(url))
        choice = result.choices[0] if getattr(result, "choices", None) else None
        msg = choice.message if choice else None
        usage = getattr(result, "usage", None)
        finish = getattr(choice, "finish_reason", None)
        rec = {"call_idx": len(self.records), "step": self.step, "kind": "perceive" if images else "plan",
               "model": model, "latency_s": round(latency, 2), "images": images, "prompt": "\n".join(text),
               "reasoning": (getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or "") if msg else "",
               "response": (msg.content or "") if msg else "",
               "input_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None),
               "finish_reason": finish, "truncated": finish == "length"}
        self.records.append(rec)
        with open(self.ep_dir / "vlm_trajectory.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _frame(self, data_url: str) -> str:
        """Save a base64 image once under frames/ and return its path relative to the episode."""
        raw = base64.b64decode(data_url.split(",", 1)[1])
        rel = f"frames/{hashlib.md5(raw).hexdigest()[:12]}.png"
        p = self.ep_dir / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
        return rel


# ── the simulator ───────────────────────────────────────────────────────────────────────────────
def make_env(setting: str, port: int, log_dir: Path):
    import gym
    import simworld_gym  # noqa: F401  (registers the environment ids)
    return gym.make(SETTINGS[setting][1], port=port, resolution=(720, 600), render_mode="rgb_array",
                    observation_type="all", record_video=False, log_dir=str(log_dir), reward_setting=REWARD,
                    disable_env_checker=True)


def wait_for_port(port: int, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2.0).close()
            return True
        except OSError:
            time.sleep(2.0)
    return False


def reset(env, options: dict, world_json: str, setting: str, port: int, log_dir: Path, attempts: int = 3):
    """env.reset with recovery: after a failure the environment is closed, the simulator's port awaited and the
    environment remade on the task's map. Returns (env, obs, info), with obs None when every attempt failed."""
    for attempt in range(1, attempts + 1):
        try:
            obs, info = env.reset(options=options)
            return env, obs, info
        except Exception as e:                                               # noqa: BLE001
            print(f"[evaluator] reset failed, attempt {attempt}/{attempts}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            if attempt == attempts:
                return env, None, None
            try:
                env.close()
            except Exception:                                                # noqa: BLE001
                pass
            if not wait_for_port(port, 120.0):
                return env, None, None
            env = make_env(setting, port, log_dir)
            options["world_json"] = world_json                               # a fresh environment loads the map again


# ── one episode ─────────────────────────────────────────────────────────────────────────────────
def distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 if len(a) >= 2 and len(b) >= 2 else 0.0


def play(env, harness, clients: dict, calls: ModelCalls, tdir: Path, map_id: str, task: str, ep_dir: Path,
         obs, info) -> dict:
    """Run the harness on the reset environment, write the episode's trace and return its record."""
    spec = harness_task_spec(tdir, map_id, task)
    spec["_initial_obs"], spec["_initial_info"] = obs, info
    n_sub = len(spec["subtasks"])
    per_step, last = [], {"info": info}
    real_step = env.step

    def step(action):
        """env.step plus the per-step record the metrics and the renderer need."""
        calls.in_step = 0
        out = real_step(action)
        info2 = out[4]
        last["info"] = info2
        ac, ag, tg = info2.get("agent_collision") or {}, info2.get("agent") or {}, info2.get("target") or {}
        per_step.append({"action": int(action), "agent_location": list(ag.get("agent_location", [])),
                         "target_location": list(tg.get("target_location", [])),
                         "moving_distance": float(info2.get("moving_distance", 0.0)),
                         **{k: int(ac.get(k, 0)) for k in ("human", "object", "building", "vehicle")},
                         "red_light_violation": int(info2.get("total_red_light_violation", 0))})
        calls.step += 1                                  # calls logged from here on plan the next action
        return out

    env.step = step
    calls.episode(ep_dir)
    t0 = time.time()
    try:
        result = harness.run_episode(AuditedEnv(env), Audited(spec, "task_spec"), clients, trace_dir=str(ep_dir))
    except Exception as e:                                                   # noqa: BLE001 — a crash is a failed episode
        traceback.print_exc()
        result = (getattr(harness, "EpisodeResult", None) or SimpleNamespace)(
            success=False, subtask_success=[False] * n_sub, steps=0, actions=[], llm_calls=[], notes=f"crash: {e}")
    finally:
        env.step = real_step

    info0, info1 = info or {}, last["info"] or {}
    target = list((info0.get("target") or {}).get("target_location", []))
    d0 = distance(list((info0.get("agent") or {}).get("agent_location", [])), target)
    d1 = distance(list((info1.get("agent") or {}).get("agent_location", [])), target)
    ac = info1.get("agent_collision") or {}
    llm_calls = [c for c in (result.llm_calls or []) if isinstance(c, dict)]
    sub = [bool(b) for b in result.subtask_success]
    rec = {"episode": ep_dir.name, "success": bool(result.success), "subtask_success": sub, "n_subtasks": n_sub,
           "subtask_sr": sum(sub) / n_sub if n_sub else 0.0, "steps": int(result.steps),
           "input_tokens": sum(c.get("input_tokens") or 0 for c in llm_calls),
           "output_tokens": sum(c.get("output_tokens") or 0 for c in llm_calls),
           "wall_s": round(time.time() - t0, 1), "notes": result.notes,
           "static_collision": int(ac.get("object", 0)) + int(ac.get("building", 0)),
           "dynamic_collision": int(ac.get("human", 0)) + int(ac.get("vehicle", 0)),
           "red_light_violation": int(info1.get("total_red_light_violation", 0)),
           "moving_distance": float(info1.get("moving_distance", 0.0)),
           "initial_dist_to_target": round(d0, 1), "final_dist_to_target": round(d1, 1),
           "distance_progress": round(max(0.0, min(1.0, 1.0 - d1 / d0)), 4) if d0 > 1e-3 else 0.0}
    rec["total_tokens"] = rec["input_tokens"] + rec["output_tokens"]
    names = getattr(harness, "ACTION_NAMES", None) or {}
    _jsonl(ep_dir / "env_step_log.jsonl", per_step)
    _jsonl(ep_dir / "actions.jsonl", [{"step": i, "action": int(a), "action_name": action_name(names, a)}
                                      for i, a in enumerate(result.actions)])
    _jsonl(ep_dir / "llm_calls.jsonl", llm_calls)
    _jsonl(ep_dir / "observations.jsonl", getattr(result, "observations", None) or [])
    _dump(ep_dir / "verdict.json", {k: rec[k] for k in ("episode", "success", "subtask_success", "n_subtasks",
                                                        "subtask_sr", "steps", "wall_s", "notes")})
    _dump(ep_dir / "gt.json", ground_truth(tdir))                            # for the diagnosis, never the harness
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True, help="harness directory (agent.py, or the older harness.py)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tasks", help="JSON list of {map, task} records (task_spec/map_20_10_task_dist_21_0_1.json is the task example)")
    src.add_argument("--subset", help=f"a subset of {TASK_SPEC.name}: search_hard_50 | val_hard_10 | search_easy_50 | val_easy_10")
    ap.add_argument("--out", required=True, help="output directory: score.json, traces/, read_audit.json")
    ap.add_argument("--setting", default=os.environ.get("NAV_SETTING") or os.environ.get("SETTING") or "hard",
                    choices=sorted(SETTINGS), help="hard = TrafficWorld (the paper's main setting); easy = SimpleWorld")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    (out / "traces").mkdir(parents=True, exist_ok=True)
    items = _load(a.tasks) if a.tasks else _load(TASK_SPEC)[a.subset]
    if isinstance(items, dict):
        items = items.get("items") or [{"map": m, "task": t} for m in items["maps"] for t in items["tasks_per_map"]]
    if not items:
        raise SystemExit("no task records")

    bound_unrealcv_rpc(float(os.environ.get("UNREALCV_RPC_TIMEOUT_S", "15")))
    if BASELINE_DIR.is_dir():
        sys.path.insert(0, str(BASELINE_DIR))                                # the harnesses' helper imports
    import openai
    import simworld_gym
    root = Path(simworld_gym.__file__).resolve().parent / "envs" / "setting"  # where the gym reads task data
    harness = import_harness(Path(a.harness))
    model = os.environ.get("NAV_MODEL", "Qwen3-VL-30B-A3B-Instruct")
    clients = {model: openai.OpenAI(base_url=os.environ.get("NAV_API_BASE", "http://127.0.0.1:8000/v1"), api_key="EMPTY")}
    calls = ModelCalls(float(os.environ.get("VLM_CALL_TIMEOUT_S", "180")), int(os.environ.get("VLM_CALL_RETRIES", "3")),
                       int(os.environ.get("MAX_VLM_CALLS_PER_STEP", "30")))
    port = int(os.environ.get("UE_PORT", "9000"))
    setting_dir = SETTINGS[a.setting][0]
    env = make_env(a.setting, port, out / "env_logs")

    rows, spawned = [], None
    for item in items:
        map_id, task = str(item["map"]), str(item["task"])
        rel = f"single_agent_world/{setting_dir}/map_road_{map_id}/{task}"
        tdir = root / rel
        if not (tdir / "task_config.json").exists() or (a.setting == "hard" and not (tdir / "traffic.json").exists()):
            print(f"[evaluator] skip ep_{map_id}_{task}: no task data under {tdir}", flush=True)
            continue
        options = {"task_path": rel, "agent_json": f"{rel}/task_config.json"}
        if a.setting == "hard":
            options["traffic_json"] = _load(tdir / "traffic.json")           # TrafficWorld.reset indexes it as a dict
        if map_id != spawned:
            options["world_json"] = f"{rel}/progen_world.json"               # a new map: load the level
        env, obs, info = reset(env, options, f"{rel}/progen_world.json", a.setting, port, out / "env_logs")
        if obs is None:
            print("[evaluator] the simulator did not come back; stopping", flush=True)
            break
        spawned = map_id
        ep_dir = out / "traces" / f"ep_{map_id}_{task}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        rec = play(env, harness, clients, calls, tdir, map_id, task, ep_dir, obs, info)
        rows.append(rec)
        print(f"[evaluator] {rec['episode']}: success={rec['success']} subtask_sr={rec['subtask_sr']:.2f} "
              f"steps={rec['steps']} tokens={rec['total_tokens']}", flush=True)
    env.close()

    n = len(rows)
    mean = lambda k: sum(float(r[k]) for r in rows) / n if n else 0.0        # noqa: E731
    k_max = max((len(r["subtask_success"]) for r in rows), default=0)
    per_position = [[r["subtask_success"][k] for r in rows if len(r["subtask_success"]) > k] for k in range(k_max)]
    audit = read_audit()
    _dump(out / "read_audit.json", audit)
    score = {"SubtaskSR": round(mean("subtask_sr"), 4), "SR": round(mean("success"), 4),
             "per_position_SR": [round(sum(v) / len(v), 4) for v in per_position if v],
             "n_episodes": n, "cost_per_episode": int(mean("total_tokens")), "wall_per_episode_s": round(mean("wall_s"), 1),
             "distance_progress_mean": round(mean("distance_progress"), 4),
             "moving_distance_mean": round(mean("moving_distance"), 1),
             "static_collision_mean": round(mean("static_collision"), 3),
             "dynamic_collision_mean": round(mean("dynamic_collision"), 3),
             "red_light_violation_mean": round(mean("red_light_violation"), 3),
             "static_collision_rate": round(sum(1 for r in rows if r["static_collision"]) / n, 3) if n else 0.0,
             "dynamic_collision_rate": round(sum(1 for r in rows if r["dynamic_collision"]) / n, 3) if n else 0.0,
             "setting": a.setting, "tasks": str(Path(a.tasks).resolve()) if a.tasks else str(TASK_SPEC), "subset": a.subset,
             "harness": str(Path(a.harness).resolve()), "model": model, "read_audit": audit["verdict"], "per_episode": rows}
    _dump(out / "score.json", score)
    print(f"[nav-eval] SubtaskSR={score['SubtaskSR']:.3f} SR={score['SR']:.3f} n={n} audit={audit['verdict']} "
          f"-> {out / 'score.json'}")


if __name__ == "__main__":
    main()
