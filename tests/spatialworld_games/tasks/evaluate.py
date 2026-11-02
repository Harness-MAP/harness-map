#!/usr/bin/env python
"""Run one harness on SpatialWorld 3D-game tasks and write score.json.

    python evaluate.py --harness DIR --tasks TASKS.json --out OUT     # or --split <game>_evolve | snake_eval
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPISODE = HERE / "episode.py"
TASK_SPEC = HERE / "task_spec"                     # the paper's split files
SPATIALWORLD_REPO = Path(os.environ.get(
    "SPATIALWORLD_REPO", str(Path.home() / "workspace" / "SpatialWorld"))).expanduser().resolve()
STEP_KEYS = ("steps_taken", "steps", "total_steps", "num_steps")


def _task(rec: dict):
    """(task_id, game, selector flag, selector) of one record."""
    game = rec["game"]
    is_seed = rec.get("level") is None
    sel = int(rec["seed"] if is_seed else rec["level"])
    task_id = rec.get("task_id") or f"{game}/{'run' if is_seed else 'level'}_{sel:03d}"
    return task_id, game, is_seed, sel


def run_episode(rec: dict, harness: Path, out_dir: Path, *, model: str, api_base: str, api_key: str,
                timeout: float, max_steps: str | None) -> dict:
    """One episode = one episode.py subprocess under this interpreter."""
    task_id, game, is_seed, sel = _task(rec)
    log_dir = out_dir / "episodes" / task_id.replace("/", "__")
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GAME_HARNESS_DIR=str(harness), SPATIALWORLD_REPO=str(SPATIALWORLD_REPO))
    env.setdefault("SPATIALWORLD_MAX_IMAGES", "32")          # the paper's image window
    env.setdefault("ANTICHEAT_GUARD", "1")
    cmd = [sys.executable, str(EPISODE), "--game", game, "--log-dir", str(log_dir),
           "--model", model, "--api-base-url", api_base, "--api-key", api_key]
    cmd += ["--seed", str(sel)] if is_seed else ["--level", str(sel)]
    if max_steps:
        cmd += ["--max-steps", str(int(max_steps))]
    row = {"task_id": task_id, "game": game, "score": 0.0, "valid": True}
    try:
        with open(log_dir / "episode.log", "w") as log:
            subprocess.run(cmd, env=env, cwd=str(SPATIALWORLD_REPO), timeout=timeout,
                           stdout=log, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        row["outcome"] = f"timed out after {int(timeout)} s"
        return row
    rf = log_dir / "eval_result.json"
    if not rf.exists():
        row["outcome"] = "no result (the episode crashed; see episode.log)"
        return row
    r = json.loads(rf.read_text())
    row["score"] = float(r.get("score", 0.0))
    steps = next((r[k] for k in STEP_KEYS if r.get(k) is not None), None)
    if steps is not None:
        row["steps"] = steps
    row["valid"] = bool(r.get("valid", True))
    for k in ("oracle_access", "no_vlm_call", "vlm_calls", "used_rich_harness"):
        if k in r:
            row[k] = r[k]
    tail = f" in {steps} steps" if steps is not None else ""
    row["outcome"] = (("solved" if row["score"] >= 1.0 else f"score {row['score']:.2f}") + tail
                      + ("" if row["valid"] else " (voided by the anti-cheat guard)"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True, help="harness directory (agent.py, or the older harness.py)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tasks", help="JSON list of task records")
    src.add_argument("--split", help=f"a split of {TASK_SPEC}: <game>_evolve | snake_eval | a task example such as maze3d_pro_level_05")
    ap.add_argument("--out", required=True, help="output directory: score.json and episodes/")
    ap.add_argument("--workers", type=int, default=None, help="parallel episodes (env GAME_EVAL_WORKERS)")
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("GAME_EPISODE_TIMEOUT_S", "1800")),
                    help="seconds per episode")
    a = ap.parse_args()

    harness = Path(a.harness).resolve()
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_file = Path(a.tasks) if a.tasks else TASK_SPEC / f"{a.split}.json"
    if not tasks_file.exists():
        raise SystemExit(f"no such task file: {tasks_file}")
    records = json.loads(tasks_file.read_text())
    if not records:
        raise SystemExit(f"{tasks_file} holds no task records")
    model = os.environ.get("GAME_MODEL", "Qwen3-VL-30B-A3B-Instruct")
    api_key = os.environ.get("GAME_API_KEY", "EMPTY")
    bases = [b.strip() for b in os.environ.get("GAME_API_BASE", "http://127.0.0.1:8000/v1").split(",") if b.strip()]
    workers = max(1, int(a.workers or os.environ.get("GAME_EVAL_WORKERS", str(len(bases)))))
    max_steps = os.environ.get("GAME_MAX_STEPS")

    # a fixed server per episode (position modulo the server list), so a re-run deals the same
    # episodes to the same servers
    jobs = [(rec, bases[i % len(bases)]) for i, rec in enumerate(records)]

    def _run(job):
        rec, base = job
        return run_episode(rec, harness, out_dir, model=model, api_base=base, api_key=api_key,
                           timeout=a.timeout, max_steps=max_steps)

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            rows = list(ex.map(_run, jobs))
    else:
        rows = [_run(j) for j in jobs]

    n = len(rows)
    score = {
        "score": round(sum(r["score"] for r in rows) / n, 4) if n else 0.0,
        "n_episodes": n,
        "games": sorted({r["game"] for r in rows}),
        "voided": [r["task_id"] for r in rows if not r["valid"]],
        "harness": str(harness),
        "model": model,
        "tasks_file": str(tasks_file.resolve()),
        "split": a.split,
        "per_episode": rows,
    }
    (out_dir / "score.json").write_text(json.dumps(score, indent=1))
    print(f"[games-eval] score={score['score']:.4f} n={n} voided={len(score['voided'])} -> {out_dir / 'score.json'}")


if __name__ == "__main__":
    main()
