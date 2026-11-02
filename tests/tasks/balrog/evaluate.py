#!/usr/bin/env python
"""evaluate.py — run one harness on BALROG-MiniHack task records and write score.json.

The evaluator of the paper's BALROG runs. BALROG itself is never modified: its EvaluatorManager takes
the agent factory as an argument, so the harness is injected through a factory that imports the
harness's entry file and calls build_agent(client_factory, prompt_builder). The one runtime patch
stamps a deterministic seed per (task, episode) so that every candidate plays the same levels and a
re-run reproduces them (common random numbers); BALROG's own seeding path still does the seeding.

    python evaluate.py --harness DIR --tasks TASKS.json --out OUT
    python evaluate.py --harness DIR --split evolve --out OUT          # a split of task_spec/

TASKS.json is a list of {"task_id": "<MiniHack task>", "episodes": n} records, optionally with
"seed_base"; `--split NAME` reads task_spec/NAME.json, the paper's own split files (evolve = the
search set; eval_levels = the same tasks on unseen levels, seed_base 1; eval_tasks = eight unseen
tasks; see task_spec/SPLITS.md). tests/tasks/balrog/*.json are single-task examples cut from evolve.json.

Environment: BALROG_REPO (the BALROG checkout at b7afe79; default ~/workspace/BALROG), BALROG_MODEL
(default Qwen3-VL-30B-A3B-Instruct), BALROG_API_BASE (the vLLM server; default http://127.0.0.1:8000/v1),
BALROG_WORKERS (parallel episodes, default 8), BALROG_SEED_BASE (overrides the records' seed_base).

Scoring: `success` is the environment's terminal flag (end_status == TASK_SUCCESSFUL), the true
post-rollout state. BALROG's `progression` is also emitted, unmodified, for leaderboard comparability,
but it is not the search signal: it tests reward >= 1.0, and a solve whose final step carries the step
penalty scores 0.99 and would count as a failure. A crashed episode counts as a failure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import uuid
from pathlib import Path

BALROG_REPO = Path(os.environ.get("BALROG_REPO", Path.home() / "workspace" / "BALROG")).expanduser()
TASK_SPEC = Path(__file__).resolve().parent / "task_spec"   # the paper's split files
ENTRY_FILES = ("agent.py", "harness.py")          # the harness entry point; the second is its older name

TASK_SUCCESSFUL = 2          # minihack.MiniHack.StepStatus: ABORTED -1, RUNNING 0, DEATH 1, SUCCESS 2
DEATH = 1

SEED_BASE = int(os.environ.get("BALROG_SEED_BASE", "0"))


# ── common random numbers: one fixed level per (task, episode) for every candidate ──────────────
def episode_seed(task: str, episode_idx: int, base: int | None = None) -> int:
    """The seed for (task, episode_idx): deterministic and identical for every candidate.

    BALROG ships `envs.env_kwargs.seed: null`, and run_episode then draws a seed from the clock and
    the PID, so every candidate would face different randomly generated levels and nothing would be
    reproducible. Setting BALROG's single scalar seed is not a fix either: all episodes of a task
    would be the same level. The seed is therefore derived per (task, episode_idx) from a fixed base:
    the same levels for every candidate, five different levels per task."""
    b = SEED_BASE if base is None else base
    h = hashlib.sha256(f"{b}|{task}|{episode_idx}".encode()).hexdigest()
    return int(h[:8], 16)


def install_crn_seeding():
    """Patch Evaluator.run_episode, in this process only, to set the deterministic seed before each
    episode through `config.envs.env_kwargs.seed`, which run_episode uses verbatim when non-null."""
    from balrog.evaluator import Evaluator
    if getattr(Evaluator, "_crn_installed", False):
        return
    original = Evaluator.run_episode

    def run_episode(self, task, agent, process_num=None, position=0, episode_idx=0):
        self.config.envs.env_kwargs.seed = episode_seed(task, episode_idx)
        return original(self, task, agent, process_num=process_num,
                        position=position, episode_idx=episode_idx)

    Evaluator.run_episode = run_episode
    Evaluator._crn_installed = True


# ── the harness as a BALROG agent ───────────────────────────────────────────────────────────────
class HarnessAgentFactory:
    """BALROG's AgentFactory interface, backed by a harness directory. Only create_agent() is called,
    once per worker process (after fork), so the import happens in the child."""

    def __init__(self, config, harness_dir: Path):
        self.config = config
        self.harness_dir = Path(harness_dir).resolve()
        self._mod = None

    def _load(self):
        if self._mod is not None:
            return self._mod
        entry = next((self.harness_dir / n for n in ENTRY_FILES if (self.harness_dir / n).exists()), None)
        if entry is None:
            raise FileNotFoundError(f"{self.harness_dir} has no {' or '.join(ENTRY_FILES)}")
        name = f"_balrog_harness_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(name, entry)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        if not hasattr(mod, "build_agent"):
            raise AttributeError(f"{entry} does not export build_agent(client_factory, prompt_builder)")
        self._mod = mod
        return mod

    def create_agent(self):
        from balrog.client import create_llm_client
        from balrog.prompt_builder import create_prompt_builder
        mod = self._load()
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)
        return mod.build_agent(client_factory, prompt_builder)


# ── BALROG config ───────────────────────────────────────────────────────────────────────────────
def build_config(tasks, episodes, workers, model, api_base, out_dir):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(BALROG_REPO / "balrog" / "config" / "config.yaml")
    cfg.agent.type = "custom"                    # unused (the harness is injected), kept truthful
    cfg.agent.max_image_history = 0              # text-only: the BALROG LLM protocol
    cfg.agent.max_text_history = 16
    cfg.eval.num_workers = int(workers)
    cfg.eval.num_episodes.minihack = int(episodes)
    cfg.eval.output_dir = str(out_dir)
    cfg.eval.save_trajectories = True            # the judge and the diagnostician read these
    cfg.envs.names = "minihack"
    cfg.tasks.minihack_tasks = list(tasks)
    cfg.client.client_name = "vllm"
    cfg.client.model_id = model
    cfg.client.base_url = api_base
    cfg.client.timeout = 300
    return cfg


# ── scoring ─────────────────────────────────────────────────────────────────────────────────────
def _se(xs):
    xs = [float(x) for x in xs]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) / math.sqrt(len(xs))


def collect(out_dir: Path, step_cap: int, tasks, eps: int) -> dict:
    """BALROG's per-episode JSONs -> score.json. A crashed episode counts as a failure: BALROG drops an
    episode whose worker raised, so its JSON is absent; every missing (task, idx) of the requested
    roster is filled in as an unsolved, step-capped episode (end_status -99)."""
    rows = []
    for f in sorted((out_dir / "minihack").glob("*/*.json")):
        if f.name.endswith("_summary.json"):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:                                        # noqa: BLE001
            continue
        if "progression" not in d:
            continue
        end = d.get("end_reason")
        end = int(end) if isinstance(end, (int, float)) or str(end).lstrip("-").isdigit() else -99
        steps = int(d.get("num_steps", 0) or 0)
        rows.append({
            "episode": f"{f.parent.name}/{f.stem.rsplit('_run_', 1)[-1]}",
            "task": f.parent.name,
            "success": 1.0 if end == TASK_SUCCESSFUL else 0.0,       # the true post-rollout state
            "progression": float(d.get("progression", 0.0) or 0.0),  # BALROG's own metric
            "died": 1.0 if end == DEATH else 0.0,
            "steps": steps,
            "ret": float(d.get("episode_return", 0.0) or 0.0),
            "end_status": end,
            "total_tokens": int(d.get("input_tokens", 0) or 0) + int(d.get("output_tokens", 0) or 0),
        })

    missing = []
    have = {r["episode"] for r in rows}
    for t in tasks:
        for i in range(int(eps)):
            key = f"{t}/{i:02d}"
            if key not in have:
                missing.append(key)
                rows.append({"episode": key, "task": t, "success": 0.0, "progression": 0.0,
                             "died": 0.0, "steps": int(step_cap), "ret": 0.0, "end_status": -99,
                             "total_tokens": 0, "crashed": True})
    rows.sort(key=lambda r: r["episode"])
    n = len(rows)
    mean = lambda k: (statistics.mean(r[k] for r in rows) if rows else 0.0)   # noqa: E731
    eff = (statistics.mean((1.0 - min(r["steps"], step_cap) / step_cap) if r["success"] else 0.0
                           for r in rows) if rows else 0.0)   # solving in fewer steps scores higher
    per_task = {}
    for r in rows:
        per_task.setdefault(r["task"], []).append(r["success"])
    return {
        "success": mean("success"),
        "progression": mean("progression"),
        "survival_rate": 1.0 - mean("died"),
        "step_efficiency": eff,
        "n_episodes": n,
        "n_missing": len(missing),                       # crashed episodes, scored as failures
        "missing": missing,
        "n_tasks": len(per_task),
        "cost_per_episode": mean("total_tokens"),
        "success_se": _se([r["success"] for r in rows]),
        "per_task_success": {k: statistics.mean(v) for k, v in sorted(per_task.items())},
        # one success rate per task, in the order of the task records (= rubric_labels in config.json)
        "per_task_vector": [statistics.mean(per_task[t]) if per_task.get(t) else 0.0 for t in tasks],
        "per_episode": rows,
    }


def main():
    global SEED_BASE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True, help="harness directory (agent.py, or the older harness.py)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tasks", help="JSON list of {task_id, episodes[, seed_base]} records")
    src.add_argument("--split", help=f"a split of {TASK_SPEC}: evolve | eval_levels | eval_tasks | headroom | quest")
    ap.add_argument("--out", required=True, help="output directory: score.json and BALROG's per-episode files")
    ap.add_argument("--episodes", type=int, default=0, help="override the records' episodes per task")
    ap.add_argument("--seed-base", type=int, default=None, help="override the records' seed_base")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("BALROG_WORKERS", "8")))
    ap.add_argument("--model", default=os.environ.get("BALROG_MODEL", "Qwen3-VL-30B-A3B-Instruct"))
    ap.add_argument("--api-base", default=os.environ.get("BALROG_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--step-cap", type=int, default=100, help="minihack max_episode_steps (BALROG default)")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks_file = Path(a.tasks) if a.tasks else TASK_SPEC / f"{a.split}.json"
    if not tasks_file.exists():
        raise SystemExit(f"no such task file: {tasks_file}")
    records = json.loads(tasks_file.read_text())
    if not records:
        raise SystemExit(f"{tasks_file} holds no task records")
    tasks = [r.get("task_id") or r["task"] for r in records]
    eps = a.episodes or max((int(r.get("episodes") or 0) for r in records), default=0) or 5
    bases = {int(r["seed_base"]) for r in records if r.get("seed_base") is not None}
    if a.seed_base is not None:
        SEED_BASE = a.seed_base
    elif bases:
        if len(bases) > 1:
            raise SystemExit(f"the records name several seed_base values {sorted(bases)}; pass --seed-base")
        SEED_BASE = bases.pop()

    sys.path.insert(0, str(BALROG_REPO))
    from balrog.evaluator import EvaluatorManager
    from balrog.utils import setup_environment
    setup_environment(original_cwd=str(BALROG_REPO))
    install_crn_seeding()

    cfg = build_config(tasks, eps, a.workers, a.model, a.api_base, out)
    factory = HarnessAgentFactory(cfg, a.harness)
    os.chdir(BALROG_REPO)                         # BALROG resolves its data paths against the CWD
    mgr = EvaluatorManager(cfg, original_cwd=str(BALROG_REPO), output_dir=str(out))
    mgr.run(factory)

    score = collect(out, a.step_cap, tasks, eps)
    score.update({"tasks_file": str(tasks_file.resolve()), "split": a.split, "harness": str(Path(a.harness).resolve()),
                  "model": a.model, "episodes_per_task": eps, "seed_base": SEED_BASE,
                  "seeds": {t: [episode_seed(t, i) for i in range(eps)] for t in tasks}})
    (out / "score.json").write_text(json.dumps(score, indent=1))
    print(f"[balrog-eval] success={score['success']:.4f}±{score['success_se']:.4f} "
          f"progression={score['progression']:.4f} survival={score['survival_rate']:.3f} "
          f"n={score['n_episodes']} crashed={score['n_missing']} -> {out / 'score.json'}")


if __name__ == "__main__":
    main()
