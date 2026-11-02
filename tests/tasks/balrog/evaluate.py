#!/usr/bin/env python
"""Run one harness on BALROG-MiniHack tasks and write score.json.

    python evaluate.py --harness DIR --tasks TASKS.json --out OUT     # or --split evolve | eval_levels | eval_tasks

BALROG is not modified: the harness's build_agent(client_factory, prompt_builder) is injected into its
EvaluatorManager, and episode i of a task plays the level seeded by sha256(f"{seed_base}|{task}|{i}"), the
same for every harness. Environment: BALROG_REPO (~/workspace/BALROG), BALROG_MODEL, BALROG_API_BASE (the
vLLM server), BALROG_WORKERS (8), BALROG_SEED_BASE (0, when the records name none).
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
TASK_SPEC = Path(__file__).resolve().parent / "task_spec"      # the paper's split files
ENTRY_FILES = ("agent.py", "harness.py")          # the harness entry point; the second is its older name
TASK_SUCCESSFUL, DEATH = 2, 1                     # minihack StepStatus: ABORTED -1, RUNNING 0, DEATH 1, SUCCESS 2


# ── common random numbers: one fixed level per (task, episode) for every harness ───────────────
def episode_seed(task: str, episode_idx: int, seed_base: int) -> int:
    """The level seed of episode `episode_idx` of `task`. BALROG's own seed is null (a clock draw) or one scalar
    (every episode the same level), so it is derived here per (task, episode) from a fixed base."""
    return int(hashlib.sha256(f"{seed_base}|{task}|{episode_idx}".encode()).hexdigest()[:8], 16)


def install_crn_seeding(seed_base: int) -> None:
    """Patch Evaluator.run_episode, in this process only, to set the seed through config.envs.env_kwargs.seed."""
    from balrog.evaluator import Evaluator
    if getattr(Evaluator, "_crn_installed", False):
        return
    original = Evaluator.run_episode

    def run_episode(self, task, agent, process_num=None, position=0, episode_idx=0):
        self.config.envs.env_kwargs.seed = episode_seed(task, episode_idx, seed_base)
        return original(self, task, agent, process_num=process_num, position=position, episode_idx=episode_idx)

    Evaluator.run_episode, Evaluator._crn_installed = run_episode, True


# ── the harness as a BALROG agent ───────────────────────────────────────────────────────────────
class HarnessAgentFactory:
    """BALROG's AgentFactory interface over a harness directory; create_agent() runs once per worker process."""

    def __init__(self, config, harness_dir):
        self.config, self.harness_dir, self._mod = config, Path(harness_dir).resolve(), None

    def _load(self):
        if self._mod is None:
            entry = next((self.harness_dir / n for n in ENTRY_FILES if (self.harness_dir / n).exists()), None)
            if entry is None:
                raise FileNotFoundError(f"{self.harness_dir} has no {' or '.join(ENTRY_FILES)}")
            spec = importlib.util.spec_from_file_location(f"_harness_{os.getpid()}_{uuid.uuid4().hex[:8]}", entry)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            if not hasattr(mod, "build_agent"):
                raise AttributeError(f"{entry} does not export build_agent(client_factory, prompt_builder)")
            self._mod = mod
        return self._mod

    def create_agent(self):
        from balrog.client import create_llm_client
        from balrog.prompt_builder import create_prompt_builder
        return self._load().build_agent(create_llm_client(self.config.client), create_prompt_builder(self.config.agent))


def build_config(tasks, episodes: int, workers: int, model: str, api_base: str, out_dir: Path):
    """BALROG's config.yaml with the text-only LLM protocol, the task list and the vLLM client."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(BALROG_REPO / "balrog" / "config" / "config.yaml")
    cfg.agent.type = "custom"                    # unused: the harness is injected
    cfg.agent.max_image_history = 0              # text-only, the BALROG LLM protocol
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
def _se(xs) -> float:
    return statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def collect(out_dir: Path, step_cap: int, tasks, episodes: int) -> dict:
    """BALROG's per-episode JSONs -> the score; an episode BALROG dropped (its worker raised) is a failure."""
    rows = []
    for f in sorted((out_dir / "minihack").glob("*/*.json")):
        if f.name.endswith("_summary.json"):
            continue
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        if "progression" not in d:
            continue
        end = d.get("end_reason")
        end = int(end) if isinstance(end, (int, float)) or str(end).lstrip("-").isdigit() else -99
        rows.append({"episode": f"{f.parent.name}/{f.stem.rsplit('_run_', 1)[-1]}", "task": f.parent.name,
                     "success": 1.0 if end == TASK_SUCCESSFUL else 0.0,          # the true post-rollout state
                     "progression": float(d.get("progression") or 0.0),         # BALROG's metric: reward >= 1, not the search signal
                     "died": 1.0 if end == DEATH else 0.0, "steps": int(d.get("num_steps") or 0),
                     "ret": float(d.get("episode_return") or 0.0), "end_status": end,
                     "total_tokens": int(d.get("input_tokens") or 0) + int(d.get("output_tokens") or 0)})
    have = {r["episode"] for r in rows}
    missing = [f"{t}/{i:02d}" for t in tasks for i in range(episodes) if f"{t}/{i:02d}" not in have]
    rows += [{"episode": k, "task": k.split("/")[0], "success": 0.0, "progression": 0.0, "died": 0.0, "steps": step_cap,
              "ret": 0.0, "end_status": -99, "total_tokens": 0, "crashed": True} for k in missing]
    rows.sort(key=lambda r: r["episode"])
    per_task: dict[str, list] = {}
    for r in rows:
        per_task.setdefault(r["task"], []).append(r["success"])
    mean = lambda k: statistics.mean(r[k] for r in rows) if rows else 0.0       # noqa: E731
    return {"success": mean("success"), "progression": mean("progression"), "survival_rate": 1.0 - mean("died"),
            # solving in fewer steps scores higher
            "step_efficiency": statistics.mean((1.0 - min(r["steps"], step_cap) / step_cap) if r["success"] else 0.0
                                               for r in rows) if rows else 0.0,
            "n_episodes": len(rows), "n_missing": len(missing), "missing": missing, "n_tasks": len(per_task),
            "cost_per_episode": mean("total_tokens"), "success_se": _se([r["success"] for r in rows]),
            "per_task_success": {k: statistics.mean(v) for k, v in sorted(per_task.items())},
            # one success rate per task, in the order of the records (config.json's rubric_labels)
            "per_task_vector": [statistics.mean(per_task[t]) if per_task.get(t) else 0.0 for t in tasks],
            "per_episode": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True, help="harness directory (agent.py, or the older harness.py)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tasks", help="JSON list of {task_id, episodes[, seed_base]} records")
    src.add_argument("--split", help=f"a split of {TASK_SPEC}: evolve | eval_levels | eval_tasks")
    ap.add_argument("--out", required=True, help="output directory: score.json and BALROG's per-episode files")
    ap.add_argument("--episodes", type=int, default=0, help="override the records' episodes per task")
    ap.add_argument("--seed-base", type=int, default=None, help="override the records' seed_base")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("BALROG_WORKERS", "8")))
    ap.add_argument("--model", default=os.environ.get("BALROG_MODEL", "Qwen3-VL-30B-A3B-Instruct"))
    ap.add_argument("--api-base", default=os.environ.get("BALROG_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--step-cap", type=int, default=100, help="minihack max_episode_steps (BALROG's default)")
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
    episodes = a.episodes or max((int(r.get("episodes") or 0) for r in records), default=0) or 5
    bases = {int(r["seed_base"]) for r in records if r.get("seed_base") is not None}
    if len(bases) > 1 and a.seed_base is None:
        raise SystemExit(f"the records name several seed_base values {sorted(bases)}; pass --seed-base")
    seed_base = (a.seed_base if a.seed_base is not None else bases.pop() if bases
                 else int(os.environ.get("BALROG_SEED_BASE", "0")))

    sys.path.insert(0, str(BALROG_REPO))
    from balrog.evaluator import EvaluatorManager
    from balrog.utils import setup_environment
    setup_environment(original_cwd=str(BALROG_REPO))
    install_crn_seeding(seed_base)
    cfg = build_config(tasks, episodes, a.workers, a.model, a.api_base, out)
    factory = HarnessAgentFactory(cfg, a.harness)
    os.chdir(BALROG_REPO)                         # BALROG resolves its data paths against the cwd
    EvaluatorManager(cfg, original_cwd=str(BALROG_REPO), output_dir=str(out)).run(factory)

    score = collect(out, a.step_cap, tasks, episodes)
    score.update({"tasks_file": str(tasks_file.resolve()), "split": a.split, "harness": str(factory.harness_dir),
                  "model": a.model, "episodes_per_task": episodes, "seed_base": seed_base,
                  "seeds": {t: [episode_seed(t, i, seed_base) for i in range(episodes)] for t in tasks}})
    (out / "score.json").write_text(json.dumps(score, indent=1))
    print(f"[balrog-eval] success={score['success']:.4f}±{score['success_se']:.4f} progression={score['progression']:.4f} "
          f"n={score['n_episodes']} crashed={score['n_missing']} -> {out / 'score.json'}")


if __name__ == "__main__":
    main()
