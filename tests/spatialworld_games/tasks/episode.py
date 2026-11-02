#!/usr/bin/env python
"""One 3D-game episode through SpatialWorld's own per-game evaluator, with the harness injected.

    python episode.py --game maze --level 6 --log-dir DIR --harness-dir HARNESS     # snake: --seed instead of --level
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import importlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

# ── locate SpatialWorld and put its game runners, engine and configs on sys.path ───────────────
SPATIALWORLD_REPO = Path(os.environ.get(
    "SPATIALWORLD_REPO", str(Path.home() / "workspace" / "SpatialWorld"))).resolve()
_GAME_SCRIPTS = SPATIALWORLD_REPO / "scripts" / "game"          # the *OpenAIEvaluator modules
_GAME_ENGINE = SPATIALWORLD_REPO / "mllm_base_agent" / "environments" / "game"  # core/games/...
_GAME_CONFIGS = SPATIALWORLD_REPO / "configs" / "game"          # per-game <game>_config modules

for _p in (str(SPATIALWORLD_REPO), str(_GAME_SCRIPTS), str(_GAME_ENGINE), str(_GAME_CONFIGS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ENTRY_FILES = ("agent.py", "harness.py")          # the harness entry point; the second is its older name

# ── per-game reuse map: SpatialWorld module + class + score field ───────────────────────────────
GAMES = {
    "maze":       {"module": "maze_openai_evaluation",       "cls": "MazeOpenAIEvaluator",
                   "score_field": "success",     "kind": "binary", "select": "level"},
    "block3d":    {"module": "block3d_openai_evaluation",    "cls": "Block3DOpenAIEvaluator",
                   "score_field": "won",         "kind": "binary", "select": "level"},
    "maze3d_pro": {"module": "maze3d_pro_openai_evaluation", "cls": "Maze3DProOpenAIEvaluator",
                   "score_field": "success",     "kind": "binary", "select": "level"},
    "rubik":      {"module": "rubik_openai_evaluation",      "cls": "RubikOpenAIEvaluator",
                   "score_field": "cube_solved", "kind": "binary", "select": "level"},
    "snake":      {"module": "snake_openai_evaluation",      "cls": "SnakeOpenAIEvaluator",
                   "score_field": "score",       "kind": "snake",  "select": "seed"},
}
SNAKE_EDGE_LENGTH = 4          # L for snake's scale-normalized score min(score / L, 1)


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _score_of(game: str, result: dict | None) -> float:
    """Normalize one episode's run_evaluation return dict to a scalar in [0, 1]."""
    if not result:
        return 0.0
    spec = GAMES[game]
    val = result.get(spec["score_field"])
    if spec["kind"] == "snake":
        try:
            return min((float(val or 0.0)) / float(SNAKE_EDGE_LENGTH), 1.0)
        except (TypeError, ValueError):
            return 0.0
    return 1.0 if _truthy(val) else 0.0


def _build_evaluator(game: str, mod, model: str, api_base: str, api_key: str, level: int):
    """maze / maze3d_pro take the level at construction; block3d / rubik / snake at run time."""
    cls = getattr(mod, GAMES[game]["cls"])
    if game in ("maze", "maze3d_pro"):
        return cls(api_base_url=api_base, api_key=api_key, model=model, level_number=level)
    return cls(api_base_url=api_base, api_key=api_key, model=model)


def _run_kwargs(game: str, max_steps: int | None, initial_prompt, level: int, log_dir: str) -> dict:
    """Per-game run_evaluation kwargs: block3d wants `level`, rubik `config_name` (a string)."""
    kw = {"initial_prompt": initial_prompt, "log_dir": log_dir}
    if max_steps is not None:
        kw["max_steps"] = int(max_steps)
    if game == "block3d":
        kw["level"] = int(level)
    elif game == "rubik":
        kw["config_name"] = str(level)
    return kw


# ── anti-cheat guards ───────────────────────────────────────────────────────────────────────────
class _GameGuard:
    """`evaluator.game` as the harness sees it: `.screen` passes through, every other read or write is recorded and raises."""
    __slots__ = ("_real", "_sink")
    _ALLOWED = frozenset({"screen"})            # the frame the model is already shown

    def __init__(self, real, sink):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_sink", sink)

    def __getattr__(self, name):
        real = object.__getattribute__(self, "_real")
        if name.startswith("__") or name in _GameGuard._ALLOWED:
            return getattr(real, name)
        object.__getattribute__(self, "_sink").add(name)          # record BEFORE raising
        raise AttributeError(
            f"anti-cheat: engine state 'evaluator.game.{name}' is off-limits; the agent may read "
            f"only the rendered frame it is shown (evaluator.game.screen / frame_data)")

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_sink").add(f"set:{name}")
        raise AttributeError(f"anti-cheat: mutating engine state 'evaluator.game.{name}' is forbidden")


class _GuardedEvaluator:
    """The evaluator as the harness sees it: `.game` is guarded, everything else is forwarded."""
    __slots__ = ("_real", "_guard")

    def __init__(self, real, sink):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_guard", _GameGuard(real.game, sink))

    def __getattr__(self, name):
        if name == "game":
            return object.__getattribute__(self, "_guard")
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):                            # forward writes to the real evaluator
        setattr(object.__getattribute__(self, "_real"), name, value)


_DATA_DIR = str((SPATIALWORLD_REPO / "data").resolve())   # ground-truth tree (scrambles, layouts, targets)


class _DiskGuard:
    """Refuses reads of the level files under SpatialWorld's data/ from harness code: os.open is patched (open(),
    FileIO and importlib all go through it), and a read is refused only when its caller is inside the harness."""
    __slots__ = ("_hdir", "_sink", "_real")

    def __init__(self, harness_dir, sink):
        self._hdir = str(Path(harness_dir).resolve())
        self._sink = sink
        self._real = os.open

    def _by_harness(self):
        import inspect
        for fr in inspect.stack()[2:30]:              # importlib's read chain is deep
            fn = fr.filename
            if self._hdir in fn:
                return True
            if "/mllm_base_agent/" in fn or "/scripts/game/" in fn:   # an engine frame first: allow
                return False
        return False

    def _osopen(self, path, *a, **k):
        try:
            sp = os.fspath(path)
            if isinstance(sp, bytes):
                sp = sp.decode("utf-8", "ignore")
            if isinstance(sp, str) and ("data/game" in sp or _DATA_DIR in sp):
                rp = os.path.realpath(sp)
                if rp.startswith(_DATA_DIR) and self._by_harness():
                    self._sink.add("disk:" + os.path.relpath(rp, _DATA_DIR))
                    raise PermissionError(f"anti-cheat: harness read of ground-truth data '{rp}' is forbidden")
        except PermissionError:
            raise
        except Exception:  # noqa: BLE001 — never break a legitimate open on a path-parse hiccup
            pass
        return self._real(path, *a, **k)

    def __enter__(self):
        os.open = self._osopen
        return self

    def __exit__(self, *x):
        os.open = self._real
        return False


def _entry_file(hdir: Path) -> Path | None:
    return next((hdir / n for n in ENTRY_FILES if (hdir / n).exists()), None)


def _load_harness(entry: Path):
    """Load the harness entry file with its directory importable, so `from tools import ...` is this harness's tools/."""
    hdir = entry.parent
    if str(hdir) not in sys.path:
        sys.path.insert(0, str(hdir))
    for _m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(_m, None)                   # never inherit another harness's tools/
    spec = importlib.util.spec_from_file_location("evolved_harness", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def run_one(*, game: str, level: int, seed: int | None, model: str, api_base: str,
                  api_key: str, max_steps: int | None, harness_dir: str | None,
                  log_dir: str) -> dict:
    """One episode through SpatialWorld's evaluator and its run_evaluation; returns the result dict."""
    if game not in GAMES:
        raise SystemExit(f"episode.py: unknown game {game!r}; known: {sorted(GAMES)}")
    spec = GAMES[game]
    mod = importlib.import_module(spec["module"])

    sel = seed if spec["select"] == "seed" else level
    if game == "snake" and seed is not None:        # the snake adapter places food with random.randint
        random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except Exception:  # noqa: BLE001
            pass

    if max_steps is None:                           # the game's own configured default
        max_steps = getattr(mod, "EVALUATION_CONFIG", {}).get("max_steps")

    evaluator = _build_evaluator(game, mod, model, api_base, api_key, level)

    # HISTORY_WINDOW caps how many prior image+text turns the evaluator feeds the model each step
    # (default: the game's EVALUATION_CONFIG); read at step time, so it applies to this process only.
    _hw = os.environ.get("HISTORY_WINDOW")
    if _hw and hasattr(evaluator, "history_window"):
        try:
            evaluator.history_window = int(_hw)
        except ValueError:
            pass

    initial_prompt = None
    hdir = Path(harness_dir) if harness_dir else None
    if hdir is not None:
        spf = hdir / "system_prompt.txt"
        if spf.exists():
            initial_prompt = spf.read_text()

    used_harness = False
    _cheat_sink: set[str] = set()                      # engine-state / disk-oracle reads
    _vlm_calls = [0]                                    # base-model delegations (0 -> not a harness)
    _guard_on = _truthy(os.environ.get("ANTICHEAT_GUARD", "1"))
    entry = _entry_file(hdir) if hdir is not None else None
    if entry is not None:
        harness = _load_harness(entry)
        if hasattr(harness, "build_agent"):
            evaluator.baseline_step = evaluator.openai_ai_model      # the pristine per-step model call
            if _guard_on:
                _pristine_step = evaluator.baseline_step

                async def _counted_step(fd, __p=_pristine_step):
                    _vlm_calls[0] += 1
                    return await __p(fd)
                evaluator.baseline_step = _counted_step
            _ev = _GuardedEvaluator(evaluator, _cheat_sink) if _guard_on else evaluator
            agent_fn = harness.build_agent(_ev)
            composed = getattr(evaluator, "current_prompt", None)   # a composed system prompt, if any
            if composed:
                initial_prompt = composed
            evaluator.openai_ai_model = agent_fn
            used_harness = True

    _disk = _DiskGuard(str(hdir), _cheat_sink) if (_guard_on and used_harness and hdir is not None) else None
    kwargs = _run_kwargs(game, max_steps, initial_prompt, level, log_dir)
    if _disk is not None:
        with _disk:
            result = await evaluator.run_evaluation(**kwargs)
    else:
        result = await evaluator.run_evaluation(**kwargs)

    score = _score_of(game, result)
    _cheat_raw = None
    _no_vlm = False
    if _cheat_sink:
        _cheat_raw = round(score, 4)
        score = 0.0
    elif _guard_on and used_harness and _vlm_calls[0] == 0:
        _no_vlm = True
        _cheat_raw = round(score, 4)
        score = 0.0
    out = {
        # the raw run_evaluation dict first, so the normalized keys below override it (snake and
        # block3d return their own raw "score")
        **(result or {}),
        "game": game, "task_id": f"{game}/{('run' if spec['select'] == 'seed' else 'level')}_{sel:03d}",
        "select": spec["select"], "level": level, "seed": seed,
        "score": round(score, 4), "score_field": spec["score_field"],
        "used_rich_harness": used_harness, "model": model,
    }
    if _cheat_sink:
        out["oracle_access"] = sorted(_cheat_sink)
        out["cheat_raw_primary"] = _cheat_raw
        out["valid"] = False
        print(f"[anti-cheat] VOIDED: the harness read engine/disk state {sorted(_cheat_sink)} -> score 0.0",
              file=sys.stderr)
    elif _no_vlm:
        out["no_vlm_call"] = True
        out["hand_coded_raw_primary"] = _cheat_raw
        out["valid"] = False
        print("[anti-cheat] VOIDED: the harness made 0 base-model calls (never used baseline_step) -> "
              "not a harness, score 0.0", file=sys.stderr)
    out["vlm_calls"] = _vlm_calls[0]
    stats = out.pop("stats", None)                  # drop the un-serializable engine stats blob
    if isinstance(stats, dict):
        out["total_time"] = stats.get("total_time")
        out["total_frames"] = stats.get("total_frames")
    return out


def _write_outputs(log_dir: str, game: str, out: dict) -> None:
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_result.json").write_text(json.dumps(out, indent=2, default=str))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (d / f"{game}_evaluation_summary_{ts}.json").write_text(json.dumps(out, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, choices=sorted(GAMES))
    ap.add_argument("--level", type=int, default=1, help="level index (maze / block3d / maze3d_pro / rubik)")
    ap.add_argument("--seed", type=int, default=None, help="snake seed (reproducible food layout)")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--model", default=os.environ.get("GAME_MODEL", "Qwen3-VL-30B-A3B-Instruct"))
    ap.add_argument("--api-base-url", default=os.environ.get("GAME_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("GAME_API_KEY", "EMPTY"))
    ap.add_argument("--max-steps", type=int, default=None, help="override the game's default step cap")
    ap.add_argument("--harness-dir", default=os.environ.get("GAME_HARNESS_DIR"),
                    help="harness directory (system_prompt.txt and/or agent.py); env GAME_HARNESS_DIR")
    a = ap.parse_args()

    # run from the SpatialWorld repo root so its relative data paths and symlinks resolve
    os.chdir(str(SPATIALWORLD_REPO))
    out = asyncio.run(run_one(
        game=a.game, level=a.level, seed=a.seed, model=a.model, api_base=a.api_base_url,
        api_key=a.api_key, max_steps=a.max_steps, harness_dir=a.harness_dir, log_dir=a.log_dir))
    _write_outputs(a.log_dir, a.game, out)
    print(json.dumps({"score": out["score"], "score_field": out["score_field"],
                      out["score_field"]: out.get(out["score_field"]),
                      "used_rich_harness": out["used_rich_harness"],
                      "eval_result": str(Path(a.log_dir) / "eval_result.json")}, default=str))


if __name__ == "__main__":
    main()
