"""Meta-agent roles (tool-using agents briefed by `skills/<role>/SKILL.md`) and `AgenticAdapter`."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from harness_map.main import Genome


# The proposer skill fixes the harness layout for every benchmark; a benchmark supplies the content.
SKILLS_DIR = Path(__file__).resolve().parent / "skills"
DEFAULT_PROPOSER_SKILL = SKILLS_DIR / "proposer"
DEFAULT_DIAGNOSTICIAN_SKILL = SKILLS_DIR / "diagnostician"  # writes diagnosis.md
DEFAULT_JUDGE_SKILL = SKILLS_DIR / "judge"                  # writes judge.json


@dataclass
class MetaAgent:
    """Settings of the `meta_agent` config block; unset skill dirs use the shipped skills."""
    backbone: str = "chat-loop"                 # chat-loop | cli:claude | cli:codex
    skill_dir: Optional[Path] = None            # proposer skill dir
    diag_skill_dir: Optional[Path] = None       # diagnostician skill dir
    judge_skill_dir: Optional[Path] = None      # judge skill dir
    done_sentinel: str = "NOTES.md"             # file the proposer writes to signal completion
    timeout_s: int = 1200                       # wall-clock limit per agent run
    max_turns: int = 120                        # chat-loop turn cap
    cli_cmd: Optional[list[str]] = None         # CLI argv override; "{cwd}" is substituted
    prompt_template: str = (                    # proposer brief with {label} {directive} {sentinel}
        "Run ONE proposer iteration for candidate `{label}`. The harness to improve "
        "is in $OUT_DIR. Follow the SKILL to edit it, then write $OUT_DIR/{sentinel} "
        "(your NOTES) to finish.\n\n{directive}"
    )
    extra_env: dict = field(default_factory=dict)   # added to every agent run's env

    def __post_init__(self):
        self.skill_dir = Path(self.skill_dir) if self.skill_dir is not None else DEFAULT_PROPOSER_SKILL
        self.diag_skill_dir = Path(self.diag_skill_dir) if self.diag_skill_dir is not None else DEFAULT_DIAGNOSTICIAN_SKILL
        self.judge_skill_dir = Path(self.judge_skill_dir) if self.judge_skill_dir is not None else DEFAULT_JUDGE_SKILL

    @classmethod
    def from_cfg(cls, d: dict | None) -> "MetaAgent":
        """Build from a config dict, ignoring keys that are not fields."""
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── chat-loop backbone: an OpenAI-compatible chat API with four tools ───────────

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI                 # lazy: the CLI backbones do not need it
        kw = {}
        base = os.environ.get("OPENAI_BASE_URL")
        if base:
            kw["base_url"] = base
        _CLIENT = OpenAI(**kw)                     # reads OPENAI_API_KEY from the environment
    return _CLIENT


def _model(ma: dict | None = None) -> str:
    return ((ma or {}).get("model") or os.environ.get("OPENAI_META_MODEL") or "gpt-5.6")


def _log(log_path: Path | None, text: str) -> None:
    if log_path is None:
        return
    with open(log_path, "a") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def _create(messages, tools=None, timeout_s: int = 600, ma: dict | None = None):
    ma = ma or {}
    kw = dict(model=_model(ma), messages=messages, timeout=timeout_s,
              max_completion_tokens=int(ma.get("max_tokens")
                                        or os.environ.get("OPENAI_MAX_COMPLETION_TOKENS", "16384")))
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    # reasoning_effort only without tools: some OpenAI-compatible endpoints reject it alongside
    # function tools. OPENAI_REASONING_EFFORT=none disables it.
    eff = os.environ.get("OPENAI_REASONING_EFFORT", "high")
    if not tools and eff and eff.lower() != "none":
        kw["reasoning_effort"] = eff
    return _client().chat.completions.create(**kw)


_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a UTF-8 text file and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer", "description": "cap (default 200000)"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write text to a file, creating parent dirs. Overwrites.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory (ls -la style).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Run a bash command from the repo root with the skill env vars set "
                       "($OUT_DIR, $PARENT_DIR, ...). Use for grep, cp -r, diff, and running "
                       "any validate step. Returns stdout+stderr (truncated).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]


def _tool_read_file(args, env, cwd) -> str:
    p = (cwd / args["path"]) if not os.path.isabs(args["path"]) else Path(args["path"])
    cap = int(args.get("max_bytes") or 200000)
    try:
        data = p.read_text(errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"ERROR reading {p}: {type(e).__name__}: {e}"
    return data[:cap] + (f"\n...[truncated at {cap} of {len(data)}]" if len(data) > cap else "")


def _tool_write_file(args, env, cwd) -> str:
    p = (cwd / args["path"]) if not os.path.isabs(args["path"]) else Path(args["path"])
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"])
    except Exception as e:  # noqa: BLE001
        return f"ERROR writing {p}: {type(e).__name__}: {e}"
    return f"wrote {len(args['content'])} bytes to {p}"


def _tool_list_dir(args, env, cwd) -> str:
    p = (cwd / args["path"]) if not os.path.isabs(args["path"]) else Path(args["path"])
    r = subprocess.run(["ls", "-la", str(p)], capture_output=True, text=True)
    return (r.stdout + r.stderr)[:8000] or f"(empty) {p}"


# run_bash guardrail (chat-loop only): refuse destructive, privileged and install commands, and
# network fetches other than curl to localhost.
_BASH_DENY = ("rm -rf /", "rm -rf ~", "rm -rf /home", "rm -rf /tmp/claude",
              "git push", "git reset --hard", "git checkout", "git clean",
              "sudo ", "pip install", "conda install", "conda remove", "wget ", ":(){")


def _bash_blocked(cmd: str) -> str | None:
    c = " ".join(cmd.split())
    for pat in _BASH_DENY:
        if pat in c:
            return f"BLOCKED by guardrail (pattern {pat!r}) — command NOT run: {c[:160]}"
    if "curl " in c and "http" in c and "127.0.0.1" not in c and "localhost" not in c:
        return f"BLOCKED by guardrail (external curl) — command NOT run: {c[:160]}"
    return None


def _tool_run_bash(args, env, cwd) -> str:
    blocked = _bash_blocked(args.get("command", ""))
    if blocked:
        return blocked
    to = int(os.environ.get("META_BASH_TIMEOUT_S", "300"))
    try:
        r = subprocess.run(["bash", "-lc", args["command"]], cwd=str(cwd), env=env,
                           capture_output=True, text=True, timeout=to)
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {to}s"
    out = r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else "")
    return (out[:12000] + (f"\n...[truncated of {len(out)}]" if len(out) > 12000 else "")) or "(no output)"


_DISPATCH = {"read_file": _tool_read_file, "write_file": _tool_write_file,
             "list_dir": _tool_list_dir, "run_bash": _tool_run_bash}


def _system_prompt(skill_dir: Path, env: dict) -> str:
    skill_md = skill_dir / "SKILL.md"
    body = skill_md.read_text() if skill_md.exists() else f"(SKILL.md missing at {skill_md})"
    passthru = ["WORKSPACE", "REPO_ROOT", "PARENT_TAG", "PARENT_DIR", "ITER_N",
                "OUT_DIR", "SETTING", "SUBSET", "TRACE_DIR", "GAME"]
    kv = "\n".join(f"  {k}={env[k]}" for k in passthru if k in env)
    preamble = (
        "You are running HEADLESS via the OpenAI API (no interactive shell, no CLI agent "
        "wrapper). You have exactly four tools: read_file, write_file, list_dir, run_bash. "
        "run_bash runs from the repo root with the skill env vars set, so `$OUT_DIR`, "
        "`$PARENT_DIR`, etc. resolve. Read text/file contents via the tools; there is no "
        "image display. Follow the skill to completion, then STOP (a short final summary, "
        "no tool call). Do NOT ask questions — act. Your job is done ONLY once the required "
        "output file exists on disk.\n\n"
        f"Runtime env values:\n{kv}\n\n"
        "===== SKILL: follow these instructions =====\n"
    )
    return preamble + body


def run_skill_agent(skill_dir: Path, prompt: str, env: dict,
                    done_check: Callable[[], bool], timeout_s: int,
                    max_turns: int | None = None, log_path: Path | None = None) -> bool:
    """Drive the chat model through a tool-use loop; True iff `done_check()` passes in time."""
    if log_path is not None:
        open(log_path, "w").close()
    repo_root = Path(env.get("REPO_ROOT") or env.get("WORKSPACE") or ".").resolve()
    messages = [{"role": "system", "content": _system_prompt(skill_dir, env)},
                {"role": "user", "content": prompt}]
    max_turns = int(max_turns or os.environ.get("META_AGENT_MAX_TURNS", "120"))
    t0 = time.time()
    _log(log_path, f"[agent chat-loop] model={_model()} skill={skill_dir.name} "
                   f"timeout={timeout_s}s max_turns={max_turns}")
    turn = 0
    for turn in range(1, max_turns + 1):
        if done_check():
            _log(log_path, f"[turn {turn}] done_check passed -> success")
            return True
        if time.time() - t0 > timeout_s:
            _log(log_path, f"[turn {turn}] wall-clock {timeout_s}s exceeded -> stop")
            break
        try:
            resp = _create(messages, tools=_TOOLS,
                           timeout_s=min(600, max(60, timeout_s - int(time.time() - t0))))
        except Exception as e:  # noqa: BLE001 — an API error costs a turn, not the run
            _log(log_path, f"[turn {turn}] API error: {type(e).__name__}: {e}")
            time.sleep(5)
            continue
        msg = resp.choices[0].message
        asst = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            asst["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]
        messages.append(asst)
        if not msg.tool_calls:
            _log(log_path, f"[turn {turn}] model finished (no tool call): {(msg.content or '')[:400]}")
            break
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = _DISPATCH.get(name)
            result = fn(args, env, repo_root) if fn else f"ERROR: unknown tool {name}"
            preview = args.get("path") or args.get("command") or ""
            _log(log_path, f"[turn {turn}] {name}({str(preview)[:120]}) -> {len(str(result))}b")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    ok = done_check()
    _log(log_path, f"[agent chat-loop] finished after {round(time.time()-t0,1)}s, {turn} turns "
                   f"-> {'success' if ok else 'no output'}")
    return ok


# ── CLI backbones: the CLI runs its own agent loop ──────────────────────────────

_CLI_DEFAULTS = {
    "cli:claude": ["claude", "-p", "--permission-mode", "bypassPermissions"],
    # Flags for codex >= 0.149: headless, without approvals or the sandbox (candidate directories
    # are not git repositories), at high reasoning effort — at lower effort the model may stop at
    # a plan without editing.
    "cli:codex":  ["codex", "exec", "-C", "{cwd}",
                   "-c", "model_reasoning_effort=\"high\"",
                   "-c", "model_temperature=1.0",
                   "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"],
}


def run_cli_agent(backbone: str, cwd: Path, skill_dir: Path, prompt: str, env: dict,
                  done_check: Callable[[], bool], timeout_s: int,
                  cli_cmd: list[str] | None = None, log_path: Path | None = None) -> bool:
    """Run a coding CLI in `cwd` with the skill brief and `prompt`; returns `done_check()`."""
    argv = [tok.replace("{cwd}", str(cwd)) for tok in (cli_cmd or _CLI_DEFAULTS.get(backbone, []))]
    if not argv:
        raise ValueError(f"no cli_cmd for backbone {backbone!r}; set meta_agent.cli_cmd")
    _bin_env = {"codex": "CODEX_BIN", "claude": "CLAUDE_BIN"}.get(argv[0])
    if _bin_env and os.environ.get(_bin_env):
        argv[0] = os.environ[_bin_env]
    skill_md = skill_dir / "SKILL.md"
    brief = (_system_prompt(skill_dir, env) + "\n\n===== TASK =====\n" + prompt) \
        if skill_md.exists() else prompt
    _log(log_path, f"[agent {backbone}] argv={argv} cwd={cwd} timeout={timeout_s}s")
    try:
        r = subprocess.run(argv + [brief], cwd=str(cwd), env={**os.environ, **env},
                           capture_output=True, text=True, timeout=timeout_s)
        _log(log_path, (r.stdout or "") + ("\n[stderr]\n" + r.stderr if r.stderr else ""))
    except subprocess.TimeoutExpired:
        _log(log_path, f"[agent {backbone}] timed out after {timeout_s}s")
    return done_check()


# ── roles: proposer, diagnostician, judge ───────────────────────────────────────

# Proposer diversity axes: child i gets axes[i % len(axes)], visible in its label and as $AXIS.
_AXES = ["spatial-grounding", "action-planning", "verification-recovery",
         "efficiency-commitment", "observation-parsing"]


def propose_candidates(parent: Genome, *, work_root: Path, rid: int, n_children: int,
                       meta_agent: MetaAgent | dict | None = None, env: dict | None = None,
                       directive: str | None = None, axes: list[str] | None = None,
                       label_prefix: str = "cand") -> list[Genome]:
    """`n_children` candidates from `parent`, one proposer run each; an unfinished run is a noop."""
    ma = meta_agent if isinstance(meta_agent, MetaAgent) else MetaAgent.from_cfg(meta_agent)
    if ma.skill_dir is None:
        raise ValueError("meta_agent.skill_dir is required (the proposer SKILL.md dir)")
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    axes = axes or _AXES
    base_env = {**(env or {}), **ma.extra_env}
    base_env.setdefault("REPO_ROOT", str(Path(base_env.get("REPO_ROOT", ".")).resolve()))
    if parent.label:
        base_env.setdefault("PARENT_TAG", parent.label)
    base_env.setdefault("PARENT_DIR", str(parent.source_dir))
    base_env["ITER_N"] = str(rid)

    out: list[Genome] = []
    for i in range(n_children):
        axis = axes[i % len(axes)]
        cand = work_root / f"cand_r{rid:03d}_{i}_{axis}"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(parent.source_dir, cand, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sentinel = cand / ma.done_sentinel
        if sentinel.exists():                         # a copied sentinel would pass done_check
            sentinel.unlink()
        cenv = {**base_env, "OUT_DIR": str(cand), "AXIS": axis}
        grad = f"Diagnostic gradient / directive:\n{directive}" if directive else ""
        prompt = ma.prompt_template.format(label=f"{label_prefix}_r{rid}_{i}_{axis}",
                                           directive=grad, sentinel=ma.done_sentinel)
        log_path = cand / "propose.log"
        done = sentinel.exists
        if ma.backbone == "chat-loop":
            run_skill_agent(ma.skill_dir, prompt, cenv, done, ma.timeout_s,
                            max_turns=ma.max_turns, log_path=log_path)
        elif ma.backbone.startswith("cli:"):
            run_cli_agent(ma.backbone, cand, ma.skill_dir, prompt, cenv, done,
                          ma.timeout_s, cli_cmd=ma.cli_cmd, log_path=log_path)
        else:
            raise ValueError(f"unknown meta_agent.backbone {ma.backbone!r}")
        out.append(Genome(source_dir=cand, parent_dir=Path(parent.source_dir),
                          label=f"{label_prefix}_r{rid}_{i}_{axis}"))
    return out


def run_diagnosis(genome_dir: Path, *, meta_agent: MetaAgent | dict | None = None,
                  trace_dir: Path | None = None, fitness_summary: str = "",
                  env: dict | None = None, log_path: Path | None = None) -> bool:
    """Run the diagnostician on an evaluated candidate; True iff it wrote `diagnosis.md`."""
    ma = meta_agent if isinstance(meta_agent, MetaAgent) else MetaAgent.from_cfg(meta_agent)
    genome_dir = Path(genome_dir)
    diag = genome_dir / "diagnosis.md"
    if diag.exists():
        diag.unlink()                                     # so done_check sees only a fresh file
    cenv = {**(env or {}), "OUT_DIR": str(genome_dir)}
    if trace_dir is not None:
        cenv["TRACE_DIR"] = str(trace_dir)
    cenv.setdefault("REPO_ROOT", str(Path(cenv.get("REPO_ROOT", ".")).resolve()))
    prompt = (f"Diagnose the just-evaluated candidate in $OUT_DIR "
              f"(result: {fitness_summary or 'see diagnostics'}). Read its harness + the traces in "
              f"$TRACE_DIR, then write $OUT_DIR/diagnosis.md per the skill.")
    done = diag.exists
    if ma.backbone == "chat-loop":
        return run_skill_agent(ma.diag_skill_dir, prompt, cenv, done, ma.timeout_s,
                               max_turns=ma.max_turns, log_path=log_path)
    if ma.backbone.startswith("cli:"):
        return run_cli_agent(ma.backbone, genome_dir, ma.diag_skill_dir, prompt, cenv, done,
                             ma.timeout_s, cli_cmd=ma.cli_cmd, log_path=log_path)
    raise ValueError(f"unknown meta_agent.backbone {ma.backbone!r}")


_VERDICTS = {"OK", "WARN", "FAIL"}


def run_judge(genome_dir: Path, *, meta_agent: MetaAgent | dict | None = None,
              trace_dir: Path | None = None, env: dict | None = None,
              log_path: Path | None = None) -> str:
    """Run the anti-cheat judge; returns the verdict it wrote (OK | WARN | FAIL) or "MISSING"."""
    ma = meta_agent if isinstance(meta_agent, MetaAgent) else MetaAgent.from_cfg(meta_agent)
    genome_dir = Path(genome_dir)
    vf = genome_dir / "judge.json"
    if vf.exists():
        vf.unlink()
    cenv = {**(env or {}), "OUT_DIR": str(genome_dir)}
    if trace_dir is not None:
        cenv["TRACE_DIR"] = str(trace_dir)
    cenv.setdefault("REPO_ROOT", str(Path(cenv.get("REPO_ROOT", ".")).resolve()))
    prompt = ("Adversarially review the just-evaluated candidate in $OUT_DIR (its agent.py + "
              "tools/*.py) and its traces in $TRACE_DIR for reward-hacking, then write "
              "$OUT_DIR/judge.json per the skill.")
    done = vf.exists
    if ma.backbone == "chat-loop":
        run_skill_agent(ma.judge_skill_dir, prompt, cenv, done, ma.timeout_s,
                        max_turns=ma.max_turns, log_path=log_path)
    elif ma.backbone.startswith("cli:"):
        run_cli_agent(ma.backbone, genome_dir, ma.judge_skill_dir, prompt, cenv, done,
                      ma.timeout_s, cli_cmd=ma.cli_cmd, log_path=log_path)
    else:
        raise ValueError(f"unknown meta_agent.backbone {ma.backbone!r}")
    try:
        v = str(json.loads(vf.read_text()).get("verdict", "")).upper()
        return v if v in _VERDICTS else "MISSING"
    except Exception:  # noqa: BLE001 — a missing or invalid verdict is MISSING, not FAIL
        return "MISSING"


def _fit_summary(fitness) -> str:
    if fitness is None:
        return "see diagnostics"
    p = getattr(fitness, "primary", None)
    return f"primary={p}" + (f", valid={getattr(fitness, 'valid', True)}" if fitness is not None else "")


# ── AgenticAdapter ──────────────────────────────────────────────────────────────


class AgenticAdapter:
    """Adapter base: set `meta_agent` and `work_root`; implement `anchor` and `evaluate[_batch]`."""

    meta_agent = None
    work_root: Path = Path("runs_adaptive")

    def _ma(self) -> MetaAgent:
        return self.meta_agent if isinstance(self.meta_agent, MetaAgent) else MetaAgent.from_cfg(self.meta_agent)

    def proposer_env(self, parent) -> dict:        # override: extra env vars for every agent run
        return {}

    def propose(self, parent, iter_n=0,
                directive=None, n_children=None, **kw):
        """`n_children` candidates under `work_root/prop_r<iter_n>`, `directive` in each brief."""
        rid = int(iter_n or 0)
        wr = Path(self.work_root) / f"prop_r{rid:03d}"
        return propose_candidates(parent, work_root=wr, rid=rid,
                                  n_children=int(n_children or 1), meta_agent=self._ma(),
                                  env=self.proposer_env(parent), directive=directive)

    def trace_dir_for(self, genome):
        """Evaluation-trace directory for the judge and diagnostician; None means code only."""
        return None

    def privileged_keys(self) -> set:
        """Oracle names the harness code must never access (default: `full_state`)."""
        return {"full_state"}

    def judge(self, genome, fitness, iter_n=0) -> str:
        """Anti-cheat judge: sets `fitness.judge`, invalidates on FAIL; returns the verdict."""
        verdict = run_judge(genome.source_dir, meta_agent=self._ma(),
                            trace_dir=self.trace_dir_for(genome), env=self.proposer_env(genome))
        if verdict in _VERDICTS:
            fitness.judge = verdict
            if verdict == "FAIL":
                fitness.valid = False
        return verdict

    def diagnose(self, genome, iter_n=0, *, fitness=None) -> bool:
        """Post-eval diagnosis: write diagnosis.md into the genome's dir for the next proposer."""
        return run_diagnosis(genome.source_dir, meta_agent=self._ma(),
                             trace_dir=self.trace_dir_for(genome),
                             fitness_summary=_fit_summary(fitness), env=self.proposer_env(genome))
